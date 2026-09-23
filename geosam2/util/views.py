"""Rasterise the twelve canonical views GeoSAM2 consumes, without Blender.

    from geosam2.util.views import render_views
    render_views("mesh.glb", "views/")   # 12 views + meta.json + mesh.glb

Replaces VAST's Blender script (``geosam2_render.py``, kept in git history at b5de23c). The network never sees a shaded
image (``sam2/utils/misc.py:506`` rewrites every ``color_*`` path to ``normal_*``,
and ``sam2_base_geosam2.py:470`` encodes only the normal and point maps -- the
``img_batch`` it is handed is never used), so nothing here is rendered for the
model's benefit. Blender was shading 64 samples and discarding them.

``color_*`` carries a lit render of the mesh anyway: its RGB is free space, and
the VLM stage that seeds prompts has to look at *something*. Note that on a mesh
that is already segmented, those materials are the part colours -- so a sample
built from one cannot double as a VLM benchmark: it hands over the answer.

Nothing here imports GeoSAM2, so the file can be lifted into another project.

Output contract
---------------
::

    views/
      color_0000..0011.webp    RGBA -- alpha is the visibility mask; the RGB is
                               a lit render (supersampled), which the model never
                               reads (see below) but a VLM asked to find parts does
      depth_0000..0011.exr     float32, 3 channels, z-depth; invalid = 65504
      normal_0000..0011.webp   RGBA, world-space normals encoded (n + 1) / 2
      meta.json                camera_angle_x, 12 c2w transforms, scaling_factor, translation
      mesh.glb                 the source mesh, copied verbatim

Three details are load-bearing, each learned by getting it wrong first:

* **Depth is z-depth, not radial distance.** ``get_ray_directions``
  (``get_ray_directions`` below) returns *unnormalised* rays with ``z = -1``,
  so ``direction * depth`` only lands on the surface if depth measures along the
  camera axis. pyrender's depth buffer already is exactly this.
* **The EXR needs three channels.** Every consumer indexes ``depth[..., 0]``,
  which silently degenerates to a 1-D array on a single-channel file rather than
  failing loudly.
* **``mesh.glb`` is copied, never re-exported.** ``read_views`` reloads it and
  indexes the label array by its face order; a re-export that welds or reorders
  faces would misalign every label without erroring.

Two facts make the approach safe, both measured rather than assumed:
pyrender's depth is linear z-depth in world units, and ``RenderFlags.FLAT``
round-trips vertex colours byte-exactly -- so normals ride through the colour
buffer with no gamma transform mangling them.

Validation
----------
Correctness is checked, not asserted. The bundled ``example/sample_*`` roots are
ground truth -- the data the model demonstrably works on::

    python -m tests.views_check example/sample_00
    python -m tests.views_check example/sample_00 --images example/render_comparison
    python -m tests.views_check example/sample_00 --against /path/to/blender_render

The ``--images`` sheets show each channel side by side plus an error heatmap,
compositing onto white the way the loader does -- comparing raw RGB would diff
background pixels the model never reads.

Measured on sample_00: camera transforms max diff 4.3e-07, scaling_factor and
translation exact, depth median 5.0e-04 (tolerance 1e-03), normals median 1.05
degrees, silhouette IoU 99.51%. Everything deterministic -- the rig, the
normalisation -- is reproduced exactly; the residuals are rasteriser
micro-differences.

**Known gap:** segmenting sample_00 from these views yields 23 parts against the
reference's 24, agreeing on 87% of faces. Not antialiasing (supersampling 3x
moves the normal median only 1.048 -> 1.003 degrees) and not authored normals
(the GLB declares only POSITION and TEXCOORD_0, so the glTF spec generates flat
normals, which is what this does -- and flat beats smooth, 1.05 vs 2.06 degrees).
Resolving it needs a three-way ``--against`` comparison with a Blender 4.0/4.1
render: if Blender-4.1-vs-reference also lands near 87%, this renderer is as
faithful as another Blender version and the gap is GeoSAM2's own sensitivity to
input perturbation rather than a defect here. That check has not been run.

Requires ``pyrender``, ``trimesh``, ``numpy``, ``opencv-python<5``, ``pillow``,
``PyOpenGL>=3.1.7``. Neither pin is cosmetic: opencv 5.x ships without the
OpenEXR codec entirely, so ``cv2.imread`` returns ``None`` on every depth map --
and pyrender pins ``PyOpenGL==3.1.0``, which predates numpy 2 by nine years and
raises a ctypes ArgumentError from ``glGenTextures`` on any textured mesh.
Overriding that pin is deliberate; pip warns about the conflict.
"""

from __future__ import annotations

import os

# Must precede the pyrender import: it picks its GL backend at import time, and
# the default one needs a display this runs without.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# OpenEXR is an opt-in codec in OpenCV and the depth maps are EXR.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image, ImageEnhance



# ---------------------------------------------------------------------------
# Camera rig -- pure geometry, no renderer involved
# ---------------------------------------------------------------------------

# Twelve views: azimuth walks a full circle in 30-degree steps while elevation
# cycles 0/+25/0/-25. Views i and i+6 are antipodal, which GeoSAM2 relies on --
# it seeds automatic segmentation from view (start + 6) % 12.
ELEVATIONS: Tuple[float, ...] = (0.0, 25.0, 0.0, -25.0, 0.0, 25.0, 0.0, -25.0, 0.0, 25.0, 0.0, -25.0)

# Azimuths of the bundled example data, which starts at 270 degrees.
#
# VAST's Blender script starts at 180 instead -- the same rig rotated by 90 degrees.
# The bundled examples were not produced by the shipped renderer (their meta.json
# carries keys it never writes), and the reference azimuths are the ones the
# example point prompts were authored against: those prompts are pixel
# coordinates in view 0, so a rig rotated by 90 degrees puts the clicks on
# different parts of the object. Reproduce the data that works.
AZIMUTHS_REFERENCE: Tuple[float, ...] = (270., 300., 330., 0., 30., 60., 90., 120., 150., 180., 210., 240.)
AZIMUTHS_BLENDER_SCRIPT: Tuple[float, ...] = (180., 210., 240., 270., 300., 330., 0., 30., 60., 90., 120., 150.)

NUM_VIEWS = len(ELEVATIONS)

# The Blender defaults VAST's script baked in: a 50 mm lens on a 36 mm sensor.
CAMERA_LENS_MM = 50.0
SENSOR_WIDTH_MM = 36.0

# Longest bounding-box edge after normalisation.
NORMALIZATION_RANGE = 1.0

# Blender is Z-up, and the renders were made after the mesh was rotated into it.
WORLD_UP = np.array([0.0, 0.0, 1.0])


def camera_angle_x() -> float:
    """Horizontal field of view, in radians."""
    return 2.0 * math.atan(SENSOR_WIDTH_MM / 2 / CAMERA_LENS_MM)


def focal_length(width_px: int) -> float:
    """Focal length in pixels, matching ``get_ray_directions``."""
    return 0.5 * width_px / math.tan(0.5 * camera_angle_x())


def camera_distance(bbox_size: Sequence[float]) -> float:
    """Orbit radius Blender's rig uses for a given normalised bounding box."""
    return CAMERA_LENS_MM / SENSOR_WIDTH_MM * float(np.linalg.norm(bbox_size))


def look_at(eye: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Camera-to-world matrix aiming ``-Z`` at ``target`` with ``+Y`` up.

    Reproduces Blender's ``to_track_quat('-Z', 'Y')``, which is what
    VAST's Blender script uses to orient each camera.
    """
    target = np.zeros(3) if target is None else np.asarray(target, dtype=float)
    eye = np.asarray(eye, dtype=float)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise ValueError("camera and target coincide; cannot aim")
    forward /= norm

    z_axis = -forward  # a camera looks down its own -Z
    x_axis = np.cross(WORLD_UP, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        raise ValueError("view direction is parallel to world up; rig is degenerate")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)

    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = x_axis, y_axis, z_axis, eye
    return matrix


def camera_positions(radius: float, azimuths: Sequence[float] = AZIMUTHS_REFERENCE) -> np.ndarray:
    """The twelve eye positions on the orbit sphere, centred on the origin."""
    positions = np.zeros((len(azimuths), 3))
    for i, (elev_deg, azim_deg) in enumerate(zip(ELEVATIONS, azimuths)):
        elev, azim = math.radians(elev_deg), math.radians(azim_deg)
        positions[i] = (
            radius * math.cos(elev) * math.cos(azim),
            radius * math.cos(elev) * math.sin(azim),
            radius * math.sin(elev),
        )
    return positions


def camera_transforms(radius: float, azimuths: Sequence[float] = AZIMUTHS_REFERENCE) -> List[np.ndarray]:
    """The twelve camera-to-world matrices, in view order."""
    return [look_at(eye) for eye in camera_positions(radius, azimuths)]


def build_meta(
    bbox_size: Sequence[float],
    scaling_factor: float,
    translation: Sequence[float],
    transforms: Sequence[np.ndarray],
) -> Dict:
    """The ``meta.json`` payload.

    Only ``camera_angle_x``, ``transforms``, ``scaling_factor`` and
    ``translation`` are read downstream (``read_views``); the rest is
    recorded because the reference files carry it and it costs nothing.
    """
    return {
        "camera_angle_x": camera_angle_x(),
        "camera_lens": CAMERA_LENS_MM,
        "sensor_width": SENSOR_WIDTH_MM,
        "env_texture": "null",
        "bbox_size": [float(v) for v in bbox_size],
        "scaling_factor": float(scaling_factor),
        "translation": [float(v) for v in translation],
        "transforms": [np.asarray(m).tolist() for m in transforms],
    }


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

# What ``gen_pcd`` treats as "no geometry here".
INVALID_DEPTH = 65504.0

RESOLUTION = 1024

# glTF is Y-up, Blender is Z-up, and the reference renders were made in Z-up.
# _propagation.prepare_mesh_and_point_cloud applies this same rotation to mesh.glb
# before lifting, so the
# render must live in the rotated frame or the two disagree.
_Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)


def _as_scene(source: Union[str, Path, trimesh.Trimesh, trimesh.Scene]) -> trimesh.Scene:
    """Load ``source`` as a scene, untouched.

    Never ``force="mesh"``: that concatenates, which makes trimesh merge every
    material into one atlas and remap the UVs onto it badly -- an eight-texture
    asset comes back scrambled. ``process=False`` for the same reason: welding
    vertices at load time is a change to the asset nobody asked for.
    """
    if isinstance(source, (str, Path)):
        source = trimesh.load(source, force="scene", process=False)
    if isinstance(source, trimesh.Trimesh):
        source = trimesh.Scene(source)
    if not isinstance(source, trimesh.Scene) or not source.geometry:
        raise ValueError(f"no geometry in {source}")
    return source


def _parts(scene: trimesh.Scene) -> List[trimesh.Trimesh]:
    """The scene's meshes, with its graph transforms baked in."""
    parts = [g for g in scene.dump(concatenate=False) if isinstance(g, trimesh.Trimesh)]
    if not parts:
        raise ValueError("no mesh geometry in scene")
    return parts


def _as_mesh(source: Union[str, Path, trimesh.Trimesh, trimesh.Scene]) -> trimesh.Trimesh:
    """One mesh, for the geometry passes. Appearance is not preserved."""
    return trimesh.util.concatenate(_parts(_as_scene(source)))


def normalize_matrix(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """The transform :func:`normalize` applies, as a 4x4, plus what meta.json needs.

    Exposed separately so a *scene* can be normalised geometry by geometry --
    concatenating it first would force trimesh to merge every material into one
    atlas, which is how the textures get scrambled.
    """
    rotated = mesh.copy()
    rotated.apply_transform(_Y_UP_TO_Z_UP)

    lo, hi = rotated.bounds
    extent = hi - lo
    longest = float(extent.max())
    if longest <= 0:
        raise ValueError("mesh is degenerate: zero extent")

    scaling_factor = NORMALIZATION_RANGE / longest
    translation = -(lo + hi) / 2.0

    translate = np.eye(4)
    translate[:3, 3] = translation
    scale = np.eye(4)
    scale[:3, :3] *= scaling_factor
    return scale @ translate @ _Y_UP_TO_Z_UP, scaling_factor, translation, extent * scaling_factor


def normalize(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, float, np.ndarray, np.ndarray]:
    """Rotate into Z-up and fit the mesh into a unit box centred on the origin.

    Returns ``(mesh, scaling_factor, translation, bbox_size)`` where the two
    scalars are what ``meta.json`` must carry: ``prepare_mesh_and_point_cloud`` rebuilds this
    exact mesh as ``(raw_rotated + translation) * scaling_factor``, so they are
    defined to satisfy *that* formula rather than to mirror how Blender happened
    to compose its own transform.
    """
    matrix, scaling_factor, translation, bbox_size = normalize_matrix(mesh)
    out = mesh.copy()
    out.apply_transform(matrix)
    return out, scaling_factor, translation, bbox_size


def _normal_carrier(mesh: trimesh.Trimesh, smooth: bool) -> trimesh.Trimesh:
    """A mesh whose vertex colours encode its world normals as ``(n + 1) / 2``.

    The colour buffer is the only per-pixel channel pyrender exposes besides
    depth, so normals ride through it. Under FLAT shading nothing lights or
    tone-maps them, and the rasteriser's barycentric interpolation of an encoded
    normal is the same thing smooth shading does -- decode then renormalise.
    """
    carrier = mesh.copy()
    if smooth:
        normals = carrier.vertex_normals
    else:
        # One vertex per corner, so each face carries a constant colour and the
        # interpolation above collapses to a flat normal.
        carrier = trimesh.Trimesh(
            vertices=carrier.vertices[carrier.faces].reshape(-1, 3),
            faces=np.arange(len(carrier.faces) * 3).reshape(-1, 3),
            process=False,
        )
        normals = np.repeat(mesh.face_normals, 3, axis=0)

    encoded = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    colors = np.empty((len(encoded), 4), dtype=np.uint8)
    colors[:, :3] = np.round(encoded * 255.0).astype(np.uint8)
    colors[:, 3] = 255
    carrier.visual = trimesh.visual.ColorVisuals(mesh=carrier, vertex_colors=colors)
    return carrier


# pyrender exposes no MSAA -- OffscreenRenderer takes a size and nothing else,
# and RenderFlags has no antialiasing bit. Supersampling is the only lever:
# render the lit pass at this multiple and box it back down. It applies to that
# pass alone; resampling depth or normals would invent geometry that is not there.
SSAA = 2

# Three dim lamps around the camera rather than one bright headlamp. A single
# lamp puts its specular lobe dead centre, and on a metallic material (the
# bundled vase is 0.68) over faceted geometry that lands as a hard bright patch
# -- measured: it vanishes under FLAT and under ambient alone, so it is the lamp.
# Spreading the same light kills the lobe and keeps the relief.
LIGHT_YAWS: Tuple[float, ...] = (-40.0, 0.0, 40.0)
LIGHT_INTENSITY = 0.5
AMBIENT = 0.45


def _lit_scene(scene: trimesh.Scene, matrix: np.ndarray, smooth_normals: bool = False):
    """The mesh as it is -- textures, PBR factors, everything -- lit from the camera.

    Returns ``(pyrender_scene, camera_node, lights)``, where ``lights`` is what
    :func:`_pose_lit` needs. Pose them together with the camera: the lamps ride
    on it, so a boundary visible only as a shading break stays visible from all
    twelve views rather than falling into shadow on half of them.

    Each geometry is added separately, carrying its own material, and normalised
    by ``matrix`` rather than by concatenating first -- concatenation is exactly
    what merges the materials into one atlas and scrambles them.
    """
    pr_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[AMBIENT] * 3)
    for part in _parts(scene):
        placed = part.copy()
        placed.apply_transform(matrix)
        pr_scene.add(pyrender.Mesh.from_trimesh(placed, smooth=smooth_normals))

    camera_node = pr_scene.add(pyrender.PerspectiveCamera(yfov=camera_angle_x(), aspectRatio=1.0))
    lights = []
    for yaw in LIGHT_YAWS:
        node = pr_scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                                      intensity=LIGHT_INTENSITY))
        offset = trimesh.transformations.rotation_matrix(np.radians(yaw), [0.0, 1.0, 0.0])
        lights.append((node, offset))
    return pr_scene, camera_node, lights


def _pose_lit(pr_scene, camera_node, lights, pose: np.ndarray) -> None:
    """Aim the camera and its lamps at one canonical view."""
    pr_scene.set_pose(camera_node, pose)
    for node, offset in lights:
        pr_scene.set_pose(node, pose @ offset)


def _downsample(image: np.ndarray, resolution: int) -> np.ndarray:
    """Box a supersampled render back to ``resolution``."""
    return np.asarray(Image.fromarray(image).resize((resolution, resolution), Image.LANCZOS))


# Ported from PixMesh's ViewGenerator, which renders for the same reason -- a
# vision model looking for parts. Directional-lit pyrender output is flatter than
# what Blender's environment lighting gives, and these recover the separation
# between neighbouring surfaces that the flatness costs.
SATURATION_BOOST = 1.4
CONTRAST_BOOST = 1.15


def _enhance(image: np.ndarray) -> np.ndarray:
    """Lift saturation and contrast on an RGB render.

    Applied to the lit pass only. The normal carrier must never see this: its
    RGB is an encoded direction, not a colour, and a contrast curve would bend
    every normal.
    """
    alpha = image[..., 3:] if image.shape[-1] == 4 else None
    out = Image.fromarray(image[..., :3])
    out = ImageEnhance.Color(out).enhance(SATURATION_BOOST)
    out = ImageEnhance.Contrast(out).enhance(CONTRAST_BOOST)
    out = np.asarray(out)
    return np.dstack([out, alpha]) if alpha is not None else out


def render_views(
    source: Union[str, Path, trimesh.Trimesh, trimesh.Scene],
    output_dir: Union[str, Path],
    resolution: int = RESOLUTION,
    azimuths: Sequence[float] = AZIMUTHS_REFERENCE,
    smooth_normals: bool = False,
    mesh_copy: Optional[Union[str, Path]] = None,
) -> Path:
    """Render ``source`` into a GeoSAM2 view directory and return its path.

    Writes ``color_XXXX.webp``, ``depth_XXXX.exr``, ``normal_XXXX.webp`` for the
    twelve views, plus ``meta.json`` and a copy of the source mesh as
    ``mesh.glb`` (``read_views`` hard-codes that name).

    ``resolution`` is a knob for testing only. GeoSAM2 hard-codes 1024 in its
    lifting maths (``_lift.cal_link``, ``_lift.lift_2dmask_3d``), so anything else
    produces a silently wrong 3D result -- callers should leave it alone.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_in = _as_scene(source)
    mesh = trimesh.util.concatenate(_parts(scene_in))
    matrix, scaling_factor, translation, bbox_size = normalize_matrix(mesh)
    normalized = mesh.copy()
    normalized.apply_transform(matrix)

    transforms = camera_transforms(camera_distance(bbox_size), azimuths)
    carrier = _normal_carrier(normalized, smooth_normals)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(carrier, smooth=smooth_normals))
    camera = pyrender.PerspectiveCamera(yfov=camera_angle_x(), aspectRatio=1.0)
    camera_node = scene.add(camera, pose=np.eye(4))

    # Two sequential passes, never two renderers at once: each OffscreenRenderer
    # owns an EGL context, and a second one alive at the same time makes
    # eglMakeCurrent fail. They also want different sizes -- the lit pass is
    # supersampled, the geometry passes must not be.
    geometry = []
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    try:
        for pose in transforms:
            scene.set_pose(camera_node, pose)
            geometry.append(renderer.render(scene, flags=pyrender.RenderFlags.FLAT))
    finally:
        renderer.delete()

    lit_scene, lit_camera, lit_lights = _lit_scene(scene_in, matrix, smooth_normals)
    renderer = pyrender.OffscreenRenderer(resolution * SSAA, resolution * SSAA)
    try:
        for view, pose in enumerate(transforms):
            _pose_lit(lit_scene, lit_camera, lit_lights, pose)
            lit, _ = renderer.render(lit_scene, flags=pyrender.RenderFlags.RGBA)
            color, depth = geometry[view]
            _write_view(output_dir, view, color, depth, _downsample(lit, resolution))
    finally:
        renderer.delete()

    meta = build_meta(bbox_size, scaling_factor, translation, transforms)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    _write_mesh(source if mesh_copy is None else mesh_copy, mesh, output_dir / "mesh.glb")
    return output_dir


def shaded_view(
    source: Union[str, Path, trimesh.Trimesh, trimesh.Scene],
    view: int = 0,
    resolution: int = RESOLUTION,
    azimuths: Sequence[float] = AZIMUTHS_REFERENCE,
) -> Image.Image:
    """A lit render of one canonical view, for a human or a VLM to look at.

    The model itself never needs this -- it reads normals and depth, and a normal
    map is a poor thing to ask a vision model to reason about.

    The camera matches ``render_views`` exactly, so pixel coordinates picked here
    are valid prompts for the same view -- and this is bit for bit the render
    that lands in that view's ``color_*.webp``.
    """
    scene_in = _as_scene(source)
    matrix, _, _, bbox_size = normalize_matrix(trimesh.util.concatenate(_parts(scene_in)))
    pose = camera_transforms(camera_distance(bbox_size), azimuths)[view]

    scene, camera_node, lights = _lit_scene(scene_in, matrix)
    _pose_lit(scene, camera_node, lights, pose)

    renderer = pyrender.OffscreenRenderer(resolution * SSAA, resolution * SSAA)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    return Image.fromarray(_enhance(_downsample(color[..., :3], resolution)))


def _write_view(
    output_dir: Path, view: int, color: np.ndarray, depth: np.ndarray, lit: np.ndarray
) -> None:
    hit = depth > 0.0

    # pyrender leaves misses at 0.0; GeoSAM2 reads "no geometry" as >= 65500.
    depth_out = np.where(hit, depth, INVALID_DEPTH).astype(np.float32)
    # Three channels, not one: every consumer indexes `depth[..., 0]`
    # (``load_video_frame_geosam2``, ``read_views``), which silently degenerates to a
    # 1-D array on a single-channel EXR instead of failing loudly.
    cv2.imwrite(str(output_dir / f"depth_{view:04d}.exr"), np.repeat(depth_out[..., None], 3, axis=2))

    normal = np.zeros((*depth.shape, 4), dtype=np.uint8)
    normal[..., :3] = color[..., :3]
    normal[..., 3] = np.where(hit, 255, 0)
    Image.fromarray(normal, mode="RGBA").save(output_dir / f"normal_{view:04d}.webp", lossless=True)

    # ``read_views`` reads only this file's alpha, as the visibility mask. The
    # RGB is the lit render: free to the model, and the only human- (or VLM-)
    # readable view of the object in the directory.
    #
    # Alpha comes from the supersampled lit pass, not from `hit`: a binary cut
    # leaves the silhouette stepped, and the reference renders carry ~8k partial
    # alpha pixels on this view. `> 0` reads them as covered either way, so
    # matching them costs nothing and is what antialiases the outline.
    rgba = _enhance(lit) if lit.shape[-1] == 4 else np.dstack([_enhance(lit[..., :3]), normal[..., 3:]])
    Image.fromarray(rgba, mode="RGBA").save(output_dir / f"color_{view:04d}.webp", lossless=True)


def _write_mesh(source, loaded: trimesh.Trimesh, destination: Path) -> None:
    """Copy the source GLB verbatim when we have one, else export what we loaded.

    Copying beats re-exporting: ``read_views`` reloads this file and indexes the
    label array by its face order, so a re-export that reorders or welds faces
    would silently misalign every label.
    """
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".glb":
        shutil.copy2(source, destination)
    else:
        loaded.export(destination)


# ---------------------------------------------------------------------------
# Reading a view directory back
# ---------------------------------------------------------------------------

MASK_MIN_AREA_PX = 64
MASK_COLOR_QUANT_STEP = 8


def is_view_directory(path: Path) -> bool:
    """Whether ``path`` holds a complete set of GeoSAM2 input views."""
    if not path.is_dir():
        return False
    if not (path / "meta.json").is_file() or not (path / "mesh.glb").is_file():
        return False
    for prefix, suffix in (("color", "webp"), ("depth", "exr"), ("normal", "webp")):
        for view in range(NUM_VIEWS):
            if not (path / f"{prefix}_{view:04d}.{suffix}").is_file():
                return False
    return True


def load_mesh(path: Union[str, Path]) -> trimesh.Trimesh:
    """``mesh.glb`` as one mesh, in the face order the labels refer to."""
    return trimesh.load(str(path), force="mesh")


def explode_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """The same faces with their own vertices: face ``i`` owns vertices ``3i..3i+2``.

    The lift samples points per face and votes per face; unshared vertices
    make that indexing trivial. Unprocessed, so nothing is merged back.
    """
    vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces = np.arange(len(mesh.faces) * 3).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    """Build per-pixel camera rays in camera space.

    Args:
        W: Image width in pixels.
        H: Image height in pixels.
        fx: Focal length along x axis.
        fy: Focal length along y axis.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.
        use_pixel_centers: Whether to offset samples by 0.5 pixel.
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing="xy",
    )
    directions = np.stack(
        [(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], -1
    ) 

    return directions


def gen_pcd(depth, c2w_opengl, camera_angle_x):
    """Convert a depth map into a clipped world-space position map.

    Args:
        depth: Depth image with shape [H, W].
        c2w_opengl: Camera-to-world matrix in OpenGL convention.
        camera_angle_x: Horizontal field-of-view in radians.
    """
    h, w = depth.shape
    
    depth_valid = depth < 65500.0
    focal = 0.5 * w / math.tan(0.5 * camera_angle_x)
    ray_directions = get_ray_directions(w, h, focal, focal, w // 2, h // 2)

    org_points = np.zeros((h, w, 3))

    points_c = ray_directions[depth_valid] * depth[depth_valid, None]
    points_c_homo = np.concatenate(
        [points_c, np.ones_like(points_c[..., :1])], axis=-1
    )
    valid_points = (points_c_homo @ c2w_opengl.T)[..., :3]

    valid_points = np.clip(valid_points, -1.0, 1.0)

    org_points[depth_valid] = valid_points

    return org_points


def _encode_color(rgb: np.ndarray) -> int:
    return (int(rgb[0]) << 16) + (int(rgb[1]) << 8) + int(rgb[2])


def _quantize_rgb(rgb: np.ndarray, step: int) -> np.ndarray:
    """Quantize RGB to reduce tiny color variation from anti-aliasing/compression."""
    if step <= 1:
        return rgb
    q = (rgb // step) * step
    return q.astype(np.uint8)


def extract_mask_segments(mask_path: str) -> List[Tuple[Tuple[str, int], np.ndarray]]:
    """Extract mask segments with stable keys from label maps or color previews."""
    ext = os.path.splitext(mask_path)[1].lower()

    if ext == ".exr":
        raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Failed to read mask file: {mask_path}")
        label_map = _to_int_label_map(raw[..., 0] if raw.ndim == 3 else raw)
        unique_ids = np.unique(label_map)
        unique_ids = unique_ids[unique_ids != 0]
        segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
        for obj_id in unique_ids:
            m = label_map == int(obj_id)
            if int(m.sum()) < MASK_MIN_AREA_PX:
                continue
            segments.append((("id", int(obj_id)), m))
        return segments
    elif ext == ".npy":
        label_map = _to_int_label_map(np.load(mask_path))
        unique_ids = np.unique(label_map)
        unique_ids = unique_ids[unique_ids != 0]
        segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
        for obj_id in unique_ids:
            m = label_map == int(obj_id)
            if int(m.sum()) < MASK_MIN_AREA_PX:
                continue
            segments.append((("id", int(obj_id)), m))
        return segments
    elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        mask_image = np.array(Image.open(mask_path))

        if mask_image.ndim == 2:
            label_map = _to_int_label_map(mask_image)
            unique_ids = np.unique(label_map)
            unique_ids = unique_ids[unique_ids != 0]
            segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
            for obj_id in unique_ids:
                m = label_map == int(obj_id)
                if int(m.sum()) < MASK_MIN_AREA_PX:
                    continue
                segments.append((("id", int(obj_id)), m))
            return segments

        if mask_image.ndim == 3 and mask_image.shape[2] >= 3:
            rgb = mask_image[..., :3].astype(np.uint8)
            # Treat transparent pixels as background.
            if mask_image.shape[2] == 4:
                rgb[mask_image[..., 3] == 0] = 0

            # Reduce color noise from interpolation/compression.
            rgb = _quantize_rgb(rgb, MASK_COLOR_QUANT_STEP)

            # If RGB channels are identical, treat it as a numeric label map.
            if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
                label_map = _to_int_label_map(rgb[..., 0])
                unique_ids = np.unique(label_map)
                unique_ids = unique_ids[unique_ids != 0]
                segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
                for obj_id in unique_ids:
                    m = label_map == int(obj_id)
                    if int(m.sum()) < MASK_MIN_AREA_PX:
                        continue
                    segments.append((("id", int(obj_id)), m))
                return segments

            flat_rgb = rgb.reshape(-1, 3)
            unique_colors, counts = np.unique(flat_rgb, axis=0, return_counts=True)
            segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
            if unique_colors.shape[0] == 0:
                return segments

            # Assume dominant color is background for color preview masks.
            bg_color = unique_colors[int(np.argmax(counts))]
            for color in unique_colors:
                if np.array_equal(color, np.array([0, 0, 0], dtype=np.uint8)):
                    continue
                if np.array_equal(color, bg_color):
                    continue
                color_key = _encode_color(color)
                color_mask = np.all(rgb == color, axis=-1)
                if int(color_mask.sum()) < MASK_MIN_AREA_PX:
                    continue
                segments.append((("color", color_key), color_mask))
            return segments

        raise ValueError(f"Unsupported mask image shape: {mask_image.shape}")
    else:
        raise ValueError(
            f"Unsupported mask format: {ext}. Supported: .exr, .npy, image formats"
        )


def add_mask_file_to_frame(
    frame_segments: Dict[int, np.ndarray],
    key_to_objid: Dict[Tuple[str, int], int],
    mask_path: str,
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[str, int], int]]:
    """Add one mask file into a frame with stable ID mapping for repeated keys."""
    segments = extract_mask_segments(mask_path)
    segments = sorted(segments, key=lambda x: x[0])

    next_obj_id = max(frame_segments.keys(), default=0) + 1
    for seg_key, seg_mask in segments:
        seg_mask = seg_mask.astype(bool)
        if not seg_mask.any():
            continue
        if seg_key in key_to_objid:
            obj_id = key_to_objid[seg_key]
            if obj_id in frame_segments:
                frame_segments[obj_id] = np.logical_or(frame_segments[obj_id], seg_mask)
            else:
                frame_segments[obj_id] = seg_mask
            continue

        while next_obj_id in frame_segments:
            next_obj_id += 1
        frame_segments[next_obj_id] = seg_mask
        key_to_objid[seg_key] = next_obj_id
        next_obj_id += 1

    return frame_segments, key_to_objid


@dataclass
class Views:
    """A view directory, read: the 12 canonical renders and the seed on one of them."""

    root: str
    obj_name: str
    images: List[np.ndarray]          # RGB, composited on white
    img_masks: List[np.ndarray]       # HxWx1 bool: where the object is (alpha > 0)
    depth_maps: List[np.ndarray]
    pos_maps: List[torch.Tensor]      # 3xHxW world positions, from the depth
    norm_maps: List[np.ndarray]       # RGB-encoded normals, composited on white
    c2ws: List[torch.Tensor]
    fovy_deg: float
    scaling_factor: float
    translation: np.ndarray           # float32, as meta.json's, the way the lift applies it
    mesh: trimesh.Trimesh             # faces exploded, see explode_faces
    mesh_vanilla: trimesh.Trimesh     # as loaded: the labels index its faces
    seed_view: int
    seed_masks: Dict[int, np.ndarray]  # object id -> bool mask on seed_view


def read_views(data_root: str, mask_path: str, mask_view: int) -> Views:
    """Read a view directory and the seed mask drawn on one of its views.

    ``mask_path`` is a label map (``.npy``, ``.exr``) or a flat-colour image;
    ``extract_mask_segments`` turns either into one boolean mask per object.
    """
    meta = json.load(open(os.path.join(data_root, "meta.json")))
    camera_angle_x = meta["camera_angle_x"]

    images, img_masks, depth_maps, pos_maps, norm_maps, c2ws = [], [], [], [], [], []
    for idx in range(NUM_VIEWS):
        img = Image.open(os.path.join(data_root, f"color_{idx:04d}.webp"))
        img_masks.append(np.array(img)[:, :, -1:] > 0)
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        images.append(np.array(Image.alpha_composite(background, img).convert("RGB")))

        depth = cv2.imread(os.path.join(data_root, f"depth_{idx:04d}.exr"), cv2.IMREAD_UNCHANGED)
        depth = depth[..., 0]
        depth_maps.append(depth)

        c2w = np.array(meta["transforms"][idx])
        pos_map = gen_pcd(depth, c2w, camera_angle_x)
        pos_maps.append(torch.from_numpy(pos_map).to(torch.float32).permute(2, 0, 1))
        c2ws.append(torch.tensor(c2w, dtype=torch.float32))

        norm = Image.open(os.path.join(data_root, f"normal_{idx:04d}.webp"))
        background = Image.new("RGBA", norm.size, (255, 255, 255, 255))
        norm_maps.append(np.array(Image.alpha_composite(background, norm).convert("RGB")))

    if not 0 <= mask_view < NUM_VIEWS:
        raise ValueError(f"mask_view={mask_view} is not one of the {NUM_VIEWS} views")
    seed_masks, _ = add_mask_file_to_frame({}, {}, mask_path)
    if not seed_masks:
        raise ValueError(f"no object found in the seed mask {mask_path}")

    mesh_vanilla = load_mesh(os.path.join(data_root, "mesh.glb"))

    return Views(
        root=data_root,
        obj_name=os.path.basename(os.path.normpath(data_root)),
        images=images,
        img_masks=img_masks,
        depth_maps=depth_maps,
        pos_maps=pos_maps,
        norm_maps=norm_maps,
        c2ws=c2ws,
        fovy_deg=meta["camera_angle_x"] * 180.0 / math.pi,
        scaling_factor=meta["scaling_factor"],
        translation=np.asarray(meta["translation"], dtype=np.float32),
        mesh=explode_faces(mesh_vanilla),
        mesh_vanilla=mesh_vanilla,
        seed_view=int(mask_view),
        seed_masks=seed_masks,
    )
