"""Rasterise the twelve canonical views GeoSAM2 consumes, without Blender.

    from utils.render import render_views
    render_views("mesh.glb", "views/")   # 12 views + meta.json + mesh.glb

Replaces the ``geosam2_render.py`` Blender stage. The network never sees a shaded
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
  (``utils/inference_utils.py:14``) returns *unnormalised* rays with ``z = -1``,
  so ``direction * depth`` only lands on the surface if depth measures along the
  camera axis. pyrender's depth buffer already is exactly this.
* **The EXR needs three channels.** Every consumer indexes ``depth[..., 0]``,
  which silently degenerates to a 1-D array on a single-channel file rather than
  failing loudly.
* **``mesh.glb`` is copied, never re-exported.** inference.py reloads it and
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

    python -m utils.render example/sample_00
    python -m utils.render example/sample_00 --images example/render_comparison
    python -m utils.render example/sample_00 --against /path/to/blender_render

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

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pyrender
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
# geosam2_render.py starts at 180 instead -- the same rig rotated by 90 degrees.
# The bundled examples were not produced by the shipped renderer (their meta.json
# carries keys it never writes), and the reference azimuths are the ones the
# example point prompts were authored against: those prompts are pixel
# coordinates in view 0, so a rig rotated by 90 degrees puts the clicks on
# different parts of the object. Reproduce the data that works.
AZIMUTHS_REFERENCE: Tuple[float, ...] = (270., 300., 330., 0., 30., 60., 90., 120., 150., 180., 210., 240.)
AZIMUTHS_BLENDER_SCRIPT: Tuple[float, ...] = (180., 210., 240., 270., 300., 330., 0., 30., 60., 90., 120., 150.)

NUM_VIEWS = len(ELEVATIONS)

# Blender defaults geosam2_render.py bakes in: a 50 mm lens on a 36 mm sensor.
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
    """Focal length in pixels, matching ``utils/inference_utils.get_ray_directions``."""
    return 0.5 * width_px / math.tan(0.5 * camera_angle_x())


def camera_distance(bbox_size: Sequence[float]) -> float:
    """Orbit radius Blender's rig uses for a given normalised bounding box."""
    return CAMERA_LENS_MM / SENSOR_WIDTH_MM * float(np.linalg.norm(bbox_size))


def look_at(eye: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Camera-to-world matrix aiming ``-Z`` at ``target`` with ``+Y`` up.

    Reproduces Blender's ``to_track_quat('-Z', 'Y')``, which is what
    geosam2_render.py uses to orient each camera.
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
    ``translation`` are read downstream (inference.py:371-447); the rest is
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

# What inference.py treats as "no geometry here" (utils/inference_utils.py:48).
INVALID_DEPTH = 65504.0

RESOLUTION = 1024

# glTF is Y-up, Blender is Z-up, and the reference renders were made in Z-up.
# inference.py:472 applies this same rotation to mesh.glb before lifting, so the
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
    scalars are what ``meta.json`` must carry: inference.py:472-480 rebuilds this
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
    ``mesh.glb`` (inference.py:434 hard-codes that name).

    ``resolution`` is a knob for testing only. GeoSAM2 hard-codes 1024 in its
    lifting maths (``utils/inference_utils.py:409,483``), so anything else
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
    # (sam2/utils/misc.py:509, inference.py:393), which silently degenerates to a
    # 1-D array on a single-channel EXR instead of failing loudly.
    cv2.imwrite(str(output_dir / f"depth_{view:04d}.exr"), np.repeat(depth_out[..., None], 3, axis=2))

    normal = np.zeros((*depth.shape, 4), dtype=np.uint8)
    normal[..., :3] = color[..., :3]
    normal[..., 3] = np.where(hit, 255, 0)
    Image.fromarray(normal, mode="RGBA").save(output_dir / f"normal_{view:04d}.webp", lossless=True)

    # inference.py:385 reads only this file's alpha, as the visibility mask. The
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

    Copying beats re-exporting: inference.py reloads this file and indexes the
    label array by its face order, so a re-export that reorders or welds faces
    would silently misalign every label.
    """
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".glb":
        shutil.copy2(source, destination)
    else:
        loaded.export(destination)


# ---------------------------------------------------------------------------
# Validation -- see the module docstring; run with `python -m utils.render`
# ---------------------------------------------------------------------------

# The lifting step calls a sample point visible when its reprojected depth lands
# within 1e-3 of the depth map (utils/inference_utils.py:394), in a space
# normalised to [-1, 1]. A depth error at that scale starts silently dropping
# points, so it is the tolerance that matters -- not an arbitrary epsilon.
DEPTH_TOLERANCE = 1e-3

# The noise floor of comparing two independent rasterisations of a dense mesh
# through an 8-bit encoding, not a quality target. Two irreducible terms:
# quantising a unit normal to 8 bits per channel costs ~0.45 degrees, and this
# class of mesh projects several triangles into one pixel (sample_00: 407k faces
# into ~200k covered pixels, median dihedral 3.9 degrees), so which triangle a
# pixel reports is a coin toss worth a degree or two.
#
# Measured on sample_00: flat normals land at 1.05 degrees, smooth at 2.06. The
# threshold sits between them deliberately -- it still catches the flat/smooth
# mix-up, which is a real error and the one this check exists to find.
NORMAL_TOLERANCE_DEG = 2.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name:22s} {self.detail}"


def _load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(
            f"cannot read {path} -- opencv-python 5.x ships without the OpenEXR codec"
        )
    return depth[..., 0] if depth.ndim == 3 else depth


def _load_normal(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def _decode_normals(rgba: np.ndarray) -> np.ndarray:
    n = rgba[..., :3].astype(np.float64) / 255.0 * 2.0 - 1.0
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


def compare_views(reference: Path, candidate: Path) -> List[Check]:
    """Diff two view directories channel by channel."""
    checks: List[Check] = []

    meta_ref = json.loads((reference / "meta.json").read_text())
    meta_new = json.loads((candidate / "meta.json").read_text())

    t_ref = np.array(meta_ref["transforms"])
    t_new = np.array(meta_new["transforms"])
    err = np.abs(t_ref - t_new).max()
    checks.append(Check("camera transforms", err < 1e-4, f"max|diff| = {err:.2e}"))

    err = abs(meta_ref["camera_angle_x"] - meta_new["camera_angle_x"])
    checks.append(Check("camera_angle_x", err < 1e-9, f"diff = {err:.2e}"))

    depth_err, normal_err, mask_iou = [], [], []
    for view in range(NUM_VIEWS):
        d_ref = _load_depth(reference / f"depth_{view:04d}.exr")
        d_new = _load_depth(candidate / f"depth_{view:04d}.exr")

        hit_ref, hit_new = d_ref < 65500.0, d_new < 65500.0
        both = hit_ref & hit_new
        union = hit_ref | hit_new
        mask_iou.append(both.sum() / max(union.sum(), 1))

        if both.any():
            depth_err.append(np.abs(d_ref[both] - d_new[both]))

        n_ref = _decode_normals(_load_normal(reference / f"normal_{view:04d}.webp"))
        n_new = _decode_normals(_load_normal(candidate / f"normal_{view:04d}.webp"))
        if both.any():
            dot = np.abs((n_ref[both] * n_new[both]).sum(-1))
            normal_err.append(np.degrees(np.arccos(np.clip(dot, 0, 1))))

    depth_all = np.concatenate(depth_err) if depth_err else np.zeros(1)
    normal_all = np.concatenate(normal_err) if normal_err else np.zeros(1)
    iou = float(np.mean(mask_iou))

    # Compare medians, not maxima: silhouette pixels legitimately disagree,
    # because the reference antialiases its edges and a rasteriser does not.
    checks.append(Check(
        "depth", float(np.median(depth_all)) < DEPTH_TOLERANCE,
        f"median {np.median(depth_all):.2e}, p95 {np.percentile(depth_all, 95):.2e} "
        f"(tolerance {DEPTH_TOLERANCE:.0e})",
    ))
    checks.append(Check(
        "normals", float(np.median(normal_all)) < NORMAL_TOLERANCE_DEG,
        f"median {np.median(normal_all):.3f} deg, p95 {np.percentile(normal_all, 95):.3f} deg",
    ))
    checks.append(Check("silhouette IoU", iou > 0.99, f"{iou * 100:.3f}%"))
    return checks


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Depth to greyscale, scaled to the object's own range so it stays readable."""
    hit = depth < 65500.0
    out = np.zeros(depth.shape, dtype=np.uint8)
    if hit.any():
        lo, hi = depth[hit].min(), depth[hit].max()
        span = max(hi - lo, 1e-9)
        out[hit] = (255 * (1.0 - (depth[hit] - lo) / span)).astype(np.uint8)
    return np.stack([out] * 3, -1)


def _heatmap(error: np.ndarray, valid: np.ndarray, vmax: float) -> np.ndarray:
    """Error magnitude as a black-to-red ramp, saturating at ``vmax``."""
    out = np.zeros((*error.shape, 3), dtype=np.uint8)
    scaled = np.clip(error / vmax, 0.0, 1.0)
    out[..., 0] = np.where(valid, (scaled * 255).astype(np.uint8), 0)
    return out


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    tile = tile.copy()
    cv2.putText(tile, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    return tile


def _as_model_sees_it(rgba: np.ndarray) -> np.ndarray:
    """Composite over white, the way ``_load_img_as_tensor`` does before inference.

    Showing raw RGB would compare pixels the model never reads: the loader
    (``sam2/utils/misc.py:98``) alpha-composites every normal map onto white, so
    whatever a renderer leaves in its masked-out background is discarded. Compare
    the input, not the file.
    """
    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    return (rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)


def write_comparison_images(
    reference: Path, candidate: Path, out_dir: Path, views: Sequence[int] = (0, 3, 6, 9)
) -> List[Path]:
    """Write one contact sheet per view: reference, candidate, and error maps.

    The numbers say the renders agree; these say *where* they disagree, which is
    what tells a silhouette artefact apart from a systematic bias.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for view in views:
        d_ref = _load_depth(reference / f"depth_{view:04d}.exr")
        d_new = _load_depth(candidate / f"depth_{view:04d}.exr")
        n_ref = _load_normal(reference / f"normal_{view:04d}.webp")
        n_new = _load_normal(candidate / f"normal_{view:04d}.webp")

        hit_ref, hit_new = d_ref < 65500.0, d_new < 65500.0
        both = hit_ref & hit_new

        depth_err = np.where(both, np.abs(d_ref - d_new), 0.0)
        angle = np.zeros(d_ref.shape)
        dot = (_decode_normals(n_ref) * _decode_normals(n_new)).sum(-1)
        angle[both] = np.degrees(np.arccos(np.clip(np.abs(dot[both]), 0, 1)))

        # Disagreeing silhouette pixels: green = reference only, red = ours.
        silhouette = np.zeros((*d_ref.shape, 3), dtype=np.uint8)
        silhouette[hit_ref & ~hit_new] = (0, 255, 0)
        silhouette[hit_new & ~hit_ref] = (255, 0, 0)

        top = np.hstack([
            _label(_as_model_sees_it(n_ref), "normal: reference"),
            _label(_as_model_sees_it(n_new), "normal: pyrender"),
            _label(_heatmap(angle, both, NORMAL_TOLERANCE_DEG), f"normal err (0-{NORMAL_TOLERANCE_DEG:g} deg)"),
        ])
        bottom = np.hstack([
            _label(_colorize_depth(d_ref), "depth: reference"),
            _label(_colorize_depth(d_new), "depth: pyrender"),
            _label(_heatmap(depth_err, both, DEPTH_TOLERANCE), f"depth err (0-{DEPTH_TOLERANCE:g})"),
        ])
        sheet = np.vstack([top, bottom, np.hstack([
            _label(silhouette, "silhouette: green=ref only, red=ours"),
            np.zeros_like(silhouette), np.zeros_like(silhouette),
        ])])

        path = out_dir / f"view_{view:04d}.webp"
        Image.fromarray(sheet).save(path, quality=90)
        written.append(path)
    return written


def validate(reference: Path, against: Optional[Path] = None, keep: Optional[Path] = None,
             images: Optional[Path] = None) -> bool:
    """Render ``reference``'s own mesh and diff it against ``reference``.

    ``against`` diffs a second, externally produced directory (a Blender render
    of the same mesh) against the same reference, so the two renderers can be
    judged on one scale.
    """
    reference = Path(reference)
    work = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="render_validate_"))

    print(f"reference : {reference}")
    print(f"rendering : {work}\n")
    render_views(reference / "mesh.glb", work)

    print(f"=== pyrender vs {reference.name} ===")
    checks = compare_views(reference, work)
    for check in checks:
        print(check)
    ok = all(c.passed for c in checks)

    if against is not None:
        print(f"\n=== {Path(against).name} vs {reference.name} ===")
        other = compare_views(reference, Path(against))
        for check in other:
            print(check)
        ok = ok and all(c.passed for c in other)

    if images is not None:
        written = write_comparison_images(reference, work, images)
        print(f"\ncomparison images -> {images}")
        for path in written:
            print(f"  {path.name}")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="A view directory to reproduce and diff against.")
    parser.add_argument("--against", type=Path, default=None,
                        help="A second view directory (e.g. a Blender render) to score on the same scale.")
    parser.add_argument("--keep", type=Path, default=None,
                        help="Render into this directory instead of a temporary one.")
    parser.add_argument("--images", type=Path, default=None,
                        help="Write side-by-side comparison sheets into this directory.")
    args = parser.parse_args()
    return 0 if validate(args.reference, args.against, args.keep, args.images) else 1


if __name__ == "__main__":
    sys.exit(main())
