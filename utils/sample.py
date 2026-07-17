"""Build a GeoSAM2 sample directory from an already-segmented mesh.

A sample is what ``inference.py`` consumes: twelve canonical views (colour,
depth, normal), the camera metadata tying them to the mesh, and the mesh itself.
``utils.render`` already produces all of that, and is validated against the
bundled reference. What it cannot produce is the seed mask, because a mask needs
to know where the parts are.

This module fills that gap for meshes that already carry the answer -- one
geometry per part. Painting each part a flat colour and rendering it from the
same cameras *is* the mask: exact by construction, no VLM, no boundary guessing.
Producing a mask for an *unsegmented* mesh is a different problem and lives in
:mod:`utils.mask_agent`.

    python -m utils.sample segmented.glb example/sample_04
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pyrender
import trimesh
from PIL import Image

from utils.mask_agent import PALETTE, snap_to_palette
from utils.render import (
    RESOLUTION,
    camera_angle_x,
    camera_distance,
    camera_transforms,
    normalize,
    render_views,
)

# Black, matching the bundled samples. inference.py's extract_mask_segments
# discards the most frequent colour *and* pure black, so a black background is
# excluded twice over -- and no palette entry is near it.
MASK_BACKGROUND: Tuple[int, int, int] = (0, 0, 0)

NUM_VIEWS = 12


def _parts_of(scene: trimesh.Scene) -> List[Tuple[str, trimesh.Trimesh]]:
    parts = [(n, g) for n, g in scene.geometry.items() if isinstance(g, trimesh.Trimesh)]
    if not parts:
        raise ValueError("no mesh geometry in the scene")
    return parts


def paint_parts(
    scene: trimesh.Scene,
) -> Tuple[trimesh.Scene, Dict[str, Tuple[int, int, int]]]:
    """A copy of ``scene`` with every part filled with a distinct flat colour.

    Returns ``(painted, colour_table)``. Raises rather than reusing a colour past
    the end of the palette: two parts sharing one would read as a single part,
    and the mask would be quietly wrong instead of loudly absent.
    """
    parts = _parts_of(scene)
    if len(parts) > len(PALETTE):
        raise ValueError(f"{len(parts)} parts but only {len(PALETTE)} distinct colours")

    painted = scene.copy()
    table: Dict[str, Tuple[int, int, int]] = {}
    for index, (name, _) in enumerate(parts):
        geometry = painted.geometry[name]
        table[name] = PALETTE[index]
        rgba = np.array([*PALETTE[index], 255], dtype=np.uint8)
        geometry.visual = trimesh.visual.ColorVisuals(
            mesh=geometry, vertex_colors=np.tile(rgba, (len(geometry.vertices), 1))
        )
    return painted, table


def _flat_scene(scene: trimesh.Scene, background: Tuple[int, int, int]):
    """A pyrender scene of ``scene``'s vertex colours, unlit.

    Returns ``(pyrender_scene, camera_node, orbit_radius)``. The camera is built
    exactly as ``render_views`` builds it, so a pixel here and the same pixel of
    ``color_XXXX.webp`` see the same surface -- which is what lets the result
    serve as that view's mask.

    ``to_geometry`` rather than concatenating ``scene.geometry``: the latter
    returns parts in local coordinates, so any mesh whose scene graph carries a
    transform would be rendered with its parts displaced.
    """
    normalized, _, _, bbox_size = normalize(scene.to_geometry())
    pr_scene = pyrender.Scene(
        bg_color=[c / 255.0 for c in background] + [1.0], ambient_light=[1.0] * 3
    )
    pr_scene.add(pyrender.Mesh.from_trimesh(normalized, smooth=False))
    camera_node = pr_scene.add(pyrender.PerspectiveCamera(yfov=camera_angle_x(), aspectRatio=1.0))
    return pr_scene, camera_node, camera_distance(bbox_size)


def flat_view(
    scene: trimesh.Scene,
    view: int = 0,
    resolution: int = RESOLUTION,
    background: Tuple[int, int, int] = MASK_BACKGROUND,
) -> Image.Image:
    """Render one canonical view of ``scene`` with unlit vertex colours."""
    pr_scene, camera_node, radius = _flat_scene(scene, background)
    pr_scene.set_pose(camera_node, camera_transforms(radius)[view])

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    try:
        color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.FLAT)
    finally:
        renderer.delete()
    return Image.fromarray(color[..., :3])


def mask_views(
    painted: trimesh.Scene,
    palette: Dict[str, Tuple[int, int, int]],
    output_dir: Union[str, Path],
    resolution: int = RESOLUTION,
    views: int = NUM_VIEWS,
) -> List[Path]:
    """Write ``mask_XXXX.png`` for every view of an already-painted scene.

    Each mask is snapped back onto the palette: the rasteriser antialiases part
    boundaries, and every blend colour wide enough to survive the area floor
    would otherwise be read as a part of its own.

    One renderer for all twelve views, not one each: an OffscreenRenderer owns an
    EGL context, and churning through a dozen of them is both slow and a way to
    provoke context errors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pr_scene, camera_node, radius = _flat_scene(painted, MASK_BACKGROUND)
    transforms = camera_transforms(radius)

    written: List[Path] = []
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    try:
        for view in range(views):
            pr_scene.set_pose(camera_node, transforms[view])
            color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.FLAT)
            snapped = snap_to_palette(
                Image.fromarray(color[..., :3]), palette, background=MASK_BACKGROUND
            )
            path = output_dir / f"mask_{view:04d}.png"
            Image.fromarray(snapped).save(path)
            written.append(path)
    finally:
        renderer.delete()
    return written


def make_sample(
    segmented_glb: Union[str, Path],
    output_dir: Union[str, Path],
    resolution: int = RESOLUTION,
) -> Path:
    """Turn a segmented GLB into a complete GeoSAM2 sample directory.

    Args:
        segmented_glb: Mesh with one geometry per part.
        output_dir: Destination, e.g. ``example/sample_04``.
        resolution: Render size; 1024 is what the model is wired for.

    Returns:
        ``output_dir``, holding the twelve views, twelve masks, ``mesh.glb`` and
        ``meta.json``.
    """
    output_dir = Path(output_dir)
    scene = trimesh.load(str(segmented_glb), force="scene")
    painted, palette = paint_parts(scene)

    # Rendered from the source, not the painted copy: mesh.glb must be the mesh
    # the labels will index, and the colour views are the object's own look.
    render_views(segmented_glb, output_dir, resolution=resolution)
    mask_views(painted, palette, output_dir, resolution=resolution)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", help="Segmented GLB, one geometry per part")
    parser.add_argument("output_dir", help="Sample directory to create")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    args = parser.parse_args()

    out = make_sample(args.mesh, args.output_dir, resolution=args.resolution)
    scene = trimesh.load(args.mesh, force="scene")
    print(f"{out}: {len(_parts_of(scene))} parts, {NUM_VIEWS} views + masks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
