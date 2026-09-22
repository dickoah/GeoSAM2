"""GeoSAM2 labels -> SegviGen's split, unchanged.

SegviGen's ``split_glb_by_texture_palette_rgb`` is the part of that project
worth keeping: texel-owned palette, chart-scoped smoothing, graph-cut + band
refine on real creases, then a per-label SDF cut in the UV atlas so boundaries
stop following triangle edges. It reads a textured GLB. GeoSAM2 produces
per-face labels. This module bakes those labels into a flat-colour texture on
the mesh's own UVs and hands the result to SegviGen's code as-is -- no port.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image

from utils.logs import get_logger

logger = get_logger("geosam2.split")

# The sibling checkout. Override with GEOSAM2_SEGVIGEN_DIR.
SEGVIGEN_DIR = Path(os.environ.get(
    "GEOSAM2_SEGVIGEN_DIR",
    Path(__file__).resolve().parents[2] / "pixmesh/backend/src/libraries/segvigen"))

# THE palette: one colour per label id, the same at every stage (raw lift,
# post-process, bake, split), so a part keeps its colour from one viewer entry
# to the next. Keyed by the id's VALUE, not its rank -- the raw lift and the
# post-process carry different id sets, and a rank-based palette would shift
# every colour between the two.
#
# The split quantises texels to a 16-step grid and folds palette entries
# closer than palette_merge_dist=32 into one part, so two labels must never be
# painted less than that apart. Kelly's colours are not (several pairs sit ~30
# apart: 19 labels came out as 16 parts). Farthest-point on the RGB grid, with
# black (unmapped texels) and the unassigned grey pre-taken, keeps every pair
# >= _MIN_SEP.
_MIN_SEP = 64.0
UNASSIGNED_LABELS = (0, 999)
UNASSIGNED_RGB = (120, 120, 120)
_sequence: list = []


def _colour_sequence(n: int) -> list:
    """The first ``n`` farthest-point colours, cached: colour k never changes."""
    if len(_sequence) >= n:
        return _sequence[:n]
    # 32..224: near-white reads as "unassigned" and near-black as background.
    step = np.arange(32, 225, 16, dtype=np.float64)
    grid = np.stack(np.meshgrid(step, step, step, indexing="ij"), -1).reshape(-1, 3)
    taken = np.vstack([np.zeros((1, 3)), np.array([UNASSIGNED_RGB], np.float64),
                       np.array(_sequence, np.float64).reshape(-1, 3)])
    while len(_sequence) < n:
        d = np.linalg.norm(grid[:, None] - taken[None], axis=2).min(axis=1)
        pick = grid[int(np.argmax(d))]
        if d.max() < _MIN_SEP:
            logger.warning("[palette] colour %d: closest pair down to %.0f (< %.0f), "
                           "the split may fold two parts", len(_sequence), d.max(), _MIN_SEP)
        _sequence.append(tuple(int(c) for c in pick))
        taken = np.vstack([taken, pick])
    return _sequence[:n]


def _load_split_module():
    """Import SegviGen's ``util.split`` without running ``util/__init__``.

    The package init pulls in the VLM guidance stack and rembg, neither of
    which this environment has or needs. A synthetic package with the same
    ``__path__`` lets ``split.py``'s ``from ._common import ...`` resolve.
    """
    if "segvigen_util.split" in sys.modules:
        return sys.modules["segvigen_util.split"]
    util_dir = SEGVIGEN_DIR / "util"
    if not (util_dir / "split.py").is_file():
        raise FileNotFoundError(
            f"SegviGen's util/split.py not found under {SEGVIGEN_DIR}; "
            "set GEOSAM2_SEGVIGEN_DIR")
    pkg = types.ModuleType("segvigen_util")
    pkg.__path__ = [str(util_dir)]
    sys.modules["segvigen_util"] = pkg
    for name in ("_common", "split"):
        spec = importlib.util.spec_from_file_location(
            f"segvigen_util.{name}", util_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"segvigen_util.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["segvigen_util.split"]


def split_presets() -> Dict[str, dict]:
    """SegviGen's SPLIT_PRESETS, read from its checkout (no copy to drift)."""
    spec = importlib.util.spec_from_file_location(
        "segvigen_presets", SEGVIGEN_DIR / "segvigen" / "presets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SPLIT_PRESETS


def to_linear_u8(rgb) -> np.ndarray:
    """sRGB palette colour -> the linear value glTF stores in COLOR_0.

    Textures are sRGB in glTF and vertex colours are linear; a viewer converts
    the latter on output. The palette is authored in sRGB (it is what the
    swatches and the baked texture show), so a vertex-colour export has to be
    encoded, or the same part comes out lighter than its texture.
    """
    c = np.asarray(rgb, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.round(lin * 255.0).astype(np.uint8)


def label_palette(labels: np.ndarray) -> Dict[int, Tuple[int, int, int]]:
    """``{label id: rgb}`` for every id in ``labels``; unassigned ids are grey."""
    ids = [int(i) for i in np.unique(labels)]
    parts = [i for i in ids if i not in UNASSIGNED_LABELS]
    seq = _colour_sequence(max(parts, default=-1) + 1)
    return {i: (UNASSIGNED_RGB if i in UNASSIGNED_LABELS else seq[i]) for i in ids}


def bake_labels_to_glb(mesh_path: str, face_labels: np.ndarray, out_glb: str,
                       size: int = 2048) -> Dict[int, Tuple[int, int, int]]:
    """Write ``mesh_path`` with a flat-colour texture carrying ``face_labels``.

    The mesh keeps its own UVs; each face's UV triangle is filled with its
    label's colour. Loaded exactly as ``_build_scene`` loads it (force="mesh",
    default processing) so the face order the labels index is the same.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    labels = np.asarray(face_labels).reshape(-1).astype(np.int64)
    if len(labels) != len(mesh.faces):
        raise ValueError(f"{len(labels)} labels for {len(mesh.faces)} faces")
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise ValueError("mesh has no per-vertex UVs; SegviGen's split needs an atlas")

    palette = label_palette(labels)
    img = np.zeros((size, size, 3), np.uint8)
    # u right, v up in glTF/trimesh; image rows go down.
    px = np.stack([uv[:, 0] * (size - 1), (1.0 - uv[:, 1]) * (size - 1)], axis=1)
    faces = np.asarray(mesh.faces)
    shift = 4
    for label, colour in palette.items():
        tris = (px[faces[labels == label]] * (1 << shift)).round().astype(np.int32)
        # The array is handed to PIL as RGB; cv2 writes the tuple in channel
        # order, so it is NOT reversed here (that swap painted blue parts brown).
        rgb = tuple(int(c) for c in colour)
        cv2.fillPoly(img, list(tris), rgb, lineType=cv2.LINE_8, shift=shift)
        # Thin triangles can rasterise to nothing; their edges still own texels.
        cv2.polylines(img, list(tris), True, rgb, 1, lineType=cv2.LINE_8, shift=shift)
    pil = Image.fromarray(img)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=pil, metallicFactor=0.0, roughnessFactor=1.0)
    baked = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    baked.visual = trimesh.visual.TextureVisuals(uv=uv, image=pil, material=material)
    os.makedirs(os.path.dirname(out_glb) or ".", exist_ok=True)
    baked.export(out_glb)
    logger.info("[bake] %d labels -> %s (%dx%d)", len(palette), out_glb, size, size)
    return palette


def split_with_segvigen(baked_glb: str, out_glb: Optional[str] = None,
                        **split_kwargs) -> str:
    """SegviGen's split on a baked GLB. ``split_kwargs`` go straight through."""
    split = _load_split_module()
    kwargs = dict(output_mode="vertex_colors", debug_print=True)
    kwargs.update(split_kwargs)
    return split.split_glb_by_texture_palette_rgb(baked_glb, out_glb, **kwargs)


def labels_to_parts(mesh_path: str, face_labels: np.ndarray, work_dir: str,
                    **split_kwargs) -> str:
    """The whole bridge: labels -> baked texture -> SegviGen split -> parts GLB."""
    baked = os.path.join(work_dir, "baked.glb")
    bake_labels_to_glb(mesh_path, face_labels, baked)
    return split_with_segvigen(baked, os.path.join(work_dir, "segvigen_parts.glb"),
                               **split_kwargs)
