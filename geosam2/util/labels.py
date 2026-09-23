"""Per-face labels onto the input mesh: the palette, the baked texture, the parts.

The palette is keyed by label id and shared by every stage, so a part keeps
its colour from the raw lift to the split. ``bake_labels_to_glb`` paints the
labels into a flat-colour texture on the mesh's own UVs, which is what
SegviGen's split (``geosam2.util.split``) reads. ``export_parts`` cuts the
mesh into one geometry per label, in the input's own frame.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import trimesh
from PIL import Image

from geosam2.util.logs import get_logger
from geosam2.util.views import load_mesh

logger = get_logger("geosam2.labels")

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
                       size: int = 4096) -> Dict[int, Tuple[int, int, int]]:
    """Write ``mesh_path`` with a flat-colour texture carrying ``face_labels``.

    The mesh keeps its own UVs; each face's UV triangle is filled with its
    label's colour. Loaded with ``load_mesh``, as the labels were made (force="mesh",
    default processing) so the face order the labels index is the same.
    """
    mesh = load_mesh(mesh_path)
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


# ── Parts ────────────────────────────────────────────────────────────────────

# UNASSIGNED_LABELS: 999 is the value mask_aggregation initialises its label
# volume with (_lift.py), so it survives on any face no mask ever covered --
# the reference sample_00 run carries it on 0.7% of faces -- and counting it
# as a part inflates every part count by one and paints a phantom region. 0 is
# the background label the exporter paints black.
_UNLABELED_RGBA = np.array([*UNASSIGNED_RGB, 255], dtype=np.uint8)


def _rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def export_parts(
    mesh_path: Union[str, Path], face_label: np.ndarray
) -> Tuple[trimesh.Scene, Dict[str, Any]]:
    """Paint per-face labels onto the original mesh, one geometry per part.

    The GLB that the propagation exports is rebuilt in a rotated/translated/scaled
    frame, so it cannot be shown next to the input. The label array, by contrast,
    indexes the input mesh's faces directly -- so the source mesh is reloaded and
    coloured instead, keeping both viewers aligned.

    Returns the scene and its structure (``{"name", "children": [{"name",
    "color", "faces"}]}``).
    """
    mesh_path = Path(mesh_path)
    mesh = load_mesh(mesh_path)   # the loader the labels were made with
    face_label = np.asarray(face_label).reshape(-1)

    if len(face_label) != len(mesh.faces):
        raise RuntimeError(
            f"label/mesh mismatch: {len(face_label)} labels for {len(mesh.faces)} faces"
        )

    labels = np.unique(face_label)
    part_labels = [int(v) for v in labels if int(v) not in UNASSIGNED_LABELS]
    palette = label_palette(face_label)

    scene = trimesh.Scene()
    children: List[Dict[str, Any]] = []

    # Parts are named by label id, not by rank: the raw lift and the
    # post-process share ids, so "part_003" is the same part in both.
    for label in part_labels:
        part_mask = face_label == label
        rgba = np.array([*palette[label], 255], dtype=np.uint8)
        name = f"part_{label:03d}"
        _add_part(scene, mesh, part_mask, rgba, name)
        children.append({"name": name, "color": _rgb_to_hex(rgba), "faces": int(part_mask.sum())})

    # Every unassigned label collapses into one grey region: they all mean
    # the same thing, and how much of the mesh lands here is the signal --
    # a prompt set that misses whole areas shows up as this growing.
    unassigned = np.isin(face_label, UNASSIGNED_LABELS)
    unlabeled_faces = int(unassigned.sum())
    logger.info("[lift] %d parts over %d faces, %d unassigned (%.1f%%)",
                len(part_labels), len(mesh.faces), unlabeled_faces,
                100.0 * unlabeled_faces / max(len(mesh.faces), 1))
    if not part_labels:
        logger.warning("[lift] no part survived -- the seed prompt reached no face.")
    if unlabeled_faces:
        _add_part(scene, mesh, unassigned, _UNLABELED_RGBA, "unassigned")
        children.append({
            "name": "unassigned",
            "color": _rgb_to_hex(_UNLABELED_RGBA),
            "faces": unlabeled_faces,
        })

    structure = {"name": mesh_path.stem, "children": children}
    return scene, structure


def _add_part(
    scene: trimesh.Scene,
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    rgba: np.ndarray,
    name: str,
) -> None:
    part = mesh.submesh([mask], append=True)
    # Replace the visual rather than assigning to visual.vertex_colors: a
    # textured source mesh yields TextureVisuals, which has no vertex_colors
    # setter, so the assignment would be a silent no-op and the part would
    # keep the original texture instead of its part colour. COLOR_0 is
    # linear in glTF: encode, or the part renders lighter than its texture.
    rgba = np.array([*to_linear_u8(rgba[:3]), 255], dtype=np.uint8)
    part.visual = trimesh.visual.ColorVisuals(
        mesh=part, vertex_colors=np.tile(rgba, (len(part.vertices), 1))
    )
    part.metadata["name"] = name
    scene.add_geometry(part, node_name=name, geom_name=name)
