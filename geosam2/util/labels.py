"""Per-face labels onto the input mesh: the palette, the fill, the parts.

The palette is keyed by label id and shared by every stage, so a part keeps its
colour from the raw lift to the split. ``export_parts`` cuts the mesh into one
geometry per label, in the input's own frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from geosam2.util.logs import get_logger
from geosam2.util.split import (_CONCAVE_ETA, _D_CLIP, _SCALE, _WEIGHT_FLOOR, _expand,
                                _weld_index, _welded_face_pairs)
from geosam2.util.views import load_mesh

logger = get_logger("geosam2.labels")

# THE palette: one colour per label id, the same at every stage (raw lift,
# post-process, fill, split), so a part keeps its colour from one viewer entry
# to the next. Keyed by the id's VALUE, not its rank -- the raw lift and the
# post-process carry different id sets, and a rank-based palette would shift
# every colour between the two.
#
# Colours come from the pool the guidance paints its part map with, picked the
# way it picks them (farthest-point over what is already taken), so a run's
# parts wear the hues the seed map showed. Past the pool they are generated on
# the hue circle at fixed saturation and lightness: the RGB-grid farthest point
# this used before had to reach the cube's corners and handed out saturated
# primaries. _MIN_SEP is a floor on the closest pair, warned about rather than
# enforced -- it was 64 while the texture split folded entries closer than
# palette_merge_dist=32, and nothing reads colours back now.
_MIN_SEP = 32.0
UNASSIGNED_LABELS = (0, 999)
UNASSIGNED_RGB = (120, 120, 120)
# The guidance's pool minus its grey (#7f7e80 sits 12 from UNASSIGNED_RGB and
# would read as "no label"); see guidance._KELLY_PALETTE.
_POOL = ("#dedede", "#333333", "#ebce2b", "#702c8c", "#ba1c30", "#5fa641",
         "#d485b2", "#db6917", "#4277b6", "#df8461", "#c0bd7f", "#463397",
         "#e1a11a", "#91218c", "#e8e948", "#7e1510", "#92ae31", "#6f340d",
         "#d32b1e", "#2b3514", "#96cde6")
_sequence: list = []


def _candidates() -> np.ndarray:
    """The grid colours past the pool may come from: muted, mid-lightness.

    A farthest-point walk over the whole RGB cube heads for its corners and
    hands out saturated primaries, which is what this palette used to look
    like. Bounding chroma and lightness keeps the generated ones in the same
    register as the pool's.
    """
    step = np.arange(24, 232, 8, dtype=np.float64)
    grid = np.stack(np.meshgrid(step, step, step, indexing="ij"), -1).reshape(-1, 3)
    chroma = grid.max(axis=1) - grid.min(axis=1)
    light = grid.mean(axis=1)
    return grid[(chroma >= 40) & (chroma <= 150) & (light >= 60) & (light <= 200)]


def _colour_sequence(n: int) -> list:
    """The first ``n`` palette colours, cached: colour k never changes."""
    if len(_sequence) >= n:
        return _sequence[:n]
    pool = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in _POOL]
    # Black and the unassigned grey are taken: a part must not wear either.
    taken = [(0, 0, 0), UNASSIGNED_RGB, *_sequence]
    grid = None
    while len(_sequence) < n:
        free = [c for c in pool if c not in _sequence]
        if not free:
            if grid is None:
                grid = _candidates()
            free = [tuple(int(v) for v in c) for c in grid]
        far = np.linalg.norm(np.array(free, np.float64)[:, None]
                             - np.array(taken, np.float64)[None], axis=2).min(axis=1)
        pick, gap = free[int(np.argmax(far))], float(far.max())
        if gap < _MIN_SEP:
            logger.warning("[palette] colour %d is only %.0f from another (< %.0f): "
                           "two parts may look alike", len(_sequence), gap, _MIN_SEP)
        _sequence.append(pick)
        taken.append(pick)
    return _sequence[:n]


def to_linear_u8(rgb) -> np.ndarray:
    """sRGB palette colour -> the linear value glTF stores in COLOR_0.

    Textures are sRGB in glTF and vertex colours are linear; a viewer converts
    the latter on output. The palette is authored in sRGB (it is what the
    swatches show), so a vertex-colour export has to be
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
    # bare: submesh would concatenate the texture into every part, _add_part drops it
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)

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


# ── Filling the unassigned faces ─────────────────────────────────────────────

def fill_labels(
    mesh_path: Union[str, Path],
    face_label: np.ndarray,
    lam: float = 1.0,
    thickness_weight: float = 1.0,
    crease_deg: float = 15.0,
    sweeps: int = 3,
    samples: int = 400_000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Give every unassigned face a label, following the geometry.

    Unassigned means "no evidence", never a part: those faces are the only
    free variables of an alpha-expansion graph cut over the mesh, the
    assigned faces are fixed and speak through their edges. Two terms:

    - between two faces, the length of their shared edge times how flat the
      fold is (``lam``): a boundary costs little on a crease and a lot across
      a flat surface, so labels spread over surfaces and stop at edges;
    - on each unassigned face, the label of the closest assigned surface
      (``thickness_weight``): the faces no view saw -- the back of a door
      panel -- take the label of what is on the other side of the thickness.

    Returns the labels and a small report.
    """
    mesh = load_mesh(mesh_path)
    labels = np.asarray(face_label).reshape(-1).astype(np.int64).copy()
    if len(labels) != len(mesh.faces):
        raise RuntimeError(f"label/mesh mismatch: {len(labels)} labels for {len(mesh.faces)} faces")
    un = np.isin(labels, UNASSIGNED_LABELS)
    report = {"unassigned_before": int(un.sum()), "unassigned_after": int(un.sum()),
              "from_neighbours": 0, "from_thickness": 0}
    if not un.any() or un.all():
        return labels, report

    ids = np.unique(labels[~un])                  # the real parts, sorted: searchsorted indexes them
    L = len(ids)
    area = np.asarray(mesh.area_faces, np.float64)

    # Through-thickness prior: the closest assigned surface, sampled densely.
    seen = mesh.submesh([~un], append=True, repair=False)
    points, tri = trimesh.sample.sample_surface(seen, samples, seed=0)
    _, nn = cKDTree(points).query(mesh.triangles_center[un], workers=-1)
    nearest = np.searchsorted(ids, labels[~un][tri[nn]])

    free = np.flatnonzero(un)
    cmp = np.full(len(labels), -1, np.int64)
    cmp[free] = np.arange(len(free))
    aw = np.clip(area[free] / max(float(area.mean()), 1e-12), 0.0, _D_CLIP)
    D = np.tile((thickness_weight * aw)[:, None], (1, L))
    D[np.arange(len(free)), nearest] = 0.0

    # Edges, weighted by length and fold: the split's boundary band terms.
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    inv = _weld_index(V, decimals=3)
    pairs, hs, sel = _welded_face_pairs(inv, F, drop_collapsed=True)
    ends = np.zeros((int(inv.max()) + 1, 3), np.float64)
    ends[inv] = V
    e = hs[sel]
    length = np.linalg.norm(ends[e[:, 0]] - ends[e[:, 1]], axis=1)
    a, b = pairs[:, 0], pairs[:, 1]
    n = np.asarray(mesh.face_normals, np.float64)
    c = V[F].mean(axis=1)
    angle = np.arccos(np.clip(np.einsum("ij,ij->i", n[a], n[b]), -1.0, 1.0))
    convex = np.einsum("ij,ij->i", c[b] - c[a], n[a]) < 0.0
    flat = np.clip((1.0 + np.cos(angle)) * 0.5, 0.0, 1.0)
    pen = np.where(convex | (angle < np.radians(crease_deg)), flat, flat * _CONCAVE_ETA)
    ln = length / max(float(np.mean(length)), 1e-12)
    w = lam * ln * np.maximum(pen, _WEIGHT_FLOOR)

    # An edge from a free face to a fixed one folds into the free face's unary.
    one = un[a] ^ un[b]
    fa = np.where(un[a[one]], a[one], b[one])
    fixed = np.where(un[a[one]], b[one], a[one])
    fixed_idx = np.searchsorted(ids, labels[fixed])
    D += np.bincount(cmp[fa], weights=w[one], minlength=len(free))[:, None]
    np.subtract.at(D, (cmp[fa], fixed_idx), w[one])
    both = un[a] & un[b]
    pairs_f = cmp[pairs[both]]
    w_f = np.rint(w[both] * _SCALE).astype(np.int64)

    lab_f = nearest.copy()
    for _ in range(sweeps):
        moved = 0
        for alpha in range(L):
            new = _expand(lab_f, alpha, D, pairs_f, w_f)
            moved += int((new != lab_f).sum())
            lab_f = new
        if moved == 0:
            break
    labels[free] = ids[lab_f]
    if not np.array_equal(labels[~un], np.asarray(face_label).reshape(-1)[~un]):
        raise RuntimeError("fill_labels changed an assigned face; only the unassigned ones may move")
    report.update(unassigned_after=int(np.isin(labels, UNASSIGNED_LABELS).sum()),
                  from_neighbours=int((lab_f != nearest).sum()),
                  from_thickness=int((lab_f == nearest).sum()))
    logger.info("[fill] %d unassigned faces labelled (%d by their neighbours, %d through the thickness)",
                len(free), report["from_neighbours"], report["from_thickness"])
    return labels, report

