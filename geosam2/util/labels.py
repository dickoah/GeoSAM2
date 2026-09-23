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
from scipy.spatial import cKDTree

from geosam2.util.logs import get_logger
from geosam2.util.split import (_CONCAVE_ETA, _D_CLIP, _SCALE, _WEIGHT_FLOOR, _expand,
                                _weld_index, _welded_face_pairs)
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


def _unwrap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """The same faces, in the same order, with a UV atlas from xatlas.

    Seam vertices are duplicated, so the vertex array changes; the faces keep
    their index and their positions, which is all the labels rely on.
    """
    import xatlas

    vmap, faces, uv = xatlas.parametrize(np.asarray(mesh.vertices, np.float32),
                                         np.asarray(mesh.faces, np.uint32))
    out = trimesh.Trimesh(vertices=mesh.vertices[vmap], faces=faces.astype(np.int64), process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uv)
    logger.info("[bake] xatlas: %d -> %d vertices", len(mesh.vertices), len(vmap))
    return out


# A part with fewer texels than this per face cannot carry its colour in the
# atlas: the bake paints a few texels the viewer smears, and the split has
# nothing to read. Measured: 8 texels/face at worst on the reference meshes,
# 0.01 on a desk whose UVs tile a wood pattern.
_MIN_TEXELS_PER_FACE = 4.0
_MIN_FACES_TO_JUDGE = 50


def _atlas_usable(mesh: trimesh.Trimesh, labels: np.ndarray, size: int) -> bool:
    """Whether every part of some size gets at least _MIN_TEXELS_PER_FACE texels per face.

    The UV area of each part's triangles, in texels of a ``size`` atlas, over its
    face count -- analytic, so it costs nothing on a dense mesh.
    """
    tri = np.asarray(mesh.visual.uv)[np.asarray(mesh.faces)] * size
    area = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    ids, inverse, n_faces = np.unique(labels, return_inverse=True, return_counts=True)
    texels = np.bincount(inverse, weights=area, minlength=len(ids))
    judged = n_faces >= _MIN_FACES_TO_JUDGE
    return bool(np.all(texels[judged] >= _MIN_TEXELS_PER_FACE * n_faces[judged]))


def bake_labels_to_glb(mesh_path: str, face_labels: np.ndarray, out_glb: str,
                       size: int = 4096, atlas: str = "auto") -> Dict[int, Tuple[int, int, int]]:
    """Write ``mesh_path`` with a flat-colour texture carrying ``face_labels``.

    Each face's UV triangle is filled with its label's colour. Loaded with
    ``load_mesh``, as the labels were made (force="mesh", default processing)
    so the face order the labels index is the same.

    ``atlas``: "keep" uses the mesh's own UVs; "xatlas" generates an atlas
    (same face order); "auto" keeps them when every part has room in them
    (see ``_atlas_usable``) and generates one otherwise. A mesh without UVs
    always gets one.
    """
    if atlas not in ("auto", "keep", "xatlas"):
        raise ValueError(f"atlas must be auto, keep or xatlas, got {atlas!r}")
    mesh = load_mesh(mesh_path)
    labels = np.asarray(face_labels).reshape(-1).astype(np.int64)
    if len(labels) != len(mesh.faces):
        raise ValueError(f"{len(labels)} labels for {len(mesh.faces)} faces")
    uv = getattr(mesh.visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(mesh.vertices)
    if not has_uv:
        logger.info("[bake] no UVs on the mesh: atlas generated")
    elif atlas == "xatlas":
        logger.info("[bake] atlas regenerated as asked")
    elif atlas == "auto" and not _atlas_usable(mesh, labels, size):
        logger.info("[bake] the mesh's UVs leave a part fewer than %g texels per face: atlas regenerated",
                    _MIN_TEXELS_PER_FACE)
        has_uv = False
    if not has_uv or atlas == "xatlas":
        mesh = _unwrap(mesh)
        uv = mesh.visual.uv

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

