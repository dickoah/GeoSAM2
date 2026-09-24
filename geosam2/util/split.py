"""SegviGen's split: a segmented GLB into one mesh per part.

Copied from SegviGen (pixmesh/backend/src/libraries/segvigen/util/split.py, as of
pixmesh commit d6099934) so geosam2 stops loading it from a sibling checkout.
Owned here from now on; the original module docstring follows.

Split a GLB into per-part sub-meshes from per-face labels (refinements + SDF cut).
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import trimesh
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import (breadth_first_order, connected_components,
                                  maximum_flow)
from scipy.spatial import cKDTree
from typing import Literal, Any, Dict, Optional, Tuple


# ── Shared low-level helpers (SegviGen's util/_common.py) ─────────────────────

def _load_glb(input_fname: str) -> trimesh.Scene:
    return trimesh.load(input_fname, force="scene", process=False,
                        merge_primitives=False, skip_materials=False, maintain_order=True)


def _weld_index(V: np.ndarray, decimals: Optional[int] = None,
                scale: Optional[float] = None) -> np.ndarray:
    """Map each vertex to its welded-position class."""
    V = np.asarray(V, dtype=np.float64)
    if decimals is not None:
        key = np.round(V, decimals=decimals)
    elif scale is not None:
        key = np.round(V / scale).astype(np.int64)
    else:
        raise ValueError("_weld_index needs either decimals= or scale=")
    return np.unique(key, axis=0, return_inverse=True)[1]


def _welded_face_pairs(weld: np.ndarray, F: np.ndarray,
                       drop_collapsed: bool = False,
                       dedupe: bool = False,
                       manifold_only: bool = False,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faces sharing a welded edge.

    Consecutive owners inside a welded-edge group are chained rather than
    requiring exactly two: welding fuses shells that merely touch, so this
    corpus has many non-manifold edges and a strict manifold test drops a fifth
    of the faces out of the adjacency entirely. ``manifold_only`` applies that
    strict test anyway, for label fields that must not cross shell contacts.

    ``drop_collapsed`` discards edges whose endpoints welded together (sliver
    meshes produce tens of thousands of them), ``dedupe`` removes self-pairs and
    keeps one pair per face couple. Both default to off: most call sites
    historically ran without them.

    Returns ``(pairs, sorted_half_edges, slots)`` — ``sorted_half_edges[slots]``
    gives the welded endpoints of the edge each pair sits on, which the MRF
    needs for edge length.
    """
    F = np.asarray(F, dtype=np.int64)
    nF = len(F)
    Fw = np.asarray(weld)[F]
    he = np.sort(np.stack([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]],
                          axis=1).reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(nF, dtype=np.int64), 3)
    if drop_collapsed:
        keep = he[:, 0] != he[:, 1]
        he, owner = he[keep], owner[keep]

    order = np.lexsort((owner, he[:, 1], he[:, 0]))
    hs, hown = he[order], owner[order]
    slots = np.flatnonzero(np.all(np.diff(hs, axis=0) == 0, axis=1))
    if manifold_only and len(slots):
        new = np.ones(len(hs), bool)
        new[1:] = np.any(np.diff(hs, axis=0) != 0, axis=1)
        gid = np.cumsum(new) - 1
        slots = slots[np.bincount(gid)[gid[slots]] == 2]
    pairs = np.stack([hown[slots], hown[slots + 1]], axis=1)

    if dedupe and len(pairs):
        ok = pairs[:, 0] != pairs[:, 1]
        pairs, slots = pairs[ok], slots[ok]
        if len(pairs):
            key = np.sort(pairs, axis=1)
            _, uniq = np.unique(key[:, 0] * nF + key[:, 1], return_index=True)
            pairs, slots = pairs[uniq], slots[uniq]
    return pairs, hs, slots


def _welded_face_incidence(weld: np.ndarray, F: np.ndarray,
                           dtype=np.float32) -> csr_matrix:
    """Face x welded-vertex incidence. ``M @ M.T`` links faces sharing ANY
    welded vertex."""
    F = np.asarray(F, dtype=np.int64)
    nF = len(F)
    Fw = np.asarray(weld)[F]
    rows = np.repeat(np.arange(nF, dtype=np.int64), 3)
    return coo_matrix((np.ones(3 * nF, dtype=dtype), (rows, Fw.ravel())),
                      shape=(nF, int(Fw.max()) + 1)).tocsr()


def _unwrap_uv3_for_seam(uv3: np.ndarray) -> np.ndarray:
    out = uv3.copy()
    for d in range(2):
        v = out[:, :, d]
        vmin = v.min(axis=1)
        vmax = v.max(axis=1)
        seam = (vmax - vmin) > 0.5
        if np.any(seam):
            vv = v[seam]
            vv = np.where(vv < 0.5, vv + 1.0, vv)
            out[seam, :, d] = vv
    return out


def _same_label_components(labels: np.ndarray, edges: np.ndarray, F: int):
    """Connected components of faces linked by an adjacency edge to a same-label
    neighbour. Returns ``(n_components, comp_labels)``."""
    sub_edges = edges[labels[edges[:, 0]] == labels[edges[:, 1]]]
    if len(sub_edges) == 0:
        return F, np.arange(F)
    graph = coo_matrix((np.ones(len(sub_edges), dtype=bool),
                        (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F))
    return connected_components(graph.maximum(graph.T), directed=False)


def _authored_islands(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-face authored UV chart id (identity weld = authored indexing).

    Charts are the labelling scope of every pre-cut stage: an authored seam is
    a fact about the asset, and no stage may carry a label across one. The
    hard version of that rule is ``collapse_islands_to_majority``; the soft
    stages below use the same ids to bound their adjacency.
    """
    F = np.asarray(mesh.faces, dtype=np.int64)
    inc = _welded_face_incidence(np.arange(len(mesh.vertices)), F,
                                 dtype=np.int8)
    return connected_components(inc @ inc.T, directed=False)[1]


# Unplugged for now: breaks some cases. Kept for reference.
_P1_AREA_FLOOR = 0.005          # object-area share; see the phase 1 guard


def smooth_face_labels_by_topology(
    mesh: trimesh.Trimesh,
    face_label: np.ndarray,
    small_component_min_faces: int = 50,
    postprocess_iters: int = 3,
    islands: Optional[np.ndarray] = None,
    debug_print: bool = False,
) -> np.ndarray:
    labels = face_label.copy()
    F = len(mesh.faces)
    # physical adjacency: weld coincident vertices (UV seams split them), then
    # connect faces sharing ANY welded vertex. Edge-based face_adjacency breaks
    # on the degenerate faces welding creates on sliver meshes — fragmenting a
    # contiguous part into "small" components that the phases below then eat.
    M = _welded_face_incidence(_weld_index(mesh.vertices, decimals=3),
                               mesh.faces)
    A = (M @ M.T).tocoo()
    pair = A.row < A.col
    edges_all = np.stack([A.row[pair], A.col[pair]], axis=1)
    # An authored chart border is a barrier for phases 1-2: denoise INSIDE a
    # chart, never carry a label across a seam. The size threshold below counts
    # FACES, so without the barrier a neighbouring chart eats any part a coarse
    # mesh models with few triangles (a 2-face slab covering 2.4% of the drawer;
    # 14.8% of its surface rewritten overall, against 0.5% on the dense bench).
    isl = _authored_islands(mesh) if islands is None else islands
    n_isl = int(isl.max()) + 1
    edges = edges_all[isl[edges_all[:, 0]] == isl[edges_all[:, 1]]]

    # Phase 1: every under-sized component takes the majority label of the
    # LARGE components around it in its own chart. Absolute face threshold,
    # plus an object-area floor: a real low-poly part can be 14 faces
    # (measured cliff: parts >= 0.93% of the object, noise <= 0.15%).
    n_lab = int(labels.max()) + 1 if labels.max() >= 0 else 0
    area = np.asarray(mesh.area_faces, np.float64)
    area_floor = _P1_AREA_FLOOR * area.sum()
    ea, eb = edges[:, 0], edges[:, 1]
    for _ in range(postprocess_iters if n_lab else 0):
        n_components, comp_labels = _same_label_components(labels, edges, F)
        small = np.bincount(comp_labels, minlength=n_components) \
            < small_component_min_faces
        comp_area = np.bincount(comp_labels, weights=area,
                                minlength=n_components)
        clab = np.full(n_components, -1, np.int64)
        clab[comp_labels] = labels
        small &= (clab < 0) | (comp_area < area_floor)
        if not small.any():
            break
        sa, sb = small[comp_labels[ea]], small[comp_labels[eb]]
        inner = np.concatenate([comp_labels[ea[sa & ~sb]],
                                comp_labels[eb[sb & ~sa]]])
        outer = np.concatenate([labels[eb[sa & ~sb]], labels[ea[sb & ~sa]]])
        keep = outer >= 0
        if not keep.any():
            break
        votes = np.zeros(n_components * n_lab, np.int64)
        np.add.at(votes, inner[keep] * n_lab + outer[keep], 1)
        votes = votes.reshape(n_components, n_lab)
        take = votes.sum(axis=1) > 0
        labels = np.where(take[comp_labels], votes.argmax(axis=1)[comp_labels],
                          labels)

    # Phase 2: each chart keeps its principal (area-majority) colour, and the
    # remaining small components rally to it. Area, not face count: phase 1's
    # per-face vote let the noise win inside a chart drawn with a handful of
    # big triangles (measured: 38.6% of the drawer surface flipped that way).
    snap = np.zeros(F, dtype=bool)
    principal = None
    if n_lab:
        ok = labels >= 0
        w = np.zeros(n_isl * n_lab)
        np.add.at(w, isl[ok] * n_lab + labels[ok], area[ok])
        w = w.reshape(n_isl, n_lab)
        has_ev = w.sum(axis=1) > 0
        principal = w.argmax(axis=1).astype(np.int32)
        n_components, comp_labels = _same_label_components(labels, edges, F)
        small = (np.bincount(comp_labels, minlength=n_components)
                 < small_component_min_faces)[comp_labels] & ok
        snap = small & has_ev[isl]
        labels[snap] = principal[isl[snap]]

    # Phase 3: orphan components inherit the majority of their surroundings
    # through shared WELDED VERTICES (degenerate slivers break edge-adjacency
    # but still share vertices). Components measured on the FULL adjacency —
    # a part spanning several charts must not fragment into per-chart pieces —
    # and a component wearing its chart's principal colour is not an orphan.
    _, comp_labels = _same_label_components(labels, edges_all, F)
    comp_sizes = np.bincount(comp_labels)
    orphan_mask = np.isin(comp_labels,
                          np.where(comp_sizes < small_component_min_faces)[0])
    orphan_mask |= labels < 0
    if principal is not None:
        orphan_mask &= ~(snap | ((labels >= 0) & (labels == principal[isl])))
    if orphan_mask.any() and (~orphan_mask).any():
        n_lab = int(labels.max()) + 1
        cur = labels.copy()
        cur[orphan_mask] = -1
        filled = int(orphan_mask.sum())
        while True:
            unk = cur < 0
            if not unk.any():
                break
            onehot = np.zeros((F, n_lab), np.float32)
            known = cur >= 0
            onehot[np.where(known)[0], cur[known]] = 1.0
            votes = M @ (M.T @ onehot)          # label mass via shared vertices
            votes[known] = 0
            fillable = unk & (votes.sum(axis=1) > 0)
            if not fillable.any():
                break
            cur[fillable] = votes[fillable].argmax(axis=1)
        still = cur < 0
        if still.any() and (~still).any():
            # truly disconnected islands: nearest labeled face as last resort
            cents = mesh.triangles_center
            good = np.where(~still)[0]
            _, nn = cKDTree(cents[good]).query(cents[still], workers=-1)
            cur[still] = cur[good[nn]]
        labels = cur
        if debug_print:
            print(f"  [Phase3] Absorbed {filled} orphan faces by "
                  f"vertex-topology majority")

    return labels


def _fill_gutter(label_map: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Replace uncovered gutter texels with their nearest covered label.

    Gutter padding is unreliable near island borders and distorts the SDF
    close to UV-island edges.  Propagating the nearest covered label prevents this.
    """
    src8     = (~cov).astype(np.uint8)   # 0 = covered (source), 1 = gutter
    _, ids   = cv2.distanceTransformWithLabels(
        src8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut      = np.zeros(ids.max() + 1, dtype=label_map.dtype)
    lut[ids[cov]] = label_map[cov]
    out      = label_map.copy()
    out[~cov] = lut[ids[~cov]]
    return out


def _signed_field(label_map: np.ndarray, k: int,
                  smooth: float = 1.5) -> np.ndarray:
    """Signed distance field (pixels) for label k: positive inside, negative outside.

    Guard the degenerate cases: for an all-background or all-foreground mask,
    ``cv2.distanceTransform`` returns FLT_MAX, which ``GaussianBlur`` then
    overflows to ±inf and later poisons the bilinear sampler (inf·0 = NaN).
    An empty label just gets a safe, large-negative uniform field so it
    never wins the face argmax; a full label gets a large-positive one.
    """
    mask  = (label_map == k).astype(np.uint8)
    count = int(mask.sum())
    if count == 0 or count == mask.size:
        return np.full(mask.shape, -1e6 if count == 0 else 1e6, np.float32)
    inside  = cv2.distanceTransform(mask,     cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    f = inside - outside
    if smooth > 0:
        f = cv2.GaussianBlur(f, (0, 0), smooth)
    return f


def _make_sdf_sampler(field: np.ndarray):
    """Return a bilinear sampler for *field*. UV origin is bottom-left (trimesh convention)."""
    fh, fw = field.shape

    def sample(uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        u  = np.mod(uv[:, 0], 1.0)          # repeat wrapping (safe for [0,1] too)
        v  = np.mod(uv[:, 1], 1.0)
        x  = u * (fw - 1)
        y  = (1.0 - v) * (fh - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, fw - 2)
        y0 = np.clip(np.floor(y).astype(int), 0, fh - 2)
        fx, fy = x - x0, y - y0
        return (field[y0,     x0    ] * (1 - fx) * (1 - fy)
              + field[y0,     x0 + 1] * fx        * (1 - fy)
              + field[y0 + 1, x0    ] * (1 - fx)  * fy
              + field[y0 + 1, x0 + 1] * fx         * fy)

    return sample


def _cut_along_field(V: np.ndarray, UV: np.ndarray, F: np.ndarray,
                     vals: np.ndarray, sampler,
                     tol: float = 0.05, bisect_iters: int = 16):
    """Marching-triangle cut: split every face that straddles the SDF zero.

    Cut points are cached per UV edge and per 3D edge (geo_cache) so that
    UV-seam-duplicated edges get the same 3D cut position → no cracks.
    """
    cat = np.zeros(len(vals), np.int8)
    cat[vals >  tol] =  1
    cat[vals < -tol] = -1
    fc         = cat[F]
    mixed_mask = (fc == 1).any(axis=1) & (fc == -1).any(axis=1)

    out_faces           = list(map(tuple, F[~mixed_mask]))
    V, UV               = V.copy(), UV.copy()
    new_pos, new_uv     = [], []
    cache               = {}   # UV-edge (i,j) → vertex index of cut point
    geo_cache           = {}   # 3D-edge key  → (t, exact position)
    nv                  = len(V)

    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) or 1.0
    res  = 1e-9 * diag

    def geo_edge_key(i, j):
        a = tuple(np.round(V[i] / res).astype(np.int64))
        b = tuple(np.round(V[j] / res).astype(np.int64))
        return ((a, b), False) if a <= b else ((b, a), True)

    def edge_point(i, j):
        nonlocal nv
        key = (i, j) if i < j else (j, i)
        if key in cache:
            return cache[key]
        gk, flip = geo_edge_key(i, j)
        if gk in geo_cache:
            t0, pos = geo_cache[gk]
            t = 1.0 - t0 if flip else t0
        else:
            vi            = float(vals[i])
            lo, hi, flo   = 0.0, 1.0, vi
            ui, uj        = UV[i], UV[j].copy()
            # unwrap the edge for seam-crossing faces (repeat textures):
            # interpolate along the short path, the sampler wraps back
            uj           += np.round(ui - uj)
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                fm  = float(sampler(ui + mid * (uj - ui))[0])
                if (fm >= 0) == (flo >= 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            t   = 0.5 * (lo + hi)
            pos = V[i] + t * (V[j] - V[i])
            geo_cache[gk] = (1.0 - t if flip else t, pos)
        # Snap to an existing vertex when the crossing is very close to an endpoint
        if t < 1e-4:
            cache[key] = i; return i
        if t > 1 - 1e-4:
            cache[key] = j; return j
        new_pos.append(pos)
        new_uv.append(UV[i] + t * (UV[j] - UV[i]))
        cache[key] = nv
        nv += 1
        return nv - 1

    for face in F[mixed_mask]:
        tri    = tuple(int(x) for x in face)
        c      = cat[list(tri)]
        zeros  = [i for i in range(3) if c[i] == 0]
        if zeros:
            z      = zeros[0]
            p1, p2 = tri[(z + 1) % 3], tri[(z + 2) % 3]
            m      = edge_point(p1, p2)
            out_faces += [(tri[z], p1, m), (tri[z], m, p2)]
        else:
            lone_sign = 1 if (c == 1).sum() == 1 else -1
            k          = next(i for i in range(3) if c[i] == lone_sign)
            lone, p1, p2 = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            m1, m2 = edge_point(lone, p1), edge_point(lone, p2)
            out_faces += [(lone, m1, m2), (m1, p1, p2), (m1, p2, m2)]

    if new_pos:
        V  = np.vstack([V,  np.array(new_pos)])
        UV = np.vstack([UV, np.array(new_uv)])
    return V, UV, np.array(out_faces, dtype=np.int64).reshape(-1, 3)


def _split_by_sdf_labels(V: np.ndarray, UV: np.ndarray, F: np.ndarray,
                         label_map: np.ndarray, smooth: float = 1.5):
    """Cut the mesh along every label boundary, then assign faces by SDF argmax.

    The N−1 smallest labels (by texture-pixel area) are cut sequentially.
    The final face→label assignment uses the argmax of all SDFs evaluated at
    face centroids, so no face can be left without a label.
    """
    n_labels = int(label_map.max()) + 1
    counts   = np.bincount(label_map.ravel(), minlength=n_labels)
    order    = np.argsort(counts)          # cut smallest labels first

    # Pre-compute all SDFs and samplers
    fields   = [_signed_field(label_map, k, smooth=smooth) for k in range(n_labels)]
    samplers = [_make_sdf_sampler(f) for f in fields]

    # Sequential cuts (skip the largest label — its boundary is implied)
    for lab in order[:-1]:
        vals      = samplers[lab](UV)
        V, UV, F  = _cut_along_field(V, UV, F, vals, samplers[lab])

    # Final assignment: argmax of all SDFs at face centroids
    # (per-face seam unwrap so seam-crossing centroids stay on the short path)
    cent_uv  = _unwrap_uv3_for_seam(UV[F].copy()).mean(axis=1)
    scores   = np.stack([s(cent_uv) for s in samplers])
    face_lab = scores.argmax(axis=0)
    # the cut creates new faces whose argmax can leave fresh micro-islands
    # near boundaries -> re-apply the absolute-size cleanup on the cut mesh
    face_lab = _cleanup_micro_components(V, F, face_lab, min_faces=50)

    # Filter degenerate faces (repeated vertex indices, not by area — area
    # filtering would create holes in small-but-real faces)
    ok = ((F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2]))

    parts = {}
    for lab in range(n_labels):
        Fk = F[ok & (face_lab == lab)]
        if len(Fk):
            used = np.unique(Fk)                # drop vertices no face references
            remap = -np.ones(len(V), dtype=np.int64)
            remap[used] = np.arange(len(used))
            parts[int(lab)] = (V[used], UV[used], remap[Fk])
    return parts


# ═════════════════════════════════════════════════════════════════════════════
#  MRF LABEL REFINEMENT (alpha-expansion graph cut)
#
#  Upstream stages fix labels that are wrong; none can move a boundary to the
#  crease where it belongs. Energy, minimised globally by alpha-expansion
#  (each move a scipy maximum_flow min-cut):
#
#      E(l) = sum_f  D_f(l_f)              data: -log P(label | own texels)
#           + sum_(f,g) w_fg [l_f != l_g]  Potts, cheap along concave creases
#
#  lam=20 is the measured ceiling (lam=40 drops a part on chaise1). Measured
#  negative, do not retry: discounting UV seams in the Potts weight (seams are
#  unwrapper cuts, not part boundaries — kevin 62 -> 72 open loops).
# ═════════════════════════════════════════════════════════════════════════════


# capacities must be int32; scale floats by this before rounding
_SCALE = 100
_D_CLIP = 10.0        # nats, keeps sum(D) far below the int32 ceiling
_CONCAVE_ETA = 0.15   # a valley costs this fraction of a flat edge to cut
_WEIGHT_FLOOR = 0.02  # no edge is ever free: keeps the partition from drifting


def _expand(face_label: np.ndarray, alpha: int, D: np.ndarray,
            pairs: np.ndarray, w_int: np.ndarray) -> np.ndarray:
    """One alpha-expansion move. x=1 (source side) means the face takes alpha."""
    n = len(face_label)
    active = face_label != alpha
    n_act = int(active.sum())
    if n_act == 0:
        return face_label
    compact = np.full(n, -1, dtype=np.int64)
    compact[active] = np.arange(n_act, dtype=np.int64)
    S, T = n_act, n_act + 1

    # unary: cut(S->f) <=> x=0 <=> keeps its label; cut(f->T) <=> takes alpha
    cap_s = np.rint(D[active, face_label[active]] * _SCALE).astype(np.int64)
    cap_t = np.rint(D[active, alpha] * _SCALE).astype(np.int64)

    rows = [np.full(n_act, S, np.int64), np.arange(n_act, dtype=np.int64)]
    cols = [np.arange(n_act, dtype=np.int64), np.full(n_act, T, np.int64)]
    data = [cap_s, cap_t]

    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        aa, ab = active[a], active[b]
        la, lb = face_label[a], face_label[b]

        # both free, same label: symmetric pair of n-links, energy w*[x_a != x_b]
        m = aa & ab & (la == lb)
        if m.any():
            ia, ib, ww = compact[a[m]], compact[b[m]], w_int[m]
            rows += [ia, ib]
            cols += [ib, ia]
            data += [ww, ww]

        # both free, different labels: E = w*(1 - x_a*x_b)
        #   = w*(1 - x_b) + w*(1-x_a)*x_b  ->  S->b  and  b->a
        m = aa & ab & (la != lb)
        if m.any():
            ia, ib, ww = compact[a[m]], compact[b[m]], w_int[m]
            rows += [np.full(len(ib), S, np.int64), ib]
            cols += [ib, ia]
            data += [ww, ww]

        # one side pinned to alpha: cost is w only when the free face keeps its
        # own (non-alpha) label -> a plain S-link on the free face
        for m, free_side in ((aa & ~ab, a), (ab & ~aa, b)):
            if not m.any():
                continue
            i = compact[free_side[m]]
            rows.append(np.full(len(i), S, np.int64))
            cols.append(i)
            data.append(w_int[m])

    cap = coo_matrix(
        (np.concatenate(data).astype(np.int32),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_act + 2, n_act + 2)).tocsr()
    cap.sum_duplicates()

    res = maximum_flow(cap, S, T)
    resid = (cap - res.flow).tocsr()
    resid.data = (resid.data > 0).astype(np.int32)
    resid.eliminate_zeros()
    reached = breadth_first_order(resid, S, directed=True,
                                  return_predecessors=False)

    src_side = np.zeros(n_act + 2, dtype=bool)
    src_side[reached] = True
    out = face_label.copy()
    takes = np.zeros(n, dtype=bool)
    takes[active] = src_side[:n_act]
    out[takes] = alpha
    return out


_BAND_RINGS = 3
_BAND_PRIOR = 1.0     # nats: cost of changing a face, scaled by its area
_BAND_AREA_CAP = 4.0  # low-poly slabs must not overpower lambda
_BAND_LOCK_SHARE = 0.05  # ...nor be repainted: a face this share of its mesh is a part, not a boundary
_BAND_CREASE = np.radians(15.0)


def refine_face_labels_boundary_band(
    mesh: trimesh.Trimesh,
    face_label: np.ndarray,
    lam: float = 20.0,
    debug_print: bool = False,
) -> np.ndarray:
    """Re-place part boundaries inside a +-``_BAND_RINGS`` band, across seams.

    Energy = change prior (area-capped) + dihedral-weighted length; no texel
    unary (it replays the paint dithering). Out-of-band faces leave the graph
    (a scalar lock loses to long edges — locked pairs fold into the free
    unary), adjacency is manifold-only (no shell contacts), and the concave
    discount requires a real crease (near-flat convexity is noise).
    """
    lab = np.asarray(face_label, np.int64).copy()
    labels = np.unique(lab[lab >= 0])
    if len(labels) < 2:
        return lab
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    inv = _weld_index(V, decimals=3)
    pairs, hs, sel = _welded_face_pairs(inv, F, drop_collapsed=True,
                                        manifold_only=True)
    if not len(pairs):
        return lab
    ends = np.zeros((int(inv.max()) + 1, 3), np.float64)
    ends[inv] = V
    e = hs[sel]
    length = np.linalg.norm(ends[e[:, 0]] - ends[e[:, 1]], axis=1)
    nF, L = len(F), int(lab.max()) + 1
    a, b = pairs[:, 0], pairs[:, 1]

    for _ in range(64):        # BFS fill of unlabelled faces (rare here)
        m = (lab[a] < 0) & (lab[b] >= 0)
        m2 = (lab[b] < 0) & (lab[a] >= 0)
        if not (m.any() or m2.any()):
            break
        lab[a[m]] = lab[b[m]]
        lab[b[m2]] = lab[a[m2]]

    ok = (lab[a] >= 0) & (lab[b] >= 0)
    band = np.zeros(nF, bool)
    diff = ok & (lab[a] != lab[b])
    band[a[diff]] = True
    band[b[diff]] = True
    for _ in range(_BAND_RINGS - 1):
        m = band[a] | band[b]
        band[a[m]] = True
        band[b[m]] = True
    area = np.asarray(mesh.area_faces, np.float64)
    band &= (lab >= 0) & (area < _BAND_LOCK_SHARE * area.sum())
    if not band.any():
        return lab

    # compact everything to the band: out-of-band faces have no say at all
    bidx = np.flatnonzero(band)
    cmp = np.full(nF, -1, np.int64)
    cmp[bidx] = np.arange(len(bidx))

    aw = np.clip(area / max(float(np.mean(area)), 1e-12), 0.0, _BAND_AREA_CAP)
    D = np.tile((_BAND_PRIOR * aw[bidx])[:, None], (1, L))
    D[np.arange(len(bidx)), lab[bidx]] = 0.0

    n = np.asarray(mesh.face_normals, np.float64)
    c = V[F].mean(axis=1)
    na = n[a]
    angle = np.arccos(np.clip(np.einsum("ij,ij->i", na, n[b]), -1.0, 1.0))
    convex = np.einsum("ij,ij->i", c[b] - c[a], na) < 0.0
    flat = np.clip((1.0 + np.cos(angle)) * 0.5, 0.0, 1.0)
    pen = np.where(convex | (angle < _BAND_CREASE), flat,
                   flat * _CONCAVE_ETA)
    ln = length / max(float(np.mean(length)), 1e-12)
    w_int = np.rint(lam * ln * np.maximum(pen, _WEIGHT_FLOOR)
                    * _SCALE).astype(np.int64)

    one = (band[a] ^ band[b]) & ok
    if one.any():
        fa = cmp[np.where(band[a[one]], a[one], b[one])]
        keep = lab[np.where(band[a[one]], b[one], a[one])]
        w = w_int[one] / _SCALE
        D += np.bincount(fa, weights=w, minlength=len(bidx))[:, None]
        np.subtract.at(D, (fa, keep), w)
    both = band[a] & band[b]
    pairs_b, w_b = cmp[pairs[both]], w_int[both]

    lab_b = lab[bidx]
    for _ in range(3):
        moved = 0
        for alpha in np.unique(lab_b):
            new = _expand(lab_b, int(alpha), D, pairs_b, w_b)
            moved += int((new != lab_b).sum())
            lab_b = new
        if moved == 0:
            break
    out = lab.copy()
    out[bidx] = lab_b
    if debug_print:
        print(f"  [band] {int((out != face_label).sum())} faces re-placed "
              f"({len(bidx)} in band)")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  ONE AUTHORED UV ISLAND = ONE PART
#
#  An authored atlas is cut along the object's real seams: a chart maps one
#  manufactured block, and every colour but its majority inside it is decoder
#  bleed — sitting on smooth geometry, no crease, no dihedral signal, so no
#  local stage can catch it. Islands = connected components of the face graph
#  under AUTHORED vertex indices (a seam splits the vertex). Measured to still
#  help on auto-unwrapped atlases (open loops 90 -> 46); a chart-likeness
#  guard was tried and removed — nothing discriminated, nothing to protect.
# ═════════════════════════════════════════════════════════════════════════════


def collapse_islands_to_majority(
    mesh: trimesh.Trimesh,
    face_label: np.ndarray,
    vote_label: Optional[np.ndarray] = None,
    dominance: float = 0.0,
    islands: Optional[np.ndarray] = None,
    debug_print: bool = False,
) -> np.ndarray:
    """Give every face of an island its island's area-weighted majority label.

    `dominance` > 0 spares islands whose majority holds less than that share.
    The vote runs on `vote_label` (the RAW texture labels) when given: voting
    on refined labels lets an upstream repaint tilt the majority and wipe out
    a small legitimate part (measured twice, chaise1 then bench/office).
    """
    isl = _authored_islands(mesh) if islands is None else islands
    lab = np.asarray(face_label).copy()
    vote = lab if vote_label is None else np.asarray(vote_label)
    ok = (lab >= 0) & (vote >= 0)
    if not ok.any():
        return lab

    area = np.asarray(mesh.area_faces, dtype=np.float64)
    n_isl = int(isl.max()) + 1
    n_lab = int(max(lab[ok].max(), vote[ok].max())) + 1
    weight = np.zeros(n_isl * n_lab, dtype=np.float64)
    np.add.at(weight, isl[ok] * n_lab + vote[ok], area[ok])
    weight = weight.reshape(n_isl, n_lab)

    total = weight.sum(axis=1)
    best = np.argmax(weight, axis=1)
    share = np.divide(weight[np.arange(n_isl), best], total,
                      out=np.zeros(n_isl), where=total > 0)

    take = (total > 0) & (share >= dominance)
    out = np.where(ok & take[isl], best[isl], lab)

    if debug_print:
        mixed = int(((weight > 0).sum(axis=1) > 1).sum())
        print(f"  [islands] {n_isl} ilots ({mixed} multi-couleurs), "
              f"{int((out != lab).sum())} faces reetiquetees")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  HYBRID PIPELINE — the label stages decide WHICH part each face belongs to;
#  the SDF marching-triangle cut only decides WHERE the boundary passes. The
#  cut has NO say in ownership: its label map is rasterised FROM the face
#  labels, so away from boundaries the argmax returns the stage-1 label.
# ═════════════════════════════════════════════════════════════════════════════


def _cleanup_micro_components(V: np.ndarray, F: np.ndarray,
                              face_lab: np.ndarray,
                              min_faces: int = 50,
                              iters: int = 3) -> np.ndarray:
    """Remove micro label-islands created by the cut+argmax stage.

    Stage-1 topology smoothing runs on the ORIGINAL mesh; the SDF cut then
    creates new faces whose argmax can flip tiny clusters near boundaries,
    leaving fresh micro-islands nothing cleans up.  This pass re-applies the
    legacy rule on the CUT mesh: connected components (position-welded
    adjacency, so UV seams don't split them) smaller than ``min_faces`` are
    reassigned to the neighbouring label with the most shared edges.
    """
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) or 1.0
    pairs, _, _ = _welded_face_pairs(_weld_index(V, scale=1e-6 * diag), F)

    lab = face_lab.copy()
    for _ in range(iters):
        intra = lab[pairs[:, 0]] == lab[pairs[:, 1]]
        g = coo_matrix((np.ones(int(intra.sum()), dtype=bool),
                        (pairs[intra, 0], pairs[intra, 1])),
                       shape=(len(F), len(F)))
        n, comp = connected_components(g.maximum(g.T), directed=False)
        sizes = np.bincount(comp, minlength=n)
        small = sizes < min_faces
        if not small.any():
            break
        votes: dict = {}
        inter = ~intra
        for fa, fb in pairs[inter]:
            for f_s, f_o in ((fa, fb), (fb, fa)):
                c = comp[f_s]
                if small[c] and not small[comp[f_o]]:
                    k = (int(c), int(lab[f_o]))
                    votes[k] = votes.get(k, 0) + 1
        if not votes:
            break
        best: dict = {}
        for (c, l), cnt in votes.items():
            if cnt > best.get(c, (0, None))[0]:
                best[c] = (cnt, l)
        changed = False
        for c, (cnt, l) in best.items():
            lab[comp == c] = l
            changed = True
        if not changed:
            break
    return lab


def _rasterize_face_labels(uv: np.ndarray, faces: np.ndarray,
                           face_label: np.ndarray, w: int, h: int) -> np.ndarray:
    """Rasterise per-face labels into a HxW int32 label map (texture space).

    Faces crossing the UV 0/1 border (repeat wrapping) are drawn twice with a
    ±1 offset so both sides of the seam are covered.  Texels covered by no
    face (gutter) get the nearest covered label via _fill_gutter.
    Returns a label map whose interior regions replicate the face labels
    exactly — the SDF refinement therefore cannot change region ownership.
    """
    canvas = Image.new("I", (w, h), 0)          # 0 = uncovered
    drw = ImageDraw.Draw(canvas)
    uv3 = _unwrap_uv3_for_seam(uv[faces].astype(np.float64))   # (F, 3, 2)

    x = uv3[:, :, 0] * (w - 1)
    y = (1.0 - uv3[:, :, 1]) * (h - 1)          # trimesh UV origin is bottom-left
    xw = (uv3[:, :, 0] - 1.0) * (w - 1)
    yw = (1.0 - (uv3[:, :, 1] - 1.0)) * (h - 1)
    wrap_x = uv3[:, :, 0].max(axis=1) > 1.0     # drawn twice across the seam
    wrap_y = uv3[:, :, 1].max(axis=1) > 1.0
    for i, lab in enumerate(face_label):
        if lab < 0:
            continue
        drw.polygon(list(zip(x[i].tolist(), y[i].tolist())), fill=int(lab) + 1)
        if wrap_x[i]:
            drw.polygon(list(zip(xw[i].tolist(), y[i].tolist())), fill=int(lab) + 1)
        if wrap_y[i]:
            drw.polygon(list(zip(x[i].tolist(), yw[i].tolist())), fill=int(lab) + 1)

    raw = np.asarray(canvas, dtype=np.int32)
    cov = raw > 0
    if not cov.any():
        raise ValueError("Face-label rasterisation covered no texel.")
    label_map = np.maximum(raw - 1, 0).astype(np.int32)
    label_map = _fill_gutter(label_map, cov)
    return label_map, cov


def _smooth_label_map_boundaries(label_map: np.ndarray, sigma: float,
                                 iters: int = 2,
                                 min_keep: float = 0.5) -> np.ndarray:
    """Smooth INTER-LABEL boundaries of a label map (curvature-flow style).

    Each iteration: per-label SDF blurred with ``sigma`` → argmax partition.
    This rounds the triangle-scale zigzag that face-label rasterisation
    produces, while the argmax keeps a valid partition (no gaps/overlaps).

    Anti-erosion guard: if any live label would lose more than
    ``1 - min_keep`` of its texels (thin parts eroded by the blur), sigma is
    halved and the iteration retried; below sigma=1 we stop and keep the
    current map.  Region ownership away from boundaries is untouched.
    """
    n_labels = int(label_map.max()) + 1
    orig_counts = np.bincount(label_map.ravel(), minlength=n_labels)
    alive = orig_counts > 0
    lm = label_map
    it = 0
    while it < iters and sigma >= 1.0:
        # running argmax: a stacked (n_labels, H, W) would peak at ~400 MB
        best = _signed_field(lm, 0, smooth=sigma)
        cand = np.zeros(lm.shape, np.int32)
        for k in range(1, n_labels):
            f = _signed_field(lm, k, smooth=sigma)
            m = f > best
            best[m], cand[m] = f[m], k
        counts = np.bincount(cand.ravel(), minlength=n_labels)
        if np.any(alive & (counts < min_keep * orig_counts)):
            sigma *= 0.5          # too aggressive for the thinnest part
            continue
        lm = cand
        it += 1
    return lm


# ═════════════════════════════════════════════════════════════════════════════
#  FRAGMENT REASSIGNMENT (post-split)
#
#  Texture speckle leaves each label a correct main body plus small detached
#  fragments sitting on a NEIGHBOURING label's surface; those are reassigned
#  to the label they touch. Size criterion = surface AREA (face counts and
#  bbox diagonals both mislead on slivers); the distance guard applies to the
#  host contact, never to the fragment's own main body (legitimate satellites
#  measured 3-80% of the diagonal away); mid-size components move only under
#  the orphan rule — size thresholds cannot make that call.
# ═════════════════════════════════════════════════════════════════════════════


def _rebuild_node(chunks, template_visual):
    """Assemble face chunks (possibly from several source nodes) into one
    Trimesh: concatenate, weld duplicates on (position, uv) — welding on
    position alone would erase UV seams and tear the texture — then drop
    index-degenerate faces and unreferenced vertices."""
    Vs, UVs, Fs, off = [], [], [], 0
    has_uv = all(uv is not None for _, uv, _ in chunks)
    for V, uv, F in chunks:
        used = np.unique(F)
        remap = np.full(len(V), -1, np.int64)
        remap[used] = np.arange(len(used))
        Vs.append(V[used])
        if has_uv:
            UVs.append(uv[used])
        Fs.append(remap[F] + off)
        off += len(used)
    V = np.vstack(Vs)
    F = np.vstack(Fs)
    UV = np.vstack(UVs) if has_uv else None

    # weld exact duplicates on the composite (position, uv) key
    key = np.round(V * 1e6).astype(np.int64)
    if UV is not None:
        key = np.hstack([key, np.round(UV * 1e6).astype(np.int64)])
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    F = inv[F]
    V, UV = V[first], (UV[first] if UV is not None else None)

    # index-degenerate faces created by the weld, then unreferenced vertices
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    V, F = V[used], remap[F]
    UV = UV[used] if UV is not None else None

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if isinstance(template_visual, trimesh.visual.ColorVisuals):
        rgba = np.asarray(template_visual.vertex_colors[0], np.uint8) \
            if len(template_visual.vertex_colors) else np.array([128] * 4, np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=np.tile(rgba, (len(V), 1)))
    else:
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=UV, material=template_visual.material)
    return mesh


def _pass(nodes, diag, frag_max_rel, host_min_rel, contact_max_frac, vote_k,
          debug_print):
    """One classify -> vote -> rebuild pass. Returns (new_nodes, n_moved).

    Rebuilding can weld two arriving fragments into one component whose joint
    vote differs from their individual ones, so a single pass is not always a
    fixed point — the caller iterates until it is (measured: 2-3 passes).
    """
    # ── per-label components + classification by relative area ──
    comps = []          # {node, fmask, area, rel}
    for ni, n in enumerate(nodes):
        # component id per face, welding vertices BY POSITION so UV seams
        # don't fragment the partition; the geometry itself is untouched
        M = _welded_face_incidence(_weld_index(n["V"], scale=1e-6 * diag),
                                   n["F"], dtype=bool)
        cid = connected_components(M @ M.T, directed=False)[1]
        tri = n["V"][n["F"]]
        fa = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        areas = np.zeros(cid.max() + 1)
        np.add.at(areas, cid, fa)
        main = float(areas.max()) or 1.0
        for k in range(cid.max() + 1):
            comps.append(dict(node=ni, fmask=cid == k,
                              area=float(areas[k]), rel=float(areas[k]) / main))

    hosts = [c for c in comps if c["rel"] >= host_min_rel]
    frags = [c for c in comps if c["rel"] < frag_max_rel]
    gray = [c for c in comps if frag_max_rel <= c["rel"] < host_min_rel]

    def _centroids(c):
        return nodes[c["node"]]["V"][nodes[c["node"]]["F"][c["fmask"]]].mean(1)

    # ── fragment -> host label by k-NN vote, gated on contact distance ──
    moved = []
    if hosts:
        host_cent = np.vstack([_centroids(c) for c in hosts])
        host_lab = np.concatenate([
            np.full(int(c["fmask"].sum()), c["node"]) for c in hosts])
        tree = cKDTree(host_cent)
        k = min(vote_k, len(host_cent))
        for c in frags:
            d, idx = tree.query(_centroids(c), k=k, workers=-1)
            if d.min() > contact_max_frac * diag:
                # no host within reach: keep, never delete (measured 23% of a
                # drawer destroyed when this path dropped faces)
                continue
            c["dest"] = int(np.bincount(host_lab[np.atleast_2d(idx).ravel()],
                                        minlength=len(nodes)).argmax())
            if c["dest"] != c["node"]:
                moved.append(c)

        # ── gray-zone orphan rule: a mid-size component touching NO surface
        #    of its OWN label while sitting on another's is mislabelled
        #    regardless of size — size thresholds cannot make this call ──
        if gray:
            by_lab: Dict[int, list] = {}
            for h in hosts:
                by_lab.setdefault(h["node"], []).append(_centroids(h))
            own_trees = {ni: cKDTree(np.vstack(p)) for ni, p in by_lab.items()}
            for c in gray:
                cent = _centroids(c)
                d_own, _ = own_trees[c["node"]].query(cent, workers=-1)
                if np.min(d_own) <= contact_max_frac * diag:
                    continue                 # attached to its own label: keep
                d, idx = tree.query(cent, k=k, workers=-1)
                labs = host_lab[np.atleast_2d(idx).ravel()]
                dd = np.atleast_2d(d).ravel()
                ok = (labs != c["node"]) & (dd <= contact_max_frac * diag)
                if not ok.any():
                    continue                 # floating mid-size piece: keep
                c["dest"] = int(np.bincount(labs[ok],
                                            minlength=len(nodes)).argmax())
                moved.append(c)

    # ── rebuild one geometry per label ──
    new_nodes = []
    for ni, n in enumerate(nodes):
        keep = np.ones(len(n["F"]), dtype=bool)      # own faces staying here
        for c in comps:
            if c["node"] == ni and c.get("dest", ni) != ni:
                keep &= ~c["fmask"]
        chunks = [(n["V"], n["uv"], n["F"][keep])] if keep.any() else []
        for c in comps:                              # incoming fragments
            if c.get("dest") == ni and c["node"] != ni:
                src = nodes[c["node"]]
                chunks.append((src["V"], src["uv"], src["F"][c["fmask"]]))
        n_in = sum(1 for c in comps if c.get("dest") == ni and c["node"] != ni)
        n_out = sum(1 for c in comps if c["node"] == ni and c.get("dest", ni) != ni)
        if not chunks:
            if debug_print:
                print(f"[PostProcess] {n['name']}: emptied (all components moved)")
            continue
        mesh = _rebuild_node(chunks, n["visual"])
        uv = getattr(mesh.visual, "uv", None)
        new_nodes.append(dict(
            name=n["name"], V=np.asarray(mesh.vertices, np.float64),
            uv=np.asarray(uv, np.float64) if uv is not None else None,
            F=np.asarray(mesh.faces, np.int64), visual=mesh.visual))
        if debug_print and (n_in or n_out):
            print(f"[PostProcess] {n['name']}: +{n_in} / -{n_out} fragments, "
                  f"{len(n['F'])} -> {len(mesh.faces)} faces")
    return new_nodes, len(moved)


def postprocess_split_glb(
    in_glb_path: str,
    out_glb_path: Optional[str] = None,
    frag_max_rel: float = 0.02,
    host_min_rel: float = 0.20,
    contact_max_frac: float = 0.03,
    vote_k: int = 9,
    max_passes: int = 1,
    debug_print: bool = True,
) -> Dict[str, Any]:
    """Reassign per-label speckle fragments to the label they touch.

    Only the unambiguous tail moves: fragment = component below
    ``frag_max_rel`` of its label's largest component, host = component above
    ``host_min_rel``, the grey zone in between never moves. Holes are never
    filled — a hole is the interface with the neighbouring part.

    ``max_passes`` stays at 1: pass 2 re-judges components pass 1 chose to
    KEEP, in a host landscape its own moves degraded (measured 10.3% of a
    drawer repainted; zero extra moves on the reference assets).

    ``contact_max_frac`` bounds the fragment-to-host distance (fraction of the
    object diagonal); ``vote_k`` = neighbour faces per fragment face in the
    vote. Returns a stats dict (moves, passes, per-label face counts).
    """
    if out_glb_path is None:
        out_glb_path = in_glb_path.replace(".glb", "_clean.glb")

    scene = _load_glb(in_glb_path)

    # ── flatten nodes (bake transforms), collect per-label arrays ──
    nodes = []          # {name, V, uv, F, visual}
    for node_name in scene.graph.nodes_geometry:
        T, geom_name = scene.graph[node_name]
        g = scene.geometry.get(geom_name)
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        V = np.asarray(g.vertices, np.float64)
        if T is not None and not np.allclose(T, np.eye(4)):
            V = V @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
        uv = getattr(g.visual, "uv", None)
        uv = np.asarray(uv, np.float64) if uv is not None and len(uv) == len(V) else None
        nodes.append(dict(name=geom_name, V=V, uv=uv,
                          F=np.asarray(g.faces, np.int64), visual=g.visual))
    if not nodes:
        raise ValueError(f"No triangle geometry in {in_glb_path}")

    faces_in = {n["name"]: int(len(n["F"])) for n in nodes}
    allV = np.vstack([n["V"] for n in nodes])
    diag = float(np.linalg.norm(allV.max(0) - allV.min(0))) or 1.0

    total_moved = 0
    passes = 0
    for passes in range(1, max_passes + 1):
        nodes, moved = _pass(
            nodes, diag, frag_max_rel, host_min_rel, contact_max_frac, vote_k,
            debug_print)
        total_moved += moved
        if moved == 0:
            break

    # geom_name must keep the trailing __rgb_R_G_B suffix intact:
    # downstream parses it anchored at end-of-string.
    out_scene = trimesh.Scene()
    stats: Dict[str, Any] = {"labels": {}, "moved": total_moved,
                             "passes": passes,
                             "diag": diag, "out_glb_path": out_glb_path}
    for n in nodes:
        mesh = trimesh.Trimesh(vertices=n["V"], faces=n["F"], process=False)
        mesh.visual = n["visual"]
        out_scene.add_geometry(mesh, geom_name=n["name"])
        stats["labels"][n["name"]] = dict(
            faces_in=faces_in[n["name"]], faces_out=int(len(n["F"])))
    for name in faces_in:
        if name not in stats["labels"]:
            stats["labels"][name] = dict(dropped=True, faces_in=faces_in[name])
    if debug_print:
        print(f"[PostProcess] {total_moved} fragments relabelled, "
              f"{passes} pass(es) -> {out_glb_path}")
    out_scene.export(out_glb_path)
    return stats




# ═════════════════════════════════════════════════════════════════════════════
#  SPLIT FROM PER-FACE LABELS: SegviGen's refinements and SDF cut, fed the
#  lift's labels directly.
# ═════════════════════════════════════════════════════════════════════════════


def split_glb_by_face_labels(
    in_glb_path: str,
    face_labels: np.ndarray,
    out_glb_path: Optional[str] = None,
    mode: Literal["sdf", "faces"] = "sdf",
    # ── Topology smoothing / denoising ──
    small_component_min_faces: int = 50,
    postprocess_iters: int = 3,
    # ── Boundary refinement ──
    smooth: float = 1.5,
    band_refine: bool = True,
    # ── One authored UV island = one part ──
    island_majority: bool = False,
    # ── Fragment cleanup (post-split) ──
    cleanup_fragments: bool = False,
    # ── Output ──
    min_faces_per_part: int = 1,
    debug_print: bool = True,
) -> str:
    """Split a GLB into per-part sub-meshes from labels indexing its faces.

    ``mode="sdf"`` keeps the marching-triangle cut, which
    evaluates per-label distance fields in UV space: boundaries fall inside
    triangles, as they do today, and the mesh needs usable UVs.
    ``mode="faces"`` needs none and leaves boundaries on existing edges.

    The MRF graph cut is texture-only (its data term counts a face's own
    texels), so it does not run here; ``band_refine`` places the boundaries
    instead, on dihedral angle alone.
    """
    from geosam2.util.labels import UNASSIGNED_LABELS, label_palette  # labels imports split

    if out_glb_path is None:
        out_glb_path = f"{os.path.splitext(in_glb_path)[0]}_faceseg.glb"

    scene = _load_glb(in_glb_path)
    face_labels = np.asarray(face_labels).reshape(-1)
    pal = label_palette(face_labels)
    out_scene = trimesh.Scene()
    part_count = 0
    base = os.path.splitext(os.path.basename(in_glb_path))[0]
    offset = 0
    by_label: Dict[int, list] = {}

    for node_name in scene.graph.nodes_geometry:
        geom_name = scene.graph[node_name][1]
        if geom_name is None:
            continue
        geom = scene.geometry.get(geom_name)
        if geom is None or not isinstance(geom, trimesh.Trimesh):
            continue
        mesh = geom.copy()
        T, _ = scene.graph.get(node_name)
        if T is not None:
            mesh.apply_transform(T)

        # Labels index the whole mesh's faces, node after node in scene order.
        if offset + len(mesh.faces) > len(face_labels):
            raise ValueError(
                f"{len(face_labels)} labels for at least "
                f"{offset + len(mesh.faces)} faces")
        raw_face_label, offset = face_labels[offset:offset + len(mesh.faces)], offset + len(mesh.faces)

        # -1 is "no label" downstream; the lift writes 0 for unassigned.
        raw_face_label = raw_face_label.astype(np.int64)
        ids = sorted(set(np.unique(raw_face_label).tolist()) - set(UNASSIGNED_LABELS))
        lut = np.full(int(raw_face_label.max()) + 1, -1, np.int32)
        lut[ids] = np.arange(len(ids))
        face_label = lut[raw_face_label]
        if not len(ids) or (face_label < 0).all():
            if debug_print:
                print(f"[{node_name}] no labelled face -> keep orig")
            out_scene.add_geometry(mesh, geom_name=f"{base}__{node_name}__orig")
            continue

        islands = _authored_islands(mesh)
        refine_labels = not island_majority
        if refine_labels:
            face_label = smooth_face_labels_by_topology(
                mesh, face_label,
                small_component_min_faces=small_component_min_faces,
                postprocess_iters=postprocess_iters,
                islands=islands, debug_print=debug_print)
        if refine_labels and band_refine:
            face_label = refine_face_labels_boundary_band(
                mesh, face_label, debug_print=debug_print)
        if island_majority:
            face_label = collapse_islands_to_majority(
                mesh, face_label, vote_label=face_label, islands=islands,
                debug_print=debug_print)

        used = np.unique(face_label[face_label >= 0])
        if debug_print:
            print(f"[{node_name}] faces={len(mesh.faces)} "
                  f"labels_used={len(used)} palette_size={len(ids)}")

        parts = None
        uv = getattr(mesh.visual, "uv", None)
        # one label: nothing to cut, and the 4096² maps cost ~0.5 s per node
        if mode == "sdf" and uv is not None and len(used) > 1:
            try:
                V_np = np.asarray(mesh.vertices, np.float64)
                UV_np = np.asarray(uv, np.float64)
                F_np = np.asarray(mesh.faces, np.int64)
                # the cut reads a rasterised label map, so its resolution is
                # the boundary's; 4096 is the resolution the cut was tuned on
                W = H = 4096
                label_map, cov = _rasterize_face_labels(UV_np, F_np, face_label, W, H)
                label_map = _fill_gutter(label_map, cov)
                label_map = _smooth_label_map_boundaries(label_map, 1.0, iters=1)
                parts = _split_by_sdf_labels(V_np, UV_np, F_np, label_map, smooth=smooth)
            except Exception as exc:  # pragma: no cover
                if debug_print:
                    print(f"[{node_name}] sdf cut failed ({exc}) -> face split")
                parts = None

        if parts is None:
            parts = {}
            for lab in np.unique(face_label[face_label >= 0]):
                sub = mesh.submesh([np.flatnonzero(face_label == lab)],
                                   append=True, repair=False)
                if isinstance(sub, (list, tuple)):
                    sub = sub[0] if sub else None
                if sub is None:
                    continue
                sub_uv = getattr(sub.visual, "uv", None)
                parts[int(lab)] = (
                    np.asarray(sub.vertices, np.float64),
                    np.asarray(sub_uv, np.float64) if sub_uv is not None
                    else np.zeros((len(sub.vertices), 2)),
                    np.asarray(sub.faces, np.int64),
                )

        for lab, (Vp, UVp, Fp) in parts.items():
            if len(Fp) < min_faces_per_part:
                continue
            # accumulated per LABEL, not per node: the labels span the whole
            # mesh, and a source split into material nodes would otherwise hand
            # back one part per (node, label) -- 16 for 7 labels on a sofa.
            chunk = trimesh.Trimesh(vertices=Vp, faces=Fp, process=False)
            chunk.visual = trimesh.visual.texture.TextureVisuals(uv=UVp)  # kept, as stage 6 does
            by_label.setdefault(ids[lab], []).append(chunk)

    for label_id, chunks in sorted(by_label.items()):
        m = trimesh.util.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if len(m.faces) < min_faces_per_part:
            continue
        rgb = np.array(pal[label_id], np.float64)
        # flat colour via the MATERIAL: ColorVisuals cannot carry uvs, and
        # baseColorFactor is linear
        m.visual = trimesh.visual.texture.TextureVisuals(
            uv=getattr(m.visual, "uv", None),
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[*(rgb / 255.0) ** 2.2, 1.0], metallicFactor=0.0))
        r, g, b = (int(x) for x in rgb)
        out_scene.add_geometry(
            m, geom_name=f"{base}__label_{label_id}__rgb_{r}_{g}_{b}")
        part_count += 1

    if part_count == 0:
        if debug_print:
            print("[INFO] no parts produced -> exporting original scene")
        out_scene = scene

    out_scene.export(out_glb_path)
    if debug_print:
        print(f"[INFO] exported {part_count} part(s) -> {out_glb_path}")

    if cleanup_fragments and part_count:
        try:
            postprocess_split_glb(out_glb_path, out_glb_path, debug_print=debug_print)
        except Exception as exc:
            if debug_print:
                print(f"[PostProcess] skipped ({type(exc).__name__}: {exc})")
    return out_glb_path
