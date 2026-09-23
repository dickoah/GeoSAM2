"""SegviGen's shared GLB/texture/adjacency helpers, used by the split.

Copied from SegviGen (pixmesh/backend/src/libraries/segvigen/util/_common.py, as of
pixmesh commit d6099934) so geosam2 stops loading it from a sibling checkout.
Owned here from now on; the original module docstring follows.

Shared low-level helpers: GLB texture extraction, RGB quantisation,
welded face adjacency.
"""

from __future__ import annotations

import json
import numpy as np
import struct
import trimesh
from PIL import Image
from scipy.sparse import coo_matrix, csr_matrix
from typing import Literal, Optional, Tuple

CHUNK_TYPE_JSON = 0x4E4F534A  # b'JSON'
CHUNK_TYPE_BIN = 0x004E4942   # b'BIN\0'


def load_mesh(
    input_fname: str,
    force: Literal["scene", "mesh"] = "mesh",
    process: bool = False,
) -> trimesh.Scene:
    scene = trimesh.load(input_fname,
                         force="scene",
                         process=process,
                         merge_primitives=False,
                         skip_materials=False,
                         maintain_order=True)
    if force == "scene":
        return scene

    placed = [(T, scene.geometry[name])
              for T, name in map(scene.graph.__getitem__, scene.graph.nodes_geometry)]
    parts = [g.copy().apply_transform(T) for T, g in placed
             if isinstance(g, trimesh.Trimesh)]
    authored_uv = all(getattr(g.visual, "uv", None) is not None
                      and len(g.visual.uv) == len(g.vertices) for g in parts)

    merged = trimesh.util.concatenate(parts)
    if not authored_uv:
        merged.visual = trimesh.visual.texture.TextureVisuals(
            uv=None, material=getattr(merged.visual, "material", None))
    return trimesh.Scene(merged)


def _quantize_rgb(rgb: np.ndarray, step: int) -> np.ndarray:
    if step is None or step <= 0:
        return rgb
    q = (rgb.astype(np.int32) + step // 2) // step * step
    return np.clip(q, 0, 255).astype(np.uint8)


def _load_glb_json_and_bin(glb_path: str) -> Tuple[dict, bytes]:
    data = open(glb_path, "rb").read()
    if len(data) < 12:
        raise RuntimeError("Invalid GLB: too small")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise RuntimeError("Not a GLB file (missing glTF header)")
    offset = 12
    gltf_json = None
    bin_chunk = None
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset: offset + chunk_len]
        offset += chunk_len
        if chunk_type == CHUNK_TYPE_JSON:
            gltf_json = chunk_data.decode("utf-8", errors="replace")
        elif chunk_type == CHUNK_TYPE_BIN:
            bin_chunk = chunk_data
    if gltf_json is None:
        raise RuntimeError("GLB missing JSON chunk")
    if bin_chunk is None:
        raise RuntimeError("GLB missing BIN chunk")
    return json.loads(gltf_json), bin_chunk


def _extract_basecolor_texture_image(glb_path: str, debug_print: bool = False) -> np.ndarray:
    gltf, bin_chunk = _load_glb_json_and_bin(glb_path)
    materials = gltf.get("materials", [])
    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    buffer_views = gltf.get("bufferViews", [])
    if not materials:
        raise RuntimeError("No materials in GLB")
    pbr = materials[0].get("pbrMetallicRoughness", {})
    base_tex_index = pbr.get("baseColorTexture", {}).get("index", None)
    if base_tex_index is None:
        raise RuntimeError("Material has no baseColorTexture")
    if base_tex_index >= len(textures):
        raise RuntimeError("baseColorTexture index out of range")
    tex = textures[base_tex_index]
    img_index = tex.get("source", None)
    if img_index is None or img_index >= len(images):
        raise RuntimeError("Texture has no valid image source")
    img_info = images[img_index]
    bv_index = img_info.get("bufferView", None)
    mime = img_info.get("mimeType", None)
    if bv_index is None:
        uri = img_info.get("uri", None)
        raise RuntimeError(f"Image is not embedded (bufferView missing). uri={uri}")
    if bv_index >= len(buffer_views):
        raise RuntimeError("image.bufferView out of range")
    bv = buffer_views[bv_index]
    bo = int(bv.get("byteOffset", 0))
    bl = int(bv.get("byteLength", 0))
    img_bytes = bin_chunk[bo: bo + bl]
    if debug_print:
        print(
            f"[Texture] baseColorTextureIndex={base_tex_index}, imageIndex={img_index}, "
            f"bufferView={bv_index}, mime={mime}, bytes={len(img_bytes)}"
        )
    pil = Image.open(trimesh.util.wrap_as_stream(img_bytes)).convert("RGBA")
    return np.array(pil, dtype=np.uint8)


# ── Welded face adjacency ────────────────────────────────────────────────────
#
# Welding matters because a UV seam splits a vertex, and a segmentation
# boundary likes to sit on exactly those edges: the authored indexing severs
# the adjacency there. Two notions are used across the split — faces sharing a
# welded EDGE, and faces sharing a welded VERTEX (the looser one, the only kind
# that survives the degenerate slivers welding creates).
#
# The weld tolerance stays a CALLER argument. The call sites use two different
# ones (`round(V, decimals=3)` and `round(V / (1e-6 * diag))`, the latter
# sometimes with a scene-wide diagonal rather than a per-mesh one); unifying
# them changes results and is a measured decision of its own.


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
