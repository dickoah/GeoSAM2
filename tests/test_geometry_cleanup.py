"""Self-check for clean_label_fragments (utils/inference_utils.py).

Run directly: ``python tests/test_geometry_cleanup.py``. It fails silently in
both directions -- eating a real part or leaving speckle -- and either only
shows up as a wrong part count three stages later.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geosam2._lift import clean_label_fragments


def _two_spheres(subdivisions=4):
    """Two disconnected spheres, fine enough that face spacing is well under the
    contact reach (3% of the bbox diagonal) -- the regime a generated mesh is in."""
    a = trimesh.creation.icosphere(subdivisions=subdivisions)
    b = trimesh.creation.icosphere(subdivisions=subdivisions)
    b.apply_translation((4, 0, 0))
    return trimesh.util.concatenate([a, b]), len(a.faces), len(b.faces)


def test_speckle_absorbed():
    """A patch stranded inside another label's territory moves to its host.
    It must be a minority of its OWN label: a label present only as one small
    patch is a small part, not a fragment, and is left alone."""
    mesh, n_a, _ = _two_spheres()
    lab = np.ones(len(mesh.faces), dtype=np.int64)
    lab[n_a:] = 2                      # sphere B is label 2, legitimately
    lab[:6] = 2                        # ... with 6 faces stranded inside sphere A
    out = clean_label_fragments(lab, mesh).numpy()
    assert (out[:6] == 1).all(), f"speckle not absorbed: {out[:6]}"
    assert (out[n_a:] == 2).all(), "host label damaged"


def test_small_part_survives():
    """A genuinely small part is not a fragment. Measured on sample_05: judging
    size against the largest component on the MESH (not within the label)
    called a 236-face part speckle and collapsed 12 parts into 3."""
    mesh, n_a, n_b = _two_spheres()
    lab = np.ones(len(mesh.faces), dtype=np.int64)
    lab[n_a + n_b - 40:] = 3           # a small contiguous cap on sphere B
    out = clean_label_fragments(lab, mesh).numpy()
    assert set(np.unique(out)) == {1, 3}, f"small part eaten: {np.unique(out)}"


def test_doubled_geometry():
    """Coincident triangles (generated soup) must not break the proximity
    graph: a doubled sphere is still one component."""
    from geosam2._lift import _proximity_components
    a = trimesh.creation.icosphere(subdivisions=4)
    mesh = trimesh.util.concatenate([a, a.copy()])
    comp = _proximity_components(mesh, np.ones(len(mesh.faces), np.int64))
    assert len(np.unique(comp)) == 1, f"doubled sphere fragmented into {len(np.unique(comp))}"


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        fn()
        print(f"OK  {name}")
