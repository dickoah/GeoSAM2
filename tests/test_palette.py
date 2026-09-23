"""The one label palette: stable per id, split-safe spacing, grey unassigned."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geosam2.util.bake import UNASSIGNED_RGB, _MIN_SEP, label_palette, to_linear_u8  # noqa: E402


def test_same_id_same_colour_across_label_sets():
    raw = label_palette(np.array([1, 2, 3, 4, 5, 999]))
    post = label_palette(np.array([1, 3, 5]))          # ids dropped by the post-process
    assert all(raw[i] == post[i] for i in post), "colour must follow the id, not its rank"


def test_pairs_far_enough_for_the_split():
    pal = label_palette(np.arange(1, 25))
    cols = np.array(list(pal.values()), np.float64)
    d = np.linalg.norm(cols[:, None] - cols[None], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= _MIN_SEP, f"closest pair {d.min():.0f} < {_MIN_SEP}"


def test_unassigned_is_grey():
    pal = label_palette(np.array([0, 7, 999]))
    assert pal[0] == pal[999] == UNASSIGNED_RGB


def test_linear_encoding_matches_the_srgb_curve():
    # 224 sRGB is 0.745 linear = 190; 32 sRGB is 0.0144 linear = 4; 120 is 48.
    assert tuple(to_linear_u8((32, 224, 120))) == (4, 190, 48)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK ", name)
