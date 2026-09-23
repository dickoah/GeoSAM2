"""Compare two view directories, or a fresh render against a reference one.

This is the check util/views.py was validated against VAST's Blender renders
with (the bundled example/sample_* are those renders). It also tells whether
two renders of the same mesh differ beyond rasterisation noise.

    python -m tests.views_check example/sample_01                # re-render its mesh, compare
    python -m tests.views_check ref_views/ --against new_views/  # compare two directories
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from geosam2.util.views import INVALID_DEPTH, NUM_VIEWS, RESOLUTION, render_views

# The lifting step calls a sample point visible when its reprojected depth lands
# within 1e-3 of the depth map (``_lift.cal_link``), in a space
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
