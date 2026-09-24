"""Capture run() outputs, then prove a change did not move them.

    python tests/regress.py capture BASELINE_DIR [case ...]   # on the old code
    python tests/regress.py verify  BASELINE_DIR [case ...]   # after the change

A case is a view directory holding mesh.glb and a mask_XXXX.png. With none
given, every example/sample_* that has both is used.

Each case runs the segmenter, then the split in both modes. Labels are compared
exactly; the GLBs on geometry, not bytes, since a glb embeds names and float32
buffers whose byte order trimesh does not promise across versions. A part is
compared on the surface it covers as well as its face count: the split cuts
triangles, so counts move with the cut while the surface does not.

Every artefact is compared, not just the last one: seeding k=2 neighbours
instead of 3 moved labels_post.npy while parts.glb and parts.npy stayed equal,
because the fragment cleanup absorbed the difference.

Exit code is 1 on any difference; do not pipe it through grep in CI.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _cases(argv):
    if argv:
        return [Path(a) for a in argv]
    out = []
    for d in sorted((ROOT / "example").glob("sample_*")):
        if (d / "mesh.glb").exists() and next(d.glob("mask_*.png"), None):
            out.append(d)
    return out


def _seed(case: Path):
    """The mask to seed with, and the view it belongs to, from its file name."""
    mask = sorted(case.glob("mask_*.png"))[0]
    return mask, int(re.search(r"(\d+)", mask.stem).group(1))


def _run(case: Path, out: Path):
    from geosam2.segmenter import GeoSAM2Segmenter
    from geosam2.util.split import split_glb_by_face_labels
    mask, view = _seed(case)
    seg = GeoSAM2Segmenter()
    seg.run(str(case), str(mask), view, str(out))
    for mode in ("sdf", "faces"):
        split_glb_by_face_labels(str(case / "mesh.glb"), np.load(out / "parts.npy"),
                                 str(out / f"split_{mode}.glb"), mode=mode,
                                 cleanup_fragments=False, debug_print=False)
    return out


def _fingerprint(out: Path) -> dict:
    """What must not move: the labels exactly, the parts geometry."""
    fp = {}
    for name in ("labels_raw.npy", "labels_post.npy", "parts.npy"):
        a = np.load(out / name)
        ids, counts = np.unique(a, return_counts=True)
        fp[name] = {
            "sha": hashlib.sha256(np.ascontiguousarray(a)).hexdigest(),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "per_label": {str(i): int(c) for i, c in zip(ids.tolist(), counts.tolist())},
        }
    for name in ("parts.glb", "split_sdf.glb", "split_faces.glb"):
        scene = trimesh.load(out / name, process=False)
        geo = scene.geometry if hasattr(scene, "geometry") else {"": scene}
        fp[name] = {
            "n_parts": len(geo),
            "names": sorted(geo),
            "faces": {k: int(len(g.faces)) for k, g in sorted(geo.items())},
            "verts": {k: int(len(g.vertices)) for k, g in sorted(geo.items())},
            # the split cuts triangles, so face counts move with the cut while the
            # surface a part covers does not: 4 decimals is ~0.01% of one part
            "area": {k: round(float(g.area), 4) for k, g in sorted(geo.items())},
            # rounded: a glb round-trip is float32, exact bytes are not the contract
            "volume": {k: round(float(g.volume), 6) for k, g in sorted(geo.items())},
        }
    return fp


def capture(baseline: Path, argv):
    baseline.mkdir(parents=True, exist_ok=True)
    for case in _cases(argv):
        out = baseline / case.name
        print(f"[capture] {case.name} ...", flush=True)
        _run(case, out / "run")
        (out / "fingerprint.json").write_text(json.dumps(_fingerprint(out / "run"), indent=1))
        print(f"[capture] {case.name} -> {out/'fingerprint.json'}")


def verify(baseline: Path, argv):
    bad = []
    for case in _cases(argv):
        ref_file = baseline / case.name / "fingerprint.json"
        if not ref_file.exists():
            print(f"[skip] {case.name}: no baseline")
            continue
        ref = json.loads(ref_file.read_text())
        print(f"[verify] {case.name} ...", flush=True)
        got = _fingerprint(_run(case, baseline / case.name / "new"))
        for key in sorted(ref):
            if ref[key] == got.get(key):
                print(f"   OK   {key}")
                continue
            bad.append((case.name, key))
            print(f"   DIFF {key}")
            for field in sorted(ref[key]):
                r, g = ref[key][field], got.get(key, {}).get(field)
                if r != g:
                    print(f"        {field}: baseline={_short(r)} now={_short(g)}")
    print()
    if bad:
        print(f"REGRESSION in {len(bad)} artefact(s): " +
              ", ".join(f"{c}/{k}" for c, k in bad))
        return 1
    print("identical to baseline")
    return 0


def _short(v, n=90):
    s = json.dumps(v)
    return s if len(s) <= n else s[:n] + "..."


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("capture", "verify"):
        print(__doc__)
        raise SystemExit(2)
    action, base = sys.argv[1], Path(sys.argv[2])
    raise SystemExit((capture if action == "capture" else verify)(base, sys.argv[3:]) or 0)
