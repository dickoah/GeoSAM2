"""Run the seed-view picker over a folder of meshes and write an HTML report.

    python tests/pick_report.py MESH_DIR --out report.html [--views CACHE] [--marks marks.json] [--runs N]

Each mesh is rendered once into CACHE/<name>/ (skipped when the twelve views are
there), then picked ``--runs`` times; every pick's JSON lands in
CACHE/<name>/pick_<run>/stats.txt and in CACHE/results.json, so ``--no-pick``
rebuilds the report from what is cached. ``marks.json`` maps a mesh name to the
rig views (0-11) the user accepts; the report frames them and scores the picks.
"""
import argparse
import base64
import html
import json
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

HEIGHT = {0.0: "LEVEL", 25.0: "ABOVE", -25.0: "BELOW"}


def _views(mesh: Path, cache: Path) -> Path:
    out = cache / mesh.stem
    if not (out / "color_0011.webp").exists():
        from geosam2.util.views import render_views
        t = time.time()
        render_views(str(mesh), out)
        print(f"  rendered in {time.time() - t:.0f} s")
    return out


def _pick(views: Path, run: int) -> dict:
    from geosam2.util.guidance import pick_best_view, view_on_white
    from geosam2.util.views import NUM_VIEWS
    debug = views / f"pick_{run}"
    t = time.time()
    view = pick_best_view({v: view_on_white(views, v) for v in range(NUM_VIEWS)}, debug_dir=str(debug))
    text = (debug / "stats.txt").read_text()
    choice = json.loads(text[text.index("{"):]) if "{" in text else None
    return {"view": view, "seconds": round(time.time() - t, 1), "choice": choice}


def _thumb(path: Path, size: int = 200) -> str:
    img = Image.open(path).convert("RGBA")
    white = Image.new("RGB", img.size, (255, 255, 255))
    white.paste(img.convert("RGB"), (0, 0), img.getchannel("A"))
    white.thumbnail((size, size))
    buf = BytesIO()
    white.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _report(results: dict, marks: dict, cache: Path, out: Path) -> None:
    from geosam2.util.views import ELEVATIONS, NUM_VIEWS
    sections, hits, scored, calls, failed, seconds = [], 0, 0, 0, 0, 0.0
    for name, runs in sorted(results.items(), key=lambda kv: kv[0].lower()):
        picks = [r["view"] for r in runs]
        votes = {v: picks.count(v) for v in set(picks)}
        calls += len(runs)
        failed += sum(r["choice"] is None for r in runs)
        seconds += sum(r["seconds"] for r in runs)
        truth = marks.get(name)
        verdict = ""
        if truth is not None:
            ok = sum(v in truth for v in picks)
            hits, scored = hits + ok, scored + len(picks)
            verdict = f"<span class='{'ok' if ok == len(picks) else 'ko'}'>{ok}/{len(picks)} on marks</span>"
        figures = []
        for v in range(NUM_VIEWS):
            cls = "truth" if truth and v in truth else ""
            badge = f"<i>{votes[v]}</i>" if v in votes else ""
            figures.append(f"<figure class='{cls}'><img src='{_thumb(cache / name / f'color_{v:04d}.webp')}'>"
                           f"<span class=marks>{badge}</span><figcaption>{v} {HEIGHT[ELEVATIONS[v]]}</figcaption></figure>")
        details = []
        for i, r in enumerate(runs):
            c = r["choice"]
            if c is None:
                details.append(f"run {i}: the call failed, fell back to view {r['view']}")
                continue
            cov = {}
            for e in c["seen"]:
                for t in set(e["tiles"]):
                    cov.setdefault(t, []).append(e["part"])
            per_tile = " ".join(f"{t}:{len(cov.get(t, []))}" for t in range(1, NUM_VIEWS + 1))
            seen = "\n".join(f"  {e['part']}: {sorted(set(e['tiles']))}" for e in c["seen"])
            details.append(f"run {i}: view {r['view']} (tile {r['view'] + 1}) in {r['seconds']} s\n"
                           f"parts per tile: {per_tile}\n{seen}\nanalysis: {c['analysis']}\nreason: {c['reason']}")
        marks_txt = f"<small>marks: {truth}</small>" if truth is not None else "<small>no marks</small>"
        sections.append(f"<section><h2>{html.escape(name)} {verdict}{marks_txt}</h2><div class=grid>{''.join(figures)}</div>"
                        f"<details><summary>audit</summary><pre>{html.escape(chr(10).join(details))}</pre></details></section>")
    summary = (f"<p>{len(results)} meshes, {calls} picks, {failed} failed calls, {seconds / max(calls, 1):.1f} s per pick"
               + (f", <b>{hits}/{scored} on marks</b>" if scored else "") + ".</p>")
    out.write_text(f"""<!doctype html><html lang=en><head><meta charset=utf-8><title>seed-view picks</title><style>
:root{{--bg:#fafaf9;--surface:#fff;--border:#d6d3d1;--ink:#292524;--ink2:#57534e;--ink3:#78716c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--surface:#292524;--border:#44403c;--ink:#f5f5f4;--ink2:#d6d3d1;--ink3:#a8a29e}}}}
body{{font:14px system-ui,sans-serif;background:var(--bg);color:var(--ink);margin:auto;padding:16px;max-width:1400px}}
section{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin:12px 0}}
h2{{font-size:15px;margin:0 0 8px;display:flex;gap:10px;align-items:baseline}}h2 small{{color:var(--ink3);font-weight:400}}
.ok{{color:#16a34a}}.ko{{color:#dc2626}}
.grid{{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:4px}}
@media (max-width:900px){{.grid{{grid-template-columns:repeat(6,minmax(0,1fr))}}}}
figure{{margin:0;position:relative;border:2px solid transparent;border-radius:6px;background:#fff}}
figure.truth{{border-color:#2563eb;box-shadow:0 0 0 2px #2563eb}}
figure img{{width:100%;display:block;border-radius:4px}}
figcaption{{font:11px ui-monospace,monospace;color:#57534e;padding:1px 4px}}
.marks{{position:absolute;top:3px;right:3px}}
.marks i{{font:bold 11px ui-monospace,monospace;color:#fff;background:#16a34a;border-radius:9px;min-width:18px;height:18px;display:flex;align-items:center;justify-content:center;font-style:normal}}
details{{margin-top:6px}}summary{{cursor:pointer;color:var(--ink2);font:12px ui-monospace,monospace}}
pre{{white-space:pre-wrap;font:11px ui-monospace,monospace;color:var(--ink2);background:var(--bg);padding:8px;border-radius:6px}}
</style></head><body><h1 style="font-size:20px">Seed-view picks</h1>
<p style="color:var(--ink2)">Blue frame: the views marked acceptable. Green badge: how many runs picked that view. Views 0-11 in rig order.</p>
{summary}{''.join(sections)}</body></html>""")
    print(f"report -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=Path, default=None, help="render cache (default: MESH_DIR/../<name>_views)")
    ap.add_argument("--marks", type=Path, default=None, help="json {name: [rig views]}")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None, help="mesh names to run")
    ap.add_argument("--no-pick", action="store_true", help="rebuild the report from cached picks")
    a = ap.parse_args()
    cache = a.views or a.mesh_dir.parent / f"{a.mesh_dir.name}_views"
    cache.mkdir(parents=True, exist_ok=True)
    marks = json.loads(a.marks.read_text()) if a.marks and a.marks.exists() else {}
    store = cache / "results.json"
    results = json.loads(store.read_text()) if store.exists() else {}

    meshes = sorted(a.mesh_dir.glob("*.glb"), key=lambda p: p.name.lower())
    if a.only:
        meshes = [m for m in meshes if m.stem in a.only]
    for mesh in meshes:
        print(f"== {mesh.stem}", flush=True)
        views = _views(mesh, cache)
        if a.no_pick:
            continue
        # A named mesh is re-picked from scratch; a full pass adds runs to what is cached.
        runs = results[mesh.stem] = [] if a.only else results.get(mesh.stem, [])
        for run in range(len(runs), len(runs) + a.runs):
            runs.append(_pick(views, run))
            print(f"  run {run}: view {runs[-1]['view']} in {runs[-1]['seconds']} s", flush=True)
            store.write_text(json.dumps(results, indent=1))
    _report({k: v for k, v in results.items() if (cache / k / "color_0011.webp").exists()}, marks, cache, a.out)


if __name__ == "__main__":
    main()
