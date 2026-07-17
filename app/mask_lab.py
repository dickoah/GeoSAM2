"""Mask lab: the VLM stage that turns a GLB into a flat part map.

Upload a GLB. The app renders the PixMesh rig -- a 3/4 MAIN view plus a 2x2
describe grid -- asks the VLM to describe the object as a hierarchical assembly
tree, assigns one palette colour per leaf part, and has the VLM paint the MAIN
view into a flat part map with that palette imposed. This is PixMesh's SegviGen
guidance flow, ported; nothing here segments in 3D.

Run (geosam2 conda env, needs GEMINI_API_KEY in .env):
    python -m app.mask_lab
    # -> http://127.0.0.1:7862
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import trimesh
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from utils.mask_agent import (
    assign_palette_tree, describe_assembly, describe_grid,
    generate_part_map, leaf_parts, target_view, with_contours, _hex,
)

app = FastAPI(title="GeoSAM2 Mask Lab")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

LAB_ROOT = Path(tempfile.gettempdir()) / "geosam2_mask_lab"
LAB_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=LAB_ROOT), name="runs")


def _load_glb(upload: UploadFile) -> trimesh.Scene:
    suffix = Path(upload.filename or "mesh.glb").suffix.lower() or ".glb"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.file.read())
        tmp_path = Path(tmp.name)
    try:
        return trimesh.load(tmp_path, force="scene", process=False)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/")
def index() -> HTMLResponse:
    page = STATIC_DIR / "mask_lab.html"
    if not page.exists():
        return HTMLResponse("<h1>mask_lab.html not found</h1>", status_code=404)
    return HTMLResponse(page.read_text())


@app.get("/api/config")
def config() -> dict:
    return {"has_key": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))}


@app.post("/api/segment")
def segment(file: UploadFile = File(...)) -> dict:
    """Full VLM flow: render -> describe -> palette -> part map."""
    scene = _load_glb(file)
    run_id = f"{Path(file.filename or 'mesh').stem}_{uuid.uuid4().hex[:8]}"
    run_dir = LAB_ROOT / run_id
    run_dir.mkdir(parents=True)

    try:
        grid = describe_grid(scene)
        target = target_view(scene)
        grid.save(run_dir / "grid.png")
        target.save(run_dir / "target.png")
        with_contours(target).save(run_dir / "contours.png")

        assembly = describe_assembly(grid)
        palette = assign_palette_tree(assembly)
        part_map = generate_part_map(target, palette)
        part_map.save(run_dir / "partmap.png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    # A flat legend for the UI: which colour went to which part.
    seen, legend = set(), []
    for part in leaf_parts(assembly):
        if part.name in palette and part.name not in seen:
            seen.add(part.name)
            legend.append({"name": part.name, "color": _hex(palette[part.name])})

    (run_dir / "assembly.json").write_text(assembly.model_dump_json(indent=2))
    return {
        "run": run_id,
        "scene": assembly.scene_description,
        "grid": f"/runs/{run_id}/grid.png",
        "target": f"/runs/{run_id}/target.png",
        "contours": f"/runs/{run_id}/contours.png",
        "partmap": f"/runs/{run_id}/partmap.png",
        "parts": legend,
    }


if __name__ == "__main__":
    host = os.environ.get("GEOSAM2_MASK_LAB_HOST", "127.0.0.1")
    port = int(os.environ.get("GEOSAM2_MASK_LAB_PORT", "7862"))
    print(f"GeoSAM2 mask lab -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
