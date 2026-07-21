"""Mask lab: the VLM stage that turns a GLB into a flat part map.

Upload a GLB. The app renders the PixMesh 3/4 rig -- a MAIN view plus a 2x2
describe grid -- asks the VLM to describe the object as a hierarchical assembly
tree, assigns one palette colour per leaf part, and has the VLM paint the MAIN
view into a flat part map with that palette imposed. Nothing here segments in 3D;
the raw VLM output is shown so the generation prompt can be judged directly.

Run (geosam2 conda env, needs GEMINI_API_KEY in .env):
    python -m app.mask_lab
    # -> http://127.0.0.1:7862
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import trimesh
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from utils.logs import DEV, get_logger
from utils.mask_agent import (
    Assembly, assign_palette_tree, describe_assembly, describe_grid,
    generate_part_map, leaf_parts, target_view, with_contours, _hex,
)

logger = get_logger("geosam2.mask_lab")

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


def _legend(assembly: Assembly, palette: dict) -> list:
    """Flat colour legend for the UI: one entry per coloured leaf part, in order."""
    seen, legend = set(), []
    for part in leaf_parts(assembly):
        if part.name in palette and part.name not in seen:
            seen.add(part.name)
            legend.append({"name": part.name, "color": _hex(palette[part.name])})
    return legend


@app.post("/api/segment")
def segment(file: UploadFile = File(...)) -> dict:
    """Full VLM flow on the PixMesh 3/4 rig: render -> describe -> palette ->
    part map.

    Persists the run's target view and assembly to disk so ``/api/generate_map``
    can re-paint the map from the same describe, without a second describe call.
    """
    scene = _load_glb(file)
    run_id = f"{Path(file.filename or 'mesh').stem}_{uuid.uuid4().hex[:8]}"
    run_dir = LAB_ROOT / run_id
    run_dir.mkdir(parents=True)

    logger.info("[mask_lab] === run %s: %s ===", run_id, file.filename)
    try:
        grid = describe_grid(scene)
        target = target_view(scene)
        grid.save(run_dir / "grid.png")
        target.save(run_dir / "target.png")
        with_contours(target).save(run_dir / "contours.png")
        logger.info("[mask_lab] rendered grid + target -> %s", run_dir)

        assembly = describe_assembly(grid)
        palette = assign_palette_tree(assembly)
        # Written before the (slow, fallible) paint so a describe survives even
        # if generation fails -- generate_map can then retry off it alone.
        (run_dir / "assembly.json").write_text(assembly.model_dump_json(indent=2))

        part_map = generate_part_map(target, palette)
        part_map.save(run_dir / "partmap.png")
    except Exception as exc:
        # The traceback is the whole point of a lab run that failed; it is
        # logged here rather than reduced to the string the UI shows.
        logger.exception("[mask_lab] run %s failed", run_id)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    legend = _legend(assembly, palette)
    logger.info("[mask_lab] === run %s done: '%s', %d parts ===",
                run_id, assembly.scene_description, len(legend))
    return {
        "run": run_id,
        "scene": assembly.scene_description,
        "grid": f"/runs/{run_id}/grid.png",
        "target": f"/runs/{run_id}/target.png",
        "contours": f"/runs/{run_id}/contours.png",
        "partmap": f"/runs/{run_id}/partmap.png",
        "parts": legend,
    }


@app.post("/api/generate_map")
def generate_map(run: str = Form(...)) -> dict:
    """Re-paint only the part map for an existing run -- the dedicated map button.

    Reuses the run's stored target view and assembly (so palette and target are
    identical to the last describe); the only thing that changes between calls is
    :func:`utils.mask_agent.generate_part_map` and the prompt it carries. This is
    what makes iterating on the generation prompt an apples-to-apples comparison.
    """
    run_dir = LAB_ROOT / run
    target_path = run_dir / "target.png"
    assembly_path = run_dir / "assembly.json"
    # Names are opaque uuids from segment(); reject anything that escapes LAB_ROOT
    # or was never described, rather than paint from a half-written run.
    if run_dir.resolve().parent != LAB_ROOT.resolve():
        raise HTTPException(status_code=400, detail=f"Invalid run id '{run}'")
    if not target_path.is_file() or not assembly_path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"Run '{run}' has no stored describe; segment it first.")

    logger.info("[mask_lab] === regenerate map for run %s ===", run)
    try:
        assembly = Assembly.model_validate_json(assembly_path.read_text())
        palette = assign_palette_tree(assembly)
        target = Image.open(target_path).convert("RGB")
        part_map = generate_part_map(target, palette)
        part_map.save(run_dir / "partmap.png")
    except Exception as exc:
        logger.exception("[mask_lab] regenerate map for run %s failed", run)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    legend = _legend(assembly, palette)
    logger.info("[mask_lab] === run %s map regenerated: %d parts ===", run, len(legend))
    return {
        "run": run,
        "scene": assembly.scene_description,
        "partmap": f"/runs/{run}/partmap.png",
        "parts": legend,
    }


if __name__ == "__main__":
    host = os.environ.get("GEOSAM2_MASK_LAB_HOST", "127.0.0.1")
    port = int(os.environ.get("GEOSAM2_MASK_LAB_PORT", "7862"))
    print(f"GeoSAM2 mask lab -> http://{host}:{port}")
    logger.info("[startup] DEV=%s, runs kept in %s", DEV, LAB_ROOT)
    uvicorn.run(app, host=host, port=port)
