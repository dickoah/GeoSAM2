"""GeoSAM2 evaluation app: a small FastAPI server with a self-contained UI.

In-process, synchronous endpoints serving a single static page. No queue, no
callbacks — this is an evaluation bench, not a production service.

Run:
    python -m app.server
    # -> http://localhost:7861

/api/segment returns the segmented GLB; the part list rides along in the
X-Hierarchy response header (base64 JSON) so the GLB body never changes shape.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import io
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import trimesh
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.pipeline import REPO_ROOT, GeoSAM2Segmenter, blender_info, list_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("geosam2_app")

app = FastAPI(title="GeoSAM2 Evaluation")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SEGMENTER = GeoSAM2Segmenter()

# Segmentation runs off the event loop (see `segment`), so two requests could
# otherwise put two GeoSAM2 runs on the GPU at once and OOM. Serialise them —
# a queued request waits, but the rest of the API stays responsive meanwhile.
_segment_lock = asyncio.Lock()

# The segmenter deletes its own temp dirs, so results are copied here before
# being served or they would vanish underneath the response.
APP_TEMP = Path(tempfile.mkdtemp(prefix="geosam2_app_"))
atexit.register(lambda: shutil.rmtree(APP_TEMP, ignore_errors=True))

_run_counter = 0

# Everything the UI may set. `process` also takes the seed-prompt arguments, but
# those are derived server-side from `prompt`, never accepted raw.
_ALLOWED_SETTINGS = frozenset({
    "postprocess_pa",
    "enable_postprocess",
    "opposite_auto_segmentation",
    "mask_threshold",
    "render_samples",
})


@app.get("/")
async def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>index.html not found in app/static</h1>", status_code=404)
    return HTMLResponse(page.read_text())


@app.get("/api/status")
async def status() -> dict:
    """What the UI needs to know before it lets you press Segment."""
    checkpoint = SEGMENTER.checkpoint_path
    return {
        "checkpoint": {
            "path": str(checkpoint),
            "present": checkpoint.is_file(),
            "size_mb": round(checkpoint.stat().st_size / 1e6, 1) if checkpoint.is_file() else 0,
        },
        "blender": blender_info(),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


@app.get("/api/samples")
async def samples() -> dict:
    return {"samples": list_samples(REPO_ROOT)}


@app.get("/api/sample_mesh/{name}")
async def sample_mesh(name: str) -> FileResponse:
    """Serve a bundled sample's source mesh for the input viewer."""
    path = REPO_ROOT / "example" / name / "mesh.glb"
    if not _is_within(path, REPO_ROOT / "example") or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No mesh for sample '{name}'")
    return FileResponse(path, media_type="model/gltf-binary")


@app.post("/api/convert_to_glb")
async def convert_to_glb(file: UploadFile = File(...)) -> Response:
    """Load any mesh format and return a GLB for the input preview viewer."""
    suffix = Path(file.filename or "mesh.glb").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        scene = trimesh.load(tmp_path, force="scene")
        buffer = io.BytesIO()
        scene.export(buffer, file_type="glb")
        return Response(content=buffer.getvalue(), media_type="model/gltf-binary")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/segment")
async def segment(
    settings: str = Form(...),
    sample: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> FileResponse:
    """Segment a bundled sample or an uploaded mesh.

    ``settings`` is a JSON object of GeoSAM2 parameters. Exactly one of
    ``sample`` (a pre-rendered view directory) or ``file`` (a mesh, which needs
    Blender) must be provided.
    """
    global _run_counter

    sample_dir, upload_path = await _resolve_source(sample, file)
    source = sample_dir or upload_path

    try:
        params = _prepare_params(json.loads(settings), sample_dir)
        logger.info("Segmenting %s with %s", source, params)
        # GeoSAM2 takes ~100 s. Run it off the event loop, or every other
        # request — including the UI's own polling — stalls until it finishes.
        async with _segment_lock:
            result = await asyncio.to_thread(SEGMENTER.process, source, **params)

        if result["status"].startswith("Error"):
            logger.error("Segmentation failed: %s", result["status"])
            raise HTTPException(status_code=500, detail=result["status"])

        _run_counter += 1
        run_dir = APP_TEMP / f"run_{_run_counter:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        served = run_dir / "segmented.glb"
        shutil.copy2(result["glb_path"], served)
        _release(result["temp_dir"])

        payload = {
            "structure": result["structure"],
            "n_parts": result["n_parts"],
            "unlabeled_faces": result["unlabeled_faces"],
            "seed_view": result["seed_view"],
            "status": result["status"],
        }
        headers = {"X-Hierarchy": base64.b64encode(json.dumps(payload).encode()).decode()}
        return FileResponse(served, filename="segmented.glb",
                            media_type="model/gltf-binary", headers=headers)
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)


async def _resolve_source(
    sample: Optional[str], file: Optional[UploadFile]
) -> tuple[Optional[Path], Optional[Path]]:
    """Return ``(sample_dir, upload_path)``, exactly one of which is set."""
    if sample and file:
        raise HTTPException(status_code=400, detail="Provide either a sample or a file, not both.")
    if sample:
        path = REPO_ROOT / "example" / sample
        if not _is_within(path, REPO_ROOT / "example") or not path.is_dir():
            raise HTTPException(status_code=404, detail=f"Unknown sample '{sample}'")
        return path, None
    if file is None:
        raise HTTPException(status_code=400, detail="No input: provide a sample or upload a mesh.")

    info = blender_info()
    if not info["usable"]:
        raise HTTPException(
            status_code=503,
            detail=("Uploaded meshes must be rendered to 12 views by Blender first, "
                    f"which is unavailable: {info['note']}. Pick a bundled sample instead."),
        )
    suffix = Path(file.filename or "mesh.glb").suffix.lower() or ".glb"
    payload = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        return None, Path(tmp.name)


def _prepare_params(params: Dict[str, Any], sample_dir: Optional[Path]) -> Dict[str, Any]:
    """Turn the UI's settings into :meth:`GeoSAM2Segmenter.process` kwargs.

    The UI names a prompt file; resolving it to a path — and dispatching on what
    kind of prompt it is — stays server-side so no filesystem path crosses the
    wire. A ``.json`` is a point-prompt file (clicks, the reference pipeline); a
    ``.png``/``.npy``/``.exr`` is a ready-made seed mask.
    """
    params = dict(params)
    prompt = params.pop("prompt", None)

    # Reject unknown keys here rather than letting them reach process(**params),
    # where they raise a TypeError outside the segmenter's error handling and so
    # surface as an unhandled 500 instead of a readable message. A stale cached
    # UI posting a renamed field is the likely source.
    unknown = sorted(set(params) - _ALLOWED_SETTINGS)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown settings: {', '.join(unknown)}. "
                   f"Expected any of: {', '.join(sorted(_ALLOWED_SETTINGS))}.",
        )

    if not prompt:
        return params
    if sample_dir is None:
        raise HTTPException(status_code=400,
                            detail="Prompts are only available for bundled samples.")
    prompt_path = sample_dir / prompt
    if not _is_within(prompt_path, sample_dir) or not prompt_path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"Prompt '{prompt}' not found in {sample_dir.name}")

    if prompt_path.suffix.lower() == ".json":
        params["point_prompt_file"] = str(prompt_path)
    else:
        params["mask_path"] = str(prompt_path)
        params["mask_view"] = _view_index(prompt)
    return params


def _view_index(mask_name: str) -> int:
    """Derive the seed view from a ``mask_0000.png``-style name."""
    digits = Path(mask_name).stem.split("_")[-1]
    if not digits.isdigit():
        raise HTTPException(status_code=400,
                            detail=f"Cannot infer the view index from '{mask_name}'")
    return int(digits)


def _release(temp_dir: Optional[str]) -> None:
    """Drop a finished run's temp dir now instead of at interpreter exit."""
    if not temp_dir:
        return
    shutil.rmtree(temp_dir, ignore_errors=True)
    if temp_dir in SEGMENTER.temp_dirs:
        SEGMENTER.temp_dirs.remove(temp_dir)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # 127.0.0.1 by default: the app is a local eval bench, and binding 0.0.0.0
    # both exposes it on every interface and stops some editors from
    # auto-forwarding the port over SSH. Override with GEOSAM2_APP_HOST/PORT.
    host = os.environ.get("GEOSAM2_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("GEOSAM2_APP_PORT", "7861"))
    print(f"GeoSAM2 evaluation app -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
