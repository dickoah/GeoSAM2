"""app.py -- GeoSAM2 step-by-step segmentation app.

Same shape as the SegviGen app it mirrors: every stage is its own job, takes
file paths in and returns file paths out, so a stage can be re-run on its own
without redoing the ones before it. The SegviGen generative brick is replaced
by GeoSAM2, and because GeoSAM2 labels faces rather than painting a texture,
SegviGen's texture transfer + texture split collapse into one label -> parts
step.

    1. /api/jobs/render    mesh          -> 12 canonical views
    2. /api/jobs/pickview  views         -> the seed view SegviGen's picker chooses
    3. /api/jobs/guidance  views + view  -> SegviGen's describe/palette/paint -> the map
    4. /api/jobs/segment   views + map   -> GeoSAM2, as the paper runs it
    5. /api/jobs/bake      labels        -> the labels as a texture on the UVs
    6. /api/jobs/split     baked mesh    -> SegviGen's split (its code), one mesh per part

Run with:
    python app.py            # http://127.0.0.1:7862
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# The UI polls job status once a second; those lines drown the real log.
logging.getLogger("uvicorn.access").addFilter(
    type("_PollFilter", (logging.Filter,), {
        "filter": lambda self, r: "/api/jobs/" not in r.getMessage()
    })())

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from geosam2.segmenter import REPO_ROOT, GeoSAM2Segmenter
from geosam2.util.views import is_view_directory
from geosam2.util.logs import get_logger

# Before /api/status reports on it: utils.guidance loads the same file, but only
# once it is imported, which happens inside a job rather than at startup.
load_dotenv(Path(__file__).parent / ".env")
# SegviGen's guidance reads its models and key from os.environ at call time; its
# logfire instrumentation is optional and unconfigured here -- silence the
# warning it prints on every call.
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

logger = get_logger("geosam2.app2")

_STATIC_DIR = Path(__file__).parent / "static"
_WORK_ROOT = Path(tempfile.gettempdir()) / "geosam2_steps"
_WORK_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="GeoSAM2 -- step by step")
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_segmenter = GeoSAM2Segmenter()

# In-memory job store: a local single-user bench, so no persistence.
_jobs: Dict[str, Dict[str, Any]] = {}


def _run_job(job_id: str, fn, *args, **kwargs) -> None:
    try:
        _jobs[job_id] = {"status": "done", "result": fn(*args, **kwargs), "error": None}
    except Exception as exc:                      # noqa: BLE001 - reported to the UI
        print(traceback.format_exc(), flush=True)
        _jobs[job_id] = {"status": "error", "result": None,
                         "error": f"{type(exc).__name__}: {exc}"}


def _start_job(fn, *args, **kwargs) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "result": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, fn, *args), kwargs=kwargs,
                     daemon=True).start()
    return {"job_id": job_id}


def _require_dir(path: str) -> Path:
    p = Path(path)
    if not is_view_directory(p):
        raise HTTPException(400, f"not a complete view directory: {path}")
    return p


def _writable_views(path: str) -> Path:
    """A view directory the guidance stage may write its seed into.

    ``generate_seed`` writes ``mask_XXXX.png``
    next to the views, which is what the rest of the pipeline reads. For a
    bundled ``example/sample_*`` that would edit tracked files, so those are
    copied into the work area first; a directory already under it is used
    in place, which is what makes a stage re-runnable.
    """
    src = _require_dir(path)
    if _WORK_ROOT in src.parents:
        return src
    dst = _WORK_ROOT / f"{src.name}_{uuid.uuid4().hex[:8]}"
    shutil.copytree(src, dst)
    logger.info("[views] %s copied to %s (samples are read-only)", src.name, dst)
    return dst


# ── Static + files ─────────────────────────────────────────────────────────

@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/api/upload")
async def upload(file: UploadFile) -> dict:
    suffix = Path(file.filename or "mesh.glb").suffix or ".glb"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        shutil.copyfileobj(file.file, f)
        return {"path": f.name}


@app.get("/api/files")
def serve_file(path: str) -> FileResponse:
    if not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    # Work files are results the user is comparing; a stale cached copy under
    # the same URL (a re-run, a reload) would show the previous run's result
    # and make a parameter look like it does nothing.
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    return _jobs[job_id]


@app.get("/api/samples")
def samples() -> dict:
    """The bundled example/sample_* view directories, ready to skip step 1."""
    roots = sorted(p for p in (REPO_ROOT / "example").glob("sample_*")
                   if is_view_directory(p))
    return {"samples": [{"name": p.name, "path": str(p)} for p in roots]}


@app.get("/api/status")
def status() -> dict:
    import torch
    ckpt = _segmenter.checkpoint_path
    return {
        "checkpoint": {"path": str(ckpt), "present": ckpt.is_file()},
        "cuda": torch.cuda.is_available(),
        "gemini_key": bool(os.environ.get("GEMINI_API_KEY")
                           or os.environ.get("GOOGLE_API_KEY")),
    }


# ── 1. Render ──────────────────────────────────────────────────────────────

class RenderParams(BaseModel):
    mesh_path: str


@app.post("/api/jobs/render")
def start_render(params: RenderParams) -> dict:
    mesh_path = Path(params.mesh_path)
    if not mesh_path.is_file():
        raise HTTPException(400, f"mesh not found: {mesh_path}")

    def _run() -> dict:
        from geosam2.util.views import render_views
        out = _WORK_ROOT / f"views_{uuid.uuid4().hex[:8]}"
        render_views(str(mesh_path), out)
        return {"data_root": str(out), "mesh_path": str(out / "mesh.glb"),
                "views": [str(out / f"color_{v:04d}.webp") for v in range(12)]}

    return _start_job(_run)


# ── 2. Pick the seed view (SegviGen's picker) ─────────────────────────────

class PickViewParams(BaseModel):
    data_root: str


@app.post("/api/jobs/pickview")
def start_pickview(params: PickViewParams) -> dict:
    data_root = _require_dir(params.data_root)

    def _run() -> dict:
        from geosam2.util.guidance import VIEW_MAP, pick_seed_view
        from geosam2.util.views import AZIMUTHS_REFERENCE
        view = pick_seed_view(data_root)
        compass = ("FRONT", "FRONT-RIGHT", "RIGHT", "BACK-RIGHT",
                   "BACK", "BACK-LEFT", "LEFT", "FRONT-LEFT")
        label = compass[int(((AZIMUTHS_REFERENCE[view] + 22.5) % 360) // 45)]
        return {"seed_view": view, "label": label,
                "candidates": list(VIEW_MAP.values()),
                "view_image": str(data_root / f"color_{view:04d}.webp")}

    return _start_job(_run)


# ── 3. Guidance: SegviGen's describe / palette / paint, then points ────────

class GuidanceParams(BaseModel):
    data_root: str
    seed_view: int
    # SegviGen's own knobs (its test app exposes the same four).
    mode: str = "single"                 # "single" | "grid"
    describe_model: Optional[str] = None  # None -> utils.guidance defaults
    paint_model: Optional[str] = None
    resolution: int = 1024               # size the map is painted at


@app.post("/api/jobs/guidance")
def start_guidance(params: GuidanceParams) -> dict:
    data_root = _writable_views(params.data_root)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise HTTPException(400, "GEMINI_API_KEY is not set (see .env at the repo root).")

    def _run() -> dict:
        from geosam2.util.guidance import generate_seed
        kw = dict(size=params.resolution, mode=params.mode)
        if params.describe_model:
            kw["describe_model"] = params.describe_model
        if params.paint_model:
            kw["paint_model"] = params.paint_model
        seed = generate_seed(data_root, params.seed_view, **kw)
        return {
            "seed_view": seed.view,
            "data_root": str(data_root),   # the copy the seed was written into
            "map_path": str(seed.map_path),
            "source_path": str(data_root / f"color_{seed.view:04d}.webp"),
            "scene": seed.scene,
            "parts": [{"name": n, "hex": h, "pixels": seed.coverage.get(n, 0)}
                      for n, h in seed.parts.items()],
            "painted": sum(1 for v in seed.coverage.values() if v > 0),
        }

    return _start_job(_run)


# ── 4. GeoSAM2 itself ──────────────────────────────────────────────────────

class SegmentParams(BaseModel):
    data_root: str
    seed_view: int
    postprocess_pa: float = 0.02


def _parts_glb(data_root: Path, labels_path: Path, work: Path, name: str) -> dict:
    """Cut the source mesh into one geometry per label and write it out."""
    scene, structure = _segmenter.parts(data_root / "mesh.glb", np.load(labels_path))
    glb_path = work / f"{name}.glb"
    scene.export(glb_path)
    parts = [p for p in structure["children"] if p["name"] != "unassigned"]
    unlabeled = sum(p["faces"] for p in structure["children"] if p["name"] == "unassigned")
    return {"glb_path": str(glb_path), "labels_path": str(labels_path),
            "n_parts": len(parts), "unlabeled_faces": unlabeled,
            "parts": structure["children"]}


@app.post("/api/jobs/segment")
def start_segment(params: SegmentParams) -> dict:
    """GeoSAM2 as the paper runs it: propagate the painted map, lift, post-process.

    Deliberately stops there. Everything this project adds on top is the next
    stage, so the two can be looked at side by side.
    """
    data_root = _require_dir(params.data_root)
    mask_path = data_root / f"mask_{params.seed_view:04d}.png"
    if not mask_path.is_file():
        raise HTTPException(400, f"no painted map for view {params.seed_view}: {mask_path}")

    def _run() -> dict:
        work = _WORK_ROOT / f"seg_{uuid.uuid4().hex[:8]}"
        out_dir = work / "3d_seg"
        glb_path = _segmenter.run(data_root, mask_path, params.seed_view, out_dir,
                                  postprocess_pa=params.postprocess_pa)
        # Both of GeoSAM2's own outputs are handed back: the raw lift and its
        # post-process are two of the methods whose weight the later stages
        # are there to measure, so either can feed the bake.
        out = _parts_glb(data_root, out_dir / "labels_post.npy", work, "geosam2")
        raw = _parts_glb(data_root, out_dir / "labels_raw.npy", work, "geosam2_raw")
        out.update(work_dir=str(work), out_dir=str(out_dir), seed_view=params.seed_view,
                   mask_path=str(mask_path), glb_path=glb_path,
                   raw_labels_path=raw["labels_path"], raw_glb=raw["glb_path"])
        return out

    return _start_job(_run)


# ── 5-6. Bake the labels, then SegviGen's split ────────────────────────────

class BakeParams(BaseModel):
    """5 -- labels onto the mesh's own UVs, as a flat-colour texture."""
    data_root: str
    work_dir: str
    labels_path: str
    texture_size: int = 4096


class SplitParams(BaseModel):
    """6 -- SegviGen's split, its own knobs under its own names."""
    work_dir: str
    baked_glb: str
    color_quant_step: int = 16
    palette_min_frac: float = 0.0005
    palette_max_colors: int = 256
    palette_merge_dist: int = 32
    smooth: float = 1.5
    min_faces_per_part: int = 1
    island_majority: bool = False
    cleanup_fragments: bool = False
    output_mode: str = "vertex_colors"


SPLIT_FIELDS = ("color_quant_step", "palette_min_frac", "palette_max_colors",
                "palette_merge_dist", "smooth", "min_faces_per_part",
                "island_majority", "cleanup_fragments", "output_mode")


@app.get("/api/presets/split")
def split_presets() -> dict:
    from geosam2.util.split import SPLIT_PRESETS
    return SPLIT_PRESETS


@app.post("/api/jobs/bake")
def start_bake(params: BakeParams) -> dict:
    """5: the translation step -- labels become a texture the split can read."""
    data_root = _require_dir(params.data_root)
    work = Path(params.work_dir)
    if not work.is_dir():
        raise HTTPException(400, f"unknown work dir: {work}")
    if not Path(params.labels_path).is_file():
        raise HTTPException(400, f"labels not found: {params.labels_path} (re-run stage 4)")

    def _run() -> dict:
        from geosam2.util.labels import bake_labels_to_glb
        out = work / "segvigen"
        out.mkdir(parents=True, exist_ok=True)
        baked = out / f"baked_{params.texture_size}_{uuid.uuid4().hex[:6]}.glb"
        palette = bake_labels_to_glb(str(data_root / "mesh.glb"),
                                     np.load(params.labels_path), str(baked),
                                     size=params.texture_size)
        return {"baked_glb": str(baked), "n_labels": len(palette),
                "texture_size": params.texture_size}

    return _start_job(_run)


@app.post("/api/jobs/split")
def start_split(params: SplitParams) -> dict:
    """6: SegviGen's split, unchanged -- the only stage that cuts geometry."""
    work = Path(params.work_dir)
    if not work.is_dir():
        raise HTTPException(400, f"unknown work dir: {work}")
    if not Path(params.baked_glb).is_file():
        raise HTTPException(400, f"baked mesh not found: {params.baked_glb} (re-run stage 5)")

    def _run() -> dict:
        import trimesh

        from geosam2.util.split import split_glb_by_texture_palette_rgb
        # One file per parameter set: a fixed name would make two runs
        # indistinguishable in the viewer and on disk.
        tag = uuid.uuid4().hex[:6]
        out = work / "segvigen" / f"parts_{params.output_mode}_{tag}.glb"
        split_glb_by_texture_palette_rgb(params.baked_glb, str(out), debug_print=True,
                                         **{k: getattr(params, k) for k in SPLIT_FIELDS})
        scene = trimesh.load(out, force="scene")
        return {"segvigen_glb": str(out), "n_split_parts": len(scene.geometry),
                "output_mode": params.output_mode}

    return _start_job(_run)


if __name__ == "__main__":
    host = os.environ.get("GEOSAM2_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("GEOSAM2_APP_PORT", "7862"))
    # Before serving: a missing checkpoint otherwise only surfaces at stage 4, after
    # the render and the paid VLM stages have run.
    _segmenter.ensure_checkpoint()
    print(f"GeoSAM2 step-by-step app -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
