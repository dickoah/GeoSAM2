"""GeoSAM2 seed from SegviGen's guidance code, unchanged.

SegviGen's ``util/guidance.py`` is where the describe / palette / paint prompts
were tuned; geosam2's own copies had drifted (an older "detail-first" describe
prompt, a different grid, a different model) and on a sideboard returned two
parts for a piece with drawers and doors. This module feeds SegviGen's
functions geosam2's canonical renders and turns the map they paint into the
point prompts GeoSAM2 seeds from. Nothing of the prompts lives here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from utils.auto_prompt import prompts_from_color_map, write_prompts
from utils.logs import get_logger
from utils.split import SEGVIGEN_DIR

logger = get_logger("geosam2.guidance")

# SegviGen resolves its model from os.environ at call time; the repo's .env
# holds the Gemini key. Its logfire instrumentation is optional and unconfigured
# here -- silence the warning it prints on every call.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

# SegviGen's view names -> geosam2 canonical view index. Its picker only
# accepts its own names, in this order (tiles 1-2 must be the three-quarter
# views, the prompt says so). There is no level view at azimuth 0 in geosam2's
# ring (view 3 there looks up from below), so "front"/"back" take the nearest
# level views; "top" has no counterpart and is left out.
VIEW_MAP: Dict[str, int] = {
    "main": 1, "main_high": 5, "front": 4, "back": 10, "left": 0, "right": 6,
}
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DESCRIBE_MODEL = os.environ.get("GEOSAM2_DESCRIBE_MODEL", "google:gemini-3.1-pro-preview")
PAINT_MODEL = os.environ.get("GEOSAM2_PAINT_MODEL", "google:gemini-3.1-flash-image")


def _guidance():
    """Import SegviGen's ``util.guidance`` without running ``util/__init__``."""
    if "segvigen_util.guidance" in sys.modules:
        return sys.modules["segvigen_util.guidance"]
    util_dir = SEGVIGEN_DIR / "util"
    if "segvigen_util" not in sys.modules:
        pkg = types.ModuleType("segvigen_util")
        pkg.__path__ = [str(util_dir)]
        sys.modules["segvigen_util"] = pkg
    spec = importlib.util.spec_from_file_location(
        "segvigen_util.guidance", util_dir / "guidance.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["segvigen_util.guidance"] = mod
    spec.loader.exec_module(mod)
    return mod


def view_on_white(data_root: Path, view: int) -> Image.Image:
    """A canonical view composited on white, as SegviGen's renders are."""
    img = Image.open(Path(data_root) / f"color_{view:04d}.webp").convert("RGBA")
    out = Image.new("RGB", img.size, WHITE)
    out.paste(img.convert("RGB"), (0, 0), img.getchannel("A"))
    return out


def pick_seed_view(data_root: Path, model: str = DESCRIBE_MODEL) -> int:
    """SegviGen's view picker over geosam2's renders."""
    g = _guidance()
    shots = {name: view_on_white(data_root, v) for name, v in VIEW_MAP.items()}
    name = g.pick_best_view(shots, WHITE, model)
    return VIEW_MAP[name]


class Seed(NamedTuple):
    view: int
    scene: str
    parts: Dict[str, str]          # name -> hex, as SegviGen assigned them
    coverage: Dict[str, int]       # name -> painted pixels after snapping
    points_path: Path
    map_path: Path


GRID_VIEWS = ("front", "back", "left", "right")


def generate_seed(data_root: Path, seed_view: int, size: int = 1024,
                  describe_model: str = DESCRIBE_MODEL,
                  paint_model: str = PAINT_MODEL, mode: str = "single") -> Seed:
    """Describe, palette, paint with SegviGen's code; write GeoSAM2's seed.

    ``mode`` is SegviGen's: "single" describes the seed view itself, "grid"
    describes a labelled 2x2 grid of the four level views and tells the painter
    which parts the seed view cannot show. The painted map is snapped to the
    exact palette (the VLM lands near its colours, not on them), its white
    background turned black -- the colour geosam2's readers treat as
    background -- and one interior point per part region is written as
    ``vlm_points_XXXX.json``.
    """
    g = _guidance()
    data_root = Path(data_root)
    rendered = view_on_white(data_root, seed_view)
    view_name = next((n for n, v in VIEW_MAP.items() if v == seed_view), "main")

    pov = None
    if mode == "grid":
        shots = {n: view_on_white(data_root, VIEW_MAP[n]) for n in GRID_VIEWS}
        grid = g._assemble_grid(shots, list(GRID_VIEWS), cols=2, tile_size=size)
        description = g._vlm_describe(grid, model=describe_model, is_grid=True)
    else:
        description = g._vlm_describe(rendered, model=describe_model)
    description, table = g._assign_palette(description, "#ffffff")
    if mode == "grid":
        pov = g._compute_pov_visibility(table)
    logger.info("[segvigen describe/%s] '%s': %d parts: %s", mode,
                description.get("scene_description", "?"), len(table),
                ", ".join(table))
    painted = g._vlm_segment(rendered, description, table, model=paint_model,
                             image_size=(size, size), bg_color_hex="#ffffff",
                             view_name=view_name, pov_visibility=pov)
    if painted.size != rendered.size:
        painted = painted.resize(rendered.size, Image.NEAREST)

    palette = {n: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for n, h in table.items()}
    snapped = _snap(np.asarray(painted.convert("RGB")), list(palette.values()))
    map_path = data_root / f"mask_{seed_view:04d}.png"
    Image.fromarray(snapped).save(map_path)

    prompts = prompts_from_color_map(snapped, view_idx=seed_view, background=BLACK)
    points_path = data_root / f"vlm_points_{seed_view:04d}.json"
    write_prompts(prompts, points_path)
    coverage = {n: int(np.all(snapped == np.array(c, np.uint8), axis=-1).sum())
                for n, c in palette.items()}
    logger.info("[segvigen seed] view %d: %d/%d parts painted, %d points",
                seed_view, sum(1 for v in coverage.values() if v > 0), len(palette),
                len(prompts))
    return Seed(seed_view, description.get("scene_description", ""), table,
                coverage, points_path, map_path)


def _snap(rgb: np.ndarray, colours: List[Tuple[int, int, int]],
          max_dist: float = 60.0) -> np.ndarray:
    """Nearest palette colour per pixel (RGB); far pixels and white -> black."""
    flat = rgb.reshape(-1, 3).astype(np.int16)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    pal = np.array(colours, np.int16)
    d = np.linalg.norm(uniq[:, None, :] - pal[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    lut = pal[nearest].astype(np.uint8)
    too_far = d[np.arange(len(uniq)), nearest] > max_dist
    white = np.all(uniq >= 235, axis=1)
    lut[too_far | white] = 0
    return lut[inverse].reshape(rgb.shape)
