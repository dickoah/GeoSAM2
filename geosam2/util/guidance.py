"""Guidance: the seed view, its VLM description, the palette and the painted map.

Two halves. The first is SegviGen's ``util/guidance.py`` (pixmesh commit
d6099934), the code where the describe / palette / paint prompts were tuned,
copied here without the four functions geosam2 never reaches and owned from
now on. The second feeds those functions geosam2's canonical renders and
snaps the map they paint into the seed mask GeoSAM2 reads. Nothing of the
prompts lives in the second half.

Models and API keys are read from ``os.environ`` at call time; the library
loads no ``.env``.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import math
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.messages import BinaryImage
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

from geosam2.util.logs import get_logger
from geosam2.util.views import ELEVATIONS, NUM_VIEWS

logger = get_logger("geosam2.guidance")

# ── SegviGen's guidance ─────────────────────────────────────────────────────

try:
    import logfire
except ImportError:
    logfire = None


# ── Kelly 22-color palette ─────────────────────────────────────────────────
_KELLY_PALETTE: List[str] = [
    '#dedede', # off-white
    '#333333', # near-black
    '#ebce2b', # yellow
    '#702c8c', # purple
    '#ba1c30', # red
    '#5fa641', # green
    '#d485b2', # purplePink / pink
    '#db6917', # orange
    '#4277b6', # blue
    '#df8461', # yellowPink / mediumOrange / lightBrown / papaya
    '#c0bd7f', # buff
    '#463397', # violet / navyBlue / navy
    '#7f7e80', # grey / gray
    '#e1a11a', # orangeYellow / lightOrange / manilla
    '#91218c', # magenta
    '#e8e948', # greenYellow / lightYellow / lemon
    '#7e1510', # redBrown / brown
    '#92ae31', # yellowGreen / mediumGreen / lime
    '#6f340d', # yellowBrown / darkBrown / dirt
    '#d32b1e', # redOrange / lightRed / crimson
    '#2b3514', # oliveGreen / darkGreen / olive
    '#96cde6', # lightBlue / aqua
]

# Canonical view names in display order
CANONICAL_VIEW_NAMES: List[str] = ["front", "back", "left", "right", "top", "bottom"]

# POV occlusion hints
_POV_OPP: Dict[str, set] = {
    "front":  {"back"},
    "back":   {"front"},
    "left":   {"right"},
    "right":  {"left"},
    "top":    {"bottom", "base", "floor", "foot", "feet", "lower"},
    "bottom": {"top", "upper", "roof"},
}


# ── Camera utilities ───────────────────────────────────────────────────────


# ── Rendering ─────────────────────────────────────────────────────────────



# Camera height of each rig elevation, as the picker tags its tiles.
_HEIGHT = {0.0: "LEVEL", 25.0: "ABOVE", -25.0: "BELOW"}


# ── Grid assembly ──────────────────────────────────────────────────────────

def _assemble_grid(
    images: Dict[str, Image.Image],
    view_order: List[str],
    cols: int,
    tile_size: int = 512,
    add_labels: bool = True,
) -> Image.Image:
    present = [v for v in view_order if v in images]
    rows = math.ceil(len(present) / cols)
    # Match the renders' own backdrop, sampled at a corner: a white gutter under
    # dark tiles reads as an extra edge the model has to explain away.
    pad = images[present[0]].convert("RGB").getpixel((0, 0)) if present else (255, 255, 255)
    grid = Image.new("RGB", (cols * tile_size, rows * tile_size), pad)

    for idx, name in enumerate(present):
        tile = images[name].resize((tile_size, tile_size), Image.LANCZOS)
        row, col = divmod(idx, cols)
        x, y = col * tile_size, row * tile_size
        grid.paste(tile, (x, y))

        if add_labels:
            draw = ImageDraw.Draw(grid)
            label = name.upper()
            draw.rectangle([x + 2, y + 2, x + len(label) * 7 + 6, y + 16], fill=(0, 0, 0))
            draw.text((x + 4, y + 3), label, fill=(255, 255, 255))

    return grid


# ── POV visibility ─────────────────────────────────────────────────────────

def _compute_pov_visibility(
    color_table: Dict[str, str],
) -> Dict[str, Dict[str, List[str]]]:
    result: Dict[str, Dict[str, List[str]]] = {
        v: {"visible": [], "occluded": []} for v in CANONICAL_VIEW_NAMES
    }
    for part_name in color_table:
        words = set(part_name.lower().split())
        for view in CANONICAL_VIEW_NAMES:
            if words & _POV_OPP[view]:
                result[view]["occluded"].append(part_name)
            else:
                result[view]["visible"].append(part_name)
    return result


# ── Shared utilities ───────────────────────────────────────────────────────

def _assign_palette(
    description: Dict[str, Any],
    bg_color_hex: str = "#ffffff",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    updated = copy.deepcopy(description)
    # Walk groups AND their subgroups: a subgroup carries real parts (a drawer
    # front and its knob live one level down), and reading only group["parts"]
    # leaves them without a colour — the segmentation prompt would then have no
    # hex code to paint them with.
    parts = [p
             for obj in updated.get("objects", [])
             for group in obj.get("assembly_tree", [])
             for p in (list(group.get("parts", []))
                       + [q for sub in group.get("subgroups", [])
                          for q in sub.get("parts", [])])]

    parts_ordered: List[str] = []
    for part in parts:
        name = part.get("name", "").strip()
        if name and name not in parts_ordered:
            parts_ordered.append(name)

    # Greedy max-min-ΔE assignment over the Kelly pool, in CIELab, with the
    # background counting as an already-taken colour. Sequential Kelly order
    # measured on a 17-part chair: min pairwise ΔE 14.0 and #dedede at ΔE 11.5
    # from a white background — the 3D decoder then bleeds those pairs into
    # each other and the split merges parts that must stay apart. The greedy
    # pick lifts this to ΔE 20+ between parts and ~30 from the background, and
    # since sibling instances (Leg Front Left / Front Right…) are consecutive
    # in tree order, the most confusable parts inherit the most distant
    # colours. Deterministic: pure argmax, ties broken by pool order.
    rgb255 = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)]
                       for h in _KELLY_PALETTE + [bg_color_hex]], np.float64)
    rgb = rgb255 / 255
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = rgb @ np.array([[.4124, .3576, .1805], [.2126, .7152, .0722],
                          [.0193, .1192, .9505]]).T / [.95047, 1.0, 1.08883]
    f = np.where(xyz > (6 / 29) ** 3, xyz ** (1 / 3),
                 xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    lab = np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                    200 * (f[:, 1] - f[:, 2])], axis=1)
    dist = np.linalg.norm(lab[:, None] - lab[None, :], axis=2)

    # Past the pool, colours are GENERATED rather than recycled: a second greedy
    # cycle repeated hexes verbatim (15 duplicates on 37 parts), and two parts
    # sharing a colour is a part lost, not a contrast problem. Farthest-point
    # off an sRGB grid, in RGB — the space the split thresholds on; selecting on
    # a perceptual metric drifts into one corner of the cube and was reverted.
    step = np.arange(0, 256, 8, dtype=np.float64)
    cand = np.stack(np.meshgrid(step, step, step, indexing="ij"), -1).reshape(-1, 3)

    n_pool = len(_KELLY_PALETTE)
    taken, taken_rgb, hexes = [n_pool], [rgb255[n_pool]], []  # background pre-taken
    for _ in parts_ordered:
        free = [c for c in range(n_pool) if c not in taken]
        if free:
            pick = max(free, key=lambda c: dist[c, taken].min())
            taken.append(pick)
            taken_rgb.append(rgb255[pick])
            hexes.append(_KELLY_PALETTE[pick])
        else:
            d = np.linalg.norm(cand[:, None] - np.array(taken_rgb)[None],
                               axis=2).min(axis=1)
            grown = cand[int(np.argmax(d))]
            taken_rgb.append(grown)
            hexes.append("#%02x%02x%02x" % tuple(int(v) for v in grown))
    color_table: Dict[str, str] = dict(zip(parts_ordered, hexes))

    if len(hexes) > 1:
        C = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes],
                     np.float64)
        sep = np.linalg.norm(C[:, None] - C[None, :], axis=2)
        np.fill_diagonal(sep, np.inf)
        print(f"[Palette] {len(hexes)} parts "
              f"({max(0, len(hexes) - n_pool)} generated past the pool), "
              f"closest pair {sep.min():.1f} apart in RGB — the split folds two "
              f"colours into one below palette_merge_dist=32")
    for part in parts:
        name = part.get("name", "").strip()
        if name in color_table:
            part["assigned_color_hex"] = color_table[name]

    return updated, color_table


def _img_to_content(image: Image.Image) -> BinaryContent:
    """PNG-encode a PIL image for pydantic-ai (lossless: silhouettes matter)."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return BinaryContent(data=buf.getvalue(), media_type="image/png")


# A model is a chain: comma-separated pydantic-ai 'provider:model' names, tried in order, each on its
# provider's own key (GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY...). A link whose key is
# unset is skipped, so a gateway (openrouter:google/..., or whatever comes next) is one more link in
# .env, never a case in this code. The part map is an image generation: pydantic-ai has it natively on
# google, and an OpenAI-compatible chat link (a gateway) returns it through the completion's image
# modality (_paint_by_chat).
DESCRIBE_MODEL = (os.environ.get("GEOSAM2_DESCRIBE_MODEL")
                  or "google:gemini-3.7-flash,anthropic:claude-sonnet-5,openai:gpt-5.6-luna")
PAINT_MODEL = (os.environ.get("GEOSAM2_PAINT_MODEL")
               or "google:gemini-3.1-flash-image,google:gemini-2.5-flash-image")
PICK_MODEL = os.environ.get("GEOSAM2_PICK_MODEL") or DESCRIBE_MODEL


def _resolve_chain(chain: str) -> list:
    """The keyed links of ``chain``, in order (a bare name gets 'google:')."""
    models, skipped = [], {}
    for name in (n.strip() for n in chain.split(",") if n.strip()):
        if ":" not in name:
            name = f"google:{name}"
        try:
            models.append(infer_model(name))
        except Exception as exc:                  # noqa: BLE001 - a link that cannot be built is not this link
            # An unset key is pydantic-ai's UserError; a key left EMPTY (as .env.dist ships them) reaches
            # the provider's SDK, which raises its own error. Either way the link is out, and says why.
            skipped[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
    if not models:
        raise RuntimeError(f"No usable link in {chain!r} (see .env): "
                           + "; ".join(f"{n} -> {why}" for n, why in skipped.items()))
    logger.info("[vlm route] %s%s", " -> ".join(f"{m.system}:{m.model_name}" for m in models),
                f"  (skipped: {', '.join(skipped)})" if skipped else "")
    return models


def _resolve_model(chain: str):
    """The first keyed link of ``chain``, with the later ones as fallbacks."""
    models = _resolve_chain(chain)
    return models[0] if len(models) == 1 else FallbackModel(*models)


def _sync(coro):
    """Run a coroutine from sync code, or thread out when a loop is already running (async route)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _run_agent(agent: Agent, content: list, **run_kwargs):
    result = _sync(agent.run(content, **run_kwargs))
    # Which link of the chain answered: a fallback that fires silently is a prompt drift nobody sees.
    logger.info("[vlm answered] %s:%s", result.response.provider_name, result.response.model_name)
    return result


def _paint_by_chat(model: OpenAIChatModel, instructions: str, prompt: str, image: Image.Image) -> bytes:
    """The image out of an OpenAI-compatible chat completion asked for the image modality.

    This is how a gateway serves an image model: pydantic-ai's ImageGeneration
    has no native tool there and rejects the run before any request.
    """
    buf = BytesIO()
    image.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    reply = _sync(model.client.chat.completions.create(
        model=model.model_name, temperature=0.0, timeout=180,
        messages=[{"role": "system", "content": instructions},
                  {"role": "user", "content": [{"type": "text", "text": prompt},
                                               {"type": "image_url", "image_url": {"url": data_url}}]}],
        extra_body={"modalities": ["image", "text"]},
    ))
    message = reply.choices[0].message
    images = getattr(message, "images", None) or (message.model_extra or {}).get("images") or []
    if not images:
        raise RuntimeError(f"{model.model_name} returned no image: {(message.content or '')[:200]!r}")
    logger.info("[vlm answered] %s:%s", model.system, reply.model)
    first = images[0]
    url = first["image_url"]["url"] if isinstance(first, dict) else first.image_url.url
    return base64.b64decode(url.split(",", 1)[1])


# ── VLM calls (pydantic-ai) ────────────────────────────────────────────────

class _PartDesc(BaseModel):
    name: str = Field(description="Short part name, max 3 words, no color adjectives")
    base_color_hex: str = ""
    material: str = ""


class _SubGroupDesc(BaseModel):
    group_name: str = Field(description="Subgroup name")
    parts: List[_PartDesc] = Field(default_factory=list)


class _GroupDesc(BaseModel):
    group_name: str = Field(description="Functional group name")
    # Depth is capped at 2 levels on purpose (a nested _GroupDesc would make the
    # schema recursive, which Gemini's structured output rejects) — the prompt
    # asks for the same limit.
    subgroups: List[_SubGroupDesc] = Field(default_factory=list)
    parts: List[_PartDesc] = Field(default_factory=list)


class _ObjectDesc(BaseModel):
    category: str = Field(description="Object name")
    assembly_tree: List[_GroupDesc]


class _SceneDesc(BaseModel):
    scene_description: str = Field(description="Max 5 words")
    language: str = "en"
    objects: List[_ObjectDesc]


class _PartSeen(BaseModel):
    part: str = Field(description="A name from `parts`")
    tiles: List[int] = Field(description="Tiles where this part has its own region to paint and name; "
                                         "not mostly covered by a piece in front; edge-on counts")


class _ViewChoice(BaseModel):
    # Field order is the answering order: every part is located tile by tile before the model may
    # name one, so the tile it names is the one its own audit covers best. Each instance is its own
    # part: "twins listed once" let a dumbbell's three-quarter view count 5/5 with every rear plate a
    # crescent, and GeoSAM2 never labels a twin the seed hides.
    analysis: str = Field(description="A few lines: the object, its rarely seen parts, and the tiles that show them")
    parts: List[str] = Field(description="Pieces an artist would paint separately, each instance its own entry; "
                                         "a crowd of like pieces is one; include what hides under, behind or inside")
    seen: List[_PartSeen] = Field(description="One entry per part, in the same order")
    index: int = Field(ge=1, le=NUM_VIEWS, description="The tile your `seen` credits with the most parts; LEVEL on a tie")
    reason: str = Field(description="One sentence naming the rarely seen parts this tile shows")

    def coverage(self) -> Dict[int, List[str]]:
        """Tile number -> the parts it shows."""
        seen = {}
        for entry in self.seen:
            for t in set(entry.tiles):
                seen.setdefault(t, []).append(entry.part)
        return seen

    @model_validator(mode="after")
    def _follows_its_audit(self):
        # The tile showing the most parts, per the model's own per-tile audit; a slip goes back as a retry.
        names = [e.part for e in self.seen]
        if sorted(names) != sorted(self.parts):
            raise ValueError(f"`seen` must list every name from `parts` exactly once; got {names} for {self.parts}.")
        bad = [t for e in self.seen for t in e.tiles if not 1 <= t <= NUM_VIEWS]
        if bad:
            raise ValueError(f"Tile numbers run from 1 to {NUM_VIEWS}; got {bad}.")
        coverage = self.coverage()
        best = max(len(v) for v in coverage.values()) if coverage else 0
        tiles = sorted(t for t, v in coverage.items() if len(v) == best)
        if self.index not in tiles:
            missing = [p for p in self.parts if p not in coverage.get(self.index, [])]
            raise ValueError(f"Tile {self.index} lacks {missing} by your own `seen`, while {tiles} show {best} parts: "
                             f"`index` must be one of {tiles}. Fix `seen` if a tile was mislisted, else pick among them.")
        # The tie-break the prompt states, held here: left to the model, a dumbbell's raised three-quarter
        # view won over its level profile on looks, with both at full count.
        level = [t for t in tiles if _HEIGHT[ELEVATIONS[t - 1]] == "LEVEL"]
        if level and self.index not in level:
            raise ValueError(f"Tiles {level} show the same {best} parts and are LEVEL: on a tie a LEVEL tile wins, "
                             f"so `index` must be one of {level}.")
        return self


# GeoSAM2 segments only what the seed shows, so the pick is the tile where the most parts have a
# readable region. The validator holds the count and the LEVEL tie-break; the prompt only has to get
# the inventory right and "readable" right: a sliver left by a piece in front is not a region (it tied
# a dumbbell's three-quarter view with its profile), a slab seen edge-on is.
_VIEW_PICK_PROMPT = """A grid of renders of one 3D object, each tile badged with its number and camera height.

One tile seeds a 3D segmentation: its visible parts are painted and propagated over the mesh. 
A part without a readable region there is lost: propagation grows a partly hidden region, not a sliver, and never labels a twin the seed hides.

Fill the fields in order:
1. analysis: a few lines on the object, its rarely seen parts and the tiles that show them.
2. parts: every piece an artist would paint separately, each instance its own entry (left and right, each of four legs); 
   a crowd of like pieces you could only number (foliage, keys) is one entry. 
   Include what hides under, behind or inside: a mechanism under a seat, the soil in a pot.
3. seen: per part, the tiles where it is readable: its own region you could paint and name. 
   Partly hidden counts; mostly covered by a piece in front, leaving a crescent or an edge, does not. Foreshortening is not occlusion: a slab seen edge-on counts. 
   Pieces lined up along an axis are all readable only perpendicular to it. A crowd counts where most members show. 
   Check every tile for every part; never copy a neighbour.
4. index: a tile your seen credits with the most parts, LEVEL on a tie.
5. reason: one sentence naming the rare parts it shows."""

def pick_best_view(
    shots: Dict[int, Image.Image],
    model: str = PICK_MODEL,
    debug_dir: Optional[str] = None,
) -> int:
    """Return the rig index of the view whose tile shows the most parts.

    ``shots`` are the twelve renders on white, keyed by rig index. They are tiled
    in rig order into one 4x3 grid, each tile badged with its number and its
    camera height, and a VLM picks a tile. Falls back to view 1 when the call fails.
    """
    order = sorted(shots)
    # 4 columns: the 2048x1536 grid keeps 392 px per tile after the 1568 px cap Claude applies, where
    # 6 columns left 261; a crescent and a readable disc must tell apart.
    cols, tile = 4, 512
    grid = Image.new("RGB", (cols * tile, math.ceil(len(order) / cols) * tile), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
        height_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except OSError:
        badge_font, height_font = ImageFont.load_default(size=64), ImageFont.load_default(size=26)
    for i, v in enumerate(order):
        x, y = (i % cols) * tile, (i // cols) * tile
        grid.paste(shots[v].convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(160, 160, 160), width=2)
        draw.rectangle([x + 6, y + 6, x + 96, y + 82], fill=(0, 0, 0))
        draw.text((x + 14, y + 8), str(i + 1), fill=(255, 255, 255), font=badge_font)
        draw.text((x + 104, y + 10), _HEIGHT[ELEVATIONS[v]], fill=(0, 0, 0), font=height_font)

    default = 1 if 1 in order else order[0]
    choice, view = None, default
    span = (logfire.span("guidance.pick_view", model=model)
            if logfire else nullcontext())
    try:
        with span:
            # seed + a fixed thinking level: a server-chosen budget made two
            # identical calls diverge, and it made them slow.
            choice = _run_agent(
                Agent(output_type=_ViewChoice, retries=4),
                [_img_to_content(grid), f"{len(order)} tiles, numbered 1-{len(order)}, in a {cols}-column grid; "
                                        "the camera turns 30 degrees per tile. Heights: "
                                        + ", ".join(f"{i + 1} {_HEIGHT[ELEVATIONS[v]]}" for i, v in enumerate(order))
                                        + ". Pick the seed tile."],
                model=_resolve_model(model),
                # No temperature: Anthropic rejects any value but 1 once thinking is on (400), so a Claude
                # link failed every time and the chain fell through to the next model.
                model_settings=ModelSettings(seed=7, thinking="medium", max_tokens=8000, timeout=180),
                instructions=_VIEW_PICK_PROMPT,
            ).output
    except Exception as exc:                       # noqa: BLE001 - never fatal
        print(f"  → view VLM unavailable ({type(exc).__name__}: "
              f"{str(exc)[:160]}), falling back to view {default}")

    if choice is not None:
        view = order[choice.index - 1]
        print(f"  → parts: {', '.join(choice.parts)}")
        # The parts few tiles show are the ones that decided.
        rare = [f"{e.part} {sorted(e.tiles)}" for e in choice.seen if len(e.tiles) <= 4]
        print(f"  → analysis: {' '.join(choice.analysis.split())[:300]}")
        print(f"  → rarely seen: {'; '.join(rare) or '-'}")
        coverage = choice.coverage()
        print(f"  → tile {choice.index} shows {len(coverage.get(choice.index, []))}/{len(choice.parts)} parts")
    print(f"  → view {view}" + (f": {choice.reason}" if choice else ""))

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # The grid the model saw, winner boxed.
        i = order.index(view)
        draw.rectangle([(i % cols) * tile, (i // cols) * tile,
                        (i % cols) * tile + tile - 1, (i // cols) * tile + tile - 1],
                       outline=(0, 230, 60), width=10)
        grid.save(os.path.join(debug_dir, f"00_grid_choice_{view}.png"))
        with open(os.path.join(debug_dir, "stats.txt"), "w") as f:
            f.write(f"model: {model}\nchosen: view {view}\n\n")
            f.write(json.dumps(choice.model_dump(), indent=1) if choice is not None
                    else "the VLM call failed; fell back to the default view\n")
        print(f"  → view debug written to {debug_dir}")
    return view


def _vlm_describe(
    image: Image.Image,
    model: str = DESCRIBE_MODEL,
    is_grid: bool = False,
    preferred_language: str = "en",
) -> Dict[str, Any]:
    view_context = (
        "a grid showing the object from four canonical angles "
        "(front, back, left, right)"
        if is_grid else
        "an isometric view of the object"
    )
    system_prompt = (
        "### ROLE\n"
        "Senior 3D Product Analyst and Mechanical Design Expert.\n"
        "\n"
        "### WHAT YOU ARE PRODUCING\n"
        "An assembly tree that drives a segmentation. Every part you list is handed ONE colour, "
        "and those colours must stay far apart — past roughly twenty parts they start to look "
        "alike and the downstream decoder merges the regions back together.\n"
        "A part therefore earns its place by being worth selecting on its own, not by being a "
        "separate piece in the factory sense.\n"
        "\n"
        "### HOW TO DECIDE — TWO QUESTIONS, IN THIS ORDER\n"
        "1. **WHERE DOES THE SURFACE BREAK?** A joint, a gap, a seam, a cushion edge or a change "
        "of section ends a piece. Nothing else does.\n"
        "2. **WHICH OF THOSE PIECES DESERVES ITS OWN COLOUR?** One does when you can point at it "
        "by name. If the only name available is a number in an order you could shuffle, it is one "
        "instance of a population — list the population as ONE part.\n"
        "\n"
        "📌 Function NAMES a part; the surface decides where it ENDS; addressability decides "
        "whether it gets a colour.\n"
        "\n"
        "### OUTPUT FORMAT\n"
        "Return a SINGLE valid JSON object.\n"
    )
    user_prompt = (
        "### INPUT CONTEXT\n"
        "You will receive " + view_context + ".\n"
        "\n"
        "### YOUR TASK\n"
        "Analyze the scene to identify all objects and decompose each into its constituent parts.\n"
        "\n"
        "### CRITICAL RULES\n"
        "1. **SEGMENTABLE PARTS ONLY**: List only parts that correspond to physically separable mesh regions, decided by the two questions above.\n"
        "   - ⚠️ DO NOT hallucinate parts that are not visible.\n"
        "   - ⚠️ DO NOT invent or assume tiny structural accessories that are NOT VISIBLY DISTINCT in the images. If you cannot clearly see an element as a separate piece, it does not exist.\n"
        "   - If unsure whether an element exists as a separate piece, assume it is integrated into its parent.\n"
        "\n"
        "2. **COUNT ACCURATELY**: Examine every view you were given, systematically.\n"
        "   - Look for structure that is not obvious at first glance: stretcher beams, cross-braces, backrest support frames, base boxes.\n"
        "   - **Floating Seat/Top**: If there is a visible gap between the seat/top and the supporting frame, look for the small **spacers**, **risers** or **blocks** holding them apart. List these as separate parts if visible.\n"
        "\n"
        "3. **GROUPING**: Organize parts under logical parent groups by FUNCTION.\n"
        "   - Vertical supports → 'Support Structure' (legs, posts, columns)\n"
        "   - Horizontal connecting elements → 'Stretchers' or 'Rails'\n"
        "   - Seating surfaces → 'Seating' (Seat, Backrest, or one Seat Shell if seamless)\n"
        "   - Storage units (drawers, doors) → 'Storage' with subgroups per unit\n"
        "   - Repeated elements (slats, rungs) → group name indicating plurality (e.g., 'Backrest Slats')\n"
        "   - ⚠️ Grouping is for ORGANIZATION only. It does NOT mean parts in a group should be merged.\n"
        "\n"
        "4. **NAMING**: Short, descriptive names (max 3 words). No parentheses, slashes, or color adjectives.\n"
        "   - ✅ Good: 'Front Stretcher', 'Seat Panel', 'Top Slat'\n"
        "   - ❌ Bad: 'Rail (Lower)', 'Front/Side Bar', 'Blue Connector'\n"
        "   - ❌ Bad: any name ending in a bare index ('Slat 7', 'Caster 2') — an index means you are naming one instance of a population; name the population instead (question 2b).\n"
        "\n"
        "5. **QUESTION 1 — WHERE THE SURFACE BREAKS**\n"
        "   Follow the surface. If you can travel from one element to another without crossing a joint, a gap, a seam or a change of section, it is ONE piece — however many roles it fills on the way.\n"
        "   - ✅ a bent tube rising into a backrest frame and continuing down into the legs → 'Tube Frame' (1 piece), even though it acts as backrest, upright and leg\n"
        "   - ✅ sled base, cantilever frame, one-piece molded shell, moulded star base, legs welded to their connecting rails → 1 piece each\n"
        "   - ✅ 'Seat' + 'Backrest' (2 pieces) when a cushion edge, joint or frame visibly separates them; a rigid base box or plinth under a soft seat is its own piece\n"
        "   - ❌ 'Front Left Leg' + 'Front Right Leg' + 'Backrest Frame' + 'Seat Rail' when all four are the same unbroken tube\n"
        "   - ❌ splitting a seamless shell because the furniture nomenclature expects a seat and a backrest\n"
        "   - ⚠️ Where you can see pieces JOIN (bolts, dowels, mortise-tenon, distinct shapes meeting) they are SEPARATE pieces, not welded. A trestle base has separate columns, feet and stretcher beams.\n"
        "   - 📌 Name a continuous structure for what it IS ('Tube Frame', 'Metal Base'), never for the roles it plays. A Left/Right pair of names is only legitimate when the two are physically separate pieces.\n"
        "\n"
        "6. **QUESTION 2 — WHICH PIECES DESERVE THEIR OWN COLOUR**\n"
        "   Only the pieces question 1 left you. Of each, ask: can I point at it by name, so that someone looking at the object knows which one I mean?\n"
        "   a) **Yes → its own part.** 'Front Left Leg', 'Top Drawer Front', 'Desktop', 'Lamp Shade'.\n"
        "      - Hardware — knobs, handles, pulls — always qualifies: different material, and people select it.\n"
        "   b) **No, only a number → ONE collective part.** When the only names available are 'X 1', 'X 2' … 'X 14', in an order you could shuffle without making the description wrong, those pieces are a population. Give the population one name. This is not about how many there are: it is about whether any of them can be singled out.\n"
        "      - ✅ 'Foliage', 'Bristles', 'Fringe', 'Casters', 'Gravel', 'Chain Links'\n"
        "      - ❌ 'Leaf 1' … 'Leaf 17', 'Caster 1' … 'Caster 5', 'Tassel 1' … 'Tassel 20'\n"
        "      - ⚠️ Do not dodge this with invented positional names ('Upper Left Leaf'). The test is whether the name still picks out the same piece once the object is turned around, or once the object gains one more of them.\n"
        "      - 📌 Four legs, three backrest slats and six stacked drawers all PASS the test: front/rear, top/middle/bottom and stacking order are real, stable positions. They stay separate parts.\n"
        "   c) **Assembly-only divisions → ONE part.** Pieces that exist only because the object was built from panels or boards, and that nobody would ever select apart.\n"
        "      - ✅ the outer shell of a cabinet, sideboard, credenza, dresser or shelving unit — top, bottom, left side, right side AND back panel → 'Cabinet Carcass' (1 part)\n"
        "      - ✅ planks or slats laid side by side into one surface → 'Tabletop' (1 part), likewise a bench seat, a shelf or a pallet-style surface\n"
        "      - ❌ 'Top Panel', 'Left Side Panel', 'Right Side Panel', 'Bottom Panel', 'Back Panel'\n"
        "      - ⚠️ Not where the panels do different jobs (a desktop vs a shelf vs a side panel on a desk).\n"
        "   d) **Too small to select, or no piece at all → part of its parent.** End caps, ferrules, glides, bumpers and welded tips belong to the piece they sit on.\n"
        "      - Relief in a continuous surface is NEVER a piece, however many times it repeats: tufting buttons, quilting, grooves, ribs, mouldings, perforations, decorative stitching, leaf buds, growth tips, petal veins. The surface dips or swells, it does not break — question 1 already ended the piece elsewhere, so question 2 never sees these.\n"
        "      - A structural spacer or riser that visibly holds two major components apart IS a part.\n"
        "      - If a small element is not clearly visible as a distinct piece, it does not exist — do not invent it.\n"
        "\n"
        "7. **HIERARCHY**: Use 'subgroups' for complex nested structures (max depth: 2 levels).\n"
        "   - Separate top-level objects must NOT be nested under each other.\n"
        "\n"
        "8. **METADATA**: Extract dominant color (HEX), material, and language.\n"
        "\n"
        "### FEW-SHOT EXAMPLES\n"
        "Each entry: what you see, then the parts, then what decided it.\n"
        "\n"
        "1. SIDEBOARD. Two sled-style metal legs joined by cross-bars, under a cabinet with one cupboard and three drawers.\n"
        "   Cabinet Carcass | Left Door | Door Knob | Top Drawer Front | Top Knob | Mid Drawer Front | Mid Knob | Bot Drawer Front | Bot Knob | Metal Base Frame\n"
        "   The enclosure panels are one carcass (6c); the welded sled runs unbroken (Q1); each drawer front is nameable by its place in the stack (6a).\n"
        "\n"
        "2. DINING CHAIR. Four turned legs, a backrest frame mortised into the rear legs carrying three slats, stretchers between the legs.\n"
        "   Front Left Leg | Front Right Leg | Rear Left Leg | Rear Right Leg | Seat Panel | Backrest Frame | Top Slat | Middle Slat | Bottom Slat | Front Stretcher | Rear Stretcher | Left Stretcher | Right Stretcher\n"
        "   Mortised joints break the surface, so legs and backrest are separate (Q1); front/rear and top/middle/bottom are real positions (6a).\n"
        "\n"
        "3. DESK WITH LAMP. Wooden desk with one drawer; a small metal lamp stands on the desktop.\n"
        "   Desk: Desktop | Left Leg | Right Leg | Back Panel | Drawer Front | Drawer Handle -- Lamp: Base Stand | Stem | Shade\n"
        "   Two separate top-level objects. The desk panels do different jobs, so 6c does not merge them; the handle is hardware (6a).\n"
        "\n"
        "4. WICKER ARMCHAIR on a welded wireframe base. The shell is one woven surface; a rattan binding of another material runs along its top edge.\n"
        "   Woven Shell | Rim Binding | Metal Base Frame\n"
        "   The woven ribs are relief in a continuous surface (6d); the binding is a material change, so Q1 ends the piece there; the base is welded throughout (Q1).\n"
        "\n"
        "5. TASK CHAIR. Moulded five-star base with a caster at the end of each arm, gas cylinder, seat, backrest, two armrests.\n"
        "   Seat | Backrest | Left Armrest | Right Armrest | Gas Cylinder | Star Base | Casters\n"
        "   The star base is moulded in one piece (Q1); the casters can only be told apart by an index, so they are one part (6b).\n"
        "\n"
        "### PREFERRED LANGUAGE\n"
        "" + preferred_language + "\n"
    )
    agent = Agent(output_type=_SceneDesc, retries=2)
    span = (logfire.span("guidance.describe", model=model)
            if logfire else nullcontext())
    with span:
        result = _run_agent(
            agent, [_img_to_content(image), user_prompt],
            model=_resolve_model(model),
            model_settings=ModelSettings(temperature=0.0, max_tokens=8000,
                                         timeout=120),
            instructions=system_prompt,
        )
    return result.output.model_dump()


def _vlm_segment(
    image: Image.Image,
    description: Dict[str, Any],
    color_table: Dict[str, str],
    model: str = PAINT_MODEL,
    image_size: Tuple[int, int] = (512, 512),
    bg_color_hex: str = "#ffffff",
    view_name: str = "main",
    pov_visibility: Optional[Dict[str, List[str]]] = None,
) -> Image.Image:
    W, H = image_size

    if pov_visibility and view_name in pov_visibility:
        vis = pov_visibility[view_name]["visible"]
        occ = pov_visibility[view_name]["occluded"]
        visible_ct = {n: c for n, c in color_table.items() if n in vis} or color_table
    else:
        occ, visible_ct = [], color_table

    color_table_str = "\n".join(f"  {n} → {c}" for n, c in visible_ct.items())
    # Only stated when something was actually filtered: the generated view is
    # "main", which matches no ``_compute_pov_visibility`` key, so the
    # unconditional version announced a filtering that never happened.
    occluded_str = ("Parts that cannot be seen from this angle — do not paint "
                    f"them: {', '.join(occ)}\n\n" if occ else "")

    # The tree tells instances apart (Left Door vs Right Door) and nothing else,
    # so every colour leaves it: ``base_color_hex`` is the object's REAL colour,
    # which the table forbids, and ``assigned_color_hex`` would restate the
    # pairing rule 1 tells the model to override when regions touch.
    tree = copy.deepcopy(description)
    for group in (g for obj in tree.get("objects", [])
                  for g in obj.get("assembly_tree", [])):
        for part in (list(group.get("parts", []))
                     + [q for sub in group.get("subgroups", [])
                        for q in sub.get("parts", [])]):
            for key in ("base_color_hex", "material", "assigned_color_hex"):
                part.pop(key, None)
    json_str = json.dumps(tree, indent=2, ensure_ascii=False)

    # Short on purpose: the long form restated the flat fill three times, the
    # background four, the colour count three, and gave its top slot to a
    # resolution constraint the resize below enforces anyway. What is left leads
    # with contrast — the one rule only the model can apply, being the only
    # party that sees which regions touch.
    system_prompt = (
        "You are a 3D Segmentation Colorist producing a FLAT LABEL MAP.\n\n"
        f"You receive one {W}×{H} render of an object, seen from its "
        f"{view_name.upper()} view, and a table pairing each part with a colour. "
        "Flood-fill every part region with one uniform colour: discard the "
        "render's lighting, shadows, highlights and surface detail entirely, so "
        "that every pixel of a region carries the exact same RGB value.\n\n"
        "This map is then read by a diffusion model that MERGES regions painted "
        "in similar colours. Two regions that touch must therefore never receive "
        "two similar colours — every other rule below serves that one.\n\n"
        "Return exactly one image, with no text, legend or outline inside it.\n"
    )

    user_prompt = (
        f"### PART COLOURS — {len(visible_ct)} entries\n"
        f"{color_table_str}\n\n"
        f"{occluded_str}"
        "### ASSEMBLY TREE — context only, to tell instances apart\n"
        f"```json\n{json_str}\n```\n\n"
        "### RULES\n"
        "1. **CONTRAST FIRST — this outranks the name-to-colour pairing.** Two "
        "regions sharing a border must never receive colours that look alike. "
        "The table says which colour goes with which name, but that pairing is "
        "not what matters: if following it would put two similar colours side by "
        "side, SWAP the two entries. Keep the set of colours, change which region "
        "gets which. Across the whole image, spread them as far apart as you can.\n"
        "2. **Extend the table only if you run out.** If there are more distinct "
        "regions than entries, add a colour of your own — but it must be visibly "
        f"far from EVERY colour in the table and from the background "
        f"({bg_color_hex}), never a shade sitting between two of them.\n"
        "3. **One region, one flat colour.** No gradient, no ramp, no darkening "
        "at the edges, no texture. A label map, not a recoloured render.\n"
        "4. **The image defines the geometry.** Respect the silhouette and the "
        "visible part boundaries exactly as they are. Never invent a boundary, "
        "never split a flat surface, never draw an outline between parts — they "
        "meet edge to edge.\n"
        f"5. **The background is not a part.** Every pixel that is {bg_color_hex} "
        f"in the input stays {bg_color_hex} in the output.\n"
        "6. **Paint what you see, nothing else.** The render decides how many "
        "regions exist, not the table: leaving colours unused is normal, and "
        "several entries often name one unbroken piece that takes ONE colour. A "
        "visible sliver gets its colour, zero visible pixels gets none. Never add "
        "a band or a stripe to place a leftover colour.\n"
    )

    agent = Agent(output_type=BinaryImage,
                  capabilities=[ImageGeneration(aspect_ratio="1:1")])
    span = (logfire.span("guidance.generate_segmentation", model=model,
                         view=view_name)
            if logfire else nullcontext())
    # The chain is walked here rather than by FallbackModel: a chat link paints through
    # _paint_by_chat, the others through the agent, and either failure hands over to the next.
    with span:
        for i, link in enumerate(links := _resolve_chain(model)):
            try:
                if isinstance(link, OpenAIChatModel):
                    data = _paint_by_chat(link, system_prompt, user_prompt, image)
                else:
                    data = _run_agent(
                        agent, [user_prompt, _img_to_content(image)], model=link,
                        model_settings=ModelSettings(temperature=0.0, timeout=180),
                        instructions=system_prompt,
                    ).output.data
                break
            except Exception as exc:                  # noqa: BLE001 - the next link is the answer
                if i == len(links) - 1:
                    raise
                logger.warning("[vlm paint] %s:%s failed (%s: %s), next link",
                               link.system, link.model_name, type(exc).__name__, str(exc)[:160])
    img = Image.open(BytesIO(data)).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    return img


# ── Public entry-point: Pixmesh 2D render ─────────────────────────────────


# ── geosam2's seed on top of it ──────────────────────────────────────────────

# SegviGen's view names -> geosam2 canonical view index: the painter's context views
# and the names of the seed the app reports. There is no level view at azimuth 0 in
# geosam2's ring (view 3 there looks up from below), so "front"/"back" take the nearest
# level views; "top" has no counterpart and is left out.
VIEW_MAP: Dict[str, int] = {
    "main": 1, "main_high": 5, "front": 4, "back": 10, "left": 0, "right": 6,
}
WHITE = (255, 255, 255)



def view_on_white(data_root: Path, view: int) -> Image.Image:
    """A canonical view composited on white, as SegviGen's renders are."""
    img = Image.open(Path(data_root) / f"color_{view:04d}.webp").convert("RGBA")
    out = Image.new("RGB", img.size, WHITE)
    out.paste(img.convert("RGB"), (0, 0), img.getchannel("A"))
    return out


def pick_seed_view(data_root: Path, model: str = PICK_MODEL) -> int:
    """The rig view to seed from, picked by a VLM among all twelve."""
    return pick_best_view({v: view_on_white(data_root, v) for v in range(NUM_VIEWS)}, model)


class Seed(NamedTuple):
    view: int
    scene: str
    parts: Dict[str, str]          # name -> hex, as SegviGen assigned them
    coverage: Dict[str, int]       # name -> painted pixels after snapping
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
    background -- and written as ``mask_XXXX.png``, the seed GeoSAM2 reads.
    """
    data_root = Path(data_root)
    rendered = view_on_white(data_root, seed_view)
    view_name = next((n for n, v in VIEW_MAP.items() if v == seed_view), _HEIGHT[ELEVATIONS[seed_view]].lower())

    pov = None
    if mode == "grid":
        shots = {n: view_on_white(data_root, VIEW_MAP[n]) for n in GRID_VIEWS}
        grid = _assemble_grid(shots, list(GRID_VIEWS), cols=2, tile_size=size)
        description = _vlm_describe(grid, model=describe_model, is_grid=True)
    else:
        description = _vlm_describe(rendered, model=describe_model)
    description, table = _assign_palette(description, "#ffffff")
    if mode == "grid":
        pov = _compute_pov_visibility(table)
    logger.info("[segvigen describe/%s] '%s': %d parts: %s", mode,
                description.get("scene_description", "?"), len(table),
                ", ".join(table))
    painted = _vlm_segment(rendered, description, table, model=paint_model,
                             image_size=(size, size), bg_color_hex="#ffffff",
                             view_name=view_name, pov_visibility=pov)
    if painted.size != rendered.size:
        painted = painted.resize(rendered.size, Image.NEAREST)

    palette = {n: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for n, h in table.items()}
    snapped = _snap(np.asarray(painted.convert("RGB")), list(palette.values()))
    map_path = data_root / f"mask_{seed_view:04d}.png"
    Image.fromarray(snapped).save(map_path)

    coverage = {n: int(np.all(snapped == np.array(c, np.uint8), axis=-1).sum())
                for n, c in palette.items()}
    logger.info("[segvigen seed] view %d: %d/%d parts painted",
                seed_view, sum(1 for v in coverage.values() if v > 0), len(palette))
    return Seed(seed_view, description.get("scene_description", ""), table,
                coverage, map_path)


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
