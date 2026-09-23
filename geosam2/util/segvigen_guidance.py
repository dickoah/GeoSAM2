"""SegviGen's guidance: view pick, VLM describe, palette, painted map.

Copied from SegviGen (pixmesh/backend/src/libraries/segvigen/util/guidance.py, as of
pixmesh commit d6099934) so geosam2 stops loading it from a sibling checkout.
Owned here from now on; the original module docstring follows.

Generate flat-colour 2D segmentation guidance maps via VLM + image gen.

pydantic-ai engine: the describe step returns a schema-validated assembly
tree (any provider — gemini/claude/gpt, with cross-provider fallback), the
segmentation step uses Gemini image generation. API keys come from the
environment; logfire (optional) instruments every call.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import numpy as np
import tempfile
import trimesh
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.settings import ModelSettings

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

# Candidates offered to the single-mode picker. bottom is left out: an object
# that sits on the ground never segments best from underneath.
VIEW_CANDIDATES = ("main", "main_high", "front", "back", "left", "right", "top")


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


_PROVIDER_KEYS = {
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}
# Fallback chains, filtered by available keys at call time.
_DESCRIBE_FALLBACKS = ("anthropic:claude-sonnet-4-6", "openai:gpt-5-mini")
_GENERATE_FALLBACKS = ("google:gemini-2.5-flash-image",)


def _resolve_model(model: str, fallbacks: Tuple[str, ...]):
    """'provider:name' string (bare names get 'google:') + keyed fallbacks."""
    if ":" not in model:
        model = f"google:{model}"
    chain = [m for m in dict.fromkeys([model, *fallbacks])
             if any(os.environ.get(k) for k in _PROVIDER_KEYS[m.split(":", 1)[0]])]
    if not chain:
        raise RuntimeError(f"No API key configured for {model} (see .env).")
    return chain[0] if len(chain) == 1 else FallbackModel(*chain)


def _run_agent(agent: Agent, content: list, **run_kwargs):
    """run_sync, or thread out when a loop is already running (async route)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return agent.run_sync(content, **run_kwargs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(
            lambda: asyncio.run(agent.run(content, **run_kwargs))).result()


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


class _ViewChoice(BaseModel):
    # Field order is the answering order, and it is the whole point here: every
    # step the model used to skip when it was only asked for in the prose is now
    # a field it must write before it can reach `index`. It dropped the family
    # on an ambiguously shaped object, and it never listed the bulb of a lamp —
    # so the tile that alone showed it lost to one that merely looked cleaner.
    parts: List[str] = Field(
        description="Every part of the object, short names. Repeated "
                    "interchangeable elements count once. Text, logos and "
                    "surface texture are not parts.")
    essential: List[str] = Field(
        description="The parts the object needs to do its job — what it lights, "
                    "holds, displays, supports, pours. Names from `parts`.")
    family: Literal["A", "B"] = Field(
        description="A = parts around a volume, B = parts along one axis")
    axis: str = Field(
        description="For B, the dominant axis (horizontal / vertical / the "
                    "direction it runs). For A, write 'none'.")
    eligible: List[int] = Field(
        description="Tiles allowed by your family: [1, 2] for A; for B the "
                    "tiles perpendicular to the axis, plus any tile added for "
                    "an essential part seen nowhere else.")
    reasoning: str = Field(description="Your per-part scan, brief")
    index: int = Field(description="Chosen tile — MUST be one of `eligible`")
    reason: str = Field(description="One sentence")

# The traced-outline definition and the per-part scan were measured on 22
# assets. Step 0 came later: a three-quarter view wins on compact objects but
# tilts the axis of anything strung along one, stacking the sequence — and
# whenever tiles 1-2 are eligible the model gravitates to them, so they must be
# earned through the essential-part exception, never offered. It is eliminatory;
# offered as a mere hint, the model read it and picked on separation anyway.
_VIEW_PICK_PROMPT = (
    "### TASK\n"
    "One image: 7 renders of the SAME object, numbered 1-7 in a black badge.\n"
    "Tiles 1 and 2 are three-quarter views. Pick the ONE tile to segment from.\n"
    "\n"
    "The chosen tile is the only image used downstream: an artist flood-fills each\n"
    "visible part with a flat colour, and those regions are projected back onto the\n"
    "mesh. A part that is missing, or that blends into its neighbour, is lost.\n"
    "So the best tile is the one showing the MOST parts — and an `essential` part\n"
    "lost outranks any count.\n"
    "\n"
    "### STEP 1 — FILL `parts` AND `essential` FIRST\n"
    "Inventory the object across all 7 tiles before you judge any tile. A part you\n"
    "never listed cannot weigh on the choice — that is how the one tile showing a\n"
    "lamp's bulb lost to a tile that merely looked cleaner.\n"
    "Look for the small ones: a bulb under a shade, a latch, a spout. Something\n"
    "visible in a single tile still belongs in `parts`.\n"
    "\n"
    "Then mark in `essential` the parts the object needs to do its job — the bulb\n"
    "of a lamp, the plates of a dumbbell, the seat of a chair. Not its bulk: the\n"
    "widest part is often not the essential one.\n"
    "\n"
    "### STEP 2 — FILL `family`, `axis` AND `eligible`\n"
    "Are the parts distributed around a volume, or strung along one axis?\n"
    "\n"
    "A. AROUND A VOLUME — front, side and top each carry different parts.\n"
    "   -> eligible = [1, 2]. A flat-on view would stack those faces.\n"
    "\n"
    "B. ALONG ONE AXIS — a sequence of elements on a line, often symmetric.\n"
    "   A thin support plus one wide element still counts as B when the parts\n"
    "   follow the support.\n"
    "   -> Tiles 1 and 2 tilt that axis toward the camera: the sequence\n"
    "      foreshortens and its elements stack. NOT eligible.\n"
    "   -> eligible = the tiles looking perpendicular to the axis.\n"
    "\n"
    "`index` MUST be one of `eligible`. A tile outside it cannot be the answer,\n"
    "however good it looks. ONE exception: an `essential` part hidden in every\n"
    "eligible tile adds the single tile that shows it — say why in `reasoning`.\n"
    "\n"
    "### SEPARATION\n"
    "A part counts when you could trace its outline: a visible boundary, a change of\n"
    "brightness or material, or a gap. Foreshortening is fine as long as the region\n"
    "stays distinct. It does not count when it blends into a same-coloured neighbour.\n"
    "\n"
    "### STEP 3 — SCAN, THEN PICK (write your work in `reasoning`)\n"
    "1. Take the parts from `parts` ONE AT A TIME. For each, scan the tiles and note\n"
    "   where it is visible and separated. Finish a part before the next — a\n"
    "   tile-by-tile scan forgets the small parts. If you cannot actually see it in\n"
    "   a tile, do not list that tile.\n"
    "2. An eligible tile that loses an `essential` part is DISQUALIFIED, whatever\n"
    "   else it shows. A part lying in a horizontal plane needs a raised view; a\n"
    "   part tucked under another needs a level one.\n"
    "3. Among the tiles still standing, most parts wins. On a tie, the lower number.\n"
    "\n"
    "### DO NOT\n"
    "- Credit a tile for a part hidden behind or inside the object.\n"
    "- Prefer a tile for looking sharper, more detailed or more symmetrical.\n"
)


def pick_best_view(
    shots: Dict[str, Image.Image],
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    model: str = "gemini-3-flash-preview",
    debug_dir: Optional[str] = None,
) -> str:
    """Return the key of the candidate view that shows the most parts.

    ``shots`` are the textured renders, keyed by ``VIEW_CANDIDATES`` name; the
    caller owns the rendering, which is the only part the two apps do
    differently. They are tiled into one badged grid and a VLM picks a tile.

    The VLM decides alone. An earlier version scored the tiles on silhouette
    span and edge density and let those numbers override the model on ties, but
    edge density rewards exactly the flat-on views that stack parts on top of
    each other — on a 13-asset ground truth it never once picked the wanted
    view, and it pulled correct VLM answers off the right tile. Falling back to
    the three-quarter view when the call fails beats falling back to a metric
    that is wrong on purpose.
    """
    order = [v for v in VIEW_CANDIDATES if v in shots]
    if not order:
        raise ValueError("no candidate view was rendered")

    cols, tile = 4, 512
    grid = _assemble_grid(shots, order, cols=cols, tile_size=tile, add_labels=False)
    # The badge is the model's only handle on a tile — load_default() is 11px.
    draw = ImageDraw.Draw(grid)
    try:
        badge_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
    except OSError:
        badge_font = ImageFont.load_default(size=64)
    # Invert the badge on a dark backdrop — black on black is no handle at all.
    dark_bg = sum(grid.getpixel((0, 0))) / 3 < 128
    badge_fill = (255, 255, 255) if dark_bg else (0, 0, 0)
    badge_ink = (0, 0, 0) if dark_bg else (255, 255, 255)
    for i in range(len(order)):
        x, y = (i % cols) * tile, (i // cols) * tile
        draw.rectangle([x, y, x + tile - 1, y + tile - 1],
                       outline=(160, 160, 160), width=2)
        draw.rectangle([x + 6, y + 6, x + 76, y + 82], fill=badge_fill)
        draw.text((x + 26, y + 8), str(i + 1), fill=badge_ink, font=badge_font)

    default = "main" if "main" in order else order[0]
    choice, view = None, default
    span = (logfire.span("guidance.pick_view", model=model)
            if logfire else nullcontext())
    try:
        with span:
            # seed + a fixed thinking level: a server-chosen budget made two
            # identical calls diverge, and it made them slow.
            choice = _run_agent(
                Agent(output_type=_ViewChoice, retries=4),
                [_img_to_content(grid), "Pick the best tile."],
                model=_resolve_model(model, _DESCRIBE_FALLBACKS),
                model_settings=ModelSettings(temperature=0.0, seed=7,
                                             thinking="low", max_tokens=4000,
                                             timeout=120),
                instructions=_VIEW_PICK_PROMPT,
            ).output
    except Exception as exc:                       # noqa: BLE001 - never fatal
        print(f"  → view VLM unavailable ({type(exc).__name__}: "
              f"{str(exc)[:160]}), falling back to {default}")

    if choice is not None:
        if 1 <= choice.index <= len(order):
            view = order[choice.index - 1]
        else:
            print(f"  → model returned tile {choice.index}, out of range — "
                  f"falling back to {default}")
    if choice is not None:
        print(f"  → parts: {', '.join(choice.parts)}")
        print(f"  → essential: {', '.join(choice.essential)}")
        print(f"  → family {choice.family} ({choice.axis}), "
              f"eligible {choice.eligible}")
    print(f"  → {view}" + (f": {choice.reason}" if choice else ""))

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # The grid the model saw, winner boxed; the individual tiles add nothing.
        i = order.index(view)
        draw.rectangle([(i % cols) * tile, (i // cols) * tile,
                        (i % cols) * tile + tile - 1, (i // cols) * tile + tile - 1],
                       outline=(0, 230, 60), width=10)
        grid.save(os.path.join(debug_dir, f"00_grid_choice_{view}.png"))
        with open(os.path.join(debug_dir, "stats.txt"), "w") as f:
            f.write(f"model: {model}\nchosen: {view}\n\n")
            for i, v in enumerate(order, 1):
                f.write(f"  tile {i}  {v}\n")
            if choice is not None:
                f.write(f"\nparts: {', '.join(choice.parts)}\n"
                        f"essential: {', '.join(choice.essential)}\n"
                        f"\nfamily {choice.family} — axis: {choice.axis}\n"
                        f"eligible tiles: {choice.eligible}\n"
                        f"\nmodel picked tile {choice.index} — {choice.reason}\n"
                        f"\nreasoning:\n{choice.reasoning}\n")
            else:
                f.write("\nthe VLM call failed; fell back to the default view\n")
        print(f"  → view debug written to {debug_dir}")
    return view


def _vlm_describe(
    image: Image.Image,
    model: str = "gemini-2.5-flash",
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
            model=_resolve_model(model, _DESCRIBE_FALLBACKS),
            model_settings=ModelSettings(temperature=0.0, max_tokens=8000,
                                         timeout=120),
            instructions=system_prompt,
        )
    return result.output.model_dump()


def _vlm_segment(
    image: Image.Image,
    description: Dict[str, Any],
    color_table: Dict[str, str],
    model: str = "gemini-3-pro-image",
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
    with span:
        result = _run_agent(
            agent, [user_prompt, _img_to_content(image)],
            model=_resolve_model(model, _GENERATE_FALLBACKS),
            model_settings=ModelSettings(temperature=0.0, timeout=180),
            instructions=system_prompt,
        )
    img = Image.open(BytesIO(result.output.data)).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    return img


# ── Public entry-point: Pixmesh 2D render ─────────────────────────────────

