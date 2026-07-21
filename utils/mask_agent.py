"""Ask a VLM to paint a part map, and turn it into GeoSAM2 seed prompts.

GeoSAM2 segments what you click on. Its promptless mode is a derived behaviour
the paper never measures, and it shows -- so an arbitrary mesh needs seed clicks
from somewhere. This asks Gemini for them:

    render view 0 -> describe its parts -> assign a colour to each
      -> have Gemini repaint the render in those colours
      -> snap to the palette, extract one click per region

The colour map is a means, not the product: :mod:`utils.auto_prompt` reduces each
region to a single interior point, so the least trustworthy part of a generated
image -- its boundaries -- never reaches GeoSAM2. That also lands on the format
the reference pipeline uses, which is the one validated end to end.

Adapted from PixMesh's SegviGen brick, with three deliberate departures:

* **Flat colours, not shaded.** SegviGen tells the model to preserve shading,
  because its consumer is a global DINOv3 embedding that wants a render-like
  image. Here the map is quantised, so shading is noise.
* **A flat part list, not a nested assembly tree.** SegviGen's palette assignment
  never walks ``subgroups``, so parts nested under one silently get no colour and
  vanish. Without nesting there is nothing to forget.
* **Validated, not assumed.** SegviGen returns the generated image untouched.
  Here it is snapped to the palette in LAB and checked for coverage, because a
  colour the model invented is a part that does not exist.

Needs ``GEMINI_API_KEY`` (a ``.env`` at the repo root is loaded automatically).
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pyrender
import trimesh
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from utils.auto_prompt import MIN_AREA_PX, prompts_from_color_map
from utils.logs import get_logger
from utils.render import (
    RESOLUTION, camera_angle_x, shaded_view,
    _lit_scene, _pose_lit, _downsample, _enhance,
)

logger = get_logger("geosam2.mask_agent")

# Load the repo-root .env before pydantic-ai resolves a provider. Its Google
# provider reads GOOGLE_API_KEY or GEMINI_API_KEY, though it only names the
# first when it fails.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# PixMesh names gemini-3-pro-preview and gemini-3-pro-image-preview. The first
# still appears in models.list but 404s on generateContent -- listed is not the
# same as callable, so check by calling. These are the current equivalents.
DESCRIBE_MODEL = "google:gemini-3.1-pro-preview"
PAINT_MODEL = "google:gemini-3-pro-image"

# As deterministic as the API lets us be, so a prompt change is the only thing
# that moves between two runs of the same input. temperature=0 is greedy
# decoding; the fixed seed pins whatever randomness remains (tie-breaks, and the
# image model's sampler). Not a hard guarantee -- server-side batching and model
# updates still drift -- but it removes run-to-run noise from the loop. Override
# GEOSAM2_SEED to sweep seeds deliberately.
_SEED = int(os.environ.get("GEOSAM2_SEED", "1234"))


def _deterministic_settings():
    """ModelSettings pinning temperature and seed. Built lazily -- importing
    pydantic_ai at module load would demand credentials this file must not need."""
    from pydantic_ai.settings import ModelSettings

    return ModelSettings(temperature=0.0, seed=_SEED)

# Background is white because `shaded_view` renders on white, and the prompt
# leans on "leave the background alone" being trivially checkable.
BACKGROUND = (255, 255, 255)

# A deliberately plain first-pass prompt. The structured describe -> palette ->
# _paint_prompt chain is the destination (ported from PixMesh next); this is the
# baseline it has to beat, and the default the mask-lab app opens with.
DEFAULT_PROMPT = (
    "This is a render of a single 3D object on a white background.\n"
    "Repaint it as a flat part map: fill each distinct part with one uniform, "
    "solid colour, a different colour per part. No shading, no gradients, no "
    "outlines. Leave the white background untouched. Return only the image, "
    "same size as the input."
)

# Kelly's maximum-contrast colours, minus white (the background) and near-blacks
# that a shaded render swallows. Ordered by how far apart they read.
PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (243, 195, 0), (135, 86, 146), (243, 132, 0), (161, 202, 241),
    (190, 0, 50), (194, 178, 128), (132, 132, 130), (0, 136, 86),
    (230, 143, 172), (0, 103, 165), (249, 147, 121), (96, 78, 151),
    (246, 166, 0), (179, 68, 108), (220, 211, 0), (136, 45, 23),
    (141, 182, 0), (101, 69, 34), (226, 88, 34), (43, 61, 38),
)

# The palette is the ceiling: past it, two parts would share a colour and
# collapse into one. SegviGen wraps with `% len(palette)` and loses them
# silently; asking for fewer parts up front is the honest fix.
MAX_PARTS = len(PALETTE)


# ── Input views for the VLM ──────────────────────────────────────────────────
# PixMesh's ViewGenerator rig, ported: the 3/4 MAIN view (yaw -50, pitch +20)
# reads an object better than any axis-aligned view, and describe sees a 2x2 grid
# of MAIN + FRONT + BOTTOM + a rear 3/4, because bottom/rear reveal parts the
# front hides. This rig is Y-up (native GLB), distinct from render.py's Z-up
# GeoSAM2 rig -- these views feed a VLM, not the model, so they need not align to
# the 12 canonical ones. The seeding pipeline does not use this rig: it runs on a
# canonical view directly (see generate_seed_mask), so no reprojection is needed.

def _main_direction(yaw_deg: float = -50.0, pitch_deg: float = 20.0) -> np.ndarray:
    yaw, pitch = np.radians([yaw_deg, pitch_deg])
    return np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])


# (label, eye direction from centre, up) — Y-up, matching PixMesh's ViewType.
_TARGET_VIEW = ("MAIN", _main_direction(), np.array([0.0, 1.0, 0.0]))
_DESCRIBE_VIEWS = (
    _TARGET_VIEW,
    ("BACK 3Q", np.array([-0.6145, 0.2215, -0.6145]), np.array([0.0, 1.0, 0.0])),
    ("FRONT", np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
    ("BOTTOM", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
)


def _look_at_yup(eye: np.ndarray, centre: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world pose aiming -Z at ``centre``. Y-up, unlike render.look_at."""
    z = eye - centre
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def _normalize_matrix_yup(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
    """Centre a mesh on the origin and fit it in a unit box, without rotating it.

    render.normalize rotates into Z-up for the GeoSAM2 rig; here the object stays
    as authored so the PixMesh Y-up directions point where they should.
    """
    lo, hi = mesh.bounds
    longest = float((hi - lo).max()) or 1.0
    matrix = np.eye(4)
    matrix[:3, :3] /= longest
    matrix[:3, 3] = -(lo + hi) / 2.0 / longest
    return matrix, float(np.linalg.norm(hi - lo) / longest)


def _render_view(scene: trimesh.Scene, direction: np.ndarray, up: np.ndarray,
                 resolution: int = RESOLUTION, ssaa: int = 2) -> Image.Image:
    """A lit, supersampled render of ``scene`` from one Y-up direction, on white.

    Reuses render.py's lit pipeline (three spread lamps, saturation/contrast) via
    _lit_scene/_pose_lit; only the camera rig differs.
    """
    matrix, diag = _normalize_matrix_yup(scene.to_geometry())
    distance = diag / (2.0 * np.tan(camera_angle_x() / 2.0)) * 1.15
    pose = _look_at_yup(direction / np.linalg.norm(direction) * distance, np.zeros(3), up)

    pr_scene, camera_node, lights = _lit_scene(scene, matrix)
    _pose_lit(pr_scene, camera_node, lights, pose)
    renderer = pyrender.OffscreenRenderer(resolution * ssaa, resolution * ssaa)
    try:
        color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    rgba = _enhance(_downsample(color, resolution))
    white = Image.new("RGB", (resolution, resolution), BACKGROUND)
    white.paste(Image.fromarray(rgba[..., :3]), (0, 0), Image.fromarray(rgba[..., 3]))
    return white


def target_view(scene: trimesh.Scene, resolution: int = RESOLUTION) -> Image.Image:
    """The 3/4 MAIN render the part map is generated on."""
    _, direction, up = _TARGET_VIEW
    return _render_view(scene, direction, up, resolution)


def describe_grid(scene: trimesh.Scene, tile: int = 512) -> Image.Image:
    """A 2x2 grid of the four describe views, each labelled in its corner."""
    grid = Image.new("RGB", (tile * 2, tile * 2), BACKGROUND)
    for i, (label, direction, up) in enumerate(_DESCRIBE_VIEWS):
        view = _render_view(scene, direction, up, resolution=tile).copy()
        draw = ImageDraw.Draw(view)
        draw.rectangle([0, 0, tile - 1, tile - 1], outline=(0, 0, 0), width=2)
        draw.rectangle([2, 2, 12 + 7 * len(label), 18], fill=(0, 0, 0))
        draw.text((5, 4), label, fill=(255, 255, 255))
        grid.paste(view, ((i % 2) * tile, (i // 2) * tile))
    return grid


class Part(BaseModel):
    """One visually separable region of the object."""

    name: str = Field(description="Short name, e.g. 'left boot', 'helmet crest'.")
    location: str = Field(description="Where it sits, e.g. 'lower left', 'centre torso'.")


class PartList(BaseModel):
    category: str = Field(description="What the object is, in a few words.")
    parts: List[Part] = Field(description="Every visually separable part, most prominent first.")


def _agent(model: str, output_type, system: str):
    # Imported lazily: pydantic-ai resolves a provider (and demands a key) at
    # construction, so a module-level agent would make this file unimportable
    # without credentials -- including for the callers that only want the
    # palette or the snapping.
    from pydantic_ai import Agent

    return Agent(model, output_type=output_type, system_prompt=system,
                 model_settings=_deterministic_settings())


_DESCRIBE_SYSTEM = f"""You identify the separable parts of a 3D object from a render.

DETAIL FIRST. Your default is to list every visually distinct component as its own
part. Merging is the exception: merge only when two surfaces have no visible
boundary at all AND serve the same purpose. When unsure, list them separately --
a part listed and not found costs nothing, a part missed cannot be recovered.

ONLY WHAT YOU SEE. Never invent a part you cannot point at in the image. If you
cannot see where one piece ends and the next begins, it is one piece. Do not
assume fasteners, seams, or internals that are not visible.

Prefer parts a segmenter can actually separate: a distinct panel, limb, plate or
accessory. Not a surface marking, not a colour change on continuous geometry.

Return at most {MAX_PARTS} parts, most prominent first. If the object has more,
merge the least significant ones -- the list is truncated past that anyway."""


def describe(image: Image.Image, model: str = DESCRIBE_MODEL) -> PartList:
    """Ask what the object is and which parts it has.

    Structured output rather than scraping JSON out of prose: pydantic-ai
    validates against the schema and re-asks on a mismatch, so a malformed reply
    is retried instead of raising three stages later.
    """
    agent = _agent(model, PartList, _DESCRIBE_SYSTEM)
    from pydantic_ai import BinaryContent

    logger.info("[describe] asking %s about a %dx%d render", model, *image.size)
    started = time.monotonic()
    result = agent.run_sync([
        "Identify this object and list its separable parts.",
        BinaryContent(data=_png_bytes(image), media_type="image/png"),
    ])
    parts = result.output.parts[:MAX_PARTS]
    if len(result.output.parts) > MAX_PARTS:
        logger.warning("[describe] returned %d parts, truncated to the palette's %d",
                       len(result.output.parts), MAX_PARTS)
    if not parts:
        raise RuntimeError("the model found no parts in the render")
    logger.info("[describe] '%s', %d parts in %.1fs -- %s",
                result.output.category, len(parts), time.monotonic() - started,
                ", ".join(p.name for p in parts))
    return PartList(category=result.output.category, parts=parts)


def assign_palette(parts: Sequence[Part]) -> Dict[str, Tuple[int, int, int]]:
    """One distinct colour per part, in order."""
    if len(parts) > len(PALETTE):
        raise ValueError(f"{len(parts)} parts but only {len(PALETTE)} distinct colours")
    return {part.name: PALETTE[i] for i, part in enumerate(parts)}


def with_contours(image: Image.Image, color: Tuple[int, int, int] = (255, 0, 255)) -> Image.Image:
    """Overlay Canny edges on the render.

    Borrowed from SegviGen, where it is the trick that makes this work at all:
    the model is far better at filling regions someone else outlined than at
    deciding where a boundary lies. Magenta because the render is greyscale, so
    the lines cannot be mistaken for geometry.
    """
    grey = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grey, 50, 150)
    out = np.asarray(image.convert("RGB")).copy()
    out[edges > 0] = color
    return Image.fromarray(out)


def _paint_prompt(parts: PartList, palette: Dict[str, Tuple[int, int, int]]) -> str:
    table = "\n".join(
        f"  {_hex(palette[p.name])}  {p.name} ({p.location})" for p in parts.parts
    )
    return f"""Repaint this render of a {parts.category} as a flat part map.

COLOUR TABLE — use these exact RGB values, nothing else:
{table}

RULES
1. FLAT COLOUR ONLY. Fill each part with one uniform colour. No shading, no
   gradient, no highlight, no texture. This is a label map, not a picture.
2. BACKGROUND STAYS WHITE {_hex(BACKGROUND)}. Do not paint outside the object.
3. KEEP THE SHAPE. Every silhouette and boundary stays exactly where it is. Do
   not move, smooth, redraw or invent an edge.
4. The magenta lines mark boundaries. Paint inside them, then remove them — no
   magenta in your output.
5. Use each colour exactly once, on the part it names. Do not use a colour for a
   part you cannot see; leave that part's pixels to its neighbour.
6. Same resolution as the input.

Return one image."""


def paint(
    image: Image.Image,
    parts: PartList,
    palette: Dict[str, Tuple[int, int, int]],
    model: str = PAINT_MODEL,
) -> Image.Image:
    """Have the model repaint the render as a flat part map."""
    from pydantic_ai import Agent, BinaryContent, BinaryImage
    from pydantic_ai.capabilities import NativeTool
    from pydantic_ai.native_tools import ImageGenerationTool

    # 2.11 routes provider-native tools through `capabilities=[NativeTool(...)]`.
    # PixMesh's `builtin_tools=` is gone; `tools=` wants plain callables, and
    # `toolsets=` accepts the object then calls it as a factory at run time.
    agent = Agent(model, output_type=BinaryImage, model_settings=_deterministic_settings(),
                  capabilities=[NativeTool(ImageGenerationTool(aspect_ratio="1:1"))])
    logger.info("[paint] asking %s for a %d-colour part map", model, len(palette))
    started = time.monotonic()
    result = agent.run_sync([
        _paint_prompt(parts, palette),
        BinaryContent(data=_png_bytes(with_contours(image)), media_type="image/png"),
    ])
    painted = Image.open(io.BytesIO(result.output.data)).convert("RGB")
    logger.info("[paint] got %dx%d in %.1fs", *painted.size, time.monotonic() - started)
    if painted.size != image.size:
        # NEAREST, never LANCZOS: interpolating a label map manufactures colours
        # that are in no part's palette along every boundary.
        logger.warning("[paint] returned %dx%d, resizing to the input's %dx%d",
                       *painted.size, *image.size)
        painted = painted.resize(image.size, Image.NEAREST)
    return painted


# ── Structured flow, ported from PixMesh's SegviGen segmenter ────────────────
# describe (hierarchical assembly tree) -> palette -> part-map generation. The
# describe/generation prompts are PixMesh's, verbatim; they "work pretty well"
# there and are the destination this bench was built to reach.

_DESCRIBE_SYSTEM_STRUCT = '### ROLE\nSenior 3D Product Analyst and Mechanical Design Expert.\n\n### OBJECTIVES\n1. **VISUALLY INSPECT** all views to identify every distinct component.\n2. **FAVOUR DETAIL**: List every part you can visually distinguish.\n3. **GENERATE** the most complete assembly tree possible.\n\n### DETAIL-FIRST PRINCIPLE (CRITICAL — READ THIS FIRST)\nYour DEFAULT behaviour is to list every visually-distinct component as a SEPARATE part.\nMerging is the EXCEPTION, not the rule. Only merge when:\n  - Two surfaces are truly INDISTINGUISHABLE (no visible boundary at all), AND\n  - They serve the SAME functional purpose.\nWhen in doubt, LIST MORE PARTS. The downstream pipeline handles detail well; oversimplification breaks it.\n\n### THE ONLY MERGE EXCEPTION\nMerge into one part ONLY for continuous welded/molded METAL frames where joints are truly invisible:\n  - Sled bases, U-frames, spider bases, cantilever frames → 1 part.\n⚠️ This does NOT apply to:\n  - Assemblies where you can see distinct pieces (trestle columns, turned legs, stretcher beams)\n  - Seat + Backrest (ALWAYS separate — see functional zone test in rules)\n  - Anything that is NOT a welded metal frame\n\n### OUTPUT FORMAT\nReturn a SINGLE valid JSON object.\n'

_DESCRIBE_USER_STRUCT = '### INPUT CONTEXT\nYou will receive a grid of 4 images showing a 3D scene from different viewpoints.\n\n### YOUR TASK\nAnalyze the scene to identify all objects and decompose each into its constituent parts.\n\n### DETAILED OUTPUT FORMAT\nReturn a SINGLE valid JSON object with this exact structure:\n{\n  "scene_description": "<VERY SHORT description (max 5 words)>",\n  "language": "<en|fr|other>",\n  "objects": [\n    {\n      "category": "<Object Name (e.g. Workbench, Draped Cloth)>",\n      "assembly_tree": [\n        {\n          "group_name": "<Storage Unit / Functional Group>",\n          "subgroups": [  // OPTIONAL: Use for hierarchical parts like Door Sets, Drawer Units\n            {\n              "group_name": "<Subgroup Name>",\n              "parts": [{ "name": "...", "base_color_hex": "...", "material": "..." }]\n            }\n          ],\n          "parts": [ // Direct parts of this group\n            { "name": "<Part Name>", "base_color_hex": "<HEX color>", "material": "<material>" }\n          ]\n        }\n      ]\n    }\n  ]\n}\n\n### CRITICAL RULES\n1. **SEGMENTABLE PARTS ONLY**: List only parts that correspond to physically separable mesh regions.\n   - If you can SEE where one piece ends and another begins → they are SEPARATE parts.\n   - Trestle columns, pedestals, stretcher beams, decorative spindles → each is its own part if visually distinct.\n   - Only merge when two elements are truly INDISTINGUISHABLE (same shape, same surface, no visible boundary between them).\n   - ⚠️ DO NOT hallucinate parts that are not visible — but DO list everything you CAN see.\n   - ⚠️ DO NOT invent or assume tiny structural accessories that are NOT VISIBLY DISTINCT in the images. If you cannot clearly see an element as a separate piece, it does not exist.\n   - If unsure whether an element exists as a separate piece, assume it is integrated into its parent.\n\n2. **COUNT ACCURATELY**: Examine ALL 4 views systematically.\n   - Check BOTTOM VIEW and REAR VIEW for structure not obvious from the front.\n   - BOTTOM/REAR views may reveal stretcher beams, cross-braces, backrest support frames, or base boxes — list them as parts.\n   - **Floating Seat/Top**: If there is a visible gap between the seat/top and the supporting frame, look for small **spacers**, **risers**, or **blocks** that connect them. List these as separate parts if visible.\n   - Connection does NOT mean same part. Two pieces bolted or joined together are still SEPARATE parts if you can see the boundary.\n\n3. **GROUPING**: Organize parts under logical parent groups by FUNCTION.\n   - Vertical supports → \'Support Structure\' (legs, posts, columns)\n   - Horizontal connecting elements → \'Stretchers\' or \'Rails\'\n   - Seating surfaces → \'Seating\' (Seat, Backrest — always separate parts)\n   - Storage units (drawers, doors) → \'Storage\' with subgroups per unit\n   - Repeated elements (slats, rungs) → group name indicating plurality (e.g., \'Backrest Slats\')\n   - ⚠️ Grouping is for ORGANIZATION only. It does NOT mean parts in a group should be merged.\n\n4. **NAMING**: Short, descriptive names (max 3 words). No parentheses, slashes, or color adjectives.\n   - ✅ Good: \'Front Stretcher\', \'Seat Panel\', \'Top Slat\'\n   - ❌ Bad: \'Rail (Lower)\', \'Front/Side Bar\', \'Blue Connector\'\n\n5. **WHAT IS ONE PART? (Visual distinctness is the primary criterion)**:\n   a) **Visual boundary test (PRIMARY)**: If you can see a clear boundary, joint line, or shape change between two elements → they are SEPARATE parts.\n      - Trestle table: each column, each foot/base, each stretcher beam = separate parts\n      - Turned/carved legs: each leg is its own part\n      - Ornate pedestals with distinct top and bottom sections → separate parts\n      - ⚠️ Only list a part if it is a distinct geometric shape that could be cut apart from its neighbors in a 3D mesh. Sub-features of the same continuous surface (e.g. leaf buds on a branch, growth tips, individual petal veins) do NOT count as separate parts.\n   b) **Welded-frame exception (NARROW)**: Only merge into one part when the structure is a CONTINUOUS welded/molded piece with NO visible joints:\n      - Sled base, cantilever frame, tubular bent frame → 1 part each\n      - **Sled legs**: Two sled runners connecting left and right sides OFTEN form a unified base. If they look like part of the same metal structure (even if cross-bar is hidden), merge them as \'Support Frame\' or \'Metal Base\'.\n      - **Metal Frames**: Merge continuous metal support structures (legs + connecting rails) into a single \'Frame\' part unless distinct joinery is visible.\n      - ⚠️ If you can see where pieces JOIN (bolts, dowels, mortise-tenon, distinct shapes meeting) → they are SEPARATE parts, not welded.\n      - ⚠️ **STAR/SWIVEL BASE EXCEPTION**: Star-shaped or swivel bases (office chairs, task chairs) with clearly distinct radiating arms ARE separate parts — list each arm as \'Base Arm 1\', \'Base Arm 2\', … \'Base Arm N\' PLUS the central hub as \'Base Hub\'. Do NOT merge them into a single \'Swivel Base\' or \'Spider Base\' part. Their arms radiate outward with visible gaps between them and are geometrically separable.\n   c) **Magnifying-glass test**: Tiny accessories that are not individually segmentable (e.g. welded tips, small caps, bumpers, glides, feet caps) are NOT separate parts — they belong to their parent structure.\n      - Do NOT list any accessory smaller than ~5%% of the object\'s total size as a separate part, UNLESS it is a structural spacer/riser separating two major components.\n      - Simple end-caps, ferrules, or glides on the bottom of legs are PART OF THE LEG.\n      - If a small element is not clearly visible as a distinct piece across multiple views, it does not exist — do not invent it.\n   d) **Hardware exception**: Knobs, handles, and pulls are SEPARATE parts — they are functional hardware, typically different material (e.g., metal knob on wood door).\n   e) **Functional zone test (CRITICAL — OVERRIDES b)**: Even if two regions share the same material, color, and appear physically continuous (even MOLDED as one piece), they are SEPARATE parts if they serve DIFFERENT ergonomic or functional purposes.\n      - Seat vs Backrest → ALWAYS 2 separate parts, even on a one-piece molded shell/bucket chair\n      - Armrest vs Side Panel → separate if functionally distinct\n      - Desktop vs Side Panel → separate\n      - ❌ WRONG: \'Molded Seat Body\' or \'Upholstered Body\' or \'Shell Body\' combining seat + backrest into 1 part\n      - ✅ CORRECT: \'Seat\' (1 part) + \'Backrest\' (1 part) = 2 parts, even on a plastic bucket chair\n      - 📌 Rule 5b (molded-piece exception) does NOT apply to seat+backrest. The functional zone test ALWAYS wins for primary ergonomic surfaces.\n      - ⚠️ **ARMREST EXCEPTION**: If armrests are clearly molded or upholstered as a single continuous, seamless piece with the backrest (like on a tub chair or wingback chair), DO NOT separate them. Merge them into the \'Backrest\' or \'Seat Shell\'. Only separate armrests if there is a visible joint, gap, or material change.\n   f) **Structure vs Cushioning**: Separate the rigid support structure (frame, plinth, base box) from the soft cushions (seat, backrest) if visible.\n      - Sofa beds often have a visible \'Base Box\' or \'Frame\' underneath the \'Seat\' and \'Backrest\'. Check for this structural layer.\n      - If you see a hard base supporting a soft seat/back, list them as separate parts.\n   g) **Structural Enclosure / Carcass (OVERRIDES a for storage furniture)**: For cabinets, sideboards, credenzas, dressers, shelving units, and similar storage/case furniture, ALL outer panels forming the box-shaped shell (top, bottom, left side, right side, AND back panel) serve the SAME structural function — they are the **carcass**.\n      - ✅ CORRECT: Describe them as ONE part: \'Cabinet Carcass\' (1 single part encompassing all enclosure panels including the back)\n      - ❌ WRONG: Listing \'Top Panel\', \'Left Side Panel\', \'Right Side Panel\', \'Bottom Panel\', \'Back Panel\' as separate parts\n      - ⚠️ This does NOT apply to panels with different functional purposes (e.g., a desktop vs a shelf vs a side panel on a desk).\n      - 📌 The visual boundary test (5a) does NOT override this rule: panel joints on a carcass are assembly joints, not functional boundaries.\n   h) **Parallel Planks/Slats exception**: When multiple identical planks or slats are arranged side-by-side to form a SINGLE continuous functional surface (like a Tabletop, a Bench Seat, or a Shelf), merge them into ONE part, even if seams are visible.\n      - ✅ CORRECT: \'Tabletop\' (composed of 3 side-by-side planks) = 1 part\n      - ❌ WRONG: \'Left Plank\', \'Center Plank\', \'Right Plank\'\n      - ⚠️ This applies also to pallet-style surfaces or slatted seats if they form the main surface.\n\n6. **HIERARCHY**: Use \'subgroups\' for complex nested structures (max depth: 2 levels).\n   - Separate top-level objects must NOT be nested under each other.\n\n7. **METADATA**: Extract dominant color (HEX), material, and language.\n\n### COMMON MISTAKES (READ BEFORE EXAMPLES)\n❌ **OVER-SIMPLIFICATION**: Do NOT lump visually-distinct components into one part just because they are connected.\\n   - A trestle table base has separate columns, feet, and stretcher beams → list each one.\\n   - Only merge into one part when elements are truly INDISTINGUISHABLE (continuous welded metal with no visible joints).\\n❌ **WELDED SLED/RUNNER BASES**: Two sled runners connected by cross-bars with INVISIBLE welds → \'Sled Base Frame\' (1 part).\\n   - ⚠️ This applies ONLY to continuous welded metal structures, NOT to wooden or assembled structures.\\n\\n### FEW-SHOT EXAMPLES\n\n#### EXAMPLE 1: UNIFIED STRUCTURE (Sideboard with Sled Base)\nINPUT: Sideboard with two sled-style metal legs. Bottom view shows cross-bars connecting them.\nOUTPUT:\n{\n  "scene_description": "Sideboard with metal base",\n  "language": "en",\n  "objects": [\n    {\n      "category": "Sideboard",\n      "assembly_tree": [\n        {\n          "group_name": "Main Body",\n          "parts": [\n            { "name": "Cabinet Frame", "base_color_hex": "#5D5247", "material": "wood" },\n          ]\n        },\n        {\n          "group_name": "Left Storage",\n          "subgroups": [\n            {\n              "group_name": "Cupboard Unit",\n              "parts": [\n                { "name": "Left Door", "base_color_hex": "#4F4136", "material": "wood" },\n                { "name": "Door Knob", "base_color_hex": "#FFD700", "material": "metal" }\n              ]\n            }\n          ]\n        },\n        {\n          "group_name": "Right Storage",\n          "subgroups": [\n            {\n              "group_name": "Top Drawer Unit",\n              "parts": [\n                { "name": "Top Drawer Front", "base_color_hex": "#4F4136", "material": "wood" },\n                { "name": "Top Knob", "base_color_hex": "#FFD700", "material": "metal" }\n              ]\n            },\n            {\n              "group_name": "Middle Drawer Unit",\n              "parts": [\n                { "name": "Mid Drawer Front", "base_color_hex": "#4F4136", "material": "wood" },\n                { "name": "Mid Knob", "base_color_hex": "#FFD700", "material": "metal" }\n              ]\n            },\n            {\n              "group_name": "Bottom Drawer Unit",\n              "parts": [\n                { "name": "Bot Drawer Front", "base_color_hex": "#4F4136", "material": "wood" },\n                { "name": "Bot Knob", "base_color_hex": "#FFD700", "material": "metal" }\n              ]\n            }\n          ]\n        },\n        {\n          "group_name": "Support Structure",\n          "parts": [\n            { "name": "Metal Base Frame", "base_color_hex": "#1A1A1A", "material": "metal" },\n          ]\n        }\n      ]\n    }\n  ]\n}\n\n#### EXAMPLE 2: INTEGRATED SEAT (Standard Chair)\nINPUT: A four-legged dining chair with integrated seat (no visible separate rails under seat panel). Rear legs extend upward to support backrest.\nOUTPUT:\n{\n  "scene_description": "Standard dining chair",\n  "language": "en",\n  "objects": [\n    {\n      "category": "Dining Chair",\n      "assembly_tree": [\n        {\n          "group_name": "Support Structure",\n          "parts": [\n            { "name": "Front Left Leg", "base_color_hex": "#8B4513", "material": "wood" },\n            { "name": "Front Right Leg", "base_color_hex": "#8B4513", "material": "wood" },\n            { "name": "Rear Left Leg", "base_color_hex": "#8B4513", "material": "wood" },\n            { "name": "Rear Right Leg", "base_color_hex": "#8B4513", "material": "wood" }\n          ]\n        },\n        {\n          "group_name": "Seating Area",\n          "parts": [\n            { "name": "Seat Panel", "base_color_hex": "#808080", "material": "wood" }\n          ]\n        },\n        {\n          "group_name": "Backrest Slats",\n          "parts": [\n            { "name": "Top Slat", "base_color_hex": "#8B4513", "material": "wood" },\n            { "name": "Middle Slat", "base_color_hex": "#8B4513", "material": "wood" },\n            { "name": "Bottom Slat", "base_color_hex": "#8B4513", "material": "wood" }\n          ]\n        },\n        {\n          "group_name": "Reinforcement Stretchers",\n          "parts": [\n            { "name": "Front Stretcher", "base_color_hex": "#8B4513", "material": "wood" },\n            { "name": "Rear Stretcher", "base_color_hex": "#8B4513", "material": "wood" },\n            { "name": "Top Left Stretcher", "base_color_hex": "#8B4513", "material": "wood" },\n            { "name": "Top Right Stretcher", "base_color_hex": "#8B4513", "material": "wood" }\n            { "name": "Bottom Left Stretcher", "base_color_hex": "#8B4513", "material": "wood" }\n            { "name": "Bottom Right Stretcher", "base_color_hex": "#8B4513", "material": "wood" }\n          ]\n        }\n      ]\n    }\n  ]\n}\n\n#### EXAMPLE 3: MULTIPLE OBJECTS (Desk with Lamp)\nINPUT: A wooden office desk with a drawer, and a small metal lamp sitting on the desktop.\nOUTPUT:\n{\n  "scene_description": "Wooden desk with lamp",\n  "language": "en",\n  "objects": [\n    {\n      "category": "Office Desk",\n      "assembly_tree": [\n        {\n          "group_name": "Frame Structure",\n          "parts": [\n            { "name": "Desktop", "base_color_hex": "#8B5A2B", "material": "wood" },\n            { "name": "Left Leg", "base_color_hex": "#8B5A2B", "material": "wood" },\n            { "name": "Right Leg", "base_color_hex": "#8B5A2B", "material": "wood" },\n            { "name": "Back Panel", "base_color_hex": "#8B5A2B", "material": "wood" }\n          ]\n        },\n        {\n          "group_name": "Drawer Unit",\n          "parts": [\n            { "name": "Drawer Front", "base_color_hex": "#A06B3C", "material": "wood" },\n            { "name": "Drawer Handle", "base_color_hex": "#C0C0C0", "material": "metal" }\n          ]\n        }\n      ]\n    },\n    {\n      "category": "Table Lamp",\n      "assembly_tree": [\n        {\n          "group_name": "Lamp Base",\n          "parts": [\n            { "name": "Base Stand", "base_color_hex": "#2F4F4F", "material": "metal" },\n            { "name": "Stem", "base_color_hex": "#2F4F4F", "material": "metal" }\n          ]\n        },\n        {\n          "group_name": "Lamp Shade",\n          "parts": [\n            { "name": "Shade", "base_color_hex": "#F5F5DC", "material": "fabric" }\n          ]\n        }\n      ]\n    }\n  ]\n}\n\n#### EXAMPLE 4: CONNECTED METAL BASE (Wicker Armchair with Wireframe Legs)\nINPUT: A wicker bucket armchair sitting on a metal wireframe base. \nThe legs are connected by cross-braces into a single welded structure. \nThe shell has a visible rim frame, woven body, and attachment points where it connects to the base.\nOUTPUT:\n{\n  "scene_description": "Wicker armchair metal base",\n  "language": "en",\n  "objects": [\n    {\n      "category": "Armchair",\n      "assembly_tree": [\n        {\n          "group_name": "Seat Shell",\n          "parts": [\n            { "name": "Rim Frame", "base_color_hex": "#C8C8C8", "material": "rattan" },\n            { "name": "Woven Body", "base_color_hex": "#D4A060", "material": "rattan" },\n            { "name": "Structural Ribs", "base_color_hex": "#8B7355", "material": "rattan" }\n          ]\n        },\n        {\n          "group_name": "Base Structure",\n          "parts": [\n            { "name": "Metalic Base Frame", "base_color_hex": "#A0A0A0", "material": "metal" },\n          ]\n        }\n      ]\n    }\n  ]\n}\n\n'


class LeafPart(BaseModel):
    """One segmentable part -- a leaf of the assembly tree."""

    name: str = Field(description="Short part name, max 3 words, no colour adjectives.")
    base_color_hex: str = Field(default="", description="Dominant hex colour of the part.")
    material: str = Field(default="", description="Material, e.g. wood, metal, fabric.")


class Group(BaseModel):
    """A functional group. May nest one level of subgroups (PixMesh: max depth 2)."""

    group_name: str = Field(description="Functional group name.")
    subgroups: List["Group"] = Field(default_factory=list)
    parts: List[LeafPart] = Field(default_factory=list)


class SceneObject(BaseModel):
    category: str = Field(description="Object name, e.g. Sideboard, Workbench.")
    assembly_tree: List[Group] = Field(default_factory=list)


class Assembly(BaseModel):
    """The whole describe result: scene + objects, each a tree of parts."""

    scene_description: str = Field(default="", description="Very short, max 5 words.")
    language: str = Field(default="en")
    objects: List[SceneObject] = Field(default_factory=list)


def group_leaves(group: Group) -> List[LeafPart]:
    """Every leaf part under one group, in order, subgroups included."""
    out = list(group.parts)
    for sub in group.subgroups:
        out.extend(group_leaves(sub))
    return out


def leaf_parts(assembly: "Assembly") -> List[LeafPart]:
    """Every leaf part, in order, walking subgroups too.

    PixMesh's own palette assignment never recursed into ``subgroups``, so parts
    nested under one silently got no colour and vanished. Walking the whole tree
    is that bug fixed.
    """
    return [part for obj in assembly.objects
            for group in obj.assembly_tree
            for part in group_leaves(group)]


# The palette caps how many parts a mask can carry, and the VLM cannot reliably
# paint many more distinct colours than this anyway -- past ~20 it confuses them.
# PixMesh's describe is detail-first and will happily list 30+ parts, so the cap
# is stated to the model up front (merge the least significant) and enforced as a
# safety net below.
_PART_CAP = (
    "\n\n### HARD PART LIMIT\n"
    f"Return AT MOST {MAX_PARTS} leaf parts in total. First list every part you "
    f"see; if that exceeds {MAX_PARTS}, merge the least significant ones (small "
    "or thin elements, repeated slats/stretchers) into their groups until you are "
    f"at or under {MAX_PARTS}. Keep the most structurally significant parts "
    "(legs, seat, main panels, doors, drawers) as individual entries. If your "
    f"natural count is under {MAX_PARTS}, do not simplify."
)


def describe_assembly(grid: Image.Image, model: str = DESCRIBE_MODEL) -> Assembly:
    """Describe the object as a hierarchical assembly tree from the 4-view grid.

    Structured output: pydantic-ai forces the schema and re-asks on a mismatch,
    so a malformed reply is retried, not raised three stages later.
    """
    agent = _agent(model, Assembly, _DESCRIBE_SYSTEM_STRUCT)
    from pydantic_ai import BinaryContent

    logger.info("[describe_assembly] asking %s about a %dx%d 4-view grid", model, *grid.size)
    started = time.monotonic()
    result = agent.run_sync([
        _DESCRIBE_USER_STRUCT + _PART_CAP,
        BinaryContent(data=_png_bytes(grid), media_type="image/png"),
    ])
    parts = leaf_parts(result.output)
    if not parts:
        raise RuntimeError("describe found no parts")
    logger.info("[describe_assembly] '%s', %d object(s), %d leaf parts in %.1fs",
                result.output.scene_description, len(result.output.objects),
                len(parts), time.monotonic() - started)
    for obj in result.output.objects:
        for group in obj.assembly_tree:
            logger.info("[describe_assembly]   %s / %s: %s", obj.category, group.group_name,
                        ", ".join(p.name for p in group_leaves(group)) or "(empty)")
    return result.output


def assign_palette_tree(assembly: "Assembly") -> Dict[str, Tuple[int, int, int]]:
    """One distinct palette colour per leaf part, in tree order.

    Never wraps the palette -- two parts sharing a colour collapse into one. If
    the model overruns the cap despite being told (it sometimes does), the least
    prominent parts past the palette are dropped rather than colliding: the
    describe lists most-prominent first, so the tail is what to lose.
    """
    parts = leaf_parts(assembly)
    names = list(dict.fromkeys(p.name for p in parts))  # de-dup, keep order
    if len(names) < len(parts):
        logger.warning("[assign_palette_tree] %d of %d part names are duplicates, merged",
                       len(parts) - len(names), len(parts))
    if len(names) > len(PALETTE):
        logger.warning("[assign_palette_tree] %d parts over %d colours, dropping the "
                       "%d least prominent: %s", len(names), len(PALETTE),
                       len(names) - len(PALETTE), ", ".join(names[len(PALETTE):]))
        names = names[:len(PALETTE)]
    logger.info("[assign_palette_tree] %d parts coloured", len(names))
    return {name: PALETTE[i] for i, name in enumerate(names)}


# The contour colour with_contours draws, referenced by the generation prompt.
_CONTOUR_HEX = "#FF00FF"


def _generation_prompt(palette: Dict[str, Tuple[int, int, int]], size: Tuple[int, int]) -> str:
    """PixMesh's flat-label-map generation prompt, minus the multi-view
    POV/visibility machinery (heuristic on part names, which PixMesh itself flagged
    as inert). Every listed part may appear; the model decides what is visible.

    Departs from PixMesh's verbatim on RULE 4, hardened against the thin line the
    model draws between adjacent regions -- often a third part's colour (an orange
    seam between a yellow body and a purple collar) that no downstream snap can
    tell from a real thin part. The only place to kill it is here, so the rule is
    emphatic and carries the failure as a worked negative example."""
    w, h = size
    n = len(palette)
    bg = _hex(BACKGROUND)
    color_table = "\n".join(f"  {i + 1:2d}. {_hex(rgb)}  {name}"
                            for i, (name, rgb) in enumerate(palette.items()))
    return (
        "You are an expert 3D Segmentation Colorist. Your ONLY task: convert the input "
        "into a FLAT LABEL MAP where each part region is filled with one solid color.\n\n"
        "## MASTER COLOR PALETTE (ordered \u2014 use ONLY these hex codes)\n"
        f"{color_table}\n\n"
        f"## INPUT\n"
        f"A single {w}\u00d7{h} image of a 3D object. A thin magenta line ({_CONTOUR_HEX}) "
        "marks part boundaries and the object silhouette. It is a GUIDE to paint over and "
        "delete \u2014 never to trace or keep.\n\n"
        "## RULES\n\n"
        "### 1. Geometry (identical to input)\n"
        f"- Output MUST be exactly {w}\u00d7{h}, same aspect ratio. No crop, pad, or scale \u2014 1:1 pixel mapping.\n"
        "- The silhouette and internal boundaries are a STRICT MASK \u2014 reproduce them pixel-accurate to the input.\n"
        "- Do NOT invent boundaries, rings, splits, or geometry. One continuous region = one color.\n\n"
        "### 2. Flat fill (no shading)\n"
        "- Discard all lighting, shadows, highlights, texture, and surface detail.\n"
        "- Every pixel of a region = the exact same RGB. No gradients, ramps, edge-darkening, or aliasing.\n\n"
        "### 3. Background is sacred\n"
        f"- Every background pixel ({bg}) in the input stays background. Never recolor it with a part color.\n"
        f"- Where the magenta line falls on the object, paint it with the part color; outside, paint it {bg}. "
        "It must NOT appear in the output.\n\n"
        "### 4. Hard edges \u2014 NO lines of any kind (most common failure)\n"
        "- Two adjacent parts meet at a HARD EDGE: color A's last pixel sits directly against color B's first pixel. "
        "Nothing between them \u2014 no transition, no third color, not one pixel.\n"
        "- Along an A\u2194B boundary, ONLY colors A and B may appear. Never run a third part's color as a seam there.\n"
        "  Example: at a yellow\u2194purple boundary, go straight yellow\u2192purple. An orange line along that seam is WRONG, "
        "even if orange is a real part elsewhere. A color appears ONLY where its part actually is.\n"
        "- FORBIDDEN between regions: any line, stroke, outline, border, seam, halo, or darkened edge, in ANY color "
        f"(including the magenta guide {_CONTOUR_HEX}).\n"
        "- If you catch yourself drawing along a boundary, stop: fill each side flat until the two fills touch.\n\n"
        "### 5. Color count & compliance\n"
        f"- Use AT MOST {n} distinct part colors, all from the table. Use fewer if fewer regions are visible.\n"
        "- Paint any part with even a small visible sliver. Fully invisible parts (0 pixels) stay background.\n"
        "- Use the table's hex codes EXACTLY. Never invent, swap, or reuse a color for two parts.\n"
        "- No text, labels, legends, or color keys anywhere in the image.\n\n"
        "## OUTPUT\n"
        f"Return exactly ONE {w}\u00d7{h} image: the flat segmented result. No text, no borders.\n"
    )



def generate_part_map(
    target: Image.Image,
    palette: Dict[str, Tuple[int, int, int]],
    contours: bool = True,
    model: str = PAINT_MODEL,
) -> Image.Image:
    """Paint ``target`` into a flat part map using ``palette``, via the VLM.

    The structured sibling of :func:`paint_freeform`: PixMesh's generation prompt
    with the palette imposed, so the output colours are the ones we assigned.
    """
    from pydantic_ai import Agent, BinaryContent, BinaryImage
    from pydantic_ai.capabilities import NativeTool
    from pydantic_ai.native_tools import ImageGenerationTool

    sent = with_contours(target) if contours else target
    agent = Agent(model, output_type=BinaryImage, model_settings=_deterministic_settings(),
                  capabilities=[NativeTool(ImageGenerationTool(aspect_ratio="1:1"))])
    logger.info("[generate_part_map] asking %s for a %d-colour map on a %dx%d view "
                "(contours %s)", model, len(palette), *target.size,
                "on" if contours else "off")
    started = time.monotonic()
    result = agent.run_sync([
        _generation_prompt(palette, target.size),
        BinaryContent(data=_png_bytes(sent), media_type="image/png"),
    ])
    painted = Image.open(io.BytesIO(result.output.data)).convert("RGB")
    logger.info("[generate_part_map] got %dx%d in %.1fs", *painted.size,
                time.monotonic() - started)
    if painted.size != target.size:
        logger.warning("[generate_part_map] returned %dx%d, resizing to the target's "
                       "%dx%d", *painted.size, *target.size)
        painted = painted.resize(target.size, Image.NEAREST)
    return painted


# ── Seed-mask generation on a GeoSAM2 data-root ──────────────────────────────
# The VLM part map has to seed GeoSAM2, so it must live on one of the 12 canonical
# views, not on the PixMesh 3/4 rig. Every step -- describe, generation, seeding
# -- happens on the same canonical view, so no reprojection is needed.

# Canonical views at +25 elevation: the 3/4-high ones (see render.ELEVATIONS).
_HIGH_VIEWS = (1, 5, 9)


def _load_canonical(data_root: Union[str, Path], view: int) -> Image.Image:
    """A view's color_*.webp composited onto white, as the VLM should see it."""
    path = Path(data_root) / f"color_{view:04d}.webp"
    img = Image.open(path).convert("RGBA")
    white = Image.new("RGB", img.size, BACKGROUND)
    white.paste(img.convert("RGB"), (0, 0), img.getchannel("A"))
    return white


def _object_mask(data_root: Union[str, Path], view: int) -> np.ndarray:
    """Where the object is in a view -- the color map's alpha."""
    path = Path(data_root) / f"color_{view:04d}.webp"
    return np.asarray(Image.open(path).convert("RGBA").getchannel("A")) > 0


def _detail_score(data_root: Union[str, Path], view: int) -> float:
    """How much segmentable detail a view shows -- edge density over the object.

    Silhouette area is the wrong measure: a cabinet's blank back has the same
    outline as its drawered front but nothing to segment. Canny edges over the
    object count the seams, panels and hardware that make a view informative.
    """
    rgb = np.asarray(_load_canonical(data_root, view))
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 50, 150) > 0
    mask = _object_mask(data_root, view)
    return float(edges[mask].mean()) if mask.any() else 0.0


def best_target_view(data_root: Union[str, Path], candidates: Sequence[int] = _HIGH_VIEWS) -> int:
    """The 3/4-high view showing the most segmentable detail.

    Not "the one matching the PixMesh 3/4" -- that match is neither exact nor
    needed. The most-detailed view is the one the VLM can tell the most parts
    apart on, and the one worth seeding.
    """
    scores = {v: _detail_score(data_root, v) for v in candidates}
    best = max(scores, key=scores.get)
    logger.info("[best_target_view] view %d (edge density %s)", best,
                ", ".join(f"view {v}: {s:.4f}" for v, s in scores.items()))
    return best


def _canonical_grid(data_root: Union[str, Path], views: Sequence[int], tile: int = 512) -> Image.Image:
    """A labelled 2x2 grid from a data-root's color views, for describe."""
    grid = Image.new("RGB", (tile * 2, tile * 2), BACKGROUND)
    for i, v in enumerate(views[:4]):
        view = _load_canonical(data_root, v).resize((tile, tile)).copy()
        draw = ImageDraw.Draw(view)
        label = f"VIEW {v}"
        draw.rectangle([0, 0, tile - 1, tile - 1], outline=(0, 0, 0), width=2)
        draw.rectangle([2, 2, 12 + 7 * len(label), 18], fill=(0, 0, 0))
        draw.text((5, 4), label, fill=(255, 255, 255))
        grid.paste(view, ((i % 2) * tile, (i // 2) * tile))
    return grid


# Mirrors MASK_MIN_AREA_PX in inference.py: a colour region smaller than this is
# dropped when the mask is read, so a part painted below it is a part that will
# not reach GeoSAM2 however clearly the VLM drew it.
MIN_PART_PX = 64


class SeedMask(NamedTuple):
    """What :func:`generate_seed_mask` produced, and what survived.

    ``coverage`` is not decoration: a part the model listed but never painted
    lands at zero, which is the difference between "GeoSAM2 missed it" and "it
    was never asked about". ``painted`` counts the parts that cleared
    :data:`MIN_PART_PX`, i.e. the ones the mask can actually seed.
    """

    view: int
    assembly: "Assembly"
    palette: Dict[str, Tuple[int, int, int]]
    coverage: Dict[str, int]
    path: Path

    @property
    def painted(self) -> int:
        return sum(1 for px in self.coverage.values() if px >= MIN_PART_PX)


def generate_seed_mask(
    data_root: Union[str, Path],
    target_view_idx: Optional[int] = None,
) -> SeedMask:
    """Describe the object and paint a seed mask on one canonical view.

    Renders happen upstream (render_views wrote the data-root). Here: pick the
    target view, describe from a 4-view grid, assign a palette, have the VLM paint
    the target's color map, snap it to the exact palette, and write it as
    ``mask_{target:04d}.png`` in the data-root -- ready for GeoSAM2 to seed from.
    """
    data_root = Path(data_root)
    started = time.monotonic()
    logger.info("[generate_seed_mask] === %s ===", data_root)
    target = best_target_view(data_root) if target_view_idx is None else target_view_idx
    if target_view_idx is not None:
        logger.info("[generate_seed_mask] target view %d (given, not chosen)", target)

    # Describe from the target plus three views a quarter-turn apart: front,
    # sides, back, so parts hidden in the target are still seen somewhere.
    describe_views = [(target + k) % 12 for k in (0, 3, 6, 9)]
    logger.info("[generate_seed_mask] describe grid from views %s", describe_views)
    grid = _canonical_grid(data_root, describe_views)
    assembly = describe_assembly(grid)
    palette = assign_palette_tree(assembly)

    part_map = generate_part_map(_load_canonical(data_root, target), palette)
    # Snap to the exact palette so the mask carries N clean colours, not the
    # VLM's noise cloud -- GeoSAM2's extract_mask_segments keys on exact colours.
    snapped = snap_to_palette(part_map, palette, _object_mask(data_root, target))
    path = data_root / f"mask_{target:04d}.png"
    Image.fromarray(snapped).save(path)

    painted = coverage(snapped, palette)
    _log_coverage(painted)
    logger.info("[generate_seed_mask] === %s: %d/%d parts painted in %.1fs ===",
                path.name, sum(1 for px in painted.values() if px >= MIN_PART_PX),
                len(palette), time.monotonic() - started)
    return SeedMask(target, assembly, palette, painted, path)


def _log_coverage(painted: Dict[str, int], floor: int = MIN_PART_PX) -> None:
    """Report what each part actually got, and name the ones that got nothing.

    A part below ``floor`` is silently dropped downstream, so it is logged at
    WARNING here -- otherwise the only trace of it is a part count that is one
    lower than the describe promised. ``floor`` differs per consumer: the mask
    path is read at :data:`MIN_PART_PX`, the point-prompt path drops regions
    under ``auto_prompt.MIN_AREA_PX``, so callers pass their own.
    """
    for name, px in sorted(painted.items(), key=lambda kv: -kv[1]):
        logger.debug("[coverage]   %-28s %7d px", name, px)
    missing = [name for name, px in painted.items() if px == 0]
    thin = [f"{name} ({px}px)" for name, px in painted.items() if 0 < px < floor]
    if missing:
        logger.warning("[coverage] %d part(s) never painted: %s",
                       len(missing), ", ".join(missing))
    if thin:
        logger.warning("[coverage] %d part(s) under the %d px floor, dropped when the "
                       "mask is read: %s", len(thin), floor, ", ".join(thin))


def paint_freeform(
    image: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    model: str = PAINT_MODEL,
    contours: bool = True,
) -> Image.Image:
    """Send ``image`` to the VLM with an arbitrary prompt; return what it draws.

    The unstructured sibling of :func:`paint`: no part list, no palette, no
    snapping -- just the render (optionally with Canny edges overlaid, the trick
    that makes the model fill regions someone else outlined) and a free prompt.
    For iterating on the prompt in the mask-lab app before the structured chain
    is ported over.
    """
    from pydantic_ai import Agent, BinaryContent, BinaryImage
    from pydantic_ai.capabilities import NativeTool
    from pydantic_ai.native_tools import ImageGenerationTool

    sent = with_contours(image) if contours else image
    agent = Agent(model, output_type=BinaryImage, model_settings=_deterministic_settings(),
                  capabilities=[NativeTool(ImageGenerationTool(aspect_ratio="1:1"))])
    logger.info("[paint_freeform] asking %s (contours %s, %d-char prompt)",
                model, "on" if contours else "off", len(prompt))
    started = time.monotonic()
    result = agent.run_sync([
        prompt,
        BinaryContent(data=_png_bytes(sent), media_type="image/png"),
    ])
    painted = Image.open(io.BytesIO(result.output.data)).convert("RGB")
    logger.info("[paint_freeform] got %dx%d in %.1fs", *painted.size,
                time.monotonic() - started)
    if painted.size != image.size:
        logger.warning("[paint_freeform] returned %dx%d, resizing to the input's %dx%d",
                       *painted.size, *image.size)
        painted = painted.resize(image.size, Image.NEAREST)
    return painted


# Roughly the LAB distance at which two colours stop being a blend of each other
# and start being different colours. Only used to report, never to reject.
_OFF_PALETTE_LAB = 25.0


def snap_to_palette(
    image: Image.Image,
    palette: Dict[str, Tuple[int, int, int]],
    object_mask: Optional[np.ndarray] = None,
    background: Tuple[int, int, int] = BACKGROUND,
) -> np.ndarray:
    """Force every pixel onto the nearest palette entry, returning RGB.

    The model returns colours *near* the ones it was given, and blends them at
    every boundary. Left alone, each blend reads as a part of its own. Nearest
    neighbour is measured in LAB, not RGB: RGB distance is not perceptual, so it
    happily maps a dark blend onto a colour a human would never confuse it with.

    ``object_mask`` forces everything outside the silhouette to background,
    which is free here -- the renderer knows exactly where the object is, so a
    model that painted over the edge cannot invent geometry.
    """
    rgb = np.asarray(image.convert("RGB"))
    entries = np.array([background] + list(palette.values()), dtype=np.uint8)

    lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_entries = cv2.cvtColor(entries.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).astype(np.float32)[0]

    distance = np.linalg.norm(lab_image[:, :, None, :] - lab_entries[None, None, :, :], axis=-1)
    nearest = np.argmin(distance, axis=-1)
    snapped = entries[nearest]

    # How far the model's colours were from the ones it was given. A large share
    # over the threshold means it invented colours rather than blending between
    # palette entries, and every one of those pixels has just been forced onto a
    # part it may not belong to.
    off = np.take_along_axis(distance, nearest[..., None], axis=-1)[..., 0] > _OFF_PALETTE_LAB
    if object_mask is not None:
        off &= object_mask
        snapped[~object_mask] = background
        # Over the object only: dividing by the whole frame would dilute the
        # rate by however much background is in view, so the 5% threshold below
        # would mean something different for a tight crop than a loose one.
        denom = int(object_mask.sum())
    else:
        denom = off.size
    share = float(off.sum()) / denom if denom else 0.0
    if share > 0.05:
        logger.warning("[snap_to_palette] %.1f%% of object pixels were over %d LAB from "
                       "any palette colour -- the model painted off-palette",
                       share * 100, _OFF_PALETTE_LAB)
    else:
        logger.info("[snap_to_palette] %d colours, %.1f%% off-palette pixels",
                    len(palette), share * 100)
    return snapped


def coverage(snapped: np.ndarray, palette: Dict[str, Tuple[int, int, int]]) -> Dict[str, int]:
    """Pixels each part actually got. Zeros are parts the model never painted."""
    return {
        name: int(np.all(snapped == np.array(color, np.uint8), axis=-1).sum())
        for name, color in palette.items()
    }


def generate_prompts(
    source: Union[str, Path],
    view_idx: int = 0,
    resolution: int = RESOLUTION,
    debug_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[Dict], PartList, Dict[str, int]]:
    """Mesh or view directory in, GeoSAM2 point prompts out.

    Returns ``(prompts, parts, coverage)``. Coverage is not decoration: a part
    the model listed but never painted lands at zero, which is the difference
    between "GeoSAM2 missed it" and "it was never asked about".
    """
    mesh = Path(source)
    if mesh.is_dir():
        mesh = mesh / "mesh.glb"

    logger.info("[generate_prompts] === %s, view %d ===", mesh, view_idx)
    started = time.monotonic()
    render = shaded_view(mesh, view=view_idx, resolution=resolution)
    parts = describe(render)
    palette = assign_palette(parts.parts)

    painted = paint(render, parts, palette)
    object_mask = np.any(np.asarray(render.convert("RGB")) < 250, axis=-1)
    snapped = snap_to_palette(painted, palette, object_mask)

    prompts = prompts_from_color_map(snapped, view_idx=view_idx, background=BACKGROUND)
    painted_px = coverage(snapped, palette)
    _log_coverage(painted_px, floor=MIN_AREA_PX)
    logger.info("[generate_prompts] === %d prompts from %d parts in %.1fs ===",
                len(prompts), len(palette), time.monotonic() - started)

    if debug_dir is not None:
        debug = Path(debug_dir)
        debug.mkdir(parents=True, exist_ok=True)
        render.save(debug / f"view{view_idx:04d}_render.png")
        with_contours(render).save(debug / f"view{view_idx:04d}_contours.png")
        painted.save(debug / f"view{view_idx:04d}_painted.png")
        Image.fromarray(snapped).save(debug / f"view{view_idx:04d}_snapped.png")
        logger.info("[generate_prompts] debug images -> %s", debug)

    return prompts, parts, painted_px


def _hex(rgb: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
