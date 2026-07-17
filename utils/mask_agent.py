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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

from utils.auto_prompt import prompts_from_color_map
from utils.render import RESOLUTION, shaded_view

# Load the repo-root .env before pydantic-ai resolves a provider. Its Google
# provider reads GOOGLE_API_KEY or GEMINI_API_KEY, though it only names the
# first when it fails.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# PixMesh names gemini-3-pro-preview and gemini-3-pro-image-preview. The first
# still appears in models.list but 404s on generateContent -- listed is not the
# same as callable, so check by calling. These are the current equivalents.
DESCRIBE_MODEL = "google:gemini-3.1-pro-preview"
PAINT_MODEL = "google:gemini-3-pro-image"

# Background is white because `shaded_view` renders on white, and the prompt
# leans on "leave the background alone" being trivially checkable.
BACKGROUND = (255, 255, 255)

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

    return Agent(model, output_type=output_type, system_prompt=system)


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

    result = agent.run_sync([
        "Identify this object and list its separable parts.",
        BinaryContent(data=_png_bytes(image), media_type="image/png"),
    ])
    parts = result.output.parts[:MAX_PARTS]
    if not parts:
        raise RuntimeError("the model found no parts in the render")
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
    agent = Agent(model, output_type=BinaryImage,
                  capabilities=[NativeTool(ImageGenerationTool(aspect_ratio="1:1"))])
    result = agent.run_sync([
        _paint_prompt(parts, palette),
        BinaryContent(data=_png_bytes(with_contours(image)), media_type="image/png"),
    ])
    painted = Image.open(io.BytesIO(result.output.data)).convert("RGB")
    if painted.size != image.size:
        # NEAREST, never LANCZOS: interpolating a label map manufactures colours
        # that are in no part's palette along every boundary.
        painted = painted.resize(image.size, Image.NEAREST)
    return painted


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
    snapped = entries[np.argmin(distance, axis=-1)]

    if object_mask is not None:
        snapped[~object_mask] = background
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

    render = shaded_view(mesh, view=view_idx, resolution=resolution)
    parts = describe(render)
    palette = assign_palette(parts.parts)

    painted = paint(render, parts, palette)
    object_mask = np.any(np.asarray(render.convert("RGB")) < 250, axis=-1)
    snapped = snap_to_palette(painted, palette, object_mask)

    prompts = prompts_from_color_map(snapped, view_idx=view_idx, background=BACKGROUND)

    if debug_dir is not None:
        debug = Path(debug_dir)
        debug.mkdir(parents=True, exist_ok=True)
        render.save(debug / f"view{view_idx:04d}_render.png")
        with_contours(render).save(debug / f"view{view_idx:04d}_contours.png")
        painted.save(debug / f"view{view_idx:04d}_painted.png")
        Image.fromarray(snapped).save(debug / f"view{view_idx:04d}_snapped.png")

    return prompts, parts, coverage(snapped, palette)


def _hex(rgb: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
