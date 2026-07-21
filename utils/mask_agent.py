"""Ask a VLM to paint a flat part map, and snap it into a GeoSAM2 seed mask.

GeoSAM2 segments what you seed, so an arbitrary mesh needs that seed from
somewhere. This asks Gemini: describe the object as a hierarchical assembly tree
-> assign one palette colour per leaf part -> repaint a view in those colours ->
snap to the exact palette. ``generate_seed_mask`` runs it on one of the 12
canonical views and writes the mask GeoSAM2 seeds from.

Ported from PixMesh's SegviGen brick. Needs ``GEMINI_API_KEY`` (a ``.env`` at the
repo root is loaded automatically).
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, BinaryImage
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.settings import ModelSettings

from utils import prompts
from utils.logs import get_logger

logger = get_logger("geosam2.mask_agent")

# .env at the repo root, loaded before pydantic-ai reads GEMINI/GOOGLE_API_KEY.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────


# gemini-3-pro-preview 404s on generateContent despite being listed; these are the callable equivalents.
DESCRIBE_MODEL = "google:gemini-3.1-pro-preview"
PAINT_MODEL = "google:gemini-3-pro-image"

# temperature=0 + fixed seed: a prompt change is then the only thing moving between runs.
_SEED = int(os.environ.get("GEOSAM2_SEED", "1234"))
_SETTINGS = ModelSettings(temperature=0.0, seed=_SEED)

# Black: GeoSAM2's mask reader skips pure black as background, and it reads better empirically.
BACKGROUND = (0, 0, 0)

# Kelly's maximum-contrast colours, minus white and near-blacks, ordered by separation.
PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (243, 195, 0), (135, 86, 146), (243, 132, 0), (161, 202, 241),
    (190, 0, 50), (194, 178, 128), (132, 132, 130), (0, 136, 86),
    (230, 143, 172), (0, 103, 165), (249, 147, 121), (96, 78, 151),
    (246, 166, 0), (179, 68, 108), (220, 211, 0), (136, 45, 23),
    (141, 182, 0), (101, 69, 34), (226, 88, 34), (43, 61, 38),
)

# The ceiling: past it, two parts would share a colour and collapse into one.
MAX_PARTS = len(PALETTE)

# The magenta with_contours draws, referenced by the generation prompt.
_CONTOUR_HEX = "#FF00FF"

# Mirrors inference.py's MASK_MIN_AREA_PX: a region below this is dropped when the mask is read.
MIN_PART_PX = 64

# LAB distance past which two colours are different, not a blend. Reporting only.
_OFF_PALETTE_LAB = 25.0


# ────────────────────────────────────────────────────────────────────────────
# Schema — describe assembly tree
# ────────────────────────────────────────────────────────────────────────────


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

    def leaves(self) -> List[LeafPart]:
        """Every leaf part under this group, in order, subgroups included."""
        out = list(self.parts)
        for sub in self.subgroups:
            out.extend(sub.leaves())
        return out


class SceneObject(BaseModel):
    category: str = Field(description="Object name, e.g. Sideboard, Workbench.")
    assembly_tree: List[Group] = Field(default_factory=list)


class Assembly(BaseModel):
    """The whole describe result: scene + objects, each a tree of parts."""

    scene_description: str = Field(default="", description="Very short, max 5 words.")
    language: str = Field(default="en")
    objects: List[SceneObject] = Field(default_factory=list)

    def leaf_parts(self) -> List[LeafPart]:
        """Every leaf part, in order, walking subgroups too.

        PixMesh's own palette assignment never recursed into ``subgroups``, so
        parts nested under one silently got no colour and vanished. Walking the
        whole tree is that bug fixed.
        """
        return [part for obj in self.objects
                for group in obj.assembly_tree
                for part in group.leaves()]


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


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def with_contours(
    image: Image.Image, 
    color: Tuple[int, int, int] = (255, 0, 255)
) -> Image.Image:
    """Overlay magenta Canny edges: the model fills outlined regions far better than it finds boundaries."""
    grey = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grey, 50, 150)
    out = np.asarray(image.convert("RGB")).copy()
    out[edges > 0] = color
    return Image.fromarray(out)


def _load_canonical(data_root: Union[str, Path], view: int) -> Image.Image:
    """A view's color_*.webp composited onto white, as the VLM should see it."""
    img = Image.open(Path(data_root) / f"color_{view:04d}.webp").convert("RGBA")
    white = Image.new("RGB", img.size, BACKGROUND)
    white.paste(img.convert("RGB"), (0, 0), img.getchannel("A"))
    return white


def _canonical_grid(data_root: Union[str, Path], views: Sequence[int], tile: int = 512) -> Image.Image:
    """A labelled 2x2 grid from a data-root's color views, for describe."""
    grid = Image.new("RGB", (tile * 2, tile * 2), BACKGROUND)
    for i, v in enumerate(views[:4]):
        view = _load_canonical(data_root, v).resize((tile, tile)).copy()
        draw = ImageDraw.Draw(view)
        label = f"VIEW {v}"
        # White border/label so tiles stay separable on the black background.
        draw.rectangle([0, 0, tile - 1, tile - 1], outline=(255, 255, 255), width=2)
        draw.rectangle([2, 2, 12 + 7 * len(label), 18], fill=(255, 255, 255))
        draw.text((5, 4), label, fill=(0, 0, 0))
        grid.paste(view, ((i % 2) * tile, (i // 2) * tile))
    return grid


def _log_coverage(painted: Dict[str, int]) -> None:
    """Report what each part got, and name the ones a mask read would drop.

    A part below :data:`MIN_PART_PX` is silently dropped by extract_mask_segments,
    so it is logged at WARNING -- otherwise the only trace is a part count one
    lower than the describe promised.
    """
    for name, px in sorted(painted.items(), key=lambda kv: -kv[1]):
        logger.debug("[coverage]   %-28s %7d px", name, px)
    missing = [name for name, px in painted.items() if px == 0]
    thin = [f"{name} ({px}px)" for name, px in painted.items() if 0 < px < MIN_PART_PX]
    if missing:
        logger.warning("[coverage] %d part(s) never painted: %s",
                       len(missing), ", ".join(missing))
    if thin:
        logger.warning("[coverage] %d part(s) under the %d px floor, dropped when the "
                       "mask is read: %s", len(thin), MIN_PART_PX, ", ".join(thin))


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def describe_assembly(grid: Image.Image, model: str = DESCRIBE_MODEL) -> Assembly:
    """Describe the object as a hierarchical assembly tree from the 4-view grid.

    Structured output: pydantic-ai forces the schema and re-asks on a mismatch,
    so a malformed reply is retried, not raised three stages later.
    """
    agent = Agent(model, output_type=Assembly, system_prompt=prompts.DESCRIBE_SYSTEM,
                  model_settings=_SETTINGS)
    logger.info("[describe_assembly] asking %s about a %dx%d 4-view grid", model, *grid.size)
    started = time.monotonic()
    result = agent.run_sync([
        prompts.DESCRIBE_USER + prompts.part_cap(MAX_PARTS),
        BinaryContent(data=_png_bytes(grid), media_type="image/png"),
    ])
    parts = result.output.leaf_parts()
    if not parts:
        raise RuntimeError("describe found no parts")
    logger.info("[describe_assembly] '%s', %d object(s), %d leaf parts in %.1fs",
                result.output.scene_description, len(result.output.objects),
                len(parts), time.monotonic() - started)
    for obj in result.output.objects:
        for group in obj.assembly_tree:
            logger.info("[describe_assembly]   %s / %s: %s", obj.category, group.group_name,
                        ", ".join(p.name for p in group.leaves()) or "(empty)")
    return result.output


def assign_palette_tree(assembly: "Assembly") -> Dict[str, Tuple[int, int, int]]:
    """One distinct palette colour per leaf part, in tree order.

    Never wraps the palette -- two parts sharing a colour collapse into one. If
    the model overruns the cap despite being told (it sometimes does), the least
    prominent parts past the palette are dropped rather than colliding: the
    describe lists most-prominent first, so the tail is what to lose.
    """
    parts = assembly.leaf_parts()
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


def generate_part_map(
    target: Image.Image,
    palette: Dict[str, Tuple[int, int, int]],
    contours: bool = True,
    model: str = PAINT_MODEL,
) -> Image.Image:
    """Paint ``target`` into a flat part map using ``palette``, via the VLM.

    PixMesh's generation prompt with the palette imposed, so the output colours
    are the ones we assigned.
    """
    sent = with_contours(target) if contours else target
    agent = Agent(model, output_type=BinaryImage, model_settings=_SETTINGS,
                  capabilities=[NativeTool(ImageGenerationTool(aspect_ratio="1:1"))])
    logger.info("[generate_part_map] asking %s for a %d-colour map on a %dx%d view "
                "(contours %s)", model, len(palette), *target.size,
                "on" if contours else "off")
    started = time.monotonic()
    result = agent.run_sync([
        prompts.generation_prompt(palette, target.size, BACKGROUND, _CONTOUR_HEX),
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


def snap_to_palette(
    image: Image.Image,
    palette: Dict[str, Tuple[int, int, int]],
    object_mask: Optional[np.ndarray] = None,
    background: Tuple[int, int, int] = BACKGROUND,
) -> np.ndarray:
    """Force every pixel onto the nearest palette entry (LAB, not RGB), returning RGB.

    ``object_mask`` sends everything outside the silhouette to background, so a
    model that painted over the edge cannot invent geometry.
    """
    rgb = np.asarray(image.convert("RGB"))
    entries = np.array([background] + list(palette.values()), dtype=np.uint8)

    lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_entries = cv2.cvtColor(entries.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).astype(np.float32)[0]

    distance = np.linalg.norm(lab_image[:, :, None, :] - lab_entries[None, None, :, :], axis=-1)
    nearest = np.argmin(distance, axis=-1)
    snapped = entries[nearest]

    # Pixels far from any palette colour = the model painted off-palette (report only).
    off = np.take_along_axis(distance, nearest[..., None], axis=-1)[..., 0] > _OFF_PALETTE_LAB
    if object_mask is not None:
        off &= object_mask
        snapped[~object_mask] = background
        denom = int(object_mask.sum())  # over the object, so the threshold is crop-independent
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


# Canonical view the VLM paints on and GeoSAM2 seeds from (el +25, az 300 -- a 3/4-high read), picked in the mask lab.
SEED_VIEW = 1


def seed_view_inputs(
    data_root: Union[str, Path],
    seed_view: int = SEED_VIEW
) -> Tuple[Image.Image, Image.Image]:
    """The (describe grid, paint target) for ``seed_view`` on a rendered data-root.

    Shared by :func:`generate_seed_mask` and the mask lab, so the lab tunes the
    prompt on exactly the views the server paints and seeds on. The grid is the
    seed view plus three quarter-turns (front, sides, back), so parts the seed
    view hides are still seen somewhere.
    """
    data_root = Path(data_root)
    describe_views = [(seed_view + k) % 12 for k in (0, 3, 6, 9)]
    logger.info("[seed_view_inputs] view %d, describe grid from views %s",
                seed_view, describe_views)
    return _canonical_grid(data_root, describe_views), _load_canonical(data_root, seed_view)


def generate_seed_mask(
    data_root: Union[str, Path], 
    seed_view: int = SEED_VIEW
) -> SeedMask:
    """Describe, paint a part map on canonical ``seed_view``, and write it as the
    GeoSAM2 seed mask.

    Renders happen upstream (render_views wrote the data-root's color views);
    :func:`seed_view_inputs` builds the grid + target, the VLM paints, and the
    snapped map is written as ``mask_{seed_view:04d}.png`` -- ready to seed.
    """
    data_root = Path(data_root)
    started = time.monotonic()
    logger.info("[generate_seed_mask] === %s (view %d) ===", data_root, seed_view)

    grid, target = seed_view_inputs(data_root, seed_view)
    assembly = describe_assembly(grid)
    palette = assign_palette_tree(assembly)

    part_map = generate_part_map(target, palette)
    mask = np.asarray(part_map.convert("RGB"))  # raw VLM output, no snapping
    path = data_root / f"mask_{seed_view:04d}.png"
    Image.fromarray(mask).save(path)

    painted = {name: int(np.all(mask == np.array(c, np.uint8), axis=-1).sum())
               for name, c in palette.items()}
    _log_coverage(painted)
    logger.info("[generate_seed_mask] === %s: %d/%d parts in %.1fs ===",
                path.name, sum(1 for px in painted.values() if px >= MIN_PART_PX),
                len(palette), time.monotonic() - started)
    return SeedMask(seed_view, assembly, palette, painted, path)
