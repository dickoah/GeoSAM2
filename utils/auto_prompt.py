"""Turn a colour segmentation map into GeoSAM2 point prompts.

GeoSAM2 is prompt-controllable: it segments what you click on, and the automatic
mode is a derived behaviour the paper never measures. So an arbitrary mesh needs
seed clicks from somewhere. This module produces them from a colour map -- one
painted by a VLM, exported from another tool, or authored by hand.

Point prompts rather than the colour map itself, deliberately. ``--mask-path``
does accept a colour PNG, but it takes the mask at face value: every blurry edge
pixel and every off-palette colour becomes geometry. Prompts collapse a region to
a single interior point, so the parts of a generated map that are least
trustworthy -- its boundaries -- stop mattering. It is also the format the
reference pipeline uses, which is the one validated end to end.

Deterministic and VLM-free: everything here runs on an image. The stage that
*generates* that image lives in :mod:`utils.mask_agent`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

# Below this, a colour region is noise rather than a part. Matches
# MASK_MIN_AREA_PX in inference.py, which drops the same regions when reading a
# colour mask, so both paths agree on what counts as a part.
MIN_AREA_PX = 100

# A part can survive as several disconnected blobs (a handle seen through a gap,
# a symmetric pair sharing a label). Each gets its own click, but tiny fragments
# are noise -- clicking one would ask SAM2 to grow a part from a speck.
MIN_COMPONENT_PX = 50

BLACK = np.zeros(3, dtype=np.uint8)


def _rgb(image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    image = np.asarray(image)
    return image[..., :3] if image.ndim == 3 and image.shape[2] >= 3 else image


def interior_point(mask: np.ndarray) -> Tuple[int, int]:
    """A pixel comfortably inside ``mask``, as ``(x, y)``.

    The centroid is the obvious choice and the wrong one: for a C-shaped or
    hollow region it lands outside the mask entirely, and the prompt would then
    describe a different part. The distance transform's peak is the point
    furthest from any edge, so it is always inside and maximally unambiguous.
    """
    distance = ndimage.distance_transform_edt(mask)
    flat = int(np.argmax(distance))
    y, x = np.unravel_index(flat, mask.shape)
    return int(x), int(y)


def segment_colors(
    rgb: np.ndarray, background: Optional[Sequence[int]] = None
) -> List[Tuple[Tuple[int, int, int], np.ndarray]]:
    """Split a colour map into ``(colour, mask)`` parts, background dropped.

    Mirrors ``extract_mask_segments`` (inference.py:120): the most frequent
    colour is the background, pure black is ignored, and regions under the area
    floor are dropped.
    """
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    if len(colors) == 0:
        return []

    bg = np.asarray(background, dtype=np.uint8) if background is not None \
        else colors[int(np.argmax(counts))]

    segments = []
    for color in colors:
        if np.array_equal(color, BLACK) or np.array_equal(color, bg):
            continue
        mask = np.all(rgb == color, axis=-1)
        if int(mask.sum()) < MIN_AREA_PX:
            continue
        segments.append((tuple(int(c) for c in color), mask))
    return segments


def prompts_from_color_map(
    image: Union[str, Path, Image.Image, np.ndarray],
    view_idx: int = 0,
    background: Optional[Sequence[int]] = None,
    negatives: bool = False,
) -> List[Dict]:
    """Point prompts seeding ``view_idx``, one object per colour.

    Emits the schema ``single_view_point_prompt_infer.py`` validates:
    ``{frame_idx, obj_id, point: [x, y], label}``, where label 1 includes and 0
    excludes. Coordinates are pixels of the view they were read from -- the
    caller must pass the view the map was rendered on, or the clicks land on
    another part of the object.

    ``negatives`` adds one exclusion click per part, at its nearest neighbour's
    anchor, to push a mask off the part it is most likely to bleed into.

    Off by default, because they measurably hurt. Scored on sample_01 by adjusted
    Rand against feeding the same map to ``--mask-path``: no negatives 0.77, one
    negative per part 0.58, and the obvious "exclude every other part" 0.21 --
    there, 91 negatives against 13 positives drown the prompt and SAM2 abandons
    whole regions. Keep the flag for experimenting; do not turn it on by reflex.
    """
    rgb = _rgb(image)
    segments = segment_colors(rgb, background)

    # Sort by area, largest first, so obj_id ordering is stable across runs
    # rather than following numpy's colour sort.
    segments.sort(key=lambda s: int(s[1].sum()), reverse=True)

    anchors: List[Tuple[int, Tuple[int, int]]] = []
    prompts: List[Dict] = []

    for index, (_, mask) in enumerate(segments):
        obj_id = index + 1
        labelled, count = ndimage.label(mask)
        for component in range(1, count + 1):
            blob = labelled == component
            if int(blob.sum()) < MIN_COMPONENT_PX:
                continue
            x, y = interior_point(blob)
            prompts.append({"frame_idx": view_idx, "obj_id": obj_id,
                            "point": [float(x), float(y)], "label": 1})
            anchors.append((obj_id, (x, y)))

    if negatives:
        for obj_id, (x, y) in anchors:
            others = [(o, p) for o, p in anchors if o != obj_id]
            if not others:
                continue
            _, (nx, ny) = min(others, key=lambda a: (a[1][0] - x) ** 2 + (a[1][1] - y) ** 2)
            prompts.append({"frame_idx": view_idx, "obj_id": obj_id,
                            "point": [float(nx), float(ny)], "label": 0})

    return prompts


def write_prompts(prompts: List[Dict], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts, indent=2))
    return path


def summarize(prompts: List[Dict]) -> str:
    objects = sorted({p["obj_id"] for p in prompts})
    positive = sum(1 for p in prompts if p.get("label", 1) == 1)
    frames = sorted({p["frame_idx"] for p in prompts})
    return (f"{len(objects)} objects, {len(prompts)} prompts "
            f"({positive} positive / {len(prompts) - positive} negative), view {frames}")
