"""VLM prompts for the mask-agent pipeline, kept out of the logic module.

Two constants (PixMesh's describe system/user prompts, verbatim) and two builders
that take the caller's palette and limits -- so this module imports nothing from
mask_agent and there is no import cycle. The generation prompt departs from
PixMesh's on RULE 4, hardened against the thin line the model draws between two
regions (a third part's colour along an A|B seam).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple


def _hex(rgb: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


DESCRIBE_SYSTEM = """### ROLE
Senior 3D Product Analyst and Mechanical Design Expert.

### OBJECTIVES
1. **VISUALLY INSPECT** all views to identify every distinct component.
2. **FAVOUR DETAIL**: List every part you can visually distinguish.
3. **GENERATE** the most complete assembly tree possible.

### DETAIL-FIRST PRINCIPLE (CRITICAL — READ THIS FIRST)
Your DEFAULT behaviour is to list every visually-distinct component as a SEPARATE part.
Merging is the EXCEPTION, not the rule. Only merge when:
  - Two surfaces are truly INDISTINGUISHABLE (no visible boundary at all), AND
  - They serve the SAME functional purpose.
When in doubt, LIST MORE PARTS. The downstream pipeline handles detail well; oversimplification breaks it.

### THE ONLY MERGE EXCEPTION
Merge into one part ONLY for continuous welded/molded METAL frames where joints are truly invisible:
  - Sled bases, U-frames, spider bases, cantilever frames → 1 part.
⚠️ This does NOT apply to:
  - Assemblies where you can see distinct pieces (trestle columns, turned legs, stretcher beams)
  - Seat + Backrest (ALWAYS separate — see functional zone test in rules)
  - Anything that is NOT a welded metal frame

### OUTPUT FORMAT
Return a SINGLE valid JSON object.
"""


DESCRIBE_USER = """### INPUT CONTEXT
You will receive a grid of 4 images showing a 3D scene from different viewpoints.

### YOUR TASK
Analyze the scene to identify all objects and decompose each into its constituent parts.

### DETAILED OUTPUT FORMAT
Return a SINGLE valid JSON object with this exact structure:
{
  "scene_description": "<VERY SHORT description (max 5 words)>",
  "language": "<en|fr|other>",
  "objects": [
    {
      "category": "<Object Name (e.g. Workbench, Draped Cloth)>",
      "assembly_tree": [
        {
          "group_name": "<Storage Unit / Functional Group>",
          "subgroups": [  // OPTIONAL: Use for hierarchical parts like Door Sets, Drawer Units
            {
              "group_name": "<Subgroup Name>",
              "parts": [{ "name": "...", "base_color_hex": "...", "material": "..." }]
            }
          ],
          "parts": [ // Direct parts of this group
            { "name": "<Part Name>", "base_color_hex": "<HEX color>", "material": "<material>" }
          ]
        }
      ]
    }
  ]
}

### CRITICAL RULES
1. **SEGMENTABLE PARTS ONLY**: List only parts that correspond to physically separable mesh regions.
   - If you can SEE where one piece ends and another begins → they are SEPARATE parts.
   - Trestle columns, pedestals, stretcher beams, decorative spindles → each is its own part if visually distinct.
   - Only merge when two elements are truly INDISTINGUISHABLE (same shape, same surface, no visible boundary between them).
   - ⚠️ DO NOT hallucinate parts that are not visible — but DO list everything you CAN see.
   - ⚠️ DO NOT invent or assume tiny structural accessories that are NOT VISIBLY DISTINCT in the images. If you cannot clearly see an element as a separate piece, it does not exist.
   - If unsure whether an element exists as a separate piece, assume it is integrated into its parent.

2. **COUNT ACCURATELY**: Examine ALL 4 views systematically.
   - Check BOTTOM VIEW and REAR VIEW for structure not obvious from the front.
   - BOTTOM/REAR views may reveal stretcher beams, cross-braces, backrest support frames, or base boxes — list them as parts.
   - **Floating Seat/Top**: If there is a visible gap between the seat/top and the supporting frame, look for small **spacers**, **risers**, or **blocks** that connect them. List these as separate parts if visible.
   - Connection does NOT mean same part. Two pieces bolted or joined together are still SEPARATE parts if you can see the boundary.

3. **GROUPING**: Organize parts under logical parent groups by FUNCTION.
   - Vertical supports → 'Support Structure' (legs, posts, columns)
   - Horizontal connecting elements → 'Stretchers' or 'Rails'
   - Seating surfaces → 'Seating' (Seat, Backrest — always separate parts)
   - Storage units (drawers, doors) → 'Storage' with subgroups per unit
   - Repeated elements (slats, rungs) → group name indicating plurality (e.g., 'Backrest Slats')
   - ⚠️ Grouping is for ORGANIZATION only. It does NOT mean parts in a group should be merged.

4. **NAMING**: Short, descriptive names (max 3 words). No parentheses, slashes, or color adjectives.
   - ✅ Good: 'Front Stretcher', 'Seat Panel', 'Top Slat'
   - ❌ Bad: 'Rail (Lower)', 'Front/Side Bar', 'Blue Connector'

5. **WHAT IS ONE PART? (Visual distinctness is the primary criterion)**:
   a) **Visual boundary test (PRIMARY)**: If you can see a clear boundary, joint line, or shape change between two elements → they are SEPARATE parts.
      - Trestle table: each column, each foot/base, each stretcher beam = separate parts
      - Turned/carved legs: each leg is its own part
      - Ornate pedestals with distinct top and bottom sections → separate parts
      - ⚠️ Only list a part if it is a distinct geometric shape that could be cut apart from its neighbors in a 3D mesh. Sub-features of the same continuous surface (e.g. leaf buds on a branch, growth tips, individual petal veins) do NOT count as separate parts.
   b) **Welded-frame exception (NARROW)**: Only merge into one part when the structure is a CONTINUOUS welded/molded piece with NO visible joints:
      - Sled base, cantilever frame, tubular bent frame → 1 part each
      - **Sled legs**: Two sled runners connecting left and right sides OFTEN form a unified base. If they look like part of the same metal structure (even if cross-bar is hidden), merge them as 'Support Frame' or 'Metal Base'.
      - **Metal Frames**: Merge continuous metal support structures (legs + connecting rails) into a single 'Frame' part unless distinct joinery is visible.
      - ⚠️ If you can see where pieces JOIN (bolts, dowels, mortise-tenon, distinct shapes meeting) → they are SEPARATE parts, not welded.
      - ⚠️ **STAR/SWIVEL BASE EXCEPTION**: Star-shaped or swivel bases (office chairs, task chairs) with clearly distinct radiating arms ARE separate parts — list each arm as 'Base Arm 1', 'Base Arm 2', … 'Base Arm N' PLUS the central hub as 'Base Hub'. Do NOT merge them into a single 'Swivel Base' or 'Spider Base' part. Their arms radiate outward with visible gaps between them and are geometrically separable.
   c) **Magnifying-glass test**: Tiny accessories that are not individually segmentable (e.g. welded tips, small caps, bumpers, glides, feet caps) are NOT separate parts — they belong to their parent structure.
      - Do NOT list any accessory smaller than ~5%% of the object's total size as a separate part, UNLESS it is a structural spacer/riser separating two major components.
      - Simple end-caps, ferrules, or glides on the bottom of legs are PART OF THE LEG.
      - If a small element is not clearly visible as a distinct piece across multiple views, it does not exist — do not invent it.
   d) **Hardware exception**: Knobs, handles, and pulls are SEPARATE parts — they are functional hardware, typically different material (e.g., metal knob on wood door).
   e) **Functional zone test (CRITICAL — OVERRIDES b)**: Even if two regions share the same material, color, and appear physically continuous (even MOLDED as one piece), they are SEPARATE parts if they serve DIFFERENT ergonomic or functional purposes.
      - Seat vs Backrest → ALWAYS 2 separate parts, even on a one-piece molded shell/bucket chair
      - Armrest vs Side Panel → separate if functionally distinct
      - Desktop vs Side Panel → separate
      - ❌ WRONG: 'Molded Seat Body' or 'Upholstered Body' or 'Shell Body' combining seat + backrest into 1 part
      - ✅ CORRECT: 'Seat' (1 part) + 'Backrest' (1 part) = 2 parts, even on a plastic bucket chair
      - 📌 Rule 5b (molded-piece exception) does NOT apply to seat+backrest. The functional zone test ALWAYS wins for primary ergonomic surfaces.
      - ⚠️ **ARMREST EXCEPTION**: If armrests are clearly molded or upholstered as a single continuous, seamless piece with the backrest (like on a tub chair or wingback chair), DO NOT separate them. Merge them into the 'Backrest' or 'Seat Shell'. Only separate armrests if there is a visible joint, gap, or material change.
   f) **Structure vs Cushioning**: Separate the rigid support structure (frame, plinth, base box) from the soft cushions (seat, backrest) if visible.
      - Sofa beds often have a visible 'Base Box' or 'Frame' underneath the 'Seat' and 'Backrest'. Check for this structural layer.
      - If you see a hard base supporting a soft seat/back, list them as separate parts.
   g) **Structural Enclosure / Carcass (OVERRIDES a for storage furniture)**: For cabinets, sideboards, credenzas, dressers, shelving units, and similar storage/case furniture, ALL outer panels forming the box-shaped shell (top, bottom, left side, right side, AND back panel) serve the SAME structural function — they are the **carcass**.
      - ✅ CORRECT: Describe them as ONE part: 'Cabinet Carcass' (1 single part encompassing all enclosure panels including the back)
      - ❌ WRONG: Listing 'Top Panel', 'Left Side Panel', 'Right Side Panel', 'Bottom Panel', 'Back Panel' as separate parts
      - ⚠️ This does NOT apply to panels with different functional purposes (e.g., a desktop vs a shelf vs a side panel on a desk).
      - 📌 The visual boundary test (5a) does NOT override this rule: panel joints on a carcass are assembly joints, not functional boundaries.
   h) **Parallel Planks/Slats exception**: When multiple identical planks or slats are arranged side-by-side to form a SINGLE continuous functional surface (like a Tabletop, a Bench Seat, or a Shelf), merge them into ONE part, even if seams are visible.
      - ✅ CORRECT: 'Tabletop' (composed of 3 side-by-side planks) = 1 part
      - ❌ WRONG: 'Left Plank', 'Center Plank', 'Right Plank'
      - ⚠️ This applies also to pallet-style surfaces or slatted seats if they form the main surface.

6. **HIERARCHY**: Use 'subgroups' for complex nested structures (max depth: 2 levels).
   - Separate top-level objects must NOT be nested under each other.

7. **METADATA**: Extract dominant color (HEX), material, and language.

### COMMON MISTAKES (READ BEFORE EXAMPLES)
❌ **OVER-SIMPLIFICATION**: Do NOT lump visually-distinct components into one part just because they are connected.\\n   - A trestle table base has separate columns, feet, and stretcher beams → list each one.\\n   - Only merge into one part when elements are truly INDISTINGUISHABLE (continuous welded metal with no visible joints).\\n❌ **WELDED SLED/RUNNER BASES**: Two sled runners connected by cross-bars with INVISIBLE welds → 'Sled Base Frame' (1 part).\\n   - ⚠️ This applies ONLY to continuous welded metal structures, NOT to wooden or assembled structures.\\n\\n### FEW-SHOT EXAMPLES

#### EXAMPLE 1: UNIFIED STRUCTURE (Sideboard with Sled Base)
INPUT: Sideboard with two sled-style metal legs. Bottom view shows cross-bars connecting them.
OUTPUT:
{
  "scene_description": "Sideboard with metal base",
  "language": "en",
  "objects": [
    {
      "category": "Sideboard",
      "assembly_tree": [
        {
          "group_name": "Main Body",
          "parts": [
            { "name": "Cabinet Frame", "base_color_hex": "#5D5247", "material": "wood" },
          ]
        },
        {
          "group_name": "Left Storage",
          "subgroups": [
            {
              "group_name": "Cupboard Unit",
              "parts": [
                { "name": "Left Door", "base_color_hex": "#4F4136", "material": "wood" },
                { "name": "Door Knob", "base_color_hex": "#FFD700", "material": "metal" }
              ]
            }
          ]
        },
        {
          "group_name": "Right Storage",
          "subgroups": [
            {
              "group_name": "Top Drawer Unit",
              "parts": [
                { "name": "Top Drawer Front", "base_color_hex": "#4F4136", "material": "wood" },
                { "name": "Top Knob", "base_color_hex": "#FFD700", "material": "metal" }
              ]
            },
            {
              "group_name": "Middle Drawer Unit",
              "parts": [
                { "name": "Mid Drawer Front", "base_color_hex": "#4F4136", "material": "wood" },
                { "name": "Mid Knob", "base_color_hex": "#FFD700", "material": "metal" }
              ]
            },
            {
              "group_name": "Bottom Drawer Unit",
              "parts": [
                { "name": "Bot Drawer Front", "base_color_hex": "#4F4136", "material": "wood" },
                { "name": "Bot Knob", "base_color_hex": "#FFD700", "material": "metal" }
              ]
            }
          ]
        },
        {
          "group_name": "Support Structure",
          "parts": [
            { "name": "Metal Base Frame", "base_color_hex": "#1A1A1A", "material": "metal" },
          ]
        }
      ]
    }
  ]
}

#### EXAMPLE 2: INTEGRATED SEAT (Standard Chair)
INPUT: A four-legged dining chair with integrated seat (no visible separate rails under seat panel). Rear legs extend upward to support backrest.
OUTPUT:
{
  "scene_description": "Standard dining chair",
  "language": "en",
  "objects": [
    {
      "category": "Dining Chair",
      "assembly_tree": [
        {
          "group_name": "Support Structure",
          "parts": [
            { "name": "Front Left Leg", "base_color_hex": "#8B4513", "material": "wood" },
            { "name": "Front Right Leg", "base_color_hex": "#8B4513", "material": "wood" },
            { "name": "Rear Left Leg", "base_color_hex": "#8B4513", "material": "wood" },
            { "name": "Rear Right Leg", "base_color_hex": "#8B4513", "material": "wood" }
          ]
        },
        {
          "group_name": "Seating Area",
          "parts": [
            { "name": "Seat Panel", "base_color_hex": "#808080", "material": "wood" }
          ]
        },
        {
          "group_name": "Backrest Slats",
          "parts": [
            { "name": "Top Slat", "base_color_hex": "#8B4513", "material": "wood" },
            { "name": "Middle Slat", "base_color_hex": "#8B4513", "material": "wood" },
            { "name": "Bottom Slat", "base_color_hex": "#8B4513", "material": "wood" }
          ]
        },
        {
          "group_name": "Reinforcement Stretchers",
          "parts": [
            { "name": "Front Stretcher", "base_color_hex": "#8B4513", "material": "wood" },
            { "name": "Rear Stretcher", "base_color_hex": "#8B4513", "material": "wood" },
            { "name": "Top Left Stretcher", "base_color_hex": "#8B4513", "material": "wood" },
            { "name": "Top Right Stretcher", "base_color_hex": "#8B4513", "material": "wood" }
            { "name": "Bottom Left Stretcher", "base_color_hex": "#8B4513", "material": "wood" }
            { "name": "Bottom Right Stretcher", "base_color_hex": "#8B4513", "material": "wood" }
          ]
        }
      ]
    }
  ]
}

#### EXAMPLE 3: MULTIPLE OBJECTS (Desk with Lamp)
INPUT: A wooden office desk with a drawer, and a small metal lamp sitting on the desktop.
OUTPUT:
{
  "scene_description": "Wooden desk with lamp",
  "language": "en",
  "objects": [
    {
      "category": "Office Desk",
      "assembly_tree": [
        {
          "group_name": "Frame Structure",
          "parts": [
            { "name": "Desktop", "base_color_hex": "#8B5A2B", "material": "wood" },
            { "name": "Left Leg", "base_color_hex": "#8B5A2B", "material": "wood" },
            { "name": "Right Leg", "base_color_hex": "#8B5A2B", "material": "wood" },
            { "name": "Back Panel", "base_color_hex": "#8B5A2B", "material": "wood" }
          ]
        },
        {
          "group_name": "Drawer Unit",
          "parts": [
            { "name": "Drawer Front", "base_color_hex": "#A06B3C", "material": "wood" },
            { "name": "Drawer Handle", "base_color_hex": "#C0C0C0", "material": "metal" }
          ]
        }
      ]
    },
    {
      "category": "Table Lamp",
      "assembly_tree": [
        {
          "group_name": "Lamp Base",
          "parts": [
            { "name": "Base Stand", "base_color_hex": "#2F4F4F", "material": "metal" },
            { "name": "Stem", "base_color_hex": "#2F4F4F", "material": "metal" }
          ]
        },
        {
          "group_name": "Lamp Shade",
          "parts": [
            { "name": "Shade", "base_color_hex": "#F5F5DC", "material": "fabric" }
          ]
        }
      ]
    }
  ]
}

#### EXAMPLE 4: CONNECTED METAL BASE (Wicker Armchair with Wireframe Legs)
INPUT: A wicker bucket armchair sitting on a metal wireframe base. 
The legs are connected by cross-braces into a single welded structure. 
The shell has a visible rim frame, woven body, and attachment points where it connects to the base.
OUTPUT:
{
  "scene_description": "Wicker armchair metal base",
  "language": "en",
  "objects": [
    {
      "category": "Armchair",
      "assembly_tree": [
        {
          "group_name": "Seat Shell",
          "parts": [
            { "name": "Rim Frame", "base_color_hex": "#C8C8C8", "material": "rattan" },
            { "name": "Woven Body", "base_color_hex": "#D4A060", "material": "rattan" },
            { "name": "Structural Ribs", "base_color_hex": "#8B7355", "material": "rattan" }
          ]
        },
        {
          "group_name": "Base Structure",
          "parts": [
            { "name": "Metalic Base Frame", "base_color_hex": "#A0A0A0", "material": "metal" },
          ]
        }
      ]
    }
  ]
}

"""


def part_cap(max_parts: int) -> str:
    """The HARD PART LIMIT suffix appended to :data:`DESCRIBE_USER`."""
    return (
        "\n\n### HARD PART LIMIT\n"
        f"Return AT MOST {max_parts} leaf parts in total. First list every part you "
        f"see; if that exceeds {max_parts}, merge the least significant ones (small "
        "or thin elements, repeated slats/stretchers) into their groups until you are "
        f"at or under {max_parts}. Keep the most structurally significant parts "
        "(legs, seat, main panels, doors, drawers) as individual entries. If your "
        f"natural count is under {max_parts}, do not simplify."
    )


def generation_prompt(palette: Dict[str, Tuple[int, int, int]], size: Tuple[int, int],
                      background: Tuple[int, int, int], contour_hex: str) -> str:
    """Flat-label-map generation prompt with ``palette`` imposed, over ``size``."""
    w, h = size
    n = len(palette)
    bg = _hex(background)
    color_table = "\n".join(f"  {i + 1:2d}. {_hex(rgb)}  {name}"
                            for i, (name, rgb) in enumerate(palette.items()))
    return (
        "You are an expert 3D Segmentation Colorist. Your ONLY task: convert the input "
        "into a FLAT LABEL MAP where each part region is filled with one solid color.\n\n"
        "## MASTER COLOR PALETTE (ordered \u2014 use ONLY these hex codes)\n"
        f"{color_table}\n\n"
        f"## INPUT\n"
        f"A single {w}\u00d7{h} image of a 3D object. A thin magenta line ({contour_hex}) "
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
        f"(including the magenta guide {contour_hex}).\n"
        "- If you catch yourself drawing along a boundary, stop: fill each side flat until the two fills touch.\n\n"
        "### 5. Color count & compliance\n"
        f"- Use AT MOST {n} distinct part colors, all from the table. Use fewer if fewer regions are visible.\n"
        "- Paint any part with even a small visible sliver. Fully invisible parts (0 pixels) stay background.\n"
        "- Use the table's hex codes EXACTLY. Never invent, swap, or reuse a color for two parts.\n"
        "- No text, labels, legends, or color keys anywhere in the image.\n\n"
        "## OUTPUT\n"
        f"Return exactly ONE {w}\u00d7{h} image: the flat segmented result. No text, no borders.\n"
    )
