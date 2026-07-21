"""GeoSAM2 evaluation pipeline: a mesh in, a part-coloured GLB out.

This module is the single seam between the evaluation app and GeoSAM2. The app
never sees a label array, a view directory, or a subprocess -- it hands a mesh to
``GeoSAM2Segmenter.process`` and gets a ``trimesh.Scene`` back.

GeoSAM2 cannot consume a mesh directly: it needs 12 canonical views first, which
``utils/render.py`` rasterises in-process. Passing a directory that already holds
those views skips that stage, which is how the bundled ``example/sample_*`` roots
are used.

The segmentation itself is driven by running ``inference.py`` as a subprocess.
That is deliberate for this placeholder -- swapping it for direct imports is the
next step, and it only touches ``_run_inference``.

Every stage logs at INFO, and the subprocesses' own output is relayed line by
line as it arrives rather than captured and shown only on failure: a run takes
minutes, and silence for that long is indistinguishable from a hang.
"""

from __future__ import annotations

import colorsys
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh

from utils.logs import DEV, get_logger
from utils.render import render_views

logger = get_logger("geosam2.pipeline")

REPO_ROOT = Path(__file__).resolve().parents[1]

NUM_VIEWS = 12

# Labels that mean "no part here", not "a part numbered this".
#
# 999 is the value mask_aggregation initialises its label volume with
# (utils/inference_utils.py:409), so it survives on any face no mask ever
# covered. It is not rare -- the reference sample_00 run carries it on 0.7% of
# faces -- and counting it as a part inflates every part count by one and paints
# a phantom region. 0 is the background label the exporter paints black
# (:500); it does not appear in practice, but it costs nothing to treat alike.
UNASSIGNED_LABELS = (0, 999)

_UNLABELED_RGBA = np.array([120, 120, 120, 255], dtype=np.uint8)


def _distinct_palette(count: int) -> np.ndarray:
    """Build ``count`` visually distinct RGBA colours.

    Uses a golden-angle walk around the hue circle rather than a fixed
    matplotlib colormap: GeoSAM2 routinely returns more than 20 parts, and a
    tab20-style palette would start recycling colours at that point, which reads
    as two parts sharing an identity.
    """
    colours = np.zeros((max(count, 1), 4), dtype=np.uint8)
    golden_ratio_conjugate = 0.618033988749895
    for i in range(count):
        hue = (i * golden_ratio_conjugate) % 1.0
        saturation = 0.55 + 0.20 * (i % 3) / 2.0
        value = 0.95 - 0.20 * (i % 2)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colours[i] = (int(r * 255), int(g * 255), int(b * 255), 255)
    return colours


def _rgba_to_hex(rgba: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgba[0]), int(rgba[1]), int(rgba[2]))


def is_view_directory(path: Path) -> bool:
    """Whether ``path`` holds a complete set of GeoSAM2 input views."""
    if not path.is_dir():
        return False
    if not (path / "meta.json").is_file() or not (path / "mesh.glb").is_file():
        return False
    for prefix, suffix in (("color", "webp"), ("depth", "exr"), ("normal", "webp")):
        for view in range(NUM_VIEWS):
            if not (path / f"{prefix}_{view:04d}.{suffix}").is_file():
                return False
    return True


class GeoSAM2Segmenter:
    """Runs GeoSAM2 end-to-end and returns a scene of coloured parts.

    Stateless with respect to run parameters: the constructor only resolves
    paths, every knob lives on :meth:`process`. That lets the app hold one
    instance and call it per request.
    """

    def __init__(
        self,
        repo_root: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_cfg: str = "configs/geosam2.yaml",
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.repo_root / "ckpt" / "geosam2.pt"
        self.model_cfg = model_cfg
        self.temp_dirs: List[str] = []

    # -- public API ---------------------------------------------------------

    def process(
        self,
        source: Union[str, Path, trimesh.Trimesh, trimesh.Scene],
        postprocess_pa: float = 0.02,
        enable_postprocess: bool = True,
        opposite_auto_segmentation: bool = True,
        mask_path: Optional[Union[str, Path]] = None,
        mask_view: Optional[int] = None,
        point_prompt_file: Optional[Union[str, Path]] = None,
        seed_view: Optional[int] = None,
        mask_threshold: float = 0.0,
        vlm_mask: bool = False,
    ) -> Dict[str, Any]:
        """Segment ``source`` and return a result dict.

        ``source`` is either a mesh (path/Trimesh/Scene), which is rendered to 12
        views first, or a directory already holding those views.

        The seed prompt is given either as ``point_prompt_file`` (clicks, which
        are turned into a 2D mask first — this is the reference pipeline the
        paper evaluates) or as a ready-made ``mask_path`` + ``mask_view``. With
        neither, GeoSAM2 falls back to automatic mask generation, which the paper
        does not measure.

        Failures come back as a dict with the same shape and a ``status``
        starting with ``"Error"``, so the caller has one code path -- except
        under ``utils.logs.DEV`` (the default), where the exception is logged
        with its traceback and re-raised at the point it happened. A bench that
        turns a stack trace into a string, three stages after the fact, cannot
        be debugged; set ``GEOSAM2_DEV=0`` for the never-raises behaviour.
        """
        result = self._empty_result()
        work_dir = Path(tempfile.mkdtemp(prefix="geosam2_eval_"))
        self.temp_dirs.append(str(work_dir))
        result["temp_dir"] = str(work_dir)
        started = time.monotonic()
        logger.info("[segment] === %s (vlm_mask=%s, postprocess=%s, auto=%s) ===",
                    source if isinstance(source, (str, Path)) else type(source).__name__,
                    vlm_mask, enable_postprocess, opposite_auto_segmentation)

        try:
            # Raised rather than returned early, so the one handler below decides
            # what a failure looks like -- a traceback under DEV, an error status
            # otherwise. A silent early return would bypass both.
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"checkpoint not found at {self.checkpoint_path}")
            if point_prompt_file is not None and mask_path is not None:
                raise ValueError("Provide either point_prompt_file or mask_path, not both")

            data_root, render_log = self._resolve_views(source, work_dir)
            result["log"] += render_log
            result["data_root"] = str(data_root)

            if vlm_mask:
                # Auto-generate the seed mask with the VLM: describe the object
                # from a grid, paint a part map on the chosen canonical view, and
                # seed GeoSAM2 from it. No prompt file, no manual mask.
                if mask_path is not None or point_prompt_file is not None:
                    raise ValueError("vlm_mask cannot combine with a mask or prompt file")
                from utils.mask_agent import SEED_VIEW, generate_seed_mask

                seed = generate_seed_mask(
                    data_root, seed_view if seed_view is not None else SEED_VIEW)
                mask_path = seed.path
                mask_view = seed.view
                result["seed_mask_path"] = str(mask_path)
                result["seed_view"] = seed.view
                result["vlm_scene"] = seed.assembly.scene_description
                result["vlm_parts"] = seed.painted
                result["vlm_coverage"] = seed.coverage
                # `painted`, not `len(palette)`: the palette is what was asked
                # for, and a part the VLM never drew cannot seed anything.
                result["log"] += (f"VLM seed mask on view {seed.view}: "
                                  f"{seed.assembly.scene_description}, "
                                  f"{seed.painted}/{len(seed.palette)} parts painted\n")
            else:
                logger.info("[segment] no VLM stage (vlm_mask=False)")

            if point_prompt_file is not None:
                mask_path, mask_view, prompt_log = self._run_point_prompts(
                    data_root=data_root,
                    point_prompt_file=point_prompt_file,
                    seed_view=seed_view,
                    work_dir=work_dir,
                    mask_threshold=mask_threshold,
                )
                result["log"] += prompt_log
                result["seed_mask_path"] = str(mask_path)
                result["seed_view"] = mask_view

            # A mask given directly (not via prompt/VLM) still has a seed view;
            # record it so the UI shows that view's maps.
            if mask_view is not None and result["seed_view"] is None:
                result["seed_view"] = mask_view

            output_dir = work_dir / "seg"
            output_dir.mkdir(parents=True, exist_ok=True)

            result["log"] += self._run_inference(
                data_root=data_root,
                output_dir=output_dir,
                postprocess_pa=postprocess_pa,
                enable_postprocess=enable_postprocess,
                opposite_auto_segmentation=opposite_auto_segmentation,
                mask_path=mask_path,
                mask_view=mask_view,
            )

            labels_path = self._find_labels(output_dir)
            result["labels_path"] = str(labels_path)
            logger.info("[labels] %s", labels_path.name)

            scene, structure, n_parts, unlabeled_faces = self._build_scene(
                data_root / "mesh.glb", np.load(labels_path)
            )
            glb_path = work_dir / "segmented.glb"
            scene.export(glb_path)

            result.update(
                status=f"OK - {n_parts} parts",
                glb_path=str(glb_path),
                scene=scene,
                structure=structure,
                n_parts=n_parts,
                unlabeled_faces=unlabeled_faces,
            )
            logger.info("[segment] === OK: %d parts, %d unlabeled faces, %.1fs ===",
                        n_parts, unlabeled_faces, time.monotonic() - started)
            return result

        except Exception as exc:
            logger.exception("[segment] === FAILED after %.1fs: %s ===",
                             time.monotonic() - started, exc)
            if DEV:
                raise
            result["status"] = f"Error: {exc}"
            return result

    def cleanup(self) -> None:
        """Remove every temp directory this instance created."""
        for path in self.temp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.temp_dirs.clear()

    def __del__(self) -> None:
        self.cleanup()

    # -- stages -------------------------------------------------------------

    def _resolve_views(
        self,
        source: Union[str, Path, trimesh.Trimesh, trimesh.Scene],
        work_dir: Path,
    ) -> Tuple[Path, str]:
        """Return a directory holding the 12 views, rendering them if needed."""
        if isinstance(source, (str, Path)):
            candidate = Path(source)
            if candidate.is_dir():
                if not is_view_directory(candidate):
                    raise ValueError(
                        f"{candidate} is not a complete view directory "
                        f"(needs meta.json, mesh.glb and {NUM_VIEWS} color/depth/normal views)"
                    )
                logger.info("[views] pre-rendered, %s", candidate)
                return candidate, f"Using pre-rendered views: {candidate}\n"
            mesh_path = candidate
        else:
            mesh_path = work_dir / "input_mesh.glb"
            source.export(mesh_path)
            logger.info("[views] exported in-memory %s to %s",
                        type(source).__name__, mesh_path.name)

        if not mesh_path.is_file():
            raise FileNotFoundError(f"mesh not found: {mesh_path}")
        return self._render(mesh_path, work_dir / "views")

    def _render(self, mesh_path: Path, output_dir: Path) -> Tuple[Path, str]:
        """Rasterise the twelve views in-process.

        Uses utils/render.py rather than shelling out to Blender: the model only
        reads geometry buffers, so there is nothing to shade, and the Blender
        script only runs under 4.0/4.1 anyway.
        """
        logger.info("[render] %d views of %s -> %s", NUM_VIEWS, mesh_path.name, output_dir)
        started = time.monotonic()
        render_views(mesh_path, output_dir)
        if not is_view_directory(output_dir):
            raise RuntimeError(f"render produced an incomplete view directory at {output_dir}")
        logger.info("[render] done in %.1fs", time.monotonic() - started)
        return output_dir, f"Rendered 12 views: {output_dir}\n"

    def _run_point_prompts(
        self,
        data_root: Path,
        point_prompt_file: Union[str, Path],
        seed_view: Optional[int],
        work_dir: Path,
        mask_threshold: float,
    ) -> Tuple[Path, int, str]:
        """Turn a point-prompt file into a 2D seed mask (step 1 of the reference
        pipeline). Returns ``(mask_path, view_idx, log)``."""
        prompt_path = Path(point_prompt_file)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"point prompt file not found: {prompt_path}")

        view_idx = _resolve_seed_view(prompt_path, seed_view)
        out_dir = work_dir / "2d_seg"
        out_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable, str(self.repo_root / "single_view_point_prompt_infer.py"),
            "--data-root", str(data_root),
            "--view-idx", str(view_idx),
            "--point-prompt-file", str(prompt_path),
            "--output-dir", str(out_dir),
            "--sam2-checkpoint", str(self.checkpoint_path),
            "--model-cfg", self.model_cfg,
            "--mask-threshold", str(mask_threshold),
        ]
        logger.info("[prompts] %s -> view %d", prompt_path.name, view_idx)
        code, log = self._run_logged(command, "prompts")
        if code != 0:
            raise RuntimeError(
                f"single_view_point_prompt_infer.py failed (exit {code}). Log:\n{log}"
            )

        mask_path = out_dir / f"mask_view{view_idx:04d}.npy"
        if not mask_path.is_file():
            raise RuntimeError(f"point-prompt step wrote no mask at {mask_path}. Log:\n{log}")
        logger.info("[prompts] seed mask %s", mask_path.name)
        return mask_path, view_idx, log

    def _run_inference(
        self,
        data_root: Path,
        output_dir: Path,
        postprocess_pa: float,
        enable_postprocess: bool,
        opposite_auto_segmentation: bool,
        mask_path: Optional[Union[str, Path]],
        mask_view: Optional[int],
    ) -> str:
        if mask_path is not None and mask_view is None:
            raise ValueError("mask_view is required when mask_path is provided")
        if mask_path is None and not opposite_auto_segmentation:
            # inference.py exits 0 without writing anything in this combination.
            raise ValueError(
                "Nothing to segment: with no mask prompt, opposite auto-segmentation "
                "must stay enabled or GeoSAM2 produces no labels."
            )

        command = [
            sys.executable, str(self.repo_root / "inference.py"),
            "--data-root", str(data_root),
            "--output-dir", str(output_dir),
            "--sam2-checkpoint", str(self.checkpoint_path),
            "--model-cfg", self.model_cfg,
            "--postprocess-pa", str(postprocess_pa),
            "--enable-postprocess" if enable_postprocess else "--no-enable-postprocess",
            "--opposite-auto-segmentation" if opposite_auto_segmentation
            else "--no-opposite-auto-segmentation",
        ]
        if mask_path is not None:
            command += ["--mask-path", str(mask_path), "--mask-view", str(mask_view)]
            logger.info("[inference] seeding view %s from %s", mask_view, Path(mask_path).name)
        else:
            logger.info("[inference] no seed prompt, automatic mask generation")

        code, log = self._run_logged(command, "inference")
        if code != 0:
            raise RuntimeError(f"inference.py failed (exit {code}). Log:\n{log}")
        return log

    def _run_logged(self, command: List[str], tag: str) -> Tuple[int, str]:
        """Run a stage subprocess, relaying its output as it arrives.

        ``capture_output=True`` hides a hundred seconds of progress and only
        surfaces it if the stage fails, which is exactly backwards while
        debugging. Streams are merged so the ordering that reaches the log is
        the one the subprocess actually produced.

        cwd matters: Hydra resolves configs/geosam2.yaml against the package.
        """
        logger.info("[%s] $ %s", tag, " ".join(command))
        started = time.monotonic()
        lines: List[str] = []
        process = subprocess.Popen(
            command, cwd=str(self.repo_root), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        with process:
            for line in process.stdout:
                line = line.rstrip()
                lines.append(line)
                logger.info("[%s] %s", tag, line)

        logger.info("[%s] exit %d in %.1fs", tag, process.returncode,
                    time.monotonic() - started)
        output = _tail("\n".join(lines))
        return process.returncode, f"$ {' '.join(command)}\n{output}\n"

    def _find_labels(self, output_dir: Path) -> Path:
        matches = sorted(glob.glob(str(output_dir / "*.npy")))
        if not matches:
            raise RuntimeError(
                f"inference.py wrote no label file to {output_dir}. It exits successfully "
                "in that case, so this usually means no masks were produced."
            )
        # Post-processing writes a second, better file; prefer it when present.
        postprocessed = [m for m in matches if "postprocessed" in Path(m).name]
        return Path(postprocessed[-1] if postprocessed else matches[-1])

    def _build_scene(
        self, mesh_path: Path, face_label: np.ndarray
    ) -> Tuple[trimesh.Scene, Dict[str, Any], int, int]:
        """Paint per-face labels onto the original mesh, one geometry per part.

        The GLB that inference.py exports is rebuilt in a rotated/translated/scaled
        frame, so it cannot be shown next to the input. The label array, by
        contrast, indexes the input mesh's faces directly -- so we reload the
        source mesh and colour that instead, keeping both viewers aligned.
        """
        # Must match load_mesh_with_faces() in utils/inference_utils.py, which
        # loads with force="mesh" and trimesh's default processing, or the face
        # ordering the labels refer to would not be ours.
        mesh = trimesh.load(mesh_path, force="mesh")
        face_label = np.asarray(face_label).reshape(-1)

        if len(face_label) != len(mesh.faces):
            raise RuntimeError(
                f"label/mesh mismatch: {len(face_label)} labels for {len(mesh.faces)} faces"
            )

        labels = np.unique(face_label)
        part_labels = [int(v) for v in labels if int(v) not in UNASSIGNED_LABELS]
        palette = _distinct_palette(len(part_labels))

        scene = trimesh.Scene()
        children: List[Dict[str, Any]] = []

        for index, label in enumerate(part_labels):
            mask = face_label == label
            rgba = palette[index]
            name = f"part_{index + 1:03d}"
            self._add_part(scene, mesh, mask, rgba, name)
            children.append({"name": name, "color": _rgba_to_hex(rgba), "faces": int(mask.sum())})

        # Every unassigned label collapses into one grey region: they all mean
        # the same thing, and how much of the mesh lands here is the signal --
        # a prompt set that misses whole areas shows up as this growing.
        unassigned = np.isin(face_label, UNASSIGNED_LABELS)
        unlabeled_faces = int(unassigned.sum())
        logger.info("[lift] %d parts over %d faces, %d unassigned (%.1f%%)",
                    len(part_labels), len(mesh.faces), unlabeled_faces,
                    100.0 * unlabeled_faces / max(len(mesh.faces), 1))
        if unlabeled_faces:
            self._add_part(scene, mesh, unassigned, _UNLABELED_RGBA, "unassigned")
            children.append({
                "name": "unassigned",
                "color": _rgba_to_hex(_UNLABELED_RGBA),
                "faces": unlabeled_faces,
            })

        structure = {"name": mesh_path.stem, "children": children}
        return scene, structure, len(part_labels), unlabeled_faces

    @staticmethod
    def _add_part(
        scene: trimesh.Scene,
        mesh: trimesh.Trimesh,
        mask: np.ndarray,
        rgba: np.ndarray,
        name: str,
    ) -> None:
        part = mesh.submesh([mask], append=True)
        # Replace the visual rather than assigning to visual.vertex_colors: a
        # textured source mesh yields TextureVisuals, which has no vertex_colors
        # setter, so the assignment would be a silent no-op and the part would
        # keep the original texture instead of its part colour.
        part.visual = trimesh.visual.ColorVisuals(
            mesh=part, vertex_colors=np.tile(rgba, (len(part.vertices), 1))
        )
        part.metadata["name"] = name
        scene.add_geometry(part, node_name=name, geom_name=name)

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "status": "Error: pipeline did not run",
            "glb_path": None,
            "scene": trimesh.Scene(),
            "structure": {},
            "temp_dir": None,
            "data_root": None,
            "labels_path": None,
            "seed_mask_path": None,
            "seed_view": None,
            "vlm_scene": None,
            "vlm_parts": 0,
            "vlm_coverage": {},
            "n_parts": 0,
            "unlabeled_faces": 0,
            "log": "",
        }


def _prompt_frames(prompt_path: Path) -> set:
    """The set of views a point-prompt file carries prompts for."""
    try:
        entries = json.loads(prompt_path.read_text())
        frames = {int(e["frame_idx"]) for e in entries if "frame_idx" in e}
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"cannot read point prompts from {prompt_path.name}: {exc}")
    if not frames:
        raise ValueError(f"{prompt_path.name} carries no frame_idx entries")
    return frames


def _resolve_seed_view(prompt_path: Path, seed_view: Optional[int]) -> int:
    """Decide which view to seed, and refuse the combinations that lie.

    Prompts are pixel coordinates only valid for one specific view, but
    single_view_point_prompt_infer.py:149 throws away the frame it resolved and
    seeds ``--view-idx`` regardless: ask it for a view the file has no prompts
    for and it quietly reuses another view's points, segmenting the wrong view
    and writing it under the requested view's name. Both branches below exist to
    make that impossible rather than to guess.
    """
    frames = _prompt_frames(prompt_path)

    if seed_view is None:
        if len(frames) != 1:
            raise ValueError(
                f"{prompt_path.name} prompts {len(frames)} views ({sorted(frames)}); "
                "pass seed_view to choose which one to seed."
            )
        return next(iter(frames))

    view_idx = int(seed_view)
    if view_idx not in frames:
        raise ValueError(
            f"{prompt_path.name} has no prompts for view {view_idx} "
            f"(it prompts {sorted(frames)}); GeoSAM2 would seed view {view_idx} "
            "with another view's points instead of failing."
        )
    return view_idx


def _tail(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else "...\n" + text[-limit:]


def list_samples(repo_root: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Bundled view directories usable without Blender."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    samples = []
    for path in sorted((root / "example").glob("sample_*")):
        if is_view_directory(path):
            prompts = sorted(p.name for p in path.glob("point_prompts_*.json"))
            masks = sorted(p.name for p in path.glob("mask_*.png"))
            samples.append({"name": path.name, "path": str(path),
                            "point_prompts": prompts, "masks": masks})
    return samples
