"""GeoSAM2 stages: views in, per-face labels out, labels onto the mesh.

The step app drives GeoSAM2 one stage at a time -- seed mask, propagation,
lift -- so this module exposes the stages and nothing that chains them. The
segmentation itself runs ``inference.py`` as a subprocess, whose output is
relayed line by line as it arrives: a run takes minutes, and silence for that
long is indistinguishable from a hang.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh

from utils.logs import get_logger
from utils.split import UNASSIGNED_LABELS, UNASSIGNED_RGB, label_palette, to_linear_u8

logger = get_logger("geosam2.segmenter")

REPO_ROOT = Path(__file__).resolve().parents[1]

NUM_VIEWS = 12

# Where the pretrained weights live (README, "Pretrained weights"). geosam2.pt is 615 MB.
HF_REPO = "VAST-AI/GeoSAM2"

# UNASSIGNED_LABELS: 999 is the value mask_aggregation initialises its label
# volume with (utils/inference_utils.py), so it survives on any face no mask
# ever covered -- the reference sample_00 run carries it on 0.7% of faces --
# and counting it as a part inflates every part count by one and paints a
# phantom region. 0 is the background label the exporter paints black.

_UNLABELED_RGBA = np.array([*UNASSIGNED_RGB, 255], dtype=np.uint8)


def _rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

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
    """The stages, each callable on its own.

    Stateless with respect to run parameters: the constructor only resolves
    paths, every knob is an argument. That lets the app hold one instance and
    call it per request.
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

    def ensure_checkpoint(self) -> Path:
        """The checkpoint path, downloaded from Hugging Face first if it is missing.

        Only the default layout is fetched -- a file named as on the hub (geosam2.pt,
        geosam2-bf16.pt). A custom path that does not exist is the caller's mistake, and
        downloading something else under that name would hide it.
        """
        path = self.checkpoint_path
        if path.is_file():
            return path
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        logger.info("[checkpoint] %s missing, downloading %s from %s", path, path.name, HF_REPO)
        try:
            hf_hub_download(repo_id=HF_REPO, filename=path.name, local_dir=str(path.parent))
        except EntryNotFoundError:
            raise FileNotFoundError(
                f"checkpoint not found at {path}, and {HF_REPO} has no file named {path.name}"
            ) from None
        logger.info("[checkpoint] %s ready (%.0f MB)", path, path.stat().st_size / 1e6)
        return path

    # -- stages -------------------------------------------------------------

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
        self.ensure_checkpoint()
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
        clean_fragments: bool = False,
    ) -> str:
        if mask_path is not None and mask_view is None:
            raise ValueError("mask_view is required when mask_path is provided")
        if mask_path is None and not opposite_auto_segmentation:
            # inference.py exits 0 without writing anything in this combination.
            raise ValueError(
                "Nothing to segment: with no mask prompt, opposite auto-segmentation "
                "must stay enabled or GeoSAM2 produces no labels."
            )

        self.ensure_checkpoint()
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
        if clean_fragments:
            command.append("--clean-fragments")
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

        # Speckle cleanup that smoothing cannot do: a stranded patch of another
        # label is locally consistent, so only contact can move it. Unassigned
        # faces stay out of it -- on a partial run they are a third of the mesh
        # in one block every real part would then be measured against -- and
        # stay unassigned: how much lands there is the signal that a prompt set
        # missed an area. repair=False keeps submesh's face order equal to the
        # mask's.
        assigned = ~np.isin(face_label, list(UNASSIGNED_LABELS))
        if assigned.any():
            from utils.inference_utils import clean_label_fragments

            sub = mesh.submesh([assigned], append=True, repair=False)
            face_label = face_label.copy()
            face_label[assigned] = clean_label_fragments(face_label[assigned], sub).numpy()

        labels = np.unique(face_label)
        part_labels = [int(v) for v in labels if int(v) not in UNASSIGNED_LABELS]
        palette = label_palette(face_label)

        scene = trimesh.Scene()
        children: List[Dict[str, Any]] = []

        # Parts are named by label id, not by rank: the raw lift and the
        # post-process share ids, so "part_003" is the same part in both.
        for label in part_labels:
            mask = face_label == label
            rgba = np.array([*palette[label], 255], dtype=np.uint8)
            name = f"part_{label:03d}"
            self._add_part(scene, mesh, mask, rgba, name)
            children.append({"name": name, "color": _rgb_to_hex(rgba), "faces": int(mask.sum())})

        # Every unassigned label collapses into one grey region: they all mean
        # the same thing, and how much of the mesh lands here is the signal --
        # a prompt set that misses whole areas shows up as this growing.
        unassigned = np.isin(face_label, UNASSIGNED_LABELS)
        unlabeled_faces = int(unassigned.sum())
        logger.info("[lift] %d parts over %d faces, %d unassigned (%.1f%%)",
                    len(part_labels), len(mesh.faces), unlabeled_faces,
                    100.0 * unlabeled_faces / max(len(mesh.faces), 1))
        if not part_labels:
            # Every face unassigned: the propagation produced nothing, and the
            # grey "unassigned" blob below would be the only thing shown. Say so
            # here rather than let the caller report "0 parts" as a success.
            logger.warning("[lift] no part survived -- the seed prompt reached no "
                           "face. Automatic mode (no seed) is the usual cause; "
                           "tick VLM seed mask or pass a prompt.")
        if unlabeled_faces:
            self._add_part(scene, mesh, unassigned, _UNLABELED_RGBA, "unassigned")
            children.append({
                "name": "unassigned",
                "color": _rgb_to_hex(_UNLABELED_RGBA),
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
        # keep the original texture instead of its part colour. COLOR_0 is
        # linear in glTF: encode, or the part renders lighter than its texture.
        rgba = np.array([*to_linear_u8(rgba[:3]), 255], dtype=np.uint8)
        part.visual = trimesh.visual.ColorVisuals(
            mesh=part, vertex_colors=np.tile(rgba, (len(part.vertices), 1))
        )
        part.metadata["name"] = name
        scene.add_geometry(part, node_name=name, geom_name=name)

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
