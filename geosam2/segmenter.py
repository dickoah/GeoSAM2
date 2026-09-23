"""GeoSAM2 in one class: views and a 2D mask in, per-face labels and parts out.

Same lifecycle as segvigen's segmenters: the model loads on the first
``run()`` and stays cached, one lock per instance, ``clear_vram()`` in a
``finally``. What ``inference.py`` used to set for its whole process --
bf16 autocast, TF32, the seeds -- is scoped to each call here, so a host
process is left exactly as it was.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh

from geosam2.util.bake import UNASSIGNED_LABELS, UNASSIGNED_RGB, label_palette, to_linear_u8
from geosam2.util.logs import get_logger

logger = get_logger("geosam2.segmenter")

REPO_ROOT = Path(__file__).resolve().parents[1]

NUM_VIEWS = 12

# Where the pretrained weights live (README, "Pretrained weights"). geosam2.pt is 615 MB.
HF_REPO = "VAST-AI/GeoSAM2"
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpt" / "geosam2.pt"

# The values VAST's README suggests trying for the post-process; 0.02 is its default.
POSTPROCESS_PA_CANDIDATES = (0.01, 0.02, 0.035)

# The seed every run starts from, as inference.py's init_env() did once per process.
_SEED = 3

# UNASSIGNED_LABELS: 999 is the value mask_aggregation initialises its label
# volume with (_lift.py), so it survives on any face no mask ever covered --
# the reference sample_00 run carries it on 0.7% of faces -- and counting it
# as a part inflates every part count by one and paints a phantom region. 0 is
# the background label the exporter paints black.

_UNLABELED_RGBA = np.array([*UNASSIGNED_RGB, 255], dtype=np.uint8)

# Weights are cached per checkpoint and shared between instances, as segvigen does.
_loaded_models: Dict[str, Tuple[Any, Any]] = {}


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


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def _scoped_torch(device: torch.device):
    """What init_env() made global, for the duration of one call.

    The seeds are reset every time: a fresh subprocess always started from
    them, and the automatic mask generator and the face sampling draw from
    numpy's global generator.
    """
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    try:
        if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        np.random.seed(_SEED)
        torch.manual_seed(_SEED)
        with (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
              else contextlib.nullcontext()):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


class GeoSAM2Segmenter:
    """Views + a 2D mask on one of them -> per-face labels and one mesh per part.

    Parameters
    ----------
    checkpoint_path:
        ``geosam2.pt``; downloaded from Hugging Face when missing.
    """

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self._predictor = None
        self._mask_generator = None
        self._lock = threading.Lock()

    # -- weights -------------------------------------------------------------

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

    def load(self) -> None:
        """Build the video predictor and the automatic mask generator, once per checkpoint."""
        if self._predictor is not None:
            return
        key = str(self.ensure_checkpoint())
        if key not in _loaded_models:
            from geosam2._model import build_sam2, build_sam2_video_predictor_geosam2
            from geosam2.sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator

            logger.info("[model] loading %s", key)
            with _scoped_torch(_device()):
                # Built in this order, under these seeds: it is what inference.py did.
                sam2 = build_sam2(None, key, device="cpu", apply_postprocessing=False)
                predictor = build_sam2_video_predictor_geosam2(None, key, device="cpu")
            mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=64,
                points_per_batch=128,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.7,
                stability_score_offset=0.7,
                crop_n_layers=0,
                box_nms_thresh=0.7,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=25.0,
                use_m2m=True,
            )
            _loaded_models[key] = (predictor, mask_generator)
        self._predictor, self._mask_generator = _loaded_models[key]

    def clear_vram(self) -> None:
        """Move the weights to the CPU and flush the CUDA cache."""
        if self._predictor is not None:
            self._predictor.cpu()
            self._mask_generator.predictor.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- the run ---------------------------------------------------------------

    def run(
        self,
        views: Union[str, Path],
        mask: Union[str, Path],
        mask_view: int,
        output_dir: Union[str, Path],
        postprocess_pa: float = 0.02,
    ) -> str:
        """Propagate ``mask`` from ``mask_view`` over the 12 views and lift it to the mesh.

        Parameters
        ----------
        views:
            A view directory: the 12 canonical renders, ``meta.json`` and ``mesh.glb``.
        mask:
            The 2D seed on ``mask_view``: a label map (``.npy``, ``.exr``) or a
            flat-colour ``.png`` such as the painted guidance map.
        mask_view:
            Index of the view the mask was drawn on.
        output_dir:
            Where GeoSAM2's own exports go (``segmentation_result_*`` is the raw
            lift, ``segmentation_postprocessed_*`` the post-processed one, each as
            ``.npy`` labels and a coloured ``.glb``), plus the result below.
        postprocess_pa:
            The post-process's ``PA``; see ``POSTPROCESS_PA_CANDIDATES``.

        Returns
        -------
        str
            ``<output_dir>/parts.glb``: one geometry per label on the input mesh,
            in its own frame, after ``clean_label_fragments``. The labels it was
            built from are next to it as ``parts.npy``.
        """
        views = Path(views)
        mask = Path(mask)
        output_dir = Path(output_dir)
        if not is_view_directory(views):
            raise FileNotFoundError(f"not a complete view directory: {views}")
        if not mask.is_file():
            raise FileNotFoundError(f"mask not found: {mask}")
        if not 0 <= int(mask_view) < NUM_VIEWS:
            raise ValueError(f"mask_view must be in 0..{NUM_VIEWS - 1}, got {mask_view}")
        output_dir.mkdir(parents=True, exist_ok=True)

        from geosam2._propagation import read_data, segment_with_mask_prompts

        with self._lock:
            self.load()
            device = _device()
            try:
                with _scoped_torch(device):
                    self._predictor.to(device)
                    self._mask_generator.predictor.model.to(device)
                    logger.info("[run] %s, seed %s on view %d", views.name, mask.name, mask_view)
                    data = read_data(str(views), mask_path=str(mask), mask_view=int(mask_view))
                    start = int(mask_view)
                    seeds = [start] + ([(start + 6) % NUM_VIEWS] if (start + 6) % NUM_VIEWS != start else [])
                    results = segment_with_mask_prompts(
                        predictor=self._predictor,
                        mask_generator=self._mask_generator,
                        data=data,
                        opposite_auto_segmentation=True,
                        enable_postprocess=True,
                        postprocess_pa=postprocess_pa,
                        output_dir=str(output_dir),
                        start_frames=[start],
                        start_to_seed_views={start: seeds},
                    )
            finally:
                self.clear_vram()

        face_label = results["face_label"]
        if face_label is None:
            raise RuntimeError("GeoSAM2 produced no labels: the seed reached no face")
        labels = face_label.cpu().numpy() if torch.is_tensor(face_label) else np.asarray(face_label)
        scene, structure, labels = export_parts(views / "mesh.glb", labels)
        glb_path = output_dir / "parts.glb"
        scene.export(glb_path)
        np.save(output_dir / "parts.npy", labels)
        logger.info("[run] %d parts -> %s", len(structure["children"]), glb_path)
        return str(glb_path)


# -- labels onto the mesh --------------------------------------------------------

def export_parts(
    mesh_path: Union[str, Path], face_label: np.ndarray
) -> Tuple[trimesh.Scene, Dict[str, Any], np.ndarray]:
    """Paint per-face labels onto the original mesh, one geometry per part.

    The GLB that the propagation exports is rebuilt in a rotated/translated/scaled
    frame, so it cannot be shown next to the input. The label array, by contrast,
    indexes the input mesh's faces directly -- so the source mesh is reloaded and
    coloured instead, keeping both viewers aligned.

    Returns the scene, its structure (``{"name", "children": [{"name", "color",
    "faces"}]}``) and the labels after the fragment cleanup, which the scene was
    built from.
    """
    mesh_path = Path(mesh_path)
    # Must match load_mesh_with_faces() in _lift.py, which loads with
    # force="mesh" and trimesh's default processing, or the face ordering the
    # labels refer to would not be ours.
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
        from geosam2._lift import clean_label_fragments

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
        part_mask = face_label == label
        rgba = np.array([*palette[label], 255], dtype=np.uint8)
        name = f"part_{label:03d}"
        _add_part(scene, mesh, part_mask, rgba, name)
        children.append({"name": name, "color": _rgb_to_hex(rgba), "faces": int(part_mask.sum())})

    # Every unassigned label collapses into one grey region: they all mean
    # the same thing, and how much of the mesh lands here is the signal --
    # a prompt set that misses whole areas shows up as this growing.
    unassigned = np.isin(face_label, UNASSIGNED_LABELS)
    unlabeled_faces = int(unassigned.sum())
    logger.info("[lift] %d parts over %d faces, %d unassigned (%.1f%%)",
                len(part_labels), len(mesh.faces), unlabeled_faces,
                100.0 * unlabeled_faces / max(len(mesh.faces), 1))
    if not part_labels:
        logger.warning("[lift] no part survived -- the seed prompt reached no face.")
    if unlabeled_faces:
        _add_part(scene, mesh, unassigned, _UNLABELED_RGBA, "unassigned")
        children.append({
            "name": "unassigned",
            "color": _rgb_to_hex(_UNLABELED_RGBA),
            "faces": unlabeled_faces,
        })

    structure = {"name": mesh_path.stem, "children": children}
    return scene, structure, face_label


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
