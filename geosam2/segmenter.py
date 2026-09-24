"""GeoSAM2 in one class: a view directory and a 2D mask in, labelled parts out.

Same lifecycle as segvigen's segmenters: the model loads on the first
``run()`` and stays cached, one lock per instance, ``clear_vram()`` in a
``finally``. What VAST's script set for its whole process -- bf16 autocast,
TF32, the seeds -- is scoped to each call, so a host process is left as it was.

    seg = GeoSAM2Segmenter()
    parts_glb = seg.run(views_dir, "mask_0001.png", mask_view=1, output_dir=work)
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import trimesh

from geosam2._lift import clean_label_fragments
from geosam2._propagation import PropagationSettings, propagate
from geosam2.util.labels import UNASSIGNED_LABELS, export_parts
from geosam2.util.logs import get_logger
from geosam2.util.views import NUM_VIEWS, is_view_directory, load_mesh, read_views

logger = get_logger("geosam2.segmenter")

REPO_ROOT = Path(__file__).resolve().parents[1]


class GeoSAM2Segmenter:
    """Views + a 2D mask on one of them -> per-face labels and one mesh per part."""

    HF_REPO = "VAST-AI/GeoSAM2"                       # geosam2.pt is 615 MB
    DEFAULT_CHECKPOINT = REPO_ROOT / "ckpt" / "geosam2.pt"
    POSTPROCESS_PA_CANDIDATES = (0.01, 0.02, 0.035)   # what VAST's README suggests trying

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.DEFAULT_CHECKPOINT
        self._predictor = None
        self._mask_generator = None
        self._lock = threading.Lock()

    # ── the interface ──────────────────────────────────────────────────────────

    def ensure_checkpoint(self) -> Path:
        """The checkpoint path, downloaded from Hugging Face first if it is missing.

        Only a file named as on the hub (geosam2.pt, geosam2-bf16.pt) is fetched: a
        custom path that does not exist is the caller's mistake, and downloading
        something else under that name would hide it.
        """
        path = self.checkpoint_path
        if path.is_file():
            return path
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        logger.info("[checkpoint] %s missing, downloading %s from %s", path, path.name, self.HF_REPO)
        try:
            hf_hub_download(repo_id=self.HF_REPO, filename=path.name, local_dir=str(path.parent))
        except EntryNotFoundError:
            raise FileNotFoundError(
                f"checkpoint not found at {path}, and {self.HF_REPO} has no file named {path.name}"
            ) from None
        logger.info("[checkpoint] %s ready (%.0f MB)", path, path.stat().st_size / 1e6)
        return path

    def load(self) -> None:
        """Load the model, once per checkpoint; instances share the weights."""
        if self._predictor is not None:
            return
        key = str(self.ensure_checkpoint())
        if key not in self._cache:
            self._cache[key] = self._build(key)
        self._predictor, self._mask_generator = self._cache[key]

    def run(
        self,
        views: Union[str, Path],
        mask: Union[str, Path],
        mask_view: int,
        output_dir: Union[str, Path],
        postprocess_pa: float = 0.02,
        settings: Optional[PropagationSettings] = None,
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
            Receives ``labels_raw.npy`` (the lift as GeoSAM2 produces it),
            ``labels_post.npy`` (after its post-process) and the result below.
        postprocess_pa:
            The post-process's ``PA``; see ``POSTPROCESS_PA_CANDIDATES``.
        settings:
            GeoSAM2's other knobs (``PropagationSettings``); None = VAST's values.

        Returns
        -------
        str
            ``<output_dir>/parts.glb``: one geometry per label on the input mesh,
            in its own frame, after the fragment cleanup. The labels it was built
            from are next to it as ``parts.npy``.
        """
        views, mask, output_dir = Path(views), Path(mask), Path(output_dir)
        if not is_view_directory(views):
            raise FileNotFoundError(f"not a complete view directory: {views}")
        if not mask.is_file():
            raise FileNotFoundError(f"mask not found: {mask}")
        if not 0 <= int(mask_view) < NUM_VIEWS:
            raise ValueError(f"mask_view must be in 0..{NUM_VIEWS - 1}, got {mask_view}")
        output_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self.load()
            device = self._device()
            try:
                with self._scoped(device):
                    self._to(device)
                    logger.info("[run] %s, seed %s on view %d", views.name, mask.name, mask_view)
                    data = read_views(str(views), str(mask), int(mask_view))
                    settings = settings or PropagationSettings()
                    raw, post = propagate(self._predictor, self._generator_for(settings), data,
                                          postprocess_pa, settings)
            finally:
                self.clear_vram()

        np.save(output_dir / "labels_raw.npy", raw.cpu().numpy())
        np.save(output_dir / "labels_post.npy", post.cpu().numpy())
        scene, structure, labels = self.parts(views / "mesh.glb", post.cpu().numpy())
        glb_path = output_dir / "parts.glb"
        scene.export(glb_path)
        np.save(output_dir / "parts.npy", labels)
        logger.info("[run] %d parts -> %s", len(structure["children"]), glb_path)
        return str(glb_path)

    def parts(
        self, mesh_path: Union[str, Path], labels: np.ndarray
    ) -> Tuple[trimesh.Scene, Dict[str, Any], np.ndarray]:
        """Per-face labels -> the fragment cleanup, then one geometry per part.

        Returns the scene, its structure (``{"name", "children": [{"name",
        "color", "faces"}]}``) and the cleaned labels the scene was built from.
        """
        labels = np.asarray(labels).reshape(-1)
        # Speckle cleanup that smoothing cannot do: a stranded patch of another
        # label is locally consistent, so only contact can move it. Unassigned
        # faces stay out of it -- on a partial run they are a third of the mesh
        # in one block every real part would then be measured against -- and
        # stay unassigned: how much lands there is the signal that a prompt set
        # missed an area. repair=False keeps submesh's face order equal to the
        # mask's.
        assigned = ~np.isin(labels, list(UNASSIGNED_LABELS))
        if assigned.any():
            mesh = load_mesh(mesh_path)
            if len(labels) != len(mesh.faces):
                raise RuntimeError(f"label/mesh mismatch: {len(labels)} labels for {len(mesh.faces)} faces")
            # geometry only: submesh would concatenate the texture per call
            sub = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).submesh(
                [assigned], append=True, repair=False)
            labels = labels.copy()
            labels[assigned] = clean_label_fragments(labels[assigned], sub).numpy()
        scene, structure = export_parts(mesh_path, labels)
        return scene, structure, labels

    def clear_vram(self) -> None:
        """Move the weights to the CPU and flush the CUDA cache."""
        self._to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── the private part ───────────────────────────────────────────────────────

    _SEED = 3   # VAST's script seeded numpy and torch with it once per process

    _MASK_GENERATOR = dict(   # the automatic masks on the opposite view, as VAST configured them
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

    # Weights per checkpoint, shared between instances, as segvigen does.
    _cache: Dict[str, Tuple[Any, Any]] = {}

    def _build(self, checkpoint: str) -> Tuple[Any, Any]:
        """The video predictor and the automatic mask generator, on the CPU."""
        from geosam2._model import image_model, video_predictor
        from geosam2.sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator

        logger.info("[model] loading %s", checkpoint)
        with self._scoped(self._device()):
            # Built in this order, under these seeds: it is what VAST's script did.
            sam2 = image_model(checkpoint)
            predictor = video_predictor(checkpoint)
        return predictor, SAM2AutomaticMaskGenerator(model=sam2, **self._MASK_GENERATOR)

    def _generator_for(self, settings: PropagationSettings):
        """The cached generator, or one built on the same model when its settings differ."""
        overrides = settings.generator_overrides()
        if all(self._MASK_GENERATOR[k] == v for k, v in overrides.items()):
            return self._mask_generator
        from geosam2.sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator
        return SAM2AutomaticMaskGenerator(model=self._mask_generator.predictor.model,
                                          **{**self._MASK_GENERATOR, **overrides})

    def _to(self, device: torch.device) -> None:
        if self._predictor is not None:
            self._predictor.to(device)
            self._mask_generator.predictor.model.to(device)

    @contextlib.contextmanager
    def _scoped(self, device: torch.device):
        """What VAST's ``init_env()`` made process-wide, for the duration of one call.

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
            np.random.seed(self._SEED)
            torch.manual_seed(self._SEED)
            with (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
                  else contextlib.nullcontext()):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

    @staticmethod
    def _device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
