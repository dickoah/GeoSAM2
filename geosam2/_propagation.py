"""The propagation: a seed mask on one view -> masks on the 12 views -> per-face labels.

VAST's inference.py, reduced to the one path GeoSAM2 runs: the seed masks
propagated by the video predictor, the opposite view segmented automatically
for what they left uncovered, the masks lifted onto the faces, then the
post-process.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from geosam2._lift import (
    filter_masks, filter_mask_area, filter_mask_stability, lift_2dmask_3d,
    sample_points_on_faces_parallel, trans2bool, shrink_mask, filter_iou,
    complete_labels,
)
from geosam2.util.views import NUM_VIEWS, Views


SAMPLE_NUM = 5

# The mesh as loaded is Z-up; the cameras in meta.json are Y-up.
_CAMERA_FRAME_ROTATION = np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                   [0, 1, 0, 0], [0, 0, 0, 1]])


@dataclass
class PropagationSettings:
    """GeoSAM2's knobs. The defaults are VAST's values, the ones every result so far used."""

    # The automatic complement: where it looks, and whether it runs at all.
    auto_complement: bool = True        # segment the opposite view for what the seed left uncovered
    opposite_offset: int = 6            # which view: seed view + offset (6 = the one behind)
    # How strict the masks kept after each propagation are.
    seed_stability: float = 0.0         # min stability score of the seed's masks (0 = keep all)
    seed_area_alpha: float = 10000.0    # drop an object whose mask grows > alpha x in another view
    auto_stability: float = 0.7         # same two, for the automatic complement's masks
    auto_area_alpha: float = 1.0
    merge_iou: float = 0.8              # two objects overlapping more than this are one
    shrink_kernel: int = 5              # opening of every mask: kernel (px) ...
    shrink_iters: int = 3               # ... and iterations; erases what is thinner than ~kernel*iters
    # The automatic mask generator on the opposite view.
    points_per_side: int = 64           # a grid of N x N clicks
    pred_iou_thresh: float = 0.7        # min predicted quality of a mask
    stability_score_thresh: float = 0.7
    box_nms_thresh: float = 0.7         # redundancy suppression between masks
    min_mask_region_area: float = 25.0  # masks smaller than this (px) are dropped
    dedup_iou: float = 85.0             # automatic masks overlapping more than this (%) are one
    # The lift onto the faces.
    samples_per_face: int = SAMPLE_NUM  # points per face voting its label
    postprocess: bool = True            # complete_labels after the lift (off: post = raw)

    def generator_overrides(self) -> Dict:
        return dict(points_per_side=self.points_per_side, pred_iou_thresh=self.pred_iou_thresh,
                    stability_score_thresh=self.stability_score_thresh,
                    box_nms_thresh=self.box_nms_thresh, min_mask_region_area=self.min_mask_region_area)


def _point_cloud(views: Views, sample_num: int = SAMPLE_NUM) -> torch.Tensor:
    """The points the lift votes with: ``sample_num`` per face, in face order, in the cameras' frame."""
    mesh = views.mesh.copy()
    mesh.apply_transform(_CAMERA_FRAME_ROTATION)
    mesh.apply_translation(views.translation)
    mesh.apply_scale(views.scaling_factor)
    face_to_vertex = mesh.vertices[mesh.faces]
    points = sample_points_on_faces_parallel(face_to_vertex, num_points=sample_num)
    return torch.from_numpy(points).float().reshape(-1, 3)


def _propagate_from(predictor, inference_state, view: int, all_seg_result: Dict,
                    stability_dict: Dict, stability_thresh: float, area_alpha: float,
                    settings: PropagationSettings) -> Tuple[Dict, Dict]:
    """Propagate what was added on ``view`` to every view and merge the masks that hold up."""
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video_v2(
        inference_state, start_frame_idx=view
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: out_mask_logits[i].cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    video_segments = shrink_mask(video_segments, settings.shrink_kernel, settings.shrink_iters)
    video_segments, stability_dict = filter_mask_stability(
        video_segments, view, stability_dict, stability_score_thresh=stability_thresh)
    video_segments = filter_mask_area(video_segments, view, alpha=area_alpha)
    video_segments = trans2bool(video_segments, view)
    video_segments, all_seg_result = filter_iou(all_seg_result, video_segments, view, settings.merge_iou)

    for frame, masks in video_segments.items():
        all_seg_result[frame].update(masks)
    return all_seg_result, stability_dict


def propagate(predictor, mask_generator, views: Views, postprocess_pa: float,
              settings: PropagationSettings) -> Tuple[torch.Tensor, torch.Tensor]:
    """The pipeline: seed masks propagated over the views, the opposite view
    segmented automatically for what the seed left uncovered, both lifted onto
    the faces, then the post-process.

    Returns the raw and the post-processed per-face labels, indexing
    ``views.mesh_vanilla.faces``.
    """
    point_cloud = _point_cloud(views, settings.samples_per_face)
    inference_state = predictor.init_state(video_path=views.root,
                                           video_id_list=list(range(NUM_VIEWS)))
    all_seg_result: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(NUM_VIEWS)}
    stability_dict = {i: {j: 0 for j in range(900)} for i in range(NUM_VIEWS)}
    start = views.seed_view
    opposite = (start + settings.opposite_offset) % NUM_VIEWS

    # 1. The seed view, from the masks given. Their ids are kept; the
    # automatic masks below get ids strictly above them.
    predictor.reset_state(inference_state)
    next_obj_id = 1
    for obj_id, mask in views.seed_masks.items():
        predictor.add_new_mask(inference_state=inference_state, frame_idx=start,
                               obj_id=int(obj_id), mask=mask)
        next_obj_id = max(next_obj_id, int(obj_id) + 1)
    prior_keys = {int(k) for k in views.seed_masks}
    all_seg_result, stability_dict = _propagate_from(
        predictor, inference_state, start, all_seg_result, stability_dict,
        stability_thresh=settings.seed_stability, area_alpha=settings.seed_area_alpha,
        settings=settings)

    # 2. The opposite view: automatic masks, only where the seed's did not reach.
    if settings.auto_complement:
        uncovered = views.img_masks[opposite].squeeze().copy()
        for mask in all_seg_result[opposite].values():
            uncovered = np.logical_and(uncovered, ~mask.squeeze())
        predictor.reset_state(inference_state)
        anns = filter_masks(mask_generator.generate(
            views.images[opposite], views.pos_maps[opposite], views.norm_maps[opposite], uncovered),
            settings.dedup_iou)
        for ann in anns:
            predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=opposite, obj_id=next_obj_id,
                points=np.array([ann["point_coords"]], dtype=np.float32),
                labels=np.array([1], np.int32))
            next_obj_id += 1
        all_seg_result, _ = _propagate_from(
            predictor, inference_state, opposite, all_seg_result, stability_dict,
            stability_thresh=settings.auto_stability, area_alpha=settings.auto_area_alpha,
            settings=settings)

    # 3. Onto the faces, then the post-process.
    raw = lift_2dmask_3d(
        imgs_mask=torch.stack([torch.from_numpy(m) for m in views.img_masks], dim=0),
        depth_imgs=torch.stack([torch.from_numpy(d).unsqueeze(-1) for d in views.depth_maps], dim=0),
        c2ws=torch.stack(views.c2ws, dim=0),
        fovy_deg=views.fovy_deg,
        coord=point_cloud,
        selected_frames=list(range(NUM_VIEWS)),
        video_segments=all_seg_result,
        sample_num_per_face=settings.samples_per_face,
        prior_keys=prior_keys,
    )
    if not settings.postprocess:
        return raw, raw.clone()
    _, _, post = complete_labels(raw.clone(), views.mesh_vanilla,
                                 smooth_type="adjacent", PA=postprocess_pa)
    return raw, post
