"""
Simplified inference script for 3D mesh segmentation using SAM2.
Uses mask prompts for interactive segmentation.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import math
from typing import List, Dict, Optional, Tuple
import trimesh

from sam2.build_sam import build_sam2, build_sam2_video_predictor_geosam2
from sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator
from utils.inference_utils import (
    show_anns, show_mask, filter_mask_area, filter_mask_stability,
    lift_2dmask_3d, load_mesh_with_faces, sample_points_on_faces_parallel,
    trans2bool, shrink_mask, filter_iou, complete_labels, gen_pcd,
    smooth_labels_icm, smooth_labels_graphcut, face_vote_histogram,
    export_labelled_mesh
)


SAMPLE_NUM = 5
NUM_VIEWS = 12
MASK_MIN_AREA_PX = 64
MASK_COLOR_QUANT_STEP = 8


def _to_int_label_map(mask_like: np.ndarray) -> np.ndarray:
    if mask_like.ndim != 2:
        raise ValueError(f"Mask must be HxW, got shape: {mask_like.shape}")
    if np.issubdtype(mask_like.dtype, np.floating):
        return np.rint(mask_like).astype(np.int32)
    return mask_like.astype(np.int32)


def _encode_color(rgb: np.ndarray) -> int:
    return (int(rgb[0]) << 16) + (int(rgb[1]) << 8) + int(rgb[2])


def _quantize_rgb(rgb: np.ndarray, step: int) -> np.ndarray:
    """Quantize RGB to reduce tiny color variation from anti-aliasing/compression."""
    if step <= 1:
        return rgb
    q = (rgb // step) * step
    return q.astype(np.uint8)


def extract_mask_segments(mask_path: str) -> List[Tuple[Tuple[str, int], np.ndarray]]:
    """Extract mask segments with stable keys from label maps or color previews."""
    ext = os.path.splitext(mask_path)[1].lower()

    if ext == ".exr":
        raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Failed to read mask file: {mask_path}")
        label_map = _to_int_label_map(raw[..., 0] if raw.ndim == 3 else raw)
        unique_ids = np.unique(label_map)
        unique_ids = unique_ids[unique_ids != 0]
        segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
        for obj_id in unique_ids:
            m = label_map == int(obj_id)
            if int(m.sum()) < MASK_MIN_AREA_PX:
                continue
            segments.append((("id", int(obj_id)), m))
        return segments
    elif ext == ".npy":
        label_map = _to_int_label_map(np.load(mask_path))
        unique_ids = np.unique(label_map)
        unique_ids = unique_ids[unique_ids != 0]
        segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
        for obj_id in unique_ids:
            m = label_map == int(obj_id)
            if int(m.sum()) < MASK_MIN_AREA_PX:
                continue
            segments.append((("id", int(obj_id)), m))
        return segments
    elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        mask_image = np.array(Image.open(mask_path))

        if mask_image.ndim == 2:
            label_map = _to_int_label_map(mask_image)
            unique_ids = np.unique(label_map)
            unique_ids = unique_ids[unique_ids != 0]
            segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
            for obj_id in unique_ids:
                m = label_map == int(obj_id)
                if int(m.sum()) < MASK_MIN_AREA_PX:
                    continue
                segments.append((("id", int(obj_id)), m))
            return segments

        if mask_image.ndim == 3 and mask_image.shape[2] >= 3:
            rgb = mask_image[..., :3].astype(np.uint8)
            # Treat transparent pixels as background.
            if mask_image.shape[2] == 4:
                rgb[mask_image[..., 3] == 0] = 0

            # Reduce color noise from interpolation/compression.
            rgb = _quantize_rgb(rgb, MASK_COLOR_QUANT_STEP)

            # If RGB channels are identical, treat it as a numeric label map.
            if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
                label_map = _to_int_label_map(rgb[..., 0])
                unique_ids = np.unique(label_map)
                unique_ids = unique_ids[unique_ids != 0]
                segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
                for obj_id in unique_ids:
                    m = label_map == int(obj_id)
                    if int(m.sum()) < MASK_MIN_AREA_PX:
                        continue
                    segments.append((("id", int(obj_id)), m))
                return segments

            flat_rgb = rgb.reshape(-1, 3)
            unique_colors, counts = np.unique(flat_rgb, axis=0, return_counts=True)
            segments: List[Tuple[Tuple[str, int], np.ndarray]] = []
            if unique_colors.shape[0] == 0:
                return segments

            # Assume dominant color is background for color preview masks.
            bg_color = unique_colors[int(np.argmax(counts))]
            for color in unique_colors:
                if np.array_equal(color, np.array([0, 0, 0], dtype=np.uint8)):
                    continue
                if np.array_equal(color, bg_color):
                    continue
                color_key = _encode_color(color)
                color_mask = np.all(rgb == color, axis=-1)
                if int(color_mask.sum()) < MASK_MIN_AREA_PX:
                    continue
                segments.append((("color", color_key), color_mask))
            return segments

        raise ValueError(f"Unsupported mask image shape: {mask_image.shape}")
    else:
        raise ValueError(
            f"Unsupported mask format: {ext}. Supported: .exr, .npy, image formats"
        )

def add_mask_file_to_frame(
    frame_segments: Dict[int, np.ndarray],
    key_to_objid: Dict[Tuple[str, int], int],
    mask_path: str,
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[str, int], int]]:
    """Add one mask file into a frame with stable ID mapping for repeated keys."""
    segments = extract_mask_segments(mask_path)
    segments = sorted(segments, key=lambda x: x[0])

    next_obj_id = max(frame_segments.keys(), default=0) + 1
    for seg_key, seg_mask in segments:
        seg_mask = seg_mask.astype(bool)
        if not seg_mask.any():
            continue
        if seg_key in key_to_objid:
            obj_id = key_to_objid[seg_key]
            if obj_id in frame_segments:
                frame_segments[obj_id] = np.logical_or(frame_segments[obj_id], seg_mask)
            else:
                frame_segments[obj_id] = seg_mask
            continue

        while next_obj_id in frame_segments:
            next_obj_id += 1
        frame_segments[next_obj_id] = seg_mask
        key_to_objid[seg_key] = next_obj_id
        next_obj_id += 1

    return frame_segments, key_to_objid


def save_frame_segmentation_visualizations(
    images: List[np.ndarray],
    all_seg_result: Dict[int, Dict[int, np.ndarray]],
    save_dir: str,
    frame_tag: str = "all"
) -> None:
    """Save per-frame segmentation overlays for quick inspection."""
    os.makedirs(save_dir, exist_ok=True)

    for frame_idx, image in enumerate(images):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(image)
        ax.set_title(f"frame {frame_idx}")
        ax.axis("off")

        frame_masks = all_seg_result.get(frame_idx, {})
        for obj_id, mask in frame_masks.items():
            show_mask(mask.squeeze(), ax, obj_id=obj_id)

        out_path = os.path.join(save_dir, f"frame_{str(frame_idx).zfill(4)}_{frame_tag}.png")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def save_input_maps_visualizations(
    images: List[np.ndarray],
    pos_maps: List[torch.Tensor],
    img_masks: List[np.ndarray],
    save_dir: str,
) -> None:
    """Save RGB / position-map visualizations per frame."""
    os.makedirs(save_dir, exist_ok=True)

    for frame_idx, rgb in enumerate(images):
        pos_map = pos_maps[frame_idx].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        valid_mask = img_masks[frame_idx].squeeze().astype(bool)

        pos_vis = np.zeros_like(pos_map, dtype=np.float32)
        for c in range(3):
            channel = pos_map[..., c]
            if valid_mask.any():
                vals = channel[valid_mask]
            else:
                vals = channel.reshape(-1)
            c_min, c_max = vals.min(), vals.max()
            if c_max > c_min:
                pos_vis[..., c] = (channel - c_min) / (c_max - c_min)
            else:
                pos_vis[..., c] = 0.0
        pos_vis = (np.clip(pos_vis, 0.0, 1.0) * 255).astype(np.uint8)

        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(save_dir, f"frame_{frame_idx:04d}_rgb.png"))
        Image.fromarray(pos_vis).save(os.path.join(save_dir, f"frame_{frame_idx:04d}_pos.png"))

        panel = np.concatenate([rgb.astype(np.uint8), pos_vis], axis=1)
        Image.fromarray(panel).save(os.path.join(save_dir, f"frame_{frame_idx:04d}_panel.png"))


def save_labeled_point_cloud(
    point_cloud: torch.Tensor,
    face_label: torch.Tensor,
    sample_num_per_face: int,
    save_path: str,
) -> None:
    """Export sampled point cloud with per-point colors from face labels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    points_np = point_cloud.detach().cpu().numpy()  # [face_num * sample_num_per_face, 3]
    face_label_np = face_label.detach().cpu().numpy().astype(np.int32).reshape(-1)  # [face_num]

    point_labels = np.repeat(face_label_np, sample_num_per_face)
    if point_labels.shape[0] != points_np.shape[0]:
        min_len = min(point_labels.shape[0], points_np.shape[0])
        point_labels = point_labels[:min_len]
        points_np = points_np[:min_len]

    rng = np.random.default_rng(3)
    unique_labels = np.unique(point_labels)
    color_map = {int(lbl): rng.integers(0, 256, size=3, dtype=np.uint8) for lbl in unique_labels}
    if 0 in color_map:
        color_map[0] = np.array([0, 0, 0], dtype=np.uint8)

    colors = np.stack([color_map[int(lbl)] for lbl in point_labels], axis=0)
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    rgba = np.concatenate([colors, alpha], axis=1)

    pc = trimesh.points.PointCloud(vertices=points_np, colors=rgba)
    pc.export(save_path)


def save_labeled_point_cloud_from_pos_maps(
    pos_maps: List[torch.Tensor],
    all_seg_result: Dict[int, Dict[int, np.ndarray]],
    img_masks: List[np.ndarray],
    save_dir: str,
    tag: str,
) -> None:
    """Export point clouds directly from per-pixel pos_map XYZ and 2D mask labels."""
    os.makedirs(save_dir, exist_ok=True)

    rng = np.random.default_rng(3)
    all_labels = set([0])
    for frame_dict in all_seg_result.values():
        all_labels.update([int(k) for k in frame_dict.keys()])
    color_map = {lbl: rng.integers(0, 256, size=3, dtype=np.uint8) for lbl in sorted(all_labels)}
    color_map[0] = np.array([0, 0, 0], dtype=np.uint8)

    all_points = []
    all_colors = []

    for frame_idx in range(len(pos_maps)):
        pos = pos_maps[frame_idx].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        valid = img_masks[frame_idx].squeeze().astype(bool)

        label_map = np.zeros(valid.shape, dtype=np.int32)
        for obj_id, mask in all_seg_result.get(frame_idx, {}).items():
            m = np.array(mask)
            if m.ndim == 3:
                m = m.squeeze(0)
            m = m.astype(bool)
            label_map[m] = int(obj_id)

        keep = valid & np.isfinite(pos).all(axis=-1)
        pts = pos[keep]
        lbl = label_map[keep]
        cols = np.stack([color_map[int(x)] for x in lbl], axis=0)

        if pts.shape[0] == 0:
            continue

        all_points.append(pts)
        all_colors.append(cols)

        rgba = np.concatenate([cols, np.full((cols.shape[0], 1), 255, dtype=np.uint8)], axis=1)
        frame_pc = trimesh.points.PointCloud(vertices=pts, colors=rgba)
        frame_pc.export(os.path.join(save_dir, f"pc_pos_frame_{frame_idx:04d}_{tag}.ply"))

    if len(all_points) > 0:
        pts_all = np.concatenate(all_points, axis=0)
        cols_all = np.concatenate(all_colors, axis=0)
        rgba_all = np.concatenate([cols_all, np.full((cols_all.shape[0], 1), 255, dtype=np.uint8)], axis=1)
        pc_all = trimesh.points.PointCloud(vertices=pts_all, colors=rgba_all)
        pc_all.export(os.path.join(save_dir, f"pc_pos_all_{tag}.ply"))


def init_env() -> torch.device:
    """Initialize compute environment and set random seeds."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS."
        )
    
    np.random.seed(3)
    torch.manual_seed(3)
    return device


def read_data(
    data_root: str,
    idx_list: Optional[List[int]] = None,
    mask_path: Optional[str] = None,
    mask_view: Optional[int] = None,
) -> Dict:
    """
    Read rendered views and metadata from a data directory.
    
    Args:
        data_root: Path to the data directory containing renders and meta.json
        idx_list: List of frame indices to load
        
    Returns:
        Dictionary containing all loaded data
    """
    if idx_list is None:
        idx_list = list(range(NUM_VIEWS))

    obj_name = data_root.split("/")[-1]
    meta_path = os.path.join(data_root, "meta.json")
    meta = json.load(open(meta_path))
    camera_angle_x = meta["camera_angle_x"]

    all_images = []
    all_image_masks = []
    all_depth_maps = []
    all_pos_maps = []
    all_norm_maps = []
    all_prompt_masks = {}  # Prompt masks per frame
    frame_key_to_objid: Dict[int, Dict[Tuple[str, int], int]] = {}
    c2ws = []

    for frame_idx, idx in enumerate(idx_list):
        img_path = os.path.join(data_root, f"color_{str(idx).zfill(4)}.webp")
        img = Image.open(img_path)
        img_mask = np.array(img)[:, :, -1:] > 0  # Extract alpha mask

        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, img)
        image = np.array(composite.convert('RGB'))

        depth_path = os.path.join(data_root, f"depth_{str(idx).zfill(4)}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth[..., 0]

        c2w = np.array(meta["transforms"][idx])
        pos_map = gen_pcd(depth, c2w, camera_angle_x)
        pos_map = torch.from_numpy(pos_map).to(torch.float32).permute(2, 0, 1)

        norm_path = os.path.join(data_root, f"normal_{str(idx).zfill(4)}.webp")
        norm = Image.open(norm_path)
        background = Image.new('RGBA', norm.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, norm)
        norm_map = np.array(composite.convert('RGB'))

        all_images.append(image)
        all_image_masks.append(img_mask)
        all_depth_maps.append(depth)
        all_pos_maps.append(pos_map)
        all_norm_maps.append(norm_map)
        c2ws.append(torch.tensor(c2w, dtype=torch.float32))

    if mask_path is not None:
        if mask_view is None:
            raise ValueError("mask_view must be provided when mask_path is set")
        if mask_view not in idx_list:
            raise ValueError(f"mask_view={mask_view} not found in idx_list={idx_list}")

        frame_idx = idx_list.index(mask_view)
        frame_segments = all_prompt_masks.get(frame_idx, {})
        key_to_objid = frame_key_to_objid.get(frame_idx, {})
        frame_segments, key_to_objid = add_mask_file_to_frame(
            frame_segments,
            key_to_objid,
            mask_path,
        )
        all_prompt_masks[frame_idx] = frame_segments
        frame_key_to_objid[frame_idx] = key_to_objid

    if len(all_images) != len(idx_list):
        raise RuntimeError(
            f"Failed to load all views: loaded {len(all_images)} of {len(idx_list)} frames"
        )

    mesh, mesh_vanilla = load_mesh_with_faces(os.path.join(data_root, "mesh.glb"))

    data_ret = {
        "images": all_images,
        "img_masks": all_image_masks,
        "depth_maps": all_depth_maps,
        "pos_maps": all_pos_maps,
        "norm_maps": all_norm_maps,
        "prompt_masks": all_prompt_masks,  # Prompt masks
        "gt_masks": all_prompt_masks,  # Backward compatibility alias
        "c2ws": c2ws,
        "fovy_deg": meta['camera_angle_x'] * 180. / math.pi,
        "scaling_factor": meta["scaling_factor"],
        "translation": torch.Tensor(meta["translation"]),
        "obj_name": obj_name,
        "mesh": mesh,
        "mesh_vanilla": mesh_vanilla,
        "data_root": data_root
    }
    return data_ret


def prepare_mesh_and_point_cloud(mesh: trimesh.Trimesh, 
                                  scaling_factor: float,
                                  translation: np.ndarray,
                                  sample_num: int = SAMPLE_NUM) -> Tuple[trimesh.Trimesh, torch.Tensor]:
    """
    Transform mesh and generate point cloud for lifting 2D masks to 3D.
    
    Args:
        mesh: Input mesh
        scaling_factor: Scaling factor from metadata
        translation: Translation offset from metadata
        sample_num: Number of points to sample per face
        
    Returns:
        Tuple of (transformed_mesh, point_cloud)
    """
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    mesh.apply_transform(rotation_matrix)
    mesh.apply_translation(translation)
    mesh.apply_scale(scaling_factor)

    face_to_vertex = mesh.vertices[mesh.faces]
    object_org_coord = sample_points_on_faces_parallel(face_to_vertex, num_points=sample_num)
    point_cloud = torch.from_numpy(object_org_coord).float().reshape(-1, 3)
    
    return mesh, point_cloud


def segment_with_mask_prompts(
    predictor,
    mask_generator,
    data: Dict,
    opposite_auto_segmentation: bool = True,
    enable_postprocess: bool = False,
    postprocess_pa: float = 0.02,
    output_dir: str = "exp/inference_results",
    save_frame_vis: bool = False,
    save_pointcloud_vis: bool = False,
    start_frames: Optional[List[int]] = None,
    start_to_seed_views: Optional[Dict[int, List[int]]] = None,
    extra_smoothing: Optional[List[str]] = None,
    icm_theta: float = 15.0,
    icm_lambda: float = 0.7,
    icm_iters: int = 30,
    gc_lam: float = 1.0,
) -> Dict:
    """
    Perform segmentation using mask prompts or automatic mask generation.
    
    Args:
        predictor: SAM2 video predictor
        mask_generator: SAM2 automatic mask generator
        data: Data dictionary from read_data()
        opposite_auto_segmentation: Whether to run automatic segmentation on the opposite seed view
        enable_postprocess: Whether to run optional complete-label postprocess
        postprocess_pa: PA value passed to `complete_labels`
        output_dir: Directory to save results
        
    Returns:
        Dictionary with segmentation results
    """
    all_images = data["images"]
    all_img_masks = data["img_masks"]
    all_pos_maps = data["pos_maps"]
    all_depth_maps = data["depth_maps"]
    all_norm_maps = data["norm_maps"]
    all_prompt_masks = data.get("prompt_masks", data.get("gt_masks", {}))
    c2ws = data["c2ws"]
    fovy_deg = data["fovy_deg"]
    scaling_factor = data["scaling_factor"]
    translation = data["translation"]
    mesh = data["mesh"]
    mesh_vanilla = data["mesh_vanilla"]
    obj_name = data["obj_name"]
    video_dir = data["data_root"]

    mesh, point_cloud = prepare_mesh_and_point_cloud(
        mesh, scaling_factor.item() if torch.is_tensor(scaling_factor) else scaling_factor,
        translation.numpy() if torch.is_tensor(translation) else translation
    )
    view_indices = list(range(NUM_VIEWS))
    inference_state = predictor.init_state(video_path=video_dir, video_id_list=view_indices)

    all_seg_result = {i: {} for i in range(NUM_VIEWS)}
    all_obj_num = 1
    stability_dict = {i: {j: 0 for j in range(900)} for i in range(NUM_VIEWS)}
    prior_keys = set()
    face_label = None

    if start_frames is None:
        start_frames = [0]
    if start_to_seed_views is None:
        start_to_seed_views = {0: [0, (0 + 6) % NUM_VIEWS]}

    for start in start_frames:
        seed_views = list(dict.fromkeys(start_to_seed_views.get(start, [start])))
        for idx_ in seed_views:
            is_prompt_seed = idx_ == start
            run_tag = f"promptView{start:02d}" if is_prompt_seed else f"autoView{idx_:02d}_fromPrompt{start:02d}"
            image = all_images[idx_]
            img_mask = all_img_masks[idx_].squeeze()
            pos_map = all_pos_maps[idx_]
            norm_map = all_norm_maps[idx_]
            has_prompt_masks = idx_ in all_prompt_masks and len(all_prompt_masks[idx_]) > 0

            accelerate_mask = img_mask.copy()
            if idx_ != start:
                for key in all_seg_result[idx_].keys():
                    accelerate_mask = np.logical_and(accelerate_mask, ~all_seg_result[idx_][key].squeeze())

            sorted_anns = None
            predictor.reset_state(inference_state)

            if is_prompt_seed:
                if has_prompt_masks:
                    prompt_masks_frame = all_prompt_masks[idx_]
                    for obj_id, mask in prompt_masks_frame.items():
                        obj_id_int = int(obj_id)
                        prior_keys.add(obj_id_int)
                        predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=idx_,
                            obj_id=obj_id_int,
                            mask=mask,
                        )
                        # Keep auto-seg IDs strictly above prompt IDs to avoid collisions.
                        all_obj_num = max(all_obj_num, obj_id_int + 1)
                else:
                    if opposite_auto_segmentation:
                        pred_masks = mask_generator.generate(image, pos_map, norm_map, accelerate_mask)
                        sorted_anns = show_anns(pred_masks)

                        for idx, ann in enumerate(sorted_anns):
                            predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=idx_,
                                obj_id=all_obj_num,
                                points=np.array([ann["point_coords"]], dtype=np.float32),
                                labels=np.array([1], np.int32),
                            )
                            all_obj_num += 1
            else:
                if opposite_auto_segmentation:
                    pred_masks = mask_generator.generate(image, pos_map, norm_map, accelerate_mask)
                    sorted_anns = show_anns(pred_masks)

                    for idx, ann in enumerate(sorted_anns):
                        predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=idx_,
                            obj_id=all_obj_num,
                            points=np.array([ann["point_coords"]], dtype=np.float32),
                            labels=np.array([1], np.int32),
                        )
                        all_obj_num += 1

            if sorted_anns is not None or (is_prompt_seed and has_prompt_masks):
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video_v2(
                    inference_state, start_frame_idx=idx_
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i].cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                video_segments = shrink_mask(video_segments)
                prompt_seed_result = is_prompt_seed and has_prompt_masks
                stability_thresh = 0 if prompt_seed_result else 0.7
                area_alpha = 10000 if prompt_seed_result else 1
                video_segments, stability_dict = filter_mask_stability(video_segments, idx_, stability_dict, stability_score_thresh=stability_thresh)
                video_segments = filter_mask_area(video_segments, idx_, alpha=area_alpha)
                video_segments = trans2bool(video_segments, idx_)
                video_segments, all_seg_result = filter_iou(all_seg_result, video_segments, idx_)

                for key in video_segments.keys():
                    for key_ in video_segments[key].keys():
                        all_seg_result[key][key_] = video_segments[key][key_]

                save_this_result = (not opposite_auto_segmentation) or (not is_prompt_seed)
                if not save_this_result:
                    continue

                face_label = lift_2dmask_3d(
                    torch.stack([torch.from_numpy(m) for m in all_img_masks], dim=0),
                    torch.stack([torch.from_numpy(d).unsqueeze(-1) for d in all_depth_maps], dim=0),
                    torch.from_numpy(np.stack(all_norm_maps, axis=0)),
                    torch.stack(c2ws, dim=0),
                    fovy_deg,
                    point_cloud,
                    list(range(NUM_VIEWS)),
                    all_seg_result,
                    mesh,
                    sample_num_per_face=SAMPLE_NUM,
                    view_id=f"segmentation_result_{run_tag}",
                    uuid=obj_name,
                    prior_keys=prior_keys,
                    export_mesh=not enable_postprocess,
                    export_root=output_dir,
                )

                if enable_postprocess:
                    _, _, face_label3 = complete_labels(
                        face_label.clone(),
                        mesh_vanilla,
                        smooth_type="adjacent",
                        PA=postprocess_pa,
                    )
                    face_label = lift_2dmask_3d(
                        torch.stack([torch.from_numpy(m) for m in all_img_masks], dim=0),
                        torch.stack([torch.from_numpy(d).unsqueeze(-1) for d in all_depth_maps], dim=0),
                        torch.from_numpy(np.stack(all_norm_maps, axis=0)),
                        torch.stack(c2ws, dim=0),
                        fovy_deg,
                        point_cloud,
                        list(range(NUM_VIEWS)),
                        all_seg_result,
                        mesh,
                        sample_num_per_face=SAMPLE_NUM,
                        view_id=f"segmentation_postprocessed_{run_tag}_pa{postprocess_pa:g}",
                        uuid=obj_name,
                        face_label=face_label3,
                        prior_keys=prior_keys,
                        export_mesh=True,
                        export_root=output_dir,
                    )

                    # Extra boundary-aware smoothing passes, each exporting its
                    # own GLB next to the baseline so the seam quality can be
                    # compared side by side. The baseline above is untouched.
                    for method in (extra_smoothing or []):
                        if method in ("dihedral_icm", "majority_all"):
                            smoothed = smooth_labels_icm(
                                face_label, mesh_vanilla,
                                theta_deg=icm_theta, lambda_data=icm_lambda,
                                iters=icm_iters, dihedral=(method == "dihedral_icm"),
                            )
                        elif method == "alpha_exp":
                            smoothed = smooth_labels_graphcut(
                                face_label, mesh_vanilla,
                                theta_deg=icm_theta, lam=gc_lam,
                            )
                        elif method == "soft_alpha":
                            # Soft 12-view votes instead of the final hard labels:
                            # the histogram keeps per-view disagreement near seams
                            # and the graph-cut arbitrates it geometrically.
                            hist, ids = face_vote_histogram(
                                torch.stack([torch.from_numpy(m) for m in all_img_masks], dim=0),
                                torch.stack([torch.from_numpy(d).unsqueeze(-1) for d in all_depth_maps], dim=0),
                                torch.stack(c2ws, dim=0), fovy_deg, point_cloud,
                                all_seg_result, SAMPLE_NUM, prior_keys=prior_keys,
                            )
                            prob = (hist + 0.5) / (hist.sum(1, keepdim=True) + 0.5 * hist.shape[1])
                            smoothed = smooth_labels_graphcut(
                                face_label, mesh_vanilla,
                                unary=-torch.log(prob).numpy(), label_ids=ids,
                                theta_deg=icm_theta, lam=gc_lam,
                            )
                        else:
                            raise ValueError(f"unknown extra smoothing '{method}'")
                        moved = int((smoothed.cpu().numpy() != face_label.cpu().numpy()).sum())
                        print(f"[{method}] relabelled {moved} faces")
                        tag = f"segmentation_{method}_{run_tag}_pa{postprocess_pa:g}"
                        export_labelled_mesh(
                            mesh, smoothed, os.path.join(output_dir, f"{tag}.glb"))

                if save_pointcloud_vis:
                    pc_save_dir = os.path.join(output_dir, obj_name, "pointcloud_vis")
                    pc_save_path = os.path.join(pc_save_dir, f"labeled_pc_{run_tag}.ply")
                    save_labeled_point_cloud(
                        point_cloud=point_cloud,
                        face_label=face_label,
                        sample_num_per_face=SAMPLE_NUM,
                        save_path=pc_save_path,
                    )

                    save_labeled_point_cloud_from_pos_maps(
                        pos_maps=all_pos_maps,
                        all_seg_result=all_seg_result,
                        img_masks=all_img_masks,
                        save_dir=pc_save_dir,
                        tag=run_tag,
                    )

                if save_frame_vis:
                    vis_dir = os.path.join(output_dir, obj_name, "frame_vis")
                    save_frame_segmentation_visualizations(
                        images=all_images,
                        all_seg_result=all_seg_result,
                        save_dir=vis_dir,
                        frame_tag=run_tag,
                    )
    
    return {
        "all_seg_result": all_seg_result,
        "mesh": mesh,
        "face_label": face_label,
        "obj_name": obj_name
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GeoSAM2 mesh segmentation inference")
    parser.add_argument("--sam2-checkpoint", type=str, default="ckpt/geosam2.pt")
    parser.add_argument("--model-cfg", type=str, default="configs/geosam2.yaml")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory containing the multi-view renders (color/depth/normal) and meta.json.",
    )
    parser.add_argument(
        "--mask-root",
        type=str,
        default=None,
        help="Deprecated: automatic mask discovery is disabled; use --mask-path and --mask-view.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/3d_seg_results")
    parser.add_argument("--postprocess-pa", type=float, default=0.02)
    parser.add_argument(
        "--opposite-auto-segmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run automatic segmentation on the opposite seed view (e.g. view+6).",
    )
    parser.add_argument("--enable-postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-input-maps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-frame-vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-pointcloud-vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--extra-smoothing", nargs="*", default=[],
        choices=["dihedral_icm", "majority_all", "alpha_exp", "soft_alpha"],
        help="Extra boundary-smoothing passes on the final labels, each exporting "
             "its own GLB alongside the baseline. 'dihedral_icm' snaps seams to "
             "creases and straightens them on flat surfaces; 'majority_all' is the "
             "no-crease-weight ablation; 'alpha_exp' is the same energy optimised "
             "globally by graph-cut; 'soft_alpha' feeds the graph-cut per-view "
             "vote histograms instead of the final hard labels (best measured).",
    )
    parser.add_argument("--gc-lam", type=float, default=1.0,
                        help="Graph-cut smoothness strength for alpha_exp/soft_alpha.")
    parser.add_argument("--icm-theta", type=float, default=15.0,
                        help="Dihedral scale (deg) for crease weighting.")
    parser.add_argument("--icm-lambda", type=float, default=0.7,
                        help="Pull toward the original vote; 0 = pure smoothing.")
    parser.add_argument("--icm-iters", type=int, default=30,
                        help="Max ICM sweeps (stops early on convergence).")
    parser.add_argument(
        "--mask-path",
        type=str,
        default=None,
        help="Explicit mask file to inject (.exr/.npy/.png preview or label)",
    )
    parser.add_argument(
        "--mask-view",
        type=int,
        default=None,
        help="View index matching --mask-path",
    )
    return parser


def main():
    """Main inference function."""
    args = build_argparser().parse_args()
    device = init_env()

    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = args.model_cfg
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    predictor = build_sam2_video_predictor_geosam2(model_cfg, sam2_checkpoint, device=device)

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

    data_root = args.data_root

    mask_path = args.mask_path
    mask_view = args.mask_view

    if mask_path is not None and mask_view is None:
        raise ValueError("mask_view is required when mask_path is provided")

    data = read_data(
        data_root,
        mask_path=mask_path,
        mask_view=mask_view,
    )

    start_frames = None
    start_to_seed_views = None

    if mask_path is not None:
        # Single-start mode: start frame is exactly --mask-view.
        start = int(mask_view)
        auto_seed = (start + 6) % NUM_VIEWS
        seeds = [start]
        if auto_seed != start:
            seeds.append(auto_seed)
        start_frames = [start]
        start_to_seed_views = {start: seeds}

    if args.save_input_maps:
        input_vis_dir = os.path.join(args.output_dir, data["obj_name"], "input_maps")
        save_input_maps_visualizations(
            images=data["images"],
            norm_maps=data["norm_maps"],
            pos_maps=data["pos_maps"],
            img_masks=data["img_masks"],
            save_dir=input_vis_dir,
        )

    results = segment_with_mask_prompts(
        predictor=predictor,
        mask_generator=mask_generator,
        data=data,
        opposite_auto_segmentation=args.opposite_auto_segmentation,
        enable_postprocess=args.enable_postprocess,
        postprocess_pa=args.postprocess_pa,
        output_dir=args.output_dir,
        save_frame_vis=args.save_frame_vis,
        save_pointcloud_vis=args.save_pointcloud_vis,
        start_frames=start_frames,
        start_to_seed_views=start_to_seed_views,
        extra_smoothing=args.extra_smoothing,
        icm_theta=args.icm_theta,
        icm_lambda=args.icm_lambda,
        icm_iters=args.icm_iters,
        gc_lam=args.gc_lam,
    )

    print(f"Found {len(results['all_seg_result'][0])} objects")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
