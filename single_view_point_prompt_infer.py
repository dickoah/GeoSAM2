import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from geosam2._model import build_sam2_video_predictor_geosam2


def init_env() -> torch.device:
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

    np.random.seed(3)
    torch.manual_seed(3)
    return device


def load_prompts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list):
        raise ValueError("point prompt file must be a JSON list")
    return prompts


def choose_frame_prompts(prompts: List[Dict], view_idx: int) -> Tuple[int, List[Dict]]:
    by_frame: Dict[int, List[Dict]] = {}
    for i, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt item #{i} must be an object")
        for k in ("frame_idx", "obj_id", "point"):
            if k not in item:
                raise ValueError(f"Prompt item #{i} missing key: {k}")
        frame = int(item["frame_idx"])
        by_frame.setdefault(frame, []).append(item)

    if view_idx in by_frame:
        return view_idx, by_frame[view_idx]

    if 0 in by_frame:
        print(f"[Info] No frame_idx={view_idx} in prompt file, fallback to frame_idx=0 prompts")
        return 0, by_frame[0]

    if len(by_frame) == 1:
        only_frame = next(iter(by_frame.keys()))
        print(f"[Info] No frame_idx={view_idx} in prompt file, fallback to frame_idx={only_frame} prompts")
        return only_frame, by_frame[only_frame]

    raise ValueError(
        f"No usable prompts for view_idx={view_idx}. Prompt frames: {sorted(by_frame.keys())}"
    )


def group_obj_prompts(frame_prompts: List[Dict]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    grouped: Dict[int, Dict[str, List]] = {}
    for i, p in enumerate(frame_prompts):
        obj_id = int(p["obj_id"])
        point = p["point"]
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"Prompt item #{i} point must be [x, y]")
        label = int(p.get("label", 1))
        if label not in (0, 1):
            raise ValueError(f"Prompt item #{i} label must be 0 or 1")

        grouped.setdefault(obj_id, {"points": [], "labels": []})
        grouped[obj_id]["points"].append([float(point[0]), float(point[1])])
        grouped[obj_id]["labels"].append(label)

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for obj_id, item in grouped.items():
        out[obj_id] = (
            np.array(item["points"], dtype=np.float32),
            np.array(item["labels"], dtype=np.int32),
        )
    return out


def load_rgb(data_root: str, view_idx: int) -> np.ndarray:
    img_path = os.path.join(data_root, f"color_{view_idx:04d}.webp")
    img = Image.open(img_path)
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, img)
    return np.array(comp.convert("RGB"))


def save_outputs(label_map: np.ndarray, rgb: np.ndarray, out_dir: str, view_idx: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"mask_view{view_idx:04d}.npy"), label_map.astype(np.int32))

    rng = np.random.default_rng(3)
    vis = rgb.copy()
    obj_ids = [x for x in np.unique(label_map).tolist() if int(x) != 0]
    color_map = {int(i): rng.integers(0, 256, size=3, dtype=np.uint8) for i in obj_ids}
    for obj_id, color in color_map.items():
        vis[label_map == obj_id] = color

    Image.fromarray(vis.astype(np.uint8)).save(os.path.join(out_dir, f"mask_view{view_idx:04d}_vis.png"))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-view point-prompt segmentation")
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default="ckpt/geosam2.pt",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="configs/geosam2.yaml",
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--view-idx", type=int, required=True)
    parser.add_argument("--point-prompt-file", type=str, required=True)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="outputs/view_seg_results")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = init_env()

    predictor = build_sam2_video_predictor_geosam2(
        args.model_cfg,
        args.sam2_checkpoint,
        device=device,
    )

    prompts = load_prompts(args.point_prompt_file)
    _, frame_prompts = choose_frame_prompts(prompts, args.view_idx)
    obj_prompts = group_obj_prompts(frame_prompts)
    if len(obj_prompts) == 0:
        raise ValueError("No prompts found after parsing")

    inference_state = predictor.init_state(video_path=args.data_root, video_id_list=[args.view_idx])
    predictor.reset_state(inference_state)

    # Single-view mode: local frame index is always 0 for the provided video_id_list.
    local_frame_idx = 0
    for obj_id, (points, labels) in sorted(obj_prompts.items(), key=lambda x: x[0]):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=local_frame_idx,
            obj_id=int(obj_id),
            points=points,
            labels=labels,
        )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video_v2(
        inference_state,
        start_frame_idx=local_frame_idx,
    ):
        video_segments[out_frame_idx] = {
            int(out_obj_id): (out_mask_logits[i] > args.mask_threshold).cpu().numpy().squeeze().astype(bool)
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    if local_frame_idx not in video_segments:
        raise RuntimeError("No segmentation output for single view")

    rgb = load_rgb(args.data_root, args.view_idx)
    h, w = rgb.shape[:2]
    label_map = np.zeros((h, w), dtype=np.int32)
    for obj_id, mask in video_segments[local_frame_idx].items():
        label_map[mask] = int(obj_id)

    save_outputs(label_map, rgb, args.output_dir, args.view_idx)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
