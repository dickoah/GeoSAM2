"""Segment a data-root from its 12 per-view GT colour masks -- no SAM2.

The ideal-seed benchmark: every view already has a perfect part mask, so instead
of seeding one view and propagating, each mask votes directly on the mesh faces
(same projection as the pipeline's lift) and an alpha-expansion graph-cut
arbitrates the seams geometrically. This is the method behind
``outputs/mvgc_*/alpha_exp_strong.glb``.

Usage:
    python multiview_mask_infer.py --data-root example/sample_05 \
        --output-dir outputs/gtmv_05 [--gc-lam 3.0] [--gc-theta 20]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inference import NUM_VIEWS, SAMPLE_NUM, prepare_mesh_and_point_cloud, read_data
from utils.inference_utils import cal_link, export_labelled_mesh, smooth_labels_graphcut


def view_mask_label_maps(data_root: str) -> np.ndarray:
    """The 12 GT colour masks as one [12, 1, H, W] integer label volume.

    Colour -> id assignment is shared across views (the same part keeps the same
    id everywhere), which per-view background guessing cannot guarantee. Pure
    black is background.
    """
    palette = {}
    maps = None
    for v in range(NUM_VIEWS):
        rgb = np.asarray(Image.open(os.path.join(data_root, f"mask_{v:04d}.png")).convert("RGB"))
        if maps is None:
            maps = np.zeros((NUM_VIEWS, 1, *rgb.shape[:2]), dtype=np.float32)
        for color in map(tuple, np.unique(rgb.reshape(-1, 3), axis=0)):
            if color == (0, 0, 0):
                continue
            if color not in palette:
                palette[color] = len(palette) + 1
            maps[v, 0][np.all(rgb == color, axis=-1)] = palette[color]
    print(f"{len(palette)} parts across {NUM_VIEWS} view masks")
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gc-lam", type=float, default=3.0,
                        help="Graph-cut smoothness; 3.0 is the strong preset.")
    parser.add_argument("--gc-theta", type=float, default=20.0,
                        help="Dihedral scale (deg) for crease weighting.")
    args = parser.parse_args()

    data = read_data(args.data_root)
    mesh, coord = prepare_mesh_and_point_cloud(
        data["mesh"], data["scaling_factor"], data["translation"].numpy())
    labmaps = view_mask_label_maps(args.data_root)

    # Same projection as lift_2dmask_3d, kept soft: every sample of every face
    # reads its label in all 12 masks; the histogram keeps the disagreement.
    link = torch.ones([coord.shape[0], 3, NUM_VIEWS], dtype=torch.int)
    link[:, 0:3, :] = cal_link(
        torch.stack([torch.from_numpy(m) for m in data["img_masks"]], dim=0),
        torch.stack([torch.from_numpy(d).unsqueeze(-1) for d in data["depth_maps"]], dim=0),
        torch.stack(data["c2ws"]), data["fovy_deg"], coord).permute(1, 2, 0)
    grid = (link[:, :-1, :].permute(2, 0, 1).unsqueeze(-2).to(torch.float32)
            / (labmaps.shape[-1] - 1)) * 2 - 1
    pc = F.grid_sample(torch.from_numpy(labmaps), grid, mode="nearest").squeeze(1)
    pc[~link.permute(2, 0, 1)[:, :, -1:].to(torch.bool)] = 0
    pc = pc.squeeze().permute(1, 0).to(torch.int64)

    n_faces = coord.shape[0] // SAMPLE_NUM
    K = int(labmaps.max())
    votes = pc.reshape(n_faces, SAMPLE_NUM * NUM_VIEWS)
    hist = torch.zeros(n_faces, K + 1, dtype=torch.float32)
    hist.scatter_add_(1, votes, torch.ones_like(votes, dtype=torch.float32))
    hist = hist[:, 1:]                       # drop the no-vote column
    print(f"votes: {int((hist.sum(1) > 0).sum())}/{n_faces} faces saw a mask")

    prob = (hist + 0.5) / (hist.sum(1, keepdim=True) + 0.5 * K)
    seed = (hist.argmax(1) + 1).numpy()
    labels = smooth_labels_graphcut(
        seed, data["mesh_vanilla"], unary=-torch.log(prob).numpy(),
        label_ids=list(range(1, K + 1)), theta_deg=args.gc_theta, lam=args.gc_lam)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f"segmentation_gt_multiview_lam{args.gc_lam:g}.glb")
    export_labelled_mesh(mesh, labels, out)
    found = len(set(labels.tolist()) - {0})
    print(f"Found {found} objects")


if __name__ == "__main__":
    main()
