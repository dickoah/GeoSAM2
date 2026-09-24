"""From masks on the views to labels on the faces, and the two post-processes.

VAST's ``inference_utils``, kept to what the propagation calls: the mask
filters between propagations, the lift (``lift_2dmask_3d``), the completion
of labels over the mesh (``complete_labels``) and the fragment cleanup
(``clean_label_fragments``).
"""

import numpy as np
import torch
import math
import torch.nn.functional as F
import cv2
from geosam2.sam2.utils.amg import calculate_stability_score
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from geosam2.util.logs import get_logger
from geosam2.ext.mode_ext import mode_except_negative_one

logger = get_logger("geosam2.lift")


def compute_iou(pred, gt):
    """Compute IoU percentage between two boolean masks.

    Args:
        pred: Predicted binary mask.
        gt: Ground-truth binary mask.
    """
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union != 0:
        return (intersection / union) * 100  
    else:
        return 0


def filter_masks(anns, dedup_iou=85):
    """The automatic masks worth keeping: deduplicated by IoU, largest first.

    Args:
        anns: SAM mask annotations (each with ``segmentation``, ``predicted_iou``
            and ``area`` keys).
        dedup_iou: Overlap (%) above which two masks count as one.
    """
    if len(anns) == 0:
        return []
    sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=True)
    mask_accept_list = []
    for ann in sorted_anns:
        if len(mask_accept_list) == 0:
            mask_accept_list.append(ann)
            continue
        accept = True
        for acc_mask in mask_accept_list:
            if compute_iou(ann['segmentation'], acc_mask['segmentation']) > dedup_iou:
                accept = False
                break
        if accept:
            mask_accept_list.append(ann)

    return sorted(mask_accept_list, key=(lambda x: x['area']), reverse=True)


def filter_mask_area(video_segments, track_id, alpha=5):
    """Drop objects whose non-track-view area is too large.

    Args:
        video_segments: Dict[frame_idx, Dict[obj_id, mask]].
        track_id: Reference frame index used as anchor area.
        alpha: Max allowed area ratio vs anchor before dropping an object.
    """
    del_obj_id_list = []
    obj_id_list = video_segments[track_id].keys()
    for obj_id in obj_id_list:
        delete = False
        anchor_area = video_segments[track_id][obj_id]
        anchor_area = anchor_area.sum() if anchor_area.dtype == np.bool_ else (anchor_area>0).sum()
        for frame_id, out_mask in video_segments.items():
            if frame_id != track_id:
                pred_area = out_mask[obj_id].sum() if out_mask[obj_id].sum().dtype == np.bool_ else (out_mask[obj_id]>0).sum()
                if pred_area / max(anchor_area, 1e-6) > alpha:
                    delete = True
                    break
        if delete:
            del_obj_id_list.append(obj_id)

    for index in range(12):
        for obj_id in del_obj_id_list:
            video_segments[index].pop(obj_id, None)

    return video_segments

def trans2bool(video_segments, track_id):
    """Convert mask logits in `video_segments` to boolean masks.

    Args:
        video_segments: Dict[frame_idx, Dict[obj_id, mask/logits]].
        track_id: Anchor frame index whose object ids are iterated.
    """
    obj_id_list = video_segments[track_id].keys()
    for obj_id in obj_id_list:
        for frame_id, out_mask in video_segments.items():
            video_segments[frame_id][obj_id] = video_segments[frame_id][obj_id] > 0.
    return video_segments


def filter_mask_stability(video_segments, track_id, stability_dict, stability_score_thresh=0.92, stability_score_offset=0.7, mask_threshold=0.):
    """Filter unstable masks by SAM stability score.

    Args:
        video_segments: Dict[frame_idx, Dict[obj_id, mask/logits]].
        track_id: Anchor frame index.
        stability_dict: Output cache for per-frame per-object stability.
        stability_score_thresh: Stability threshold on anchor frame.
        stability_score_offset: Offset used by stability score computation.
        mask_threshold: Threshold used to binarize logits in stability scoring.
    """
    del_obj_id_list = []
    obj_id_list = video_segments[track_id].keys()
    for obj_id in obj_id_list:
        delete = False
        for idx in range(12):
            if idx == track_id:
                stability_score_thresh_ = stability_score_thresh
            else:
                stability_score_thresh_ = stability_score_thresh - 0.07

            mask_tmp = video_segments[idx][obj_id] > 0
            if mask_tmp.sum() > 0:
                stability_score = calculate_stability_score(torch.from_numpy(video_segments[idx][obj_id]), mask_threshold, stability_score_offset)
                delete = stability_score < stability_score_thresh_
            if delete:
                break

        if delete:
            del_obj_id_list.append(obj_id)

        for idx in list(video_segments.keys()):
            stability_score = calculate_stability_score(torch.from_numpy(video_segments[idx][obj_id]), mask_threshold, stability_score_offset)
            stability_dict[idx][obj_id] = stability_score.item()

    for index in range(12):
        for obj_id in del_obj_id_list:
            video_segments[index].pop(obj_id, None)

    return video_segments, stability_dict


def shrink_mask(video_segments, kernel_size=5, iterations=3):
    """Apply morphology to clean masks and push edge confidence apart.

    Args:
        video_segments: Dict[frame_idx, Dict[obj_id, mask/logits]].
        kernel_size: Side of the square opening kernel, in pixels.
        iterations: Erosion then dilation passes.
    """
    for frame_id in list(video_segments.keys()):
        for obj_id in list(video_segments[frame_id].keys()):
            vanilla_mask = video_segments[frame_id][obj_id] > 0

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask_ = (video_segments[frame_id][obj_id] > 0).squeeze()
            mask_ = mask_.astype(np.uint8) * 255
            mask_ = cv2.erode(mask_, kernel,iterations=iterations)
            mask_ = cv2.dilate(mask_, kernel,iterations=iterations)
            mask_ = mask_[None,:,:]
            mask_ = mask_ > 128

            video_segments[frame_id][obj_id][~mask_ & vanilla_mask] = -1024.
            video_segments[frame_id][obj_id][mask_ ^ vanilla_mask] = 1024.
    return video_segments


def get_projection_matrix(
        batch_size: int,
        fovy_deg: float,
        aspect_wh: float = 1.0,
        near: float = 0.1, far: float = 100.
    ) -> torch.FloatTensor:
    """Build OpenGL-style perspective projection matrices.

    Args:
        batch_size: Number of matrices to generate.
        fovy_deg: Vertical FOV in degrees.
        aspect_wh: Aspect ratio width/height.
        near: Near clipping plane.
        far: Far clipping plane.
    """
    fovy_deg = torch.tensor([fovy_deg] * batch_size, dtype=torch.float32)
    fovy = fovy_deg * math.pi / 180
    tan_half_fovy = torch.tan(fovy / 2)
    projection_matrix = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    projection_matrix[:, 0, 0] = 1 / (aspect_wh * tan_half_fovy)
    projection_matrix[:, 1, 1] = -1 / tan_half_fovy
    projection_matrix[:, 2, 2] = -(far + near) / (far - near)
    projection_matrix[:, 2, 3] = -2 * far * near / (far - near)
    projection_matrix[:, 3, 2] = -1
    return projection_matrix


def get_clip_space_position(pos: torch.FloatTensor, mvp_mtx: torch.FloatTensor):
    """Project 3D points into clip space with MVP matrices.

    Args:
        pos: 3D points with shape [N, 3].
        mvp_mtx: MVP matrices with shape [V, 4, 4].
    """
    pos_homo = torch.cat(
        [pos, torch.ones([pos.shape[0], 1]).to(pos)], dim=-1
    )
    return torch.matmul(pos_homo, mvp_mtx.permute(0, 2, 1))


def transform_points_homo(pos: torch.FloatTensor, mtx: torch.FloatTensor):
    """Apply homogeneous transforms to point cloud coordinates.

    Args:
        pos: 3D points with shape [N, 3].
        mtx: Transform matrices with shape [V, 4, 4].
    """
    pos_homo = torch.cat(
        [pos, torch.ones_like(pos[...,0:1])], dim=-1
    )
    # a matmul, not a [V, N, 4, 4] broadcast: that was 2.5 GB on 3.3M points
    return (pos_homo @ mtx.transpose(1, 2))[..., :3]


def cal_link(imgs_mask, depth_images, c2ws, fovy_deg, coord):
    """Link sampled 3D points to per-view image pixels and visibility.

    Args:
        imgs_mask: RGBA validity mask tensor with shape [V, H, W, 1].
        depth_images: Depth tensor with shape [V, H, W, 1].
        c2ws: Camera-to-world matrices for all views.
        fovy_deg: Vertical FOV in degrees.
        coord: Sampled 3D point cloud with shape [P, 3].
    """
    w2c = torch.linalg.inv(c2ws)
    proj_mtx = get_projection_matrix(c2ws.shape[0], fovy_deg, aspect_wh=1., near=0.1, far=10.)
    mvp_mtx = proj_mtx @ w2c

    pos_clip = get_clip_space_position(coord, mvp_mtx)
    pos_ndc = pos_clip[..., :2] / pos_clip[..., 3:4]

    pos_vs = transform_points_homo(coord, w2c)
    pos_depth = -pos_vs[..., 2:3]

    rgba_proj = F.grid_sample(
        imgs_mask.permute(0, 3, 1, 2).to(torch.float32),
        pos_ndc[:,None],
        align_corners=True,
        mode="nearest"
    ).permute(0, 2, 3, 1)[:,0]
    
    depth_proj = F.grid_sample(
        depth_images.permute(0, 3, 1, 2),
        pos_ndc[:,None],
        align_corners=True,
        mode="nearest"
    ).permute(0, 2, 3, 1)[:,0]

    depth_error = (depth_proj - pos_depth).abs()

    valid = (depth_error < 1e-3) & (rgba_proj[...,0:1] > 0.9)# & norm_mask#& (mask_view > 0.9)# & (normal_filter > 0.9)
    valid = valid.int()
    pos_pixel = (((pos_ndc + 1) / 2) * torch.tensor([imgs_mask.shape[1], imgs_mask.shape[2]]).view(1, 1, 2)).round().clamp(0, imgs_mask.shape[1]-1)
    link = torch.cat((pos_pixel, valid), dim=-1).int()
    return link

def mask_aggregation(video_segments, prior_keys=None):
    """Aggregate multi-view object masks into one label volume.

    Args:
        video_segments: Dict[frame_idx, Dict[obj_id, bool_mask]].
        prior_keys: Optional prompt-priority object ids.
    """
    obj_id_list = video_segments[0].keys()

    all_masks = np.ones((12,1,1024,1024), dtype=np.float32) * 999

    if prior_keys != None:
        temp_masks = np.zeros((12,1,1024,1024), dtype=np.float32)
    else:
        prior_keys = set()

    obj_mask_area = {key: 0 for key in obj_id_list}
    for key in obj_mask_area.keys():
        for frame_idx in range(12):
            obj_mask_area[key] += video_segments[frame_idx][key].sum()

    obj_id_list = [key for key, value in sorted(obj_mask_area.items(), key=lambda item: item[1], reverse=True)]
    for obj_id in obj_id_list:
        if obj_id in prior_keys:
            for frame_id, obj_mask_dict in video_segments.items():
                mask_ = obj_mask_dict[obj_id]
                temp_masks[frame_id][mask_] = obj_id           
        else:
            for frame_id, obj_mask_dict in video_segments.items():
                mask_ = obj_mask_dict[obj_id]
                all_masks[frame_id][mask_] = obj_id

    if len(prior_keys) > 0:
        all_masks[temp_masks!=0]=temp_masks[temp_masks!=0]
        
    return all_masks


def lift_2dmask_3d(imgs_mask, depth_imgs, c2ws, fovy_deg, coord, selected_frames,
                   video_segments, sample_num_per_face, prior_keys=None):
    """Lift 2D segmentation masks to per-face 3D labels.

    Args:
        imgs_mask: RGBA validity mask tensor for all views.
        depth_imgs: Depth maps for all views.
        c2ws: Camera-to-world matrices.
        fovy_deg: Vertical FOV in degrees.
        coord: Sampled 3D points used for lifting, ``sample_num_per_face`` per face.
        selected_frames: View indices used in current lifting pass.
        video_segments: Dict[frame_idx, Dict[obj_id, bool_mask]].
        sample_num_per_face: Number of points sampled per face.
        prior_keys: Optional object ids with prompt-priority in aggregation.
    """
    all_masks = torch.from_numpy(mask_aggregation(video_segments, prior_keys=prior_keys))
    link = cal_link(imgs_mask, depth_imgs, c2ws, fovy_deg, coord).permute(1,2,0).to(torch.int)
    grid_normalized = (link[:,:-1,:].permute(2,0,1).unsqueeze(-2).to(torch.float32) / (1024 - 1)) * 2 - 1
    pc_labels = F.grid_sample(all_masks, grid_normalized, mode="nearest").squeeze(1)
    link_ = link.permute(2,0,1)[:,:,-1:].to(torch.bool)
    pc_labels[~link_] = 0

    pc_labels = pc_labels.squeeze().permute(1,0)
    pc_label = mode_except_negative_one(pc_labels.to(torch.int32)).to(all_masks.dtype)

    pc_label = pc_label.reshape(-1, sample_num_per_face)
    return mode_except_negative_one(pc_label.to(torch.int32)).to(all_masks.dtype)


def sample_points_on_faces_parallel(face_to_vertex, num_points=3, use_vertex=False):
    """Sample points on each mesh face using barycentric coordinates.

    Args:
        face_to_vertex: Face vertex coordinates with shape [F, 3, 3].
        num_points: Number of sampled points per face.
        use_vertex: Whether to append original 3 face vertices to samples.
    """
    if use_vertex:
        num_points = num_points - 3
    num_faces = face_to_vertex.shape[0]

    # Generate random barycentric coordinates for all faces and points
    u = np.random.rand(num_faces, num_points, 1)
    v = np.random.rand(num_faces, num_points, 1)
    
    # Ensure the barycentric coordinates are valid (u + v <= 1)
    mask = (u + v > 1)
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v  # Compute the third barycentric coordinate

    # Compute sampled points using the barycentric coordinates
    sampled_points = (
        u * face_to_vertex[:, np.newaxis, 0, :] +  # Contribution from the first vertex
        v * face_to_vertex[:, np.newaxis, 1, :] +  # Contribution from the second vertex
        w * face_to_vertex[:, np.newaxis, 2, :]    # Contribution from the third vertex
    )
    if use_vertex:
        sampled_points = np.concatenate([sampled_points,face_to_vertex],axis=-2)
    return sampled_points


def complete_labels(face_labels, mesh, PA=0.025):
    """Remove tiny components and fill unlabeled faces, in place; returns them.

    Args:
        face_labels: Per-face integer labels.
        mesh: Source mesh with adjacency/geometry.
        PA: Relative threshold for tiny-component removal.
    """
    adjacency = np.asarray(mesh.face_adjacency)
    # Each face's neighbours in the order a Python set of them iterates in --
    # neither the adjacency's order nor sorted. VAST's smoothing took the FIRST
    # non-zero neighbour of such a set, so the order is the contract; the sets
    # are built as they were and read out into a -1 padded table.
    keys, vals = adjacency.ravel(), adjacency[:, ::-1].ravel()
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    neighbours = [list(set(vals[a:b].tolist())) for a, b in zip(starts, np.r_[starts[1:], len(keys)])]
    degree = np.fromiter(map(len, neighbours), np.int64, len(neighbours))
    table = np.full((len(face_labels), degree.max()), -1, np.int64)
    table[np.repeat(keys[starts], degree),
          np.arange(degree.sum()) - np.repeat(np.cumsum(degree) - degree, degree)] = \
        np.fromiter((f for nb in neighbours for f in nb), np.int64, degree.sum())

    components = sorted(label_components(face_labels, adjacency), key=len, reverse=True)
    area_faces = np.asarray(mesh.area_faces)
    components_area = [area_faces[list(comp)].sum() for comp in components]
    max_size = max(map(len, components))
    max_area = max(components_area)

    remove_comp = ({i for i, comp in enumerate(components) if len(comp) < max_size * PA}
                   & {i for i, area in enumerate(components_area) if area < max_area * PA})
    print(f"Removing {len(remove_comp)} small components")
    for i in remove_comp:
        face_labels[list(components[i])] = 0

    # Up to 64 passes; each face still at 0 takes its first non-zero neighbour's
    # label, all read from the state the pass started in. VAST's loop counted
    # the neighbours in a Counter keyed by 0-d tensors, which hash by identity,
    # so most_common() gave the first non-zero one in set order -- kept.
    # A pass that changes nothing ends the loop. Per-face Python over the
    # unfilled block took 5 s on 664k faces; this is one gather per pass.
    lab = face_labels.numpy()
    for _ in range(64):
        zero = np.flatnonzero(lab == 0)
        around = table[zero]
        labels = np.where(around >= 0, lab[np.maximum(around, 0)], 0)
        found = labels != 0
        hit = found.any(axis=1)
        if not hit.any():
            break
        lab[zero[hit]] = labels[np.arange(len(zero)), found.argmax(axis=1)][hit]

    print("Smoothing labels")
    face_unlable_idx = torch.where(face_labels == 0)[0]
    face_lable_idx = torch.where(face_labels != 0)[0]
    face_centroids = mesh.triangles_center
    unlabel_xyz = face_centroids[face_unlable_idx]
    label_xyz = face_centroids[face_lable_idx]

    unlabel_top3_indices = find_nearest_three_points(unlabel_xyz, label_xyz) # [N,3]
    nearest_labels = face_labels[face_lable_idx][unlabel_top3_indices]
    most_frequent_labels = torch.mode(nearest_labels, dim=1).values
    face_labels[face_unlable_idx] = most_frequent_labels

    labels_seen = set()
    labels_curr = face_labels.max().item() + 1
    labels_orig = labels_curr
    for comp in components:
        face = comp.pop()
        label = face_labels[face]
        comp.add(face)
        if label == 0 or label in labels_seen: # background or repeated label
            for face in comp:
                face_labels[face] = labels_curr
            labels_curr += 1
        labels_seen.add(label)
    print(f"Split {labels_curr - labels_orig} component(s) into unique labels")
    return face_labels

def label_components(face_labels, face_adjacency: np.ndarray) -> list[set]:
    """Connected groups of adjacent faces sharing one non-zero label, in order
    of their lowest face -- the order the DFS this replaces found them in.

    Args:
        face_labels: Per-face labels.
        face_adjacency: Adjacent face pairs, [E, 2].
    """
    labels = np.asarray(face_labels)
    a, b = face_adjacency.T
    keep = (labels[a] == labels[b]) & (labels[a] != 0)
    a, b = a[keep], b[keep]
    n = len(labels)
    graph = coo_matrix((np.ones(2 * len(a), np.int8), (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    _, component = connected_components(graph, directed=False)

    faces = np.flatnonzero(labels != 0)
    order = np.argsort(component[faces], kind="stable")
    faces, group = faces[order], component[faces][order]
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    return [set(faces[s:e].tolist()) for s, e in zip(starts, np.r_[starts[1:], len(faces)])]

def find_nearest_three_points(A, B):
    """Indices of the nearest 3 points in B for every point in A, [N, 3].

    A tree, not the dense [N, M] cdist VAST used: that is 319 GB on 664k faces.
    """
    np_array = isinstance(A, np.ndarray)
    A_np = A if np_array else A.numpy()
    B_np = B if np_array else B.numpy()

    k = min(3, len(B_np))
    _, indices = cKDTree(B_np).query(A_np, k=k, workers=-1)
    indices = indices.reshape(len(A_np), k)
    if k < 3:
        indices = np.repeat(indices[:, :1], 3, axis=1)

    return indices if np_array else torch.from_numpy(indices)

def filter_iou(all_seg_result, video_segments, track_id, iou_thresh=0.8):
    """Filter highly overlapping objects inside and across passes.

    Args:
        all_seg_result: Accumulated segmentation dict by frame/object.
        video_segments: Current pass segmentation dict by frame/object.
        track_id: Anchor frame used for overlap comparison.
        iou_thresh: Overlap above which two objects are one.
    """
    start = __import__('time').time()
    accept_id = []

    for obj_id in list(video_segments[0].keys()):
        accept = True
        for acpt_id in accept_id:
            mask1 = video_segments[track_id][obj_id]
            mask2 = video_segments[track_id][acpt_id]
            iou = np.sum(mask1 & mask2) / (np.sum(mask1 | mask2) + 1e-6)
            if iou > iou_thresh:
                accept = False
                break
        if accept:
            accept_id.append(obj_id)
    
    discard_list = list(set(video_segments[0].keys()) - set(accept_id))
    for obj_id in discard_list:
        for _ in range(12):
            video_segments[_].pop(obj_id)

    discard_id = []
    for obj_id_1 in list(video_segments[0].keys()):
        for obj_id_2 in list(all_seg_result[0].keys()):
            mask1 = video_segments[track_id][obj_id_1]
            mask2 = all_seg_result[track_id][obj_id_2]
            iou = np.sum(mask1 & mask2) / (np.sum(mask1 | mask2) + 1e-6)
            if iou > iou_thresh and iou <= 1:
                discard_id.append(obj_id_1)

    discard_id = list(set(discard_id))
    for obj_id in discard_id:
        for _ in range(12):
            video_segments[_].pop(obj_id)

    end = __import__('time').time()
    _ = end - start

    return video_segments, all_seg_result


# ─────────────────────────────────────────────────────────────────────────────
# Label export and fragment cleanup on the FINAL per-face labels.
# ─────────────────────────────────────────────────────────────────────────────


def _to_np_int(x) -> np.ndarray:
    """Face-label tensor/array -> contiguous int64 numpy, detached from device."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x).astype(np.int64)


# Fragment cleanup thresholds, measured on furniture assets in the PixMesh
# splitter this is ported from. Relative to the largest component OF THE SAME
# LABEL: a part is small next to the object all the time; what makes it speckle
# is being small next to the rest of its own label (mesh-relative collapsed 12
# parts into 3 on sample_05).
FRAG_MAX_REL = 0.02      # below: speckle, may move
HOST_MIN_REL = 0.20      # above: a host that can receive it; between: grey zone
CONTACT_MAX_FRAC = 0.03  # fragment-to-host reach, fraction of the bbox diagonal
VOTE_K = 9


def _contact_reach(mesh_vanilla) -> float:
    bounds = np.asarray(mesh_vanilla.bounds, dtype=np.float64)
    return CONTACT_MAX_FRAC * (float(np.linalg.norm(bounds[1] - bounds[0])) or 1.0)


def _proximity_components(mesh_vanilla, labels: np.ndarray) -> np.ndarray:
    """Connected components of same-label faces, neighbours by 3D proximity.

    Edge adjacency is useless on a generated mesh: TRELLIS output is triangle
    soup (~2400 topological components on one object), so a part that is
    visibly one piece fragments into hundreds of edge-connected components and
    every size threshold below then fires on real geometry. Two faces of one
    label are neighbours when one is among the other's nearest same-label faces
    within the contact reach -- a scale-derived radius, as upstream welds. A
    data-derived one (3x the median gap) isolated every large triangle on a
    mesh whose density varies 160x (sample_04: 1952 components for 9 labels).
    """
    centroids = np.asarray(mesh_vanilla.triangles_center, dtype=np.float64)
    reach = _contact_reach(mesh_vanilla)
    rows, cols = [], []
    for value in np.unique(labels):
        faces = np.nonzero(labels == value)[0]
        k = min(VOTE_K + 1, len(faces))
        if k < 2:
            continue
        dist, idx = cKDTree(centroids[faces]).query(centroids[faces], k=k, workers=-1)
        ok = dist[:, 1:] <= reach
        rows.append(np.repeat(faces, k - 1)[ok.ravel()])
        cols.append(faces[idx[:, 1:]][ok])
    n = len(centroids)
    if not rows:
        return np.arange(n)
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def clean_label_fragments(face_labels, mesh_vanilla) -> torch.Tensor:
    """Reassign per-label speckle fragments to the label they actually touch.

    Smoothing moves boundaries; it does not remove a patch of the seat's label
    stranded in the middle of a leg, because locally that patch is consistent.
    This does, and only for the unambiguous tail: a component under
    ``FRAG_MAX_REL`` of its own label's largest component may move onto a host
    over ``HOST_MIN_REL``. The grey zone between never moves on size alone --
    except an orphan touching no surface of its own label while sitting on
    another's, which is mislabelled whatever its size.

    Holes are never filled (a hole is the interface with a neighbouring part)
    and nothing is ever deleted: a fragment with no host within reach keeps its
    label. One pass only, as upstream: a second pass re-judges components the
    first chose to keep, in a host landscape its own moves degraded (measured
    10.3% of a drawer repainted).
    """
    lab = _to_np_int(face_labels).copy()
    centroids = np.asarray(mesh_vanilla.triangles_center, dtype=np.float64)
    area = np.asarray(mesh_vanilla.area_faces, dtype=np.float64)
    reach = _contact_reach(mesh_vanilla)

    comp = _proximity_components(mesh_vanilla, lab)
    comp_area = np.bincount(comp, weights=area)
    comp_label = np.zeros(len(comp_area), dtype=np.int64)
    comp_label[comp] = lab
    largest = np.zeros(int(comp_label.max()) + 1, dtype=np.float64)
    np.maximum.at(largest, comp_label, comp_area)
    rel = comp_area / np.maximum(largest[comp_label], 1e-12)

    is_host = (rel >= HOST_MIN_REL)[comp]
    if not is_host.any():
        return torch.from_numpy(lab)
    tree = cKDTree(centroids[is_host])
    host_label = lab[is_host]
    k = min(VOTE_K, int(is_host.sum()))

    moved = 0
    for cid in np.nonzero(rel < HOST_MIN_REL)[0]:
        faces = np.nonzero(comp == cid)[0]
        dist, idx = tree.query(centroids[faces], k=k, workers=-1)
        near = np.atleast_2d(dist) <= reach
        if not near.any():
            continue                                # out of reach: keep, never delete
        # Only neighbours within reach vote (upstream lets all k vote once the
        # nearest is in reach) -- a far neighbour says nothing about contact.
        votes = host_label[np.atleast_2d(idx)[near]]
        if rel[cid] >= FRAG_MAX_REL:
            # Grey zone: size cannot make the call, contact does. Attached to
            # its own label -> keep; otherwise only other labels may claim it.
            own = np.nonzero(is_host & (lab == comp_label[cid]))[0]
            if len(own) and cKDTree(centroids[own]).query(
                    centroids[faces], workers=-1)[0].min() <= reach:
                continue
            votes = votes[votes != comp_label[cid]]
            if not len(votes):
                continue
        dest = int(np.bincount(votes).argmax())
        if dest != comp_label[cid]:
            lab[faces] = dest
            moved += len(faces)
    logger.info("[fragments] %d faces relabelled", moved)
    return torch.from_numpy(lab)
