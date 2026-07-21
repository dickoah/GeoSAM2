import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import math
import torch.nn.functional as F
import trimesh
from collections import defaultdict, Counter
import cv2
from sam2.utils.amg import calculate_stability_score
from utils.mode_ext import mode_except_negative_one


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    """Build per-pixel camera rays in camera space.

    Args:
        W: Image width in pixels.
        H: Image height in pixels.
        fx: Focal length along x axis.
        fy: Focal length along y axis.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.
        use_pixel_centers: Whether to offset samples by 0.5 pixel.
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing="xy",
    )
    directions = np.stack(
        [(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], -1
    ) 

    return directions

def gen_pcd(depth, c2w_opengl, camera_angle_x):
    """Convert a depth map into a clipped world-space position map.

    Args:
        depth: Depth image with shape [H, W].
        c2w_opengl: Camera-to-world matrix in OpenGL convention.
        camera_angle_x: Horizontal field-of-view in radians.
    """
    h, w = depth.shape
    
    depth_valid = depth < 65500.0
    focal = 0.5 * w / math.tan(0.5 * camera_angle_x)
    ray_directions = get_ray_directions(w, h, focal, focal, w // 2, h // 2)

    org_points = np.zeros((h, w, 3))

    points_c = ray_directions[depth_valid] * depth[depth_valid, None]
    points_c_homo = np.concatenate(
        [points_c, np.ones_like(points_c[..., :1])], axis=-1
    )
    valid_points = (points_c_homo @ c2w_opengl.T)[..., :3]

    valid_points = np.clip(valid_points, -1.0, 1.0)

    org_points[depth_valid] = valid_points

    return org_points

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

def eval_per_shape_part_mean_iou(
    pred_ins: np.ndarray,
    gt_ins: dict,
    ) -> float:
    """Compute mean best-IoU score per GT part for one shape.

    Args:
        pred_ins: Predicted instance list/array with `segmentation` fields.
        gt_ins: Mapping from part id to GT binary mask.
    """

    ious = []
    for key, gt_mask in gt_ins.items():
        best_iou = 0
        for mask_ in pred_ins:
            pred_mask = mask_['segmentation']
            iou = compute_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
            ious.append(best_iou)
    return np.mean(ious)

def compute_intersect_mask1(mask1, mask2):
    """Compute intersection ratio relative to mask1 area in percent.

    Args:
        mask1: Reference mask.
        mask2: Target mask to intersect with.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    mask1_ = mask1.sum()
    if mask1_ != 0:
        return (intersection / mask1_) * 100  
    else:
        return 0

def show_anns(anns, save_path=None, borders=True):
    """Filter overlapping auto-generated masks and optionally render an overlay.

    Args:
        anns: List of SAM mask annotations (each with ``segmentation``,
            ``predicted_iou`` and ``area`` keys).
        save_path: Optional path to write the overlay image to. When ``None``
            no image is rendered and the function only returns the filtered list.
        borders: Whether to draw contour borders for each mask in the overlay.
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
            if compute_iou(ann['segmentation'], acc_mask['segmentation']) > 85:
                accept = False
                break
        if accept:
            mask_accept_list.append(ann)

    sorted_anns = sorted(mask_accept_list, key=(lambda x: x['area']), reverse=True)

    if save_path is not None:
        ax = plt.gca()
        ax.set_autoscale_on(True)

        h, w = sorted_anns[0]['segmentation'].shape
        img = np.ones((h, w, 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.9]])
            img[m] = color_mask
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        ax.imshow(img)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    return sorted_anns

def show_mask(mask, ax, obj_id=None, random_color=False):
    """Overlay one mask on a matplotlib axis.

    Args:
        mask: Binary mask with shape [H, W] (or equivalent squeezable shape).
        ax: Target matplotlib axis.
        obj_id: Optional object id used to pick a stable color.
        random_color: Whether to sample a random RGBA color.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.9])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.9])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

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


def shrink_mask(video_segments):
    """Apply morphology to clean masks and push edge confidence apart.

    Args:
        video_segments: Dict[frame_idx, Dict[obj_id, mask/logits]].
    """
    for frame_id in list(video_segments.keys()):
        for obj_id in list(video_segments[frame_id].keys()):
            vanilla_mask = video_segments[frame_id][obj_id] > 0

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_ = (video_segments[frame_id][obj_id] > 0).squeeze()
            mask_ = mask_.astype(np.uint8) * 255
            mask_ = cv2.erode(mask_, kernel,iterations=3)
            mask_ = cv2.dilate(mask_, kernel,iterations=3)
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
    pos = (pos_homo[None,:,None] * mtx.unsqueeze(1)).sum(-1)[...,:3]
    return pos

def norms_mask(norms, cam2world, threshold=0.0):
    """Compute a visibility mask by normal-camera facing direction.

    Args:
        norms: Normal vectors per pixel/point.
        cam2world: Camera-to-world matrix.
        threshold: Cosine threshold for facing criterion.
    """
    lookat = cam2world[:3, :3] @ np.array([0, 0, -1])
    return np.abs(np.dot(norms, lookat)) > threshold

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

def find_most_frequent(tensor):
    """Return mode per row, with fallback when all values are -1.

    Args:
        tensor: Tensor with shape [N, K] storing candidate labels.
    """
    values, _ = torch.mode(tensor, dim=1)

    all_neg_one = (tensor == -1).all(dim=1)
    mask = (values == -1) & (~all_neg_one)

    if mask.any():
        tensor_masked = tensor.masked_fill(tensor == -1, float('nan'))

        second_values, _ = torch.mode(tensor_masked, dim=1)

        values[mask] = second_values[mask]
    
    return values

def lift_2dmask_3d(imgs_mask, depth_imgs, norm_maps, c2ws, fovy_deg, coord, selected_frames, video_segments, mesh, sample_num_per_face, view_id, uuid, ckpt_name=None, face_label=None, prior_keys=None, export_mesh=True, export_root="outputs"):
    """Lift 2D segmentation masks to per-face 3D labels.

    Args:
        imgs_mask: RGBA validity mask tensor for all views.
        depth_imgs: Depth maps for all views.
        norm_maps: Normal maps for all views (reserved, currently unused).
        c2ws: Camera-to-world matrices.
        fovy_deg: Vertical FOV in degrees.
        coord: Sampled 3D points used for lifting.
        selected_frames: View indices used in current lifting pass.
        video_segments: Dict[frame_idx, Dict[obj_id, bool_mask]].
        mesh: Trimesh mesh whose faces are labeled.
        sample_num_per_face: Number of points sampled per face.
        view_id: Export file stem.
        uuid: Object identifier used in export path naming.
        ckpt_name: Optional run tag; when set it adds an extra export folder level.
        face_label: Optional precomputed face labels for postprocess overwrite.
        prior_keys: Optional object ids with prompt-priority in aggregation.
        export_mesh: Whether to export `.glb` and `.npy` files.
        export_root: Output root directory for mesh export.
    """
    if face_label is None:
        all_masks = mask_aggregation(video_segments,prior_keys=prior_keys)
        link = torch.ones([coord.shape[0], 3, len(selected_frames)], dtype=torch.int)
        link[:, 0:3, :] = cal_link(imgs_mask, depth_imgs, c2ws, fovy_deg, coord).permute(1,2,0)
        grid_normalized = (link[:,:-1,:].permute(2,0,1).unsqueeze(-2).to(torch.float32) / (1024 - 1)) * 2 - 1
        pc_labels = F.grid_sample(
            torch.from_numpy(all_masks),
            grid_normalized,
            mode="nearest"
        ).squeeze(1)
        link_ = link.permute(2,0,1)[:,:,-1:].to(torch.bool)
        pc_labels[~link_] = 0

        pc_labels = pc_labels.squeeze().permute(1,0)
        pc_label = mode_except_negative_one(pc_labels.to(torch.int32)).to(torch.from_numpy(all_masks).dtype)

        pc_label = pc_label.reshape(-1, sample_num_per_face)
        face_label = mode_except_negative_one(pc_label.to(torch.int32)).to(torch.from_numpy(all_masks).dtype)

    colors = np.ones((len(mesh.vertices), 3)) * 255
    color_map = {id: np.random.rand(3) * 255 for id in np.unique(face_label)}
    color_map[0] = np.zeros((3)) * 255
    for obj_id in np.unique(face_label):
        colors[mesh.faces[face_label==obj_id].flatten()] = color_map[obj_id][None,:]

    mesh_ = mesh
    fill_rgb = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    colors = np.hstack((colors, fill_rgb))
    mesh_.visual.vertex_colors = np.uint8(colors)
    if export_mesh:
        export_dir = export_root
        if ckpt_name:
            export_dir = os.path.join(export_root, ckpt_name, uuid)
        os.makedirs(export_dir, exist_ok=True)
        glb_path = os.path.join(export_dir, f"{view_id}.glb")
        np.save(glb_path.replace(".glb",".npy"), face_label.cpu().numpy())
        mesh_.export(glb_path)
        print(f"Exported labelled mesh to {glb_path}")
    return face_label

def load_mesh_with_faces(mesh_path):
    """Load mesh and rebuild a face-index-friendly trimesh object.

    Args:
        mesh_path: Input mesh file path.
    """
    mesh = trimesh.load(mesh_path,force="mesh",processed=False)

    mesh_vanilla = mesh
    vertices = mesh.vertices  # Array of vertex coordinates
    faces = mesh.faces        # Array of face indices (each row contains 3 vertex indices)
    vertices = vertices[faces].reshape(-1, 3)

    faces = np.arange(len(faces) * 3).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh, mesh_vanilla

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

def trimesh_to_blender(mesh):
    """Convert trimesh axis convention to Blender convention.

    Args:
        mesh: Trimesh object to be transformed in place.
    """
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0,-1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    mesh.apply_transform(rotation_matrix)
    return mesh

def blender_to_trimesh(mesh):
    """Convert Blender axis convention back to trimesh convention.

    Args:
        mesh: Trimesh object to be transformed in place.
    """
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0,-1, 0, 0],
        [0, 0, 0, 1]])
    mesh.apply_transform(rotation_matrix)
    return mesh

def complete_labels(face_labels, mesh, PA=0.025, smooth_type="knn"):
    """Remove tiny components and fill unlabeled faces.

    Args:
        face_labels: Per-face integer labels.
        mesh: Source mesh with adjacency/geometry.
        PA: Relative threshold for tiny-component removal.
        smooth_type: Smoothing strategy, currently `adjacent` is implemented.
    """
    mesh_graph = defaultdict(set)
    for face1, face2 in mesh.face_adjacency:
        mesh_graph[face1].add(face2)
        mesh_graph[face2].add(face1)

    components = label_components(face_labels, mesh_graph)
    threshold_percentage_size = PA
    threshold_percentage_area = PA
    components = sorted(components, key=lambda x: len(x), reverse=True)
    components_area = [
        sum([float(mesh.area_faces[face]) for face in comp]) for comp in components
    ]
    max_size = max([len(comp) for comp in components])
    max_area = max(components_area)

    remove_comp_size = set()
    remove_comp_area = set()
    for i, comp in enumerate(components):
        if len(comp)          < max_size * threshold_percentage_size:
            remove_comp_size.add(i)
        if components_area[i] < max_area * threshold_percentage_area:
            remove_comp_area.add(i)
    remove_comp = remove_comp_size.intersection(remove_comp_area)
    print(f"Removing {len(remove_comp)} small components")
    for i in remove_comp:
        for face in components[i]:
            face_labels[face]=0

    face_label1 = face_labels.clone()

    if smooth_type=="adjacent":
        smooth_iterations = 64
        iter = 0
        for iteration in range(smooth_iterations):
            iter+=1
            changes = {}
            for face in range(face_labels.shape[0]):
                if face_labels[face] != 0: 
                    continue
                labels_adj = Counter()
                for adj in mesh_graph[face]:
                    if face_labels[adj] != 0:
                        label = face_labels[adj]
                        labels_adj[label] += 1
                if len(labels_adj):
                    changes[face] = labels_adj.most_common(1)[0][0]
    
            for face, label in changes.items():
                face_labels[face] = label

    print("Smoothing labels")
    face_unlable_idx = torch.where(face_labels == 0)[0]
    face_lable_idx = torch.where(face_labels != 0)[0]
    face_centroids = mesh.triangles_center
    unlabel_xyz = face_centroids[face_unlable_idx]
    label_xyz = face_centroids[face_lable_idx]

    face_normals = mesh.face_normals
    unlabel_norm = face_normals[face_unlable_idx]
    label_norm = face_normals[face_lable_idx]

    lambda_norm = 0
    unlabel_xyz = np.concatenate([unlabel_xyz, unlabel_norm * lambda_norm], axis=-1)
    label_xyz = np.concatenate([label_xyz, label_norm * lambda_norm], axis=-1)

    unlabel_top3_indices = find_nearest_three_points(unlabel_xyz, label_xyz) # [N,3]
    nearest_labels = face_labels[face_lable_idx][unlabel_top3_indices]
    most_frequent_labels = torch.mode(nearest_labels, dim=1).values
    face_labels[face_unlable_idx] = most_frequent_labels

    face_label2 = face_labels.clone()

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
    face_label3 = face_labels.clone()

    return face_label1, face_label2, face_label3

def label_components(face_labels: dict, mesh_graph) -> list[set]:
    """Group connected faces that share identical non-zero labels.

    Args:
        face_labels: Per-face labels.
        mesh_graph: Face adjacency graph.
    """
    components = []
    visited = set()

    def dfs(source: int):
        stack = [source]
        components.append({source})
        visited.add(source)
        
        while stack:
            node = stack.pop()
            for adj in mesh_graph[node]:
                if adj not in visited and face_labels[adj]!=0 and face_labels[adj] == face_labels[node]:
                    stack.append(adj)
                    components[-1].add(adj)
                    visited.add(adj)

    for face in range(face_labels.shape[0]):
        if face not in visited and face_labels[face]!=0:
            dfs(face)

    return components

def find_nearest_three_points(A, B):
    """Find indices of the nearest 3 points in B for every point in A.

    Args:
        A: Query points, shape [N, 3], numpy or torch.
        B: Reference points, shape [M, 3], numpy or torch.
    """
    np_array = isinstance(A, np.ndarray)
    if np_array:
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

    distances = torch.cdist(A, B)  # shape [N, M]

    _, indices = torch.topk(distances, k=3, largest=False)  # shape [N, 3]

    ret = indices
    if np_array:
        ret = ret.numpy()

    return ret

def filter_iou(all_seg_result, video_segments, track_id):
    """Filter highly overlapping objects inside and across passes.

    Args:
        all_seg_result: Accumulated segmentation dict by frame/object.
        video_segments: Current pass segmentation dict by frame/object.
        track_id: Anchor frame used for overlap comparison.
    """
    start = __import__('time').time()
    accept_id = []

    for obj_id in list(video_segments[0].keys()):
        accept = True
        for acpt_id in accept_id:
            mask1 = video_segments[track_id][obj_id]
            mask2 = video_segments[track_id][acpt_id]
            iou = np.sum(mask1 & mask2) / (np.sum(mask1 | mask2) + 1e-6)
            if iou > 0.8:
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
            if iou > 0.8 and iou <= 1:
                discard_id.append(obj_id_1)

    discard_id = list(set(discard_id))
    for obj_id in discard_id:
        for _ in range(12):
            video_segments[_].pop(obj_id)

    end = __import__('time').time()
    _ = end - start

    return video_segments, all_seg_result


# ─────────────────────────────────────────────────────────────────────────────
# Boundary-aware label smoothing (added alongside complete_labels' hole-fill).
#
# complete_labels' "adjacent" pass only fills label-0 faces; it never MOVES a
# boundary, so a staircased seam from the per-face vote stays staircased. These
# operate on the FINAL per-face labels and export their own GLB, leaving the
# original pipeline untouched -- they are comparison points, not replacements.
# ─────────────────────────────────────────────────────────────────────────────


def _to_np_int(x) -> np.ndarray:
    """Face-label tensor/array -> contiguous int64 numpy, detached from device."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x).astype(np.int64)


def smooth_labels_icm(face_labels, mesh_vanilla, theta_deg=15.0, lambda_data=0.7,
                      iters=30, dihedral=True):
    """Crease-aware label smoothing by ICM on the face-adjacency graph.

    Potts model: each face keeps the label that maximises
        (sum over neighbours w_fn * [label(n) == L])  +  lambda_data * [L == vote(f)]
    with ``w_fn = exp(-dihedral / theta)`` -- so cutting across a flat surface is
    expensive (the boundary straightens into a minimal line) and cutting on a
    crease is cheap (the boundary snaps to it). ``dihedral=False`` sets every
    weight to 1 (plain all-face majority) as the ablation baseline. Unlike
    complete_labels' "adjacent" pass, this relabels EVERY face, so it actually
    moves boundaries. Runs on ``mesh_vanilla`` (welded topology) -- the exploded
    ``mesh`` has no shared edges and thus no adjacency.

    Args:
        face_labels: final per-face integer labels (tensor or array).
        mesh_vanilla: source mesh with welded vertices (real face adjacency).
        theta_deg: dihedral scale; smaller = only very sharp creases stay cheap.
        lambda_data: pull toward the original vote; 0 = pure smoothing.
        iters: max ICM sweeps (stops early once nothing changes).
        dihedral: weight cuts by crease angle; False = uniform majority.
    """
    lab = _to_np_int(face_labels).copy()
    vote = lab.copy()
    adj = np.asarray(mesh_vanilla.face_adjacency)
    ang = np.abs(np.asarray(mesh_vanilla.face_adjacency_angles))
    if len(adj) == 0:
        return torch.from_numpy(lab)
    w = np.exp(-ang / np.radians(theta_deg)) if dihedral else np.ones(len(ang))
    e0, e1 = adj[:, 0], adj[:, 1]

    uniq = np.unique(vote)
    K = len(uniq)
    lut = np.zeros(int(vote.max()) + 1, dtype=np.int64)
    lut[uniq] = np.arange(K)          # label id -> dense column; labels stay in `uniq`
    vote_col = lut[vote]

    for _ in range(iters):
        agree = np.zeros((len(lab), K), dtype=np.float64)
        np.add.at(agree, (e0, lut[lab[e1]]), w)   # neighbour across each edge votes its label
        np.add.at(agree, (e1, lut[lab[e0]]), w)
        agree[np.arange(len(lab)), vote_col] += lambda_data
        new = uniq[agree.argmax(1)]
        if np.array_equal(new, lab):
            break
        lab = new
    return torch.from_numpy(lab)


def face_vote_histogram(imgs_mask, depth_imgs, c2ws, fovy_deg, coord, video_segments,
                        sample_num, prior_keys=None):
    """Per-face label histogram across all views -- the soft version of
    lift_2dmask_3d's double ``mode()``.

    Same projection (cal_link + grid_sample on the aggregated label maps), but
    instead of collapsing samples x views to one hard label per face, every
    sample's every view keeps its vote. Near seams the views disagree; the
    histogram keeps that uncertainty for a graph-cut to arbitrate, where mode()
    freezes a noisy winner per face.

    Returns:
        (hist [n_faces, K] float tensor, label_ids list) -- column j counts
        votes for label_ids[j]; invisible samples and background vote nowhere.
    """
    all_masks = mask_aggregation(video_segments, prior_keys=prior_keys)
    n_views = all_masks.shape[0]
    link = torch.ones([coord.shape[0], 3, n_views], dtype=torch.int)
    link[:, 0:3, :] = cal_link(imgs_mask, depth_imgs, c2ws, fovy_deg, coord).permute(1, 2, 0)
    grid = (link[:, :-1, :].permute(2, 0, 1).unsqueeze(-2).to(torch.float32) / (1024 - 1)) * 2 - 1
    pc = F.grid_sample(torch.from_numpy(all_masks), grid, mode="nearest").squeeze(1)
    pc[~link.permute(2, 0, 1)[:, :, -1:].to(torch.bool)] = 0
    pc = pc.squeeze().permute(1, 0).to(torch.int64)                  # [n_points, n_views]

    label_ids = sorted(int(i) for i in np.unique(all_masks) if i not in (0, 999))
    lut = torch.zeros(max(label_ids) + 1 if label_ids else 1, dtype=torch.int64)
    for j, i in enumerate(label_ids):
        lut[i] = j + 1                                               # 0 stays "no vote"
    pc = torch.where(pc == 999, torch.zeros_like(pc), pc)
    votes = lut[pc.clamp(max=lut.shape[0] - 1)]
    n_faces = coord.shape[0] // sample_num
    votes = votes.reshape(n_faces, sample_num * n_views)
    hist = torch.zeros(n_faces, len(label_ids) + 1, dtype=torch.float32)
    hist.scatter_add_(1, votes, torch.ones_like(votes, dtype=torch.float32))
    return hist[:, 1:], label_ids


def smooth_labels_graphcut(face_labels, mesh_vanilla, unary=None, label_ids=None,
                           theta_deg=20.0, lam=1.0, sweeps=4):
    """Alpha-expansion Potts graph-cut on the face-adjacency graph (PyMaxflow).

    Same energy family as smooth_labels_icm but optimised globally per expansion
    move instead of greedily per face -- on the ideal-mask benchmark it removes
    ~2-4x more staircase (boundary edges 606->226 on sample_05, 1281->516 on
    sample_04) at unchanged agreement with the 2D masks. Cut cost is edge length
    x exp(-dihedral/theta): boundaries straighten on flat surfaces and snap to
    creases.

    Args:
        face_labels: current per-face labels (data term when ``unary`` is None).
        unary: optional [n_faces, K] cost matrix aligned with ``label_ids``
            (e.g. -log of face_vote_histogram probabilities). None = hard unary
            of ``lam`` for leaving the current label; 0/999 faces get a free
            choice and are absorbed into a neighbouring part.
        label_ids: the labels unary's columns refer to; None = unique current
            labels minus {0, 999}.
    """
    import maxflow  # optional dependency, only this smoother needs it

    lab_in = _to_np_int(face_labels)
    if label_ids is None:
        label_ids = sorted(int(i) for i in np.unique(lab_in) if i not in (0, 999))
    K = len(label_ids)
    n = len(lab_in)
    col = {i: j for j, i in enumerate(label_ids)}

    if unary is None:
        unary_np = np.full((n, K), lam, dtype=np.float64)
        for f, l in enumerate(lab_in):
            if l in col:
                unary_np[f, col[l]] = 0.0
            else:
                unary_np[f, :] = 0.0                                 # unknown: smoothing decides
    else:
        unary_np = np.asarray(unary, dtype=np.float64)

    adj = np.asarray(mesh_vanilla.face_adjacency)
    ang = np.degrees(np.abs(np.asarray(mesh_vanilla.face_adjacency_angles)))
    elen = np.linalg.norm(mesh_vanilla.vertices[mesh_vanilla.face_adjacency_edges[:, 0]]
                          - mesh_vanilla.vertices[mesh_vanilla.face_adjacency_edges[:, 1]], axis=1)
    w = lam * (elen / max(np.median(elen), 1e-12)) * np.exp(-ang / theta_deg)
    e0, e1 = adj[:, 0], adj[:, 1]

    # start from the best hard assignment the unary allows
    lab = np.array([label_ids[j] for j in unary_np.argmin(1)], dtype=np.int64)

    def energy(l):
        u = sum(unary_np[f, col[l[f]]] for f in range(n))
        return u + (w * (l[e0] != l[e1])).sum()

    E = energy(lab)
    for _ in range(sweeps):
        changed = False
        for a in label_ids:
            g = maxflow.Graph[float]()
            nodes = g.add_nodes(n)
            keep = unary_np[np.arange(n), [col[l] for l in lab]]
            take = unary_np[:, col[a]]
            for f in range(n):                                       # source=keep, sink=take alpha
                g.add_tedge(nodes[f], take[f], keep[f])
            same = lab[e0] == lab[e1]
            for i in np.nonzero(same)[0]:
                g.add_edge(nodes[e0[i]], nodes[e1[i]], w[i], w[i])
            for i in np.nonzero(~same)[0]:                           # BVZ auxiliary node
                aux = g.add_nodes(1)[0]
                wa = w[i] if lab[e0[i]] != a else 0.0
                wb = w[i] if lab[e1[i]] != a else 0.0
                g.add_edge(nodes[e0[i]], aux, wa, wa)
                g.add_edge(aux, nodes[e1[i]], wb, wb)
                g.add_tedge(aux, 0, w[i])
            g.maxflow()
            seg = np.array([g.get_segment(nodes[f]) for f in range(n)])
            new = lab.copy()
            new[seg == 1] = a
            En = energy(new)
            if En < E - 1e-9:
                lab, E, changed = new, En, True
        if not changed:
            break
    return torch.from_numpy(lab)


def export_labelled_mesh(mesh_exploded, face_labels, glb_path, color_seed=0):
    """Write a per-face-labelled mesh to ``glb_path`` (+ a sibling ``.npy``).

    Colours are deterministic per label id (seeded RNG), so different smoothing
    methods paint the same part the same colour and are visually comparable --
    unlike lift_2dmask_3d's per-run random palette. Colours the exploded
    ``mesh`` (unique verts per face) so labels don't bleed at shared vertices.
    """
    lab = _to_np_int(face_labels)
    ids = np.unique(lab)
    rng = np.random.default_rng(color_seed)
    cmap = {int(i): rng.random(3) * 255 for i in ids}
    cmap[0] = np.zeros(3)
    colors = np.ones((len(mesh_exploded.vertices), 3)) * 255
    for i in ids:
        colors[mesh_exploded.faces[lab == i].flatten()] = cmap[int(i)][None, :]
    m = mesh_exploded.copy()
    colors = np.hstack((colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)))
    m.visual.vertex_colors = np.uint8(colors)
    os.makedirs(os.path.dirname(glb_path) or ".", exist_ok=True)
    np.save(glb_path.replace(".glb", ".npy"), lab)
    m.export(glb_path)
    print(f"Exported labelled mesh to {glb_path}")