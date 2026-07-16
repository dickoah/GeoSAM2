import bpy
import sys
import os
import glob
import math
import json
import random
import shutil
import mathutils
import bpy_extras
import numpy as np
from mathutils import Vector
from mathutils import Matrix, Vector, Euler
from typing import List, Optional, Literal, Tuple, Dict


META_FILENAME = "meta.json"
RESOLUTION = 1024
RENDER_SAMPLES = 64
ENGINE = "EEVEE"
CLOSE_SHADOW = False
RENDER_DEPTH = True
RENDER_NORMAL = True

if os.environ.get("ENGINE"):
    ENGINE = os.environ.get("ENGINE")

if os.environ.get("RENDER_SAMPLES"):
    RENDER_SAMPLES = int(os.environ.get("RENDER_SAMPLES"))

if os.environ.get("RESOLUTION"):
    RESOLUTION = int(os.environ.get("RESOLUTION"))

def generate_random(left, right):
    while True:
        val = random.gauss(0, (right-left)/5)
        if val >= left and val <= right:
            return val


def generate_frames(name):
    render_type_list = [
        {'name': "depth", "suffix": ".exr", "enable": RENDER_DEPTH},
        {'name': "render_opaque", "suffix": ".webp",
            "enable": RENDER_DEPTH},
    ]
    param = [
        {
            "type": "render",
            "name": "{}",
            "height": RESOLUTION,
            "width": RESOLUTION,
        }
    ]
    variables = ["render_"+name+".webp"]

    for render_type in render_type_list:
        if render_type["enable"]:
            param.append({
                "type": render_type["name"],
                "name": "{}",
                "height": RESOLUTION,
                "width": RESOLUTION,
            })
            variables.append(render_type["name"] +
                             "_"+name+render_type["suffix"])

    for i in range(len(variables)):
        param[i]["name"] = variables[i]
    return param


def build_transformation_mat(translation,
                             rotation) -> np.ndarray:
    """ Build a transformation matrix from translation and rotation parts.

    :param translation: A (3,) vector representing the translation part.
    :param rotation: A 3x3 rotation matrix or Euler angles of shape (3,).
    :return: The 4x4 transformation matrix.
    """
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(
            f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector.")
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(Euler(rotation).to_matrix())
    else:
        raise RuntimeError(f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
                           f"(3,3) or Euler angles of shape (3,) or (3,1).")

    return mat


def reset_keyframes() -> None:
    """ Removes registered keyframes from all objects and resets frame_start and frame_end """
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0
    for a in bpy.data.actions:
        bpy.data.actions.remove(a)


def get_local2world_mat(blender_obj) -> np.ndarray:
    """ Returns the pose of the object in the form of a local2world matrix.
    :return: The 4x4 local2world matrix.
    """
    obj = blender_obj
    # Start with local2parent matrix (if obj has no parent, that equals local2world)
    matrix_world = obj.matrix_basis

    # Go up the scene graph along all parents
    while obj.parent is not None:
        # Add transformation to parent frame
        matrix_world = obj.parent.matrix_basis @ obj.matrix_parent_inverse @ matrix_world
        obj = obj.parent

    return np.array(matrix_world)


def add_camera(cam2world_matrix,camera_params) -> int:
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)

    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam_ob = bpy.context.object
    cam_ob.matrix_world = cam2world_matrix
    cam_ob_data = cam_ob.data
    cam_ob_data.type = camera_params['camera_type']
    cam_ob_data.sensor_width = camera_params['camera_sensor_width']
    if camera_params['camera_type'] == 'ORTHO':
        cam_ob_data.ortho_scale = camera_params['camera_ortho_scale']
    elif camera_params['camera_type'] == 'PERSP':
        cam_ob_data.lens = camera_params['camera_lens']


def add_camera_pose(cam2world_matrix, camera_params) -> int:
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)

    cam_ob = bpy.context.scene.camera
    cam_ob.matrix_world = cam2world_matrix
    cam_ob_data = cam_ob.data
    cam_ob_data.type = camera_params['camera_type']
    cam_ob_data.sensor_width = camera_params['camera_sensor_width']
    if camera_params['camera_type'] == 'ORTHO':
        cam_ob_data.ortho_scale = camera_params['camera_ortho_scale']
    elif camera_params['camera_type'] == 'PERSP':
        cam_ob_data.lens = camera_params['camera_lens']

    frame = bpy.context.scene.frame_end
    if bpy.context.scene.frame_end < frame + 1:
        bpy.context.scene.frame_end = frame + 1

    cam_ob.keyframe_insert(data_path='location', frame=frame)
    cam_ob.keyframe_insert(data_path='rotation_euler', frame=frame)
    cam_ob_data.keyframe_insert(data_path='type', frame=frame)
    cam_ob_data.keyframe_insert(data_path='sensor_width', frame=frame)

    if camera_params['camera_type'] == 'ORTHO':
        cam_ob_data.keyframe_insert(data_path='ortho_scale', frame=frame)
    elif camera_params['camera_type'] == 'PERSP':
        cam_ob_data.keyframe_insert(data_path='lens', frame=frame)
    return frame


def clear_normal_map():
    for material in bpy.data.materials:
        material.use_nodes = True
        node_tree = material.node_tree
        try:
            bsdf = node_tree.nodes["Principled BSDF"]
            if bsdf.inputs["Normal"].is_linked:
                for link in bsdf.inputs["Normal"].links:
                    node_tree.links.remove(link)
        except:
            pass


def enable_depth_output(output_dir: Optional[str] = '', file_prefix: str = "depth_"):

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    links = tree.links

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new('CompositorNodeRLayers')
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_z = True
    bpy.context.view_layer.use_pass_normal = True

    depth_output = tree.nodes.new('CompositorNodeOutputFile')
    depth_output.base_path = output_dir
    depth_output.name = 'DepthOutput'
    depth_output.format.file_format = 'OPEN_EXR'
    depth_output.format.color_depth = '32'
    depth_output.file_slots.values()[0].path = file_prefix

    links.new(rl.outputs["Depth"], depth_output.inputs['Image'])

def make_normal_to_rgb_node_group(node_tree, editor_type="Compositor"):
    """Create a node group that converts normal vectors to RGB colors for visualization.

    Args:
        node_tree: The node tree to add nodes to
        editor_type: Either "Shader" or "Compositor"

    Returns:
        Tuple of (input_socket, output_socket) for the conversion
    """
    link = lambda from_socket, to_socket: node_tree.links.new(from_socket, to_socket)
    sep_color_node = node_tree.nodes.new(f"{editor_type}NodeSeparateColor")

    def create_normal_to_rgb_map_node():
        node = node_tree.nodes.new(f"{editor_type}NodeMapRange")
        if editor_type == "Shader":
            node.clamp = True  # Clamp
        elif editor_type == "Compositor":
            node.use_clamp = True  # Clamp
        node.inputs["From Min"].default_value = -1.0  # From Min
        node.inputs["From Max"].default_value = 1.0  # From Max
        node.inputs["To Min"].default_value = 0.0  # To Min
        node.inputs["To Max"].default_value = 1.0  # To Max
        return node

    if editor_type == "Shader":
        map_range_node_output_socket_name = "Result"
        converter_io_node_socket_name = "Color"
    elif editor_type == "Compositor":
        map_range_node_output_socket_name = "Value"
        converter_io_node_socket_name = "Image"

    map_range_nodes = {k: create_normal_to_rgb_map_node() for k in ["R", "G", "B"]}
    comb_color_node = node_tree.nodes.new(f"{editor_type}NodeCombineColor")

    link(sep_color_node.outputs["Red"], map_range_nodes["R"].inputs["Value"])
    link(sep_color_node.outputs["Green"], map_range_nodes["G"].inputs["Value"])
    link(sep_color_node.outputs["Blue"], map_range_nodes["B"].inputs["Value"])

    link(
        map_range_nodes["R"].outputs[map_range_node_output_socket_name],
        comb_color_node.inputs["Red"],
    )
    link(
        map_range_nodes["G"].outputs[map_range_node_output_socket_name],
        comb_color_node.inputs["Green"],
    )
    link(
        map_range_nodes["B"].outputs[map_range_node_output_socket_name],
        comb_color_node.inputs["Blue"],
    )

    # return (input socket, output socket)
    return (
        sep_color_node.inputs[converter_io_node_socket_name],
        comb_color_node.outputs[converter_io_node_socket_name],
    )


def set_file_output_non_color(node):
    """Set file output node to use non-color data management."""
    if int(bpy.app.version_string[0]) >= 4:
        node.format.color_management = "OVERRIDE"
        node.format.view_settings.view_transform = "Raw"
    else:
        node.format.color_management = "OVERRIDE"
        node.format.display_settings.display_device = "None"


def enable_normals_output(
    output_dir: Optional[str] = "",
    file_prefix: str = "normal_",
    use_rgb_conversion: bool = True,
    file_format: Literal["OPEN_EXR", "WEBP", "PNG"] = "WEBP",
):
    """
    Enable normal output in pure world-space coordinates without any transformations.
    This is the simplest version that directly outputs Blender's world-space normals.

    When use_rgb_conversion=True and file_format in ["WEBP", "PNG"]:
    - World-space normals in range [-1, 1] are mapped to RGB values in range [0, 1]
    - This allows visualization in standard image formats while preserving normal data
    - To recover normals: (rgb_values * 2 - 1) gives world-space normals in [-1, 1] range

    Args:
        output_dir: Output directory for normal files
        file_prefix: Prefix for output files
        use_rgb_conversion: Whether to convert normals to RGB for better visualization
        file_format: Output file format
    """
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.scene.render.use_compositing = True

    normal_file_output = tree.nodes.new("CompositorNodeOutputFile")
    normal_file_output.base_path = output_dir
    normal_file_output.location.x = 400
    normal_file_output.file_slots.values()[0].path = file_prefix

    if use_rgb_conversion and file_format in ["WEBP", "PNG"]:
        # Convert normals to RGB for better visualization
        normal_trans_input_socket, normal_trans_output_socket = (
            make_normal_to_rgb_node_group(tree, editor_type="Compositor")
        )

        # Set alpha channel
        set_normal_alpha_node = tree.nodes.new("CompositorNodeSetAlpha")
        set_normal_alpha_node.mode = "REPLACE_ALPHA"

        # Connect nodes
        tree.links.new(rl.outputs["Normal"], normal_trans_input_socket)
        tree.links.new(
            normal_trans_output_socket, set_normal_alpha_node.inputs["Image"]
        )
        tree.links.new(rl.outputs["Alpha"], set_normal_alpha_node.inputs["Alpha"])
        tree.links.new(
            set_normal_alpha_node.outputs["Image"], normal_file_output.inputs["Image"]
        )

        # Configure output format
        if file_format == "WEBP":
            normal_file_output.format.file_format = "WEBP"
            normal_file_output.format.quality = 100
            normal_file_output.format.color_depth = "8"
        elif file_format == "PNG":
            normal_file_output.format.file_format = "PNG"
            normal_file_output.format.color_depth = "16"

        set_file_output_non_color(normal_file_output)

    else:
        # Direct output for EXR or raw normal data
        tree.links.new(rl.outputs["Normal"], normal_file_output.inputs["Image"])

        if file_format == "OPEN_EXR":
            normal_file_output.format.file_format = "OPEN_EXR"
            normal_file_output.format.color_mode = "RGBA"
            normal_file_output.format.color_depth = "32"


def get_keypoint_data(keypoint_names: Optional[List] = None):
    # Set keypoint colors
    keypoint_colors = {}
    if keypoint_names is None:
        for obj in bpy.context.scene.objects:
            if obj.type == "ARMATURE":
                for i, bone in enumerate(obj.pose.bones):
                    keypoint_colors[bone.name.lower().split(":")[-1]] = PRESET_COLORS[
                        i % len(PRESET_COLORS)
                    ]
    elif isinstance(keypoint_names, dict):
        keypoint_colors = keypoint_names
    elif isinstance(keypoint_names, list):
        keypoint_colors = {
            keypoint_names[i]: PRESET_COLORS[i] for i in range(len(keypoint_names))
        }
    else:
        raise ValueError("keypoint_names must be a list or dictionary")

    # Get camera and scene dimensions
    camera = bpy.context.scene.camera
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    # Store keypoint data for all bones
    keypoint_data = {}

    # Process each armature in the scene
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            for bone in obj.pose.bones:
                if not bone.name.lower().split(":")[-1] in keypoint_colors.keys():
                    continue

                # Get bone head in world coordinates
                head_world = obj.matrix_world @ bone.head

                # Project to 2D using Blender's projection
                head_2d = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, camera, head_world
                )

                # Convert to screen coordinates
                head_screen = (int(head_2d.x * width), int((1 - head_2d.y) * height))

                # get color
                keypoint_color = keypoint_colors.get(
                    bone.name.lower().split(":")[-1], list(keypoint_colors.values())[0]
                )

                # Get parent bone name if exists
                parent_name = bone.parent.name if bone.parent else None

                # Store data for this bone
                keypoint_data[bone.name] = {
                    "head_3d": list(head_world),
                    "head_2d": head_screen,
                    "color": keypoint_color,
                    "parent": parent_name,
                }

    return keypoint_data

def render():
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    links = tree.links

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new('CompositorNodeRLayers')
    else:
        rl = tree.nodes["Render Layers"]
    if bpy.context.scene.frame_end != bpy.context.scene.frame_start:
        bpy.context.scene.frame_end -= 1
        bpy.ops.render.render(animation=True, write_still=True)
        bpy.context.scene.frame_end += 1
    else:
        raise RuntimeError("No camera poses have been registered, therefore nothing can be rendered. A camera "
                           "pose can be registered via bproc.camera.add_camera_pose().")


def convert_position(location, center):
    position = ""
    axis = ['x', 'y', 'z']
    sub = location-center
    for i in range(len(axis)):
        if sub[i] > 0:
            position = position + "+" + axis[i]
        elif sub[i] < 0:
            position = position + "-" + axis[i]
    return position


def set_color_output(output_dir: Optional[str] = '', file_prefix: str = "color_"):
    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.use_nodes = True
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.image_settings.file_format = 'WEBP'
    scene.render.image_settings.quality = 100
    scene.render.image_settings.color_mode = 'RGBA'
    # scene.render.image_settings.color_depth = '16'
    scene.render.film_transparent = True
    scene.render.filepath = os.path.join(output_dir, file_prefix)
    pass


def eevee_init():
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = RENDER_SAMPLES
    if CLOSE_SHADOW == False:
        # EEVEE Next (Blender 4.2+) folded ambient occlusion and screen-space
        # reflections into its raytracing settings and dropped these flags. They
        # only affect the shaded colour pass, which the model never reads -- it
        # consumes the normal and depth passes, and only the colour alpha is used
        # (as a visibility mask). So skip them rather than fail the render.
        for flag in ("use_gtao", "use_ssr"):
            if hasattr(bpy.context.scene.eevee, flag):
                setattr(bpy.context.scene.eevee, flag, True)
    bpy.context.scene.render.use_high_quality_normals = True


def clear_scene(NOT_CLEAR_LIGHT=False):
    bpy.ops.object.select_all(action="DESELECT")
    if NOT_CLEAR_LIGHT:
        for obj in bpy.data.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
    else:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree

    for node in node_tree.nodes:
        node_tree.nodes.remove(node)
    reset_keyframes()


def import_models(filepath, types):
    if types == "glb":
        bpy.ops.import_scene.gltf(
            filepath=filepath)
    elif types == "obj":
        forward_axis = os.environ.get('FORWARD_AXIS', 'NEGATIVE_Z')
        up_axis = os.environ.get('UP_AXIS', 'Y')
        bpy.ops.wm.obj_import(filepath=filepath, directory=os.path.dirname(filepath),forward_axis=forward_axis, up_axis=up_axis)


def rotation_matrix(x_left, x_right, y_left, y_right):
    x_rotation = math.radians(generate_random(x_left, x_right))
    y_rotation = math.radians(generate_random(y_left, y_right))
    x_rotation_matrix = mathutils.Matrix.Rotation(x_rotation, 4, 'X')
    y_rotation_matrix = mathutils.Matrix.Rotation(y_rotation, 4, 'Y')
    final_rotation_matrix = y_rotation_matrix @ x_rotation_matrix
    return final_rotation_matrix


def scene_bbox(objects=None, ignore_small_obj=False, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in objects:
        if max(obj.dimensions * 100) < 0.1 and ignore_small_obj:
            print(f"Skipping tiny object '{obj.name}' (max dim = {max(obj.dimensions * 100):.4f})")
            continue
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            
            bbox_min = Vector(
                (min(bbox_min[i], coord[i]) for i in range(3)))
            bbox_max = Vector(
                (max(bbox_max[i], coord[i]) for i in range(3)))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def set_global_light(env_light=0.5):
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes["Background"]
    back_node.inputs["Color"].default_value = Vector(
        [env_light, env_light, env_light, 1.0]
    )
    back_node.inputs["Strength"].default_value = 1.0


def normalize_scene(normalization_range, objects):
    bpy.ops.object.empty_add(type='PLAIN_AXES')
    root_object = bpy.context.object
    for obj in scene_root_objects():
        if obj != root_object:
            _matrix_world = obj.matrix_world.copy()
            obj.parent = root_object
            obj.matrix_world = _matrix_world
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(objects)    
    scale = normalization_range / max(bbox_max - bbox_min)
    root_object.scale *= scale
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(objects,True)
    mesh_offset = - (bbox_min + bbox_max) / 2
    root_object.matrix_local.translation = mesh_offset
    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action="DESELECT")
    return root_object, bbox_max - bbox_min, scale, mesh_offset


def compute_bounding_box(mesh_objects):
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in mesh_objects:
        matrix_world = obj.matrix_world
        mesh = obj.data

        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co

            min_coords = Vector(
                (min(min_coords[i], global_coord[i]) for i in range(3)))
            max_coords = Vector(
                (max(max_coords[i], global_coord[i]) for i in range(3)))

    bbox_center = (min_coords + max_coords) / 2
    bbox_size = max_coords - min_coords

    return bbox_center, bbox_size


def change_material_blend_mode():
    for material in bpy.data.materials:
        material.use_nodes = True
        node_tree = material.node_tree
        material.blend_method = 'OPAQUE'

def change_material_blend_show_transparent(value):
    for material in bpy.data.materials:
        material.use_nodes = True
        if material.blend_method == 'BLEND':
            material.show_transparent_back = value


def get_random_points_on_sphere(center, radius, num_points):
    points = []
    for i in range(18):

        r = radius
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, 0.5*math.pi)

        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)

        flag = -1 if bool(random.randint(0, 1)) else 1
        points.append(Vector((x, y, z*flag)))
    for i in range(12):

        r = radius * 0.5
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, 0.5*math.pi)

        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)

        flag = -1 if bool(random.randint(0, 1)) else 1
        points.append(Vector((x, y, z*flag)))

    return points


def get_solid_points_on_sphere(center, radius):
    points = []
    elev_list = [0, 25, 0, -25, 0, 25, 0, -25, 0, 25, 0, -25,]
    azim_list = [180., 210., 240., 270., 300., 330., 0., 30., 60., 90., 120., 150., ]

    for i in range(len(azim_list)):
        x = center[0] + radius * math.cos(math.radians(elev_list[i])) * math.cos(math.radians(azim_list[i]))
        y = center[1] + radius * math.cos(math.radians(elev_list[i])) * math.sin(math.radians(azim_list[i]))
        z = center[2] + radius * math.sin(math.radians(elev_list[i]))
        points.append(Vector((x, y, z)))

    return points


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def process(filepath, types, output_path):
    
    random.seed()
    eevee_init()
    clear_scene()
    import_models(filepath, types)
    reset_keyframes()

    bpy.ops.object.select_by_type(type='MESH')
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(filepath, os.path.join(output_path, f"mesh.{types}"))

    mesh_objects = []  
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.visible_get() == True and obj.hide_get() == False:
            mesh_objects.append(obj)
            bpy.ops.object.select_all(action="DESELECT")
            bpy.ops.object.select_pattern(pattern=obj.name)

    for obj in mesh_objects:
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = np.deg2rad(30)

    for obj in bpy.data.objects:
        if obj.animation_data is not None:
            obj.animation_data_clear()

    clear_normal_map()
    change_material_blend_show_transparent(False)


    normalization_range = 1.0
    root_object, bbox_size, scale, mesh_offset = normalize_scene(normalization_range,mesh_objects)
    bpy.context.view_layer.update()
    root_object.rotation_euler[2] = math.radians(int(os.environ.get("FORCE_ROTATION", 0)))
    bbox_center = Vector((0,0,0))

    bpy.ops.object.camera_add(location=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object


    default_camera_lens = 50
    default_camera_senser_width = 36
    default_camera_ortho_scale = 1.4

    ratio = 1
    distance = ratio * default_camera_lens / default_camera_senser_width * \
        math.sqrt(bbox_size.x**2 + bbox_size.y**2+bbox_size.z**2)
    idx = 0

    env_texture = "null"
    set_global_light(env_light=0.5)

    camera_angle_x = 2.0*math.atan(default_camera_senser_width/2/default_camera_lens)
    out_data = {
        'camera_angle_x': camera_angle_x,
        'camera_lens': default_camera_lens,
        'sensor_width': default_camera_senser_width,
        'env_texture': env_texture,
        'bbox_size': list(bbox_size),
        'scaling_factor': scale,
        'translation': list(mesh_offset),
        'transforms': []
    }

    parent_matrix_list = [
        rotation_matrix(0, 0, 0, 0),
    ]

    camera_locations = get_solid_points_on_sphere(
        bbox_center, distance)

    positon_tag = [convert_position(camera_location,bbox_center) for camera_location in camera_locations]

    for parent_matrix in parent_matrix_list:
        camera_idx = 0

        for camera_location in camera_locations:
            _lens = 50
            _camera_location = camera_location * \
                (_lens / default_camera_lens)
            _rotation_euler = (
                bbox_center - _camera_location).to_track_quat('-Z', 'Y').to_euler()
            cam_matrix = build_transformation_mat(
                _camera_location, _rotation_euler)
            cam_matrix = listify_matrix(parent_matrix) @ cam_matrix
            camera_params = {
                'camera_type': 'PERSP',
                'camera_lens': _lens,
                'camera_sensor_width': default_camera_senser_width,
            }
            # add_camera(cam_matrix,camera_params)
            add_camera_pose(cam_matrix, camera_params)
            index = "{0:04d}".format(idx)
            out_data['transforms'].append(listify_matrix(cam_matrix))
            idx += 1
            camera_idx += 1

    set_color_output(output_dir=output_path)
    render()

    render_opaque_flag = RENDER_DEPTH
    if render_opaque_flag:
        change_material_blend_mode()
        set_color_output(output_dir=output_path, file_prefix="render_opaque_")

    if RENDER_DEPTH:
        enable_depth_output(output_dir=output_path)
    if RENDER_NORMAL:
        enable_normals_output(output_dir=output_path)
    if render_opaque_flag:
        render()

    with open(os.path.join(output_path, META_FILENAME), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

    file_prefix = "render_opaque_"
    pattern = os.path.join(output_path, f'{file_prefix}*')
    files_to_delete = glob.glob(pattern)
    for file in files_to_delete:
        os.remove(file)


if __name__ == "__main__":
    # Blender forwards script arguments after `--`, so the user-supplied
    # arguments start at sys.argv[4] (script path is at sys.argv[3]).
    if len(sys.argv) < 7:
        print(
            "Usage: blender -b -P geosam2_render.py "
            "<mesh_path> <mesh_type> <output_dir>\n"
            "Example: blender -b -P geosam2_render.py /abs/path/to/mesh.glb glb ./example/out"
        )
        sys.exit(1)
    mesh_path = sys.argv[4]
    types = sys.argv[5]
    output_path = sys.argv[6]
    ret = process(mesh_path, types, output_path)
    sys.exit(0 if ret else 1)