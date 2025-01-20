# repo modules
from src.imagine2touch.utils.utils import cropND
from src.imagine2touch.models.generate_gt_masks import mask_far_pixels
from src.imagine2touch.models.depth_correction_utils import apply_depth_correction
from robot_io.cams.realsense.realsense import Realsense

# standard libraries
import time
import numpy as np
from PIL import Image
import open3d as o3d
import open3d.visualization.gui as gui


def mask(cfg, cam_z_distance, rgb, dep, size):
    target_images, rgb_images = np.expand_dims(dep, 0), np.expand_dims(rgb, 0)
    target_images = np.array(
        [
            cropND(np.array(target_image, dtype=np.uint8), (size[0], size[1]))
            for target_image in target_images
        ],
        dtype=list,
    )
    rgb_images = np.array(
        [
            cropND(np.array(rgb_image, dtype=np.uint8), (size[0], size[1]))
            for rgb_image in rgb_images
        ],
        dtype=list,
    )
    target_images, rgbs_quantized = apply_depth_correction(
        cfg,
        dups_threshold=cam_z_distance - cfg.masks.dups_threshold,
        target_images=target_images,
        rgb_images=rgb_images,
    )
    target_masks = mask_far_pixels(
        cfg, cam_z_distance=cam_z_distance, images=target_images
    )
    return target_masks


def mask_from_depth_mesh(cfg, cam_z_distance, dep, size, view=False):
    target_images = np.expand_dims(dep, 0)
    target_masks_view = np.array(
        [
            cropND(
                np.array(target_image, dtype=np.uint8),
                (min(target_images.shape[1:]), min(target_images.shape[1:])),
            )
            for target_image in target_images
        ],
        dtype=list,
    )
    target_masks_view = mask_far_pixels(
        cfg, cam_z_distance=cam_z_distance, images=target_masks_view, pc=True, view=view
    )
    target_images = np.array(
        [
            cropND(np.array(target_image, dtype=np.uint8), (size[0], size[1]))
            for target_image in target_images
        ],
        dtype=list,
    )
    target_masks = mask_far_pixels(
        cfg, cam_z_distance=cam_z_distance, images=target_images, pc=True, view=False
    )
    return target_masks, target_images, target_masks_view


def update_po(p_object, p_o_update, normalize=False):
    p_object = p_object * p_o_update
    if normalize:
        p_object = p_object / np.sum(p_object)
    return p_object


def normalize_p(p):
    if np.any(p < 0):
        p = p - np.min(p)
    sum = np.sum(p)
    p = np.divide(p, sum)
    return p


def init_viz():
    gui_viz = gui.Application.instance
    gui_viz.initialize()
    # w = gui_viz.create_window("All objects", 1048, 512)
    w = None
    scene1 = gui.SceneWidget()
    scene2 = gui.SceneWidget()
    scene3 = gui.SceneWidget()
    scene4 = gui.SceneWidget()
    scene5 = gui.SceneWidget()
    scenes = [scene1, scene2, scene3, scene4, scene5]
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.base_color = (1.0, 1.0, 1.0, 1.0)
    mat.shader = "defaultLit"

    def on_layout(theme):
        r = w.content_rect
        scene1.frame = gui.Rect(r.x, r.y, r.width / 3, r.height / 2)
        scene2.frame = gui.Rect(r.x + r.width / 3 + 1, r.y, r.width / 3, r.height / 2)
        scene3.frame = gui.Rect(
            r.x + 2 * r.width / 3 + 2, r.y, r.width / 3, r.height / 2
        )
        scene4.frame = gui.Rect(
            r.x, r.y + r.height / 2 + 1, r.width / 2, r.height / 2
        )  # 2nd row
        scene5.frame = gui.Rect(
            r.x + r.width / 2 + 1, r.y + r.height / 2 + 1, r.width / 2, r.height / 2
        )  # 2nd row

    # w.set_on_layout(on_layout)
    return w, scenes, mat, gui_viz


def poll_gui(gui):
    if gui is not None:
        gui.run_one_tick()


def from_pcd_to_mesh(pcd):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # Extract vertices and faces
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)

    # Define z threshold
    z_threshold = 0.005  # Adjust the threshold value as needed

    # Filter vertices below the threshold
    filtered_vertices = vertices[vertices[:, 2] > z_threshold]
    filtered_vertex_indices = np.where(vertices[:, 2] < z_threshold)[0]

    # Find the corresponding faces for the filtered vertices
    filtered_face_indices = np.isin(faces, np.where(vertices[:, 2] > z_threshold)).all(
        axis=1
    )
    filtered_faces = faces[filtered_face_indices]

    # Filter vertex colors based on the filtered vertices
    filtered_vertex_colors = vertex_colors[filtered_vertex_indices]
    # Create a new mesh from the filtered vertices and faces
    mesh.vertices = o3d.utility.Vector3dVector(filtered_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(filtered_faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(filtered_vertex_colors)
    return mesh


def capture_wristcam_image():
    """
    will work only for realsense cameras
    """
    camera = Realsense(img_type="rgb_depth")
    rgb_w, depth_w = camera.get_image()
    del camera
    time.sleep(1)  # ensure camera process terminated
    return rgb_w, depth_w


def viz_PC(PC, sampled_point, normal_in_object):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=1080)
    for pc in PC:
        vis.add_geometry(pc)
    render_option = vis.get_render_option()
    render_option.point_size = 10
    ctr = vis.get_view_control()
    ctr.set_zoom(1.1)
    ctr.set_lookat([0, 0, 0])  # Look-at point (x, y, z)
    # rotate normal in object vector 45 degrees aroun x-axis
    ctr.set_up([0, 1, 0])
    ctr.set_front([0, 0, 1])
    ctr.set_constant_z_far(10)
    ctr.set_constant_z_near(0.01)
    vis.run()
    image = vis.capture_screen_float_buffer(do_render=True)
    # normalize to [0,255]
    image = (np.asarray(image) * 255).astype(np.uint8)
    # add alpha channel to make the image transparent
    image = np.concatenate(
        [image, np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)],
        axis=2,
    )
    # convert to open3d image with alpha channel
    image = o3d.geometry.Image(image)
    # destroy the visualizer
    vis.destroy_window()
    return vis
