# repo modules
import natsort
import re
from robot_io.utils.utils import pos_orn_to_matrix
from robot_io.cams.realsense.realsense import Realsense
from src.imagine2touch.utils.utils import (
    create_rotation_from_normal,
    create_homog_transformation_from_rotation,
    inverse_transform,
    point_cloud_info,
    ROBOT_IN_WORLD,
)

# Standard useful libraries
import numpy as np
import open3d as o3d
import os
import time


def capture_wristcam_image():
    """
    will work only for realsense cameras
    """
    camera = Realsense(img_type="rgb_depth")
    rgb_w, depth_w = camera.get_image()
    del camera
    time.sleep(1)  # ensure camera process terminated
    return rgb_w, depth_w


def custom_visualization_pcd(
    pcd,
    points,
    discretized_normal_array_negative,
    discretized_normal_array_positive,
    normal_safety_condition,
    safe_patch,
    unsafe_patch,
    safe_patch_center,
    safe_points=None,
):
    # color areas and draw geometries for visualization#
    points = np.append(points, discretized_normal_array_negative, axis=0)
    points = np.append(points, discretized_normal_array_positive, axis=0)
    colors = np.asarray(pcd.colors)
    colors_normal_negative = np.zeros((10, 3))  # black
    if normal_safety_condition:
        colors_normal_positive = np.zeros((10, 3)) + np.array([0, 1, 1])  # indigo
    else:
        colors_normal_positive = np.zeros((10, 3)) + np.array([1, 0, 0])  # red
    colors = np.append(colors, colors_normal_negative, axis=0)
    colors = np.append(colors, colors_normal_positive, axis=0)
    pcd_viz = o3d.geometry.PointCloud()
    pcd_viz.points = o3d.utility.Vector3dVector(points)
    color_safe = np.zeros((safe_patch[0].shape[0], 3)) + np.array([0, 1, 0])  # green
    color_unsafe = np.zeros((unsafe_patch[0].shape[0], 3)) + np.array([1, 0, 0])  # red

    colors[safe_patch] = color_safe
    colors[unsafe_patch] = color_unsafe
    point = safe_points[safe_patch_center]
    patch_center = np.where(points == point)[0][0]
    colors[patch_center] = np.array(
        [1, 1, 0]
    )  # yellow, works only if the index was from the same PCD
    pcd_viz.colors = o3d.utility.Vector3dVector(colors)
    return pcd_viz


def discretize_vector(point, vector, step, N):
    """
    -point: initial point(tail) of the vector
    -vector: target vector
    -step: unit step in meters
    -N: number of points to discretize
    """
    discretized_array_negative = np.zeros((N, 3))
    discretized_array_positive = np.zeros((N, 3))
    next_point = point  # tail of the vector
    for i in range(10):
        next_point = next_point - step * vector
        discretized_array_negative[i] = next_point
    next_point = point  # tail of the negative vector
    for i in range(10):
        next_point = next_point + step * vector
        discretized_array_positive[i] = next_point
    return discretized_array_negative, discretized_array_positive


def safety_one(orientation, tol):
    """
    Don't allow orientation vectors that approach 90 degrees angle with the up vector
    -orientation: target vector
    -tol: minimum cosine value between the up and the orientation vectors
    """
    z_in_world = np.array([0, 0, 1])
    condition_1 = orientation.dot(z_in_world) > tol
    return condition_1


def safety_two(pcd, point, nnn, sl, z_tol):
    """
    Don't allow target points, that you will encounter obstacles on their path
    point: target 3d point
    -nnn: normalized negative normal, generally a linear path vector
    -sl: patch square side length around the target point
    -z_tol: maximum tolerance to higher points in z-direction in meters
    """
    s_pcd = o3d.geometry.PointCloud(pcd)  # new point cloud for processing safety
    s_pcd.translate(
        -1 * point
    )  # translate points from "origin of world" to "origin of target patch"
    s_pcd.transform(
        create_homog_transformation_from_rotation(create_rotation_from_normal(nnn))
    )
    xs, ys, zs = point_cloud_info(s_pcd)
    condition_x = (xs < sl) & (xs > -sl)
    condition_y = (ys < sl) & (ys > -sl)
    condition_z = zs > z_tol
    counter_condition_z = zs <= z_tol
    not_safe_two = np.where(
        condition_x & condition_y & condition_z
    )  # unsafe points array
    safe_two = np.where(
        condition_x & condition_y & counter_condition_z
    )  # safe points array
    unsafe_length = not_safe_two[0].shape[0]
    condition_2 = unsafe_length == 0
    return not_safe_two, safe_two, condition_2


def safety_three(goal_point, goal_rot, flange_dimensions, threshold):
    """
    make sure the extreme positions of a cuboid shape around the robot end effector is higher than a minimum distance in the robot up direction
    -flange_dimensions: cuboid dimensions
    -threshold: minimum threshold in robot z (up) direction in meters
    """
    x = flange_dimensions[0]
    y = flange_dimensions[1]
    z = flange_dimensions[2]
    goal_tcp_in_robot = pos_orn_to_matrix(goal_point, goal_rot)
    P1_IN_ROBOT = goal_tcp_in_robot.dot(np.array([x, y, -z, 1]))[:4]
    P2_IN_ROBOT = goal_tcp_in_robot.dot(np.array([-x, y, -z, 1]))[:4]
    P3_IN_ROBOT = goal_tcp_in_robot.dot(np.array([x, -y, -z, 1]))[:4]
    P4_IN_ROBOT = goal_tcp_in_robot.dot(np.array([-x, -y, -z, 1]))[:4]
    P = np.vstack((P1_IN_ROBOT, P2_IN_ROBOT, P3_IN_ROBOT, P4_IN_ROBOT)).T
    condition_3 = min(ROBOT_IN_WORLD.dot(P)[2, :]) >= threshold
    return condition_3


def set_up_directory(cfg):
    object_folder = f"{cfg.experiment_directory}/{cfg.object_name}"
    object_images_folder = os.path.join(object_folder, f"{cfg.object_name}_images")
    rgb_folder = os.path.join(object_images_folder, "rgb")
    depth_folder = os.path.join(object_images_folder, "depth")
    poses_folder = os.path.join(object_folder, f"{cfg.object_name}_poses")
    forces_folder = os.path.join(object_folder, f"{cfg.object_name}_forces")
    tactile_folder = os.path.join(object_folder, f"{cfg.object_name}_tactile")
    meta_folder = os.path.join(object_folder, f"meta_data")
    pcds_folder = os.path.join(object_folder, f"pcds")
    folders = [
        object_folder,
        object_images_folder,
        rgb_folder,
        depth_folder,
        poses_folder,
        forces_folder,
        tactile_folder,
        meta_folder,
        pcds_folder,
    ]
    for folder in folders:
        if os.path.exists(folder):
            pass
        else:
            print(f"didn't find {folder}, creating it ..")
            os.makedirs(folder)


def pairwise_registration(source, target, voxel_size):
    print("Apply point-to-plane ICP")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_coarse,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transformation_icp = icp_fine.transformation
    information_icp = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine, icp_fine.transformation
        )
    )
    return transformation_icp, information_icp


def full_registration(
    pcds,
    voxel_size,
):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], voxel_size
            )
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False,
                    )
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    return pose_graph


def log_experiment_meta_data(cfg):
    filename = f"{cfg.experiment_directory}/{cfg.object_name}/meta_data/{cfg.object_name}_experiment_{cfg.experiment_number}.txt"
    with open(filename, "w") as f:
        f.write(f"lighting: {cfg.lighting}\n")
        f.write(f"normal_looking_distance: {cfg.normal_looking_distance}\n")
        f.write(f"pcd_offset: {cfg.pcd_offset}\n")
        f.write(f"number_of_points: {cfg.number_of_points}\n")
        f.write(f"termination: {cfg.termination}\n")
        f.write(f"sensor_id: {cfg.sensor_id}\n")
        f.write(f"pad_id: {cfg.pad_id}\n")
        f.write(f"potential problems: {cfg.potential_problems}\n")
        f.write(f"maximum_torque_threshold_contact: {cfg.T_max_contact}\n")
        f.write(f"maximum_force_threshold_contact : {cfg.f_max_contact}\n")
        f.write(f"maximum_torque_threshold_away: {cfg.T_max_away}\n")
        f.write(f"maximum_force_threshold_away: {cfg.f_max_away}\n")
        f.write(f"reskin_control_threshold: {cfg.reskin_threshold}\n")
        f.write(f"movement_wrapper_step_size: {cfg.movement_wrapper_step_size}\n")
        f.write(f"movement_wrapper_dt: {cfg.movement_wrapper_dt}\n")
        f.write(f"pcd crop: {cfg.crop}\n")
        f.write(
            f"franka_exception_pos_tolerance: {cfg.franka_exception_pos_tolerance}\n"
        )
        f.write(
            f"camera_looking_safety_tolerance_in_world: {cfg.camera_looking_safety_tolerance_in_world}\n"
        )
        f.write(
            f"reskin_ambient_recording_distance_to_contact_threshold: {cfg.reskin_ambient_recording_distance_to_contact_threshold}\n"
        )
        f.write(f"ambient_every_reading: {cfg.ambient_every_reading}\n")
        f.write(f"safety: {cfg.safety}\n")


# custom utilities
def get_spatial_info(type, dir, object, state):
    """
    parse saved spatial information from multiple experiments of the same object

    type: spatial information type; position or rotation
    dir: directory to find the information file w/o a trailing forward slash
    object: name of the target object
    state: state of the robot when recording the information; observing "prepare_contact" or contacting "final_contact"
    """
    spatial_types = ["position", "rotation"]
    if type not in spatial_types:
        raise ValueError("Invalid spatial type. Expected one of: %s" % spatial_types)
    state_types = ["prepare_contact", "final_contact", "irrelevant"]
    if state not in state_types:
        raise ValueError("Invalid state type. Expected one of: %s" % state_types)
    experiments = []
    path = f"{dir}/{object}/{object}_poses"
    if type == "position":
        regex = re.compile(f"experiment_.*_{state}$")
    else:
        regex = re.compile("experiment_.*_rotations$")
    for root, dirs, files in os.walk(path):
        for file in files:
            if regex.match(file):
                experiments.append(path + "/" + file)
    poses = np.load(experiments[0], allow_pickle=True)
    for counter, experiment in enumerate(experiments):
        if counter == 0:
            continue
        else:
            poses_i = np.load(experiment, allow_pickle=True)
            poses = np.vstack((poses, poses_i))
    return poses


def process_object_transforms_tcp_in_robot(
    poses, rotations, camera_in_tcp, single=False
):
    """move tcp in robot transforms for each collected depth image such that the
    camera replaces the tcp position and orientation in (x,y) plane

    poses: position of tcp in robot w/o camera aligning

    return tcp robot transforms when camera is looking at the target patch
    """
    if single:
        poses = np.expand_dims(poses, axis=0)
        rotations = np.expand_dims(rotations, axis=0)
    transforms = []
    OFFSET_TCP = np.array([-camera_in_tcp[:, 3][0], -camera_in_tcp[:, 3][1], 0, 1])
    for pose, rotation in zip(poses, rotations):
        e_T_tcp_in_robot = pos_orn_to_matrix(pose, rotation)
        tcp_cam_view_in_robot = e_T_tcp_in_robot.dot(OFFSET_TCP)
        transforms.append(pos_orn_to_matrix(tcp_cam_view_in_robot[:3], rotation))
    return np.asarray(transforms)


def get_object_transforms_camera_in_world(
    poses, rotations, world_in_robot, camera_in_tcp, single=False
):
    tcp_in_robots = process_object_transforms_tcp_in_robot(
        poses, rotations, camera_in_tcp, single
    )
    transforms = []
    robot_in_world = inverse_transform(world_in_robot)
    for tcp_in_robot in tcp_in_robots:
        transforms.append(robot_in_world.dot(tcp_in_robot.dot(camera_in_tcp)))
    return np.asarray(transforms)


def point_cloud_info(pcd, display=False):
    """
    Returns segregated arrays for each co-ordinate across all points
    <display>: print out pcd data
    """
    points = np.asarray(pcd.points)
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    if display:
        print("z min " + str(np.min(zs)))
        print("x min " + str(np.min(xs)))
        print("y min " + str(np.min(ys)))
        print("z max " + str(np.max(zs)))
        print("x max " + str(np.max(xs)))
        print("y max " + str(np.max(ys)))
    return xs, ys, zs
