import copy
import sys
import numpy as np
import yaml
import time
import math
from scipy.spatial.transform import Rotation
from robot_io.utils.utils import pos_orn_to_matrix
import cv2
import open3d as o3d
import os
import matplotlib.pyplot as plt
import re
import operator
from PIL import Image
import natsort

dir_path = os.path.dirname(os.path.realpath(__file__))


class NotAdaptedError(Exception):
    def __init__(self, message):
        self.message = message


def inverse_transform(tf):
    inv = np.eye(4)
    inv[:3, :3] = tf[:3, :3].T
    inv[:3, 3] = -inv[:3, :3].dot(tf[:3, 3])
    return inv


##constants
# 3D printed finger offset from default robot TCP. Change to suit your tool. used with add_ee
TOOL_OFFSET_IN_TCP = [0, 0, 0.057, 0]
# nice orientation of TCP, 180 degrees around y axis to face the tcp towards the earth, given the robot's frame is coincident with the earth's frame
FIXED_ROBOT_ORN = [0, np.pi, 0]
# Recorded with Kinect Tripod to capture the raised object placeholder on the table frame. Change to suit your setup.
# WORLD_IN_ROBOT = np.load(f"{dir_path}/utils_data/world_robot_transform.npy", allow_pickle=True)
WORLD_IN_ROBOT = np.asarray(
    [
        [1, 0, 0, 0.427],
        [0, 1, 0, 0],
        [0, 0, 1, 0.065],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)  # recorded manually
ROBOT_IN_WORLD = inverse_transform(WORLD_IN_ROBOT)
# Estimated with apriltag_detector.py followed by cam_calibration_optimization.py
WCAMERA_IN_TCP = np.load(
    f"{dir_path}/utils_data/wcamera_tcp_transform.npy", allow_pickle=True
)
# Recorded with Kinect Tripod.
WORLD_IN_CAMERA = np.load(
    f"{dir_path}/utils_data/world_camera_transform.npy", allow_pickle=True
)
# Recorded with Kinect Tripod.
CAMERA_IN_ROBOT = np.load(
    f"{dir_path}/utils_data/camera_robot_transform.npy", allow_pickle=True
)
# Aruco marker for the world, physical measurement.
WORLD_IN_MARKER = pos_orn_to_matrix([-0.35, -0.04, 0.035], [0, 0, 0.5 * np.pi])
# Aruco marker for the robot, physical measurement.
ROBOT_IN_MARKER = pos_orn_to_matrix([-0.108, 0.145, 0], [0, 0, 0])
# change to suit where you want the robot to move away
AWAY_POSE = [0.45, 0.1, 0.2]
# change to suit where you want the robot to start from
HOME_POSE = [0.45, 0, 0.1]
# Views in the robot frame chosen to cover different areas of the object, that is placed on the world placeholder in our setting,
# They will have similar size of points in the presence of calibration errors, also accounting for robot's singularities
VIEWS = [
    [0.37, 0, 0.1],
    [0.37, 0.05, 0.1],
    [0.37, -0.05, 0.1],
    [0.32, 0.0, 0.1],
    [0.42, 0, 0.1],
]
ORNS = [
    FIXED_ROBOT_ORN,
    [np.pi / 4, np.pi, 0],
    [-np.pi / 4, np.pi, 0],
    [-np.pi / 4, np.pi - np.pi / 4, 0],
    [0, np.pi + np.pi / 4, 0],
]
# Masks for the kinect images, to be used with the kinect calibration to manually select the area of interest.
MASK_KINECT_1 = (850, 150)
MASK_KINECT_2 = (1130, 500)


# custom tailored for kuka fixture experiment
def getTransform(config):
    # angle between "corner point translated from origin of robot to origin of sensor" and "x-axis of robot"
    theta = np.arctan2(
        config["target_corner"][1] - config["target_origin"][1],
        config["target_corner"][0] - config["target_origin"][0],
    )
    # angle between robot frame and sensor frame
    theta = (np.pi / 4) - theta
    transform_matrix = np.matrix(
        [
            [np.cos(theta), np.sin(theta), 0, config["target_origin"][0]],
            [-np.sin(theta), np.cos(theta), 0, config["target_origin"][1]],
            [0, 0, 1, config["target_origin"][2]],
            [0, 0, 0, 1],
        ]
    )
    return transform_matrix


def search_folder(start_path, target_folder):
    for dirpath, dirnames, filenames in os.walk(start_path):
        if target_folder in dirnames:
            return os.path.join(dirpath, target_folder)
    return None


# apply transform
def transform_sensor_to_robot(vector_sensor, transform):
    vector_sensor_homog = 1 * np.array(
        [vector_sensor[0], vector_sensor[1], vector_sensor[2], 1]
    )
    return np.dot(transform, vector_sensor_homog)


# record origin with right button and corner with left custom for space mouse
def main_record_config(robot, space_mouse, a_gripper, path):
    config = ""
    if a_gripper != 0:
        if a_gripper == 1:
            sensor_frame_dict1 = {
                "target_origin": [
                    float(robot.get_state()["tcp_pos"][0]),
                    float(robot.get_state()["tcp_pos"][1]),
                    float(robot.get_state()["tcp_pos"][2]),
                ]
            }
            with open(path, "w") as config:
                yaml.dump(sensor_frame_dict1, config, default_flow_style=False)
        else:
            print(a_gripper)
            sensor_frame_dict2 = {
                "target_corner": [
                    float(robot.get_state()["tcp_pos"][0]),
                    float(robot.get_state()["tcp_pos"][1]),
                    float(robot.get_state()["tcp_pos"][2]),
                ]
            }
            with open(path, "a") as config:
                yaml.dump(sensor_frame_dict2, config, default_flow_style=False)
        space_mouse.clear_gripper_input()
    return config


# make plane
def make_plane(n, z, points=True):
    samples = []
    s = 0.017  # sensor side length
    x = np.zeros(int(n))
    y = np.zeros(int(n))
    for i in range(len(x) + 1):
        if i == 0:
            px = 0
        if i > 0:
            px += s / (2 * n)
        for j in range(len(y) + 1):
            sample = []
            if j == 0:
                py = 0
            if j > 0:
                py += s / (2 * n)
            if points:
                if (px == 0) and (py == 0):
                    sample = [px, py, z, 1]
                    mirrored_sample = [-px, -py, z, 1]
                else:
                    sample = [px, py, np.nan, 1]
                    mirrored_sample = [-px, -py, np.nan, 1]
            else:
                sample = [px, py, z, 1]
                mirrored_sample = [-px, -py, z, 1]
            samples.append(np.asarray(sample))
            if not np.array_equal(sample, mirrored_sample):
                samples.append(np.asarray(mirrored_sample))
    samples = np.asarray(samples)
    samples[:, 0] = np.sort(samples[:, 0])
    samples[:, 1] = np.sort(samples[:, 1])
    samples[:, 2] = np.sort(samples[:, 2])
    samples[:, 3] = np.sort(samples[:, 3])
    return np.asarray(samples)


# square
def make_square(iterations=20):
    samples = []
    for j in range(iterations):
        n = 4  # steps per array
        s = 0.017  # sensor side length
        x = np.zeros(n)
        y = np.zeros(n)
        for i in range(len(x) + 1):
            sample = ()
            if i == 0:
                px = 0
            if i > 0:
                px += s / n
            sample = sample + (px,)
            for j in range(len(y) + 1):
                sample = ()
                if j == 0:
                    py = 0
                if j > 0:
                    py += s / n
                sample = sample + (px, py, 0.01)  # 1 cm above sensor in z direction
                samples.append((sample))
    return samples


def _debounce_state(robot):
    time.sleep(0.1)
    state = robot.get_state()
    wrench = state["force_torque"]
    return state, wrench


def debounce_tcp_pose(robot, delay=False):
    tcp_pose = robot.get_tcp_pose()
    if delay:
        time.sleep(1)
    else:
        time.sleep(0.1)
    tcp_pose = robot.get_tcp_pose()
    return tcp_pose


def _move_on_axis(
    axisid,
    stop,
    direction,
    goal_pose_virtual,
    max_dist_per_step,
    contact=False,
):
    """
    -computes the position error on a given axis.
    -if "stop" is set to true; it raises the goal reached flag and returns the computed error; takes precedence over direct.
    -- factor: the factor by which the error is divided; It is > 1 for errors > max_dist_per_step, and = 1 for errors <= max_dist_per_step.
    returns:
     -goal_reached: boolean
     -computed error
    """
    error_vector = np.array((0, 0, 0), dtype=float)
    if not stop:
        if contact:
            error_vector[axisid] = max_dist_per_step * direction[axisid]
        else:
            error_vector[axisid] = max_dist_per_step * -direction[axisid]
        goal_pose_virtual += error_vector
        return False, goal_pose_virtual
    else:
        return True, goal_pose_virtual


def move_to_pt_with_v_safe(
    robot,
    goal_point,
    goal_rot,
    p_i=0.0005,
    dt=1 / 100,
    f_max=8,
    T_max=4,
    end=False,
    direction=None,
    sensor_process=None,
    contact=False,
    goal_distance_tol=0.002,
    max_goal_distance=0.15,
    reskin_threshold=40,
    reskin_ambient_recording_distance_to_contact_threshold=0.02,
    max_increment_for_contact=3,
):
    """
    - move towards a goal point with step distance p_i*direction in each axis
    - maximum step velocity is always limited by the robot velocity. If the waypoint is not reached before dt seconds, the next iteration will override.
    - dt controls the average velocity to the goal point "but not the actual movement velocity"
    -- direction: 3D orientaion vector. If provided, the robot will move along it.
    returns:
        - contact state: -1 failed
                         0  no contact intended
                         1  undefined
                         2  reskin stopped
                         3  wrench stopped
                         4  contacted air
        - robot state
        - reskin norm in case of contact
        - ambient recording in case of contact
    """
    # function control variables
    goal_reached_x = False
    goal_reached_y = False
    goal_reached_z = False
    stop = False
    force_control = False
    previous_reskin = 0
    previous_reskin_norm = 0
    ambient = False
    ambient_recording = None
    stored_increament = 0
    time.sleep(0.1)
    goal_pose_virtual = robot.get_state()["tcp_pos"]
    # refuse far goals
    current_norm_error = np.linalg.norm(
        np.subtract(goal_point, robot.get_state()["tcp_pos"]), 2
    )
    if current_norm_error > max_goal_distance:
        print(
            f"goal point is {current_norm_error} away. That's too far, not going to move"
        )
        return -1, robot.get_state(), None, None

    ### movement loop
    while not end and not (goal_reached_x and goal_reached_y and goal_reached_z):
        current_norm_error = np.linalg.norm(
            np.subtract(goal_point, robot.get_state()["tcp_pos"]), 2
        )

        # allow a maximum of goal_distance_tol, increase it to avoid oscillations in trade off with accuracy
        if current_norm_error <= goal_distance_tol:
            stop = True

        # update control variables and the next way point
        goal_reached_x, goal_pose_virtual = _move_on_axis(
            0,
            stop,
            direction,
            goal_pose_virtual,
            p_i,
            contact=contact,
        )
        goal_reached_y, goal_pose_virtual = _move_on_axis(
            1,
            stop,
            direction,
            goal_pose_virtual,
            p_i,
            contact=contact,
        )
        goal_reached_z, goal_pose_virtual = _move_on_axis(
            2,
            stop,
            direction,
            goal_pose_virtual,
            p_i,
            contact=contact,
        )

        # move the robot to the next way point
        if (not goal_reached_x) or (not goal_reached_y) or (not goal_reached_z):
            # if goal_rot is not the same as the current rotation, return -
            dot_product = np.dot(goal_rot, robot.get_state()["tcp_orn"])
            if np.degrees(2 * np.arccos(np.abs(dot_product))) > 5:
                print("goal rotation is not the same as the current rotation")
                return -1, robot.get_state(), None, None
            else:
                robot.move_async_cart_pos_abs_lin(goal_pose_virtual, goal_rot)
                time.sleep(dt)
        else:
            state = robot.get_state()
            wrench = state["force_torque"]
            state, wrench = _debounce_state(robot)
            print(f"pose tolerance control stopped")
            if not contact:
                print("no contact intended, moved freely")
                return 0, state, None, None
            ## update control variables
            elif stored_increament < max_increment_for_contact:
                force_control = True
                goal_reached_x = False
                goal_reached_y = False
                goal_reached_z = False
                stop = False
                # update the goal point on the alignment direction till you feel a specific force
                print("goal point increased by 2 mm")
                goal_point = goal_point + 0.002 * direction
                stored_increament += 1
                # update the error based on the new goal point
                current_norm_error = np.linalg.norm(
                    np.subtract(goal_point, robot.get_state()["tcp_pos"]), 2
                )
            else:
                print("contacted air")
                return 4, state, None, None

        if sensor_process is not None:
            # get last 2 readings from reskin
            reskin_reading = sensor_process.get_data(num_samples=2)
            current_reskin = filter_reskin(reskin_reading, multiple_samples=True)
            current_reskin_norm = filter_reskin(
                reskin_reading, multiple_samples=True, norm=True
            )
            if previous_reskin_norm == current_reskin_norm or current_reskin_norm == 0:
                raise Exception(
                    "sensor data corrupted while attempting contact, check cables"
                )
            if previous_reskin_norm > 0:
                reskin_norm_difference = np.linalg.norm(
                    current_reskin - previous_reskin, 2
                )
                if reskin_norm_difference > reskin_threshold:
                    print(f"reskin control stopped {reskin_norm_difference}")
                    # stopped by reskin
                    if ambient_recording is not None:
                        print(current_reskin - ambient_recording)
                    return 2, state, reskin_norm_difference, ambient_recording
            if not force_control:
                previous_reskin_norm = current_reskin_norm
                previous_reskin = current_reskin
            else:
                if previous_reskin_norm > 0:
                    print(
                        f"force control mode, initial reskin reading fixed {previous_reskin}"
                    )
                    print(f"current reading {current_reskin}")
            if (
                not ambient
                and np.linalg.norm(
                    np.subtract(goal_point, robot.get_state()["tcp_pos"]), 2
                )
                < reskin_ambient_recording_distance_to_contact_threshold
            ):
                reskin_reading = sensor_process.get_data(num_samples=10)
                ambient_recording = filter_reskin(reskin_reading, multiple_samples=True)
                print("took ambient reading while moving")
                ambient = True
        # frame dangers
        current_tcp_in_robot_pos = robot.get_state()["tcp_pos"]
        current_tcp_in_robot_pos_homog = np.append(current_tcp_in_robot_pos, 1)
        current_tcp_in_robot = pos_orn_to_matrix(current_tcp_in_robot_pos, goal_rot)
        if contact:
            # Allow focal point of the camera to be maximally less than the world height (z-axis)
            if (
                ROBOT_IN_WORLD.dot(current_tcp_in_robot.dot(WCAMERA_IN_TCP))[:, 3][2]
                < -0.005
            ):
                print("can't continue further camera danger")
                return -1, robot.get_state(), None, ambient_recording
            # Allow Re-Skin to be minimally 5 mm above the world height (z-axis) accounting for robot-world calibration errors,
            # Re-Skin dimensions from the tcp are hardcoded
            if not safety_three(
                current_tcp_in_robot_pos, goal_rot, [0.0085, 0.0085, 0], 0.005
            ):
                print("won't make a contact with an object")
                return 4, robot.get_state(), None, ambient_recording
            # Allow robot flange to be minimally 6 cm (height of the world holder) bleow the world height (z-axis), robot flange dimensions from the tcp hardcoded
            if not safety_three(
                current_tcp_in_robot_pos, goal_rot, [0.015, 0.09, 0.058], -0.06
            ):
                print("can't continue further, robot frame danger")
                return -1, robot.get_state(), None, ambient_recording
        elif ROBOT_IN_WORLD.dot(current_tcp_in_robot_pos_homog)[2] > 0.1:
            return -1, None, None, None

        state = robot.get_state()
        wrench = state["force_torque"]
        # additional control for wrench
        if (
            np.linalg.norm(wrench[:3], 2) > f_max
            or np.linalg.norm(wrench[3:], 2) > T_max
        ):
            print(
                f"wrench control stopped, force norm {np.linalg.norm(wrench[:3],2)}, torque norm {np.linalg.norm(wrench[3:],2)}"
            )
            # stopped by the robot integrated wrench sensor
            return 3, state, None, ambient_recording

    state, wrench = _debounce_state(robot)
    print(f"passed all checks without returning a state, check the code {state}")
    print(end, goal_reached_x, goal_reached_y, goal_reached_z, current_norm_error)
    return 1, state, None, None


def step_move_robot_with_device(
    robot, device_pose, start_pose, goal_rot, v=0.005, dt=1 / 20
):
    goal_pos = start_pose
    goal_pos += device_pose * v * dt
    robot.move_async_cart_pos_abs_ptp(goal_pos, goal_rot)
    time.sleep(dt)


def euler_to_rotation_matrix(zyx):
    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_euler("zyx", zyx, degrees=False)
    inv = rot.inv()
    # Convert to matrix
    rot_matrix = rot.as_matrix()
    inv_rot_matrix = inv.as_matrix()
    return rot_matrix, inv_rot_matrix


def eulertoquat(xyz):
    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_euler("xyz", xyz, degrees=False)
    # Convert to quaternions
    rot_quat = rot.as_quat()
    return rot_quat


def add_ee(pos, goal_orientation):
    TOOL_OFFSET_IN_ROBOT = [0, 0, 0, 0]
    TOOL_OFFSET_IN_ROBOT[:3] = (
        euler_to_rotation_matrix(
            [goal_orientation[2], goal_orientation[1], goal_orientation[0]]
        )[1]
    ).dot(TOOL_OFFSET_IN_TCP[:3])
    TOOL_OFFSET_IN_ROBOT[2] = -TOOL_OFFSET_IN_ROBOT[2]
    return np.add(pos, TOOL_OFFSET_IN_ROBOT[:3])


def euler_from_vector(roll, vector):
    yaw = np.arctan2(vector[1], np.sqrt(np.square(vector[0]) + np.square(vector[2])))
    pitch = np.arctan2(vector[0], vector[2])
    return [-yaw, pitch, roll]


def extract_rotation(T):
    """
    remove translation and scale from a homogenous transform
    outputs a 4*4 rotation matrix
    """
    basis_one = T[:, 0]
    scale_one = np.linalg.norm(basis_one, 2)
    basis_two = T[:, 1]
    scale_two = np.linalg.norm(basis_two, 2)
    basis_three = T[:, 2]
    scale_three = np.linalg.norm(basis_three, 2)
    dummy_translation = [0, 0, 0, 1]
    rot = np.array(
        [
            basis_one / scale_one,
            basis_two / scale_two,
            basis_three / scale_three,
            dummy_translation,
        ]
    ).T
    return rot


def point3d_to_pixel(point, fx, fy, cx, cy):
    u = np.divide(point[0] * fx, point[2]) + cx
    v = np.divide(point[1] * fy, point[2]) + cy
    return (u, v)


def threed_vector_to_homog(x):
    x = 1 * np.array([x[0], x[1], x[2], 1])
    return x


def homog_vector_to_3d(x):
    x = np.array([x[0], x[1], x[2]])
    return x


def show_rgb_and_depth(rgb, dep, wait):
    cv2.imshow("image", rgb[:, :, ::-1])  # use BGR for cv
    cv2.imshow("depth", dep)
    cv2.waitKey(wait)


def create_rotation_from_normal(normal):
    """
    creates a 3*3 rotation matrix from an orientation vector with zero roll around it
    <normal>: negative normalized orientation vectot to use as new z-axis
    """

    # Normalize the input normal vector
    normal = normal / np.linalg.norm(normal, 2)

    # Set the z-axis as the provided normal vector
    basis_three = -normal

    # Calculate a perpendicular vector to the z-axis for the x-axis
    basis_one = np.cross(np.array([0, 0, 1]), basis_three)
    basis_one = basis_one / np.linalg.norm(basis_one, 2)

    # if basis one is zero vector use x-axis
    if np.linalg.norm(basis_one, 2) == 0:
        print("basis one is zero vector")
        basis_one = np.array([1, 0, 0])

    # Calculate the y-axis as the cross product of the z-axis and x-axis
    basis_two = np.cross(basis_three, basis_one)
    basis_two = basis_two / np.linalg.norm(basis_two, 2)

    # Construct the rotation matrix
    rot = np.vstack((basis_one, basis_two, basis_three))

    return rot


def create_homog_transformation_from_rotation(rot):
    """
    Returns a 4*4 homogenous transformation matrix corresponding to a 3*3 rotation matrix
    """
    T = np.zeros((4, 4))
    T[:3, :3] = rot
    T[:, 3] = [0, 0, 0, 1]
    T[3, :] = [0, 0, 0, 1]
    return T


def get_crop_indeces(hom_point_in_cam, tcp_in_c, intrinsic, resolution=None):
    sr = 0.0085  # sensor_radius
    # Sensor corners
    # Top right
    # Bottom right
    # Top left
    # Bottom left
    sensor_corners_tcp_frame = (
        np.array([[1, 1, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0], [-1, -1, 0, 0]]) * sr
    )

    sensor_corners_cam_frame = tcp_in_c.dot(sensor_corners_tcp_frame.T)
    target_contact_cam_frame = (
        sensor_corners_cam_frame.T + hom_point_in_cam
    )  # translation in camera space
    # Non- normalized crop
    crop = intrinsic.dot(
        target_contact_cam_frame.T
    )  # from cam space to image space (Transpose to apply the transformation to multiple column vectors)
    crop /= crop[-1]
    return crop


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


def mask_rgb_and_dep(im, dep_im, mask_p1, mask_p2, show=False):
    """
    crop a target portion of the rgb and depth images w/o changing resolution
    """
    if show:
        cv2.rectangle(im, mask_p1, mask_p2, [0, 255, 0], 1)  # hardcoded from image
        cv2.imshow("image", im[:, :, ::-1])  # show current image test
        cv2.imshow("depth", dep_im)
        cv2.waitKey(0)
    else:
        mask = np.zeros(im.shape[:2], dtype="uint8")
        cv2.rectangle(mask, mask_p1, mask_p2, 255, -1)  # hardcoded from image
        im = cv2.bitwise_and(im, im, mask=mask)
        dep_im = cv2.bitwise_and(dep_im * 1000, dep_im * 1000, mask=mask)
    return im, dep_im


def segment_point_cloud(
    pcd,
    minz,
    maxz,
    minx,
    maxx,
    miny,
    maxy,
    voxel,
    voxel_size=0.0012,
    statistical_filter=False,
    radius_filter=False,
    radius=0.005,
    nb_points=100,
):
    """
    filter out noisy points and distance threshold the pcd
    """
    xs, ys, zs = point_cloud_info(pcd)
    original_filter = (
        (zs > minz)
        & (zs <= maxz)
        & (xs > minx)
        & (xs < maxx)
        & (ys > miny)
        & (ys < maxy)
    )
    indices = np.where(original_filter)
    pcd = pcd.select_by_index(indices[0], invert=False)
    if voxel:
        pcd = pcd.voxel_down_sample(voxel_size)
        print("voxelized")
    # remove points that are on average farther away from their neighbors , if neighbours are 50
    if statistical_filter:
        pcd = pcd.remove_statistical_outlier(200, 0.2)
        pcd = pcd[0]  # get filtered pcd
        print("statistical filter applied")
    if radius_filter:
        pcd = pcd.remove_radius_outlier(
            nb_points, radius
        )  # remove points that have less than n neighbors in l meters
        print("radius filter applied")
        pcd = pcd[0]  # get filtered pcd

    return pcd


def convert_image_to_point_cloud(
    camera,
    im,
    dep_im,
    extrinsic,
    minz=1,
    maxz=1,
    minx=1,
    maxx=1,
    miny=1,
    maxy=1,
    mask_p1=None,
    mask_p2=None,
    display=False,
    voxel=False,
    radius_filter=False,
    reconstruct_patches=False,
    cx_offset=0,
    cy_offset=0,
    segment=True,
):
    phc = o3d.camera.PinholeCameraIntrinsic()
    if camera == "kinect":
        phc = np.array(
            [[912.103516, 0, 953.576111], [0, 911.885925, 555.760559], [0, 0, 1]]
        )
        distortions = np.array(
            [
                3.745310e-01,
                -2.447607e00,
                1.634000e-03,
                1.300000e-04,
                1.401331e00,
                2.575030e-01,
                -2.275811e00,
                1.331136e00,
            ]
        )
        im = cv2.undistort(im, phc, distortions)
        # dep_im = cv2.undistort(dep_im,phc,distortions)
        phc.set_intrinsics(
            width=1920,
            height=1080,
            fx=912.103516,
            fy=911.885925,
            cx=953.576111,
            cy=555.760559,
        )
    else:
        # print("intrinsics of realsense loaded")
        if reconstruct_patches:
            phc.set_intrinsics(
                width=640,
                height=360,
                fx=322.37493896484375,
                fy=322.0753479003906,
                cx=314.71563720703125 - cx_offset,
                cy=183.8709716796875 - cy_offset,
            )
        else:
            phc.set_intrinsics(
                width=640,
                height=360,
                fx=322.37493896484375,
                fy=322.0753479003906,
                cx=314.71563720703125,
                cy=183.8709716796875,
            )

    if (mask_p1 is None) or (mask_p2 is None):
        dep_im = dep_im * 1000
    else:
        im, dep_im = mask_rgb_and_dep(
            im, dep_im, mask_p1, mask_p2
        )  # this function additionaly scales the depth image
    dep = o3d.geometry.Image(dep_im.astype(np.uint16))

    if im is not None:
        im = o3d.geometry.Image((im).astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im, dep, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, phc, extrinsic, project_valid_depth_only=True
        )
    else:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            dep, phc, extrinsic, project_valid_depth_only=True
        )
        xs, ys, zs = point_cloud_info(pcd)
        original_filter = (zs > 0) & (zs <= 0.05)
        indices = np.where(original_filter)
        pcd = pcd.select_by_index(indices[0], invert=False)

    if not reconstruct_patches and segment:
        pcd = segment_point_cloud(pcd, minz, maxz, minx, maxx, miny, maxy, voxel)

    point_cloud_info(pcd, display)
    return pcd


def plot_reskin(reskin_recording):
    """
    Works only for a list of more than one reading
    """
    if len(reskin_recording) > 1:
        reskin_reading = np.array(reskin_recording, dtype=object)
        reskin_reading = np.squeeze(reskin_reading)[
            :, 2
        ]  # extract lists of magnetometers values and temperatures as array of lists
        reskin_reading = list(
            reskin_reading
        )  # convert to list of lists then to nd array
        reskin_reading = np.asarray(reskin_reading, dtype=object)
        reskin_reading = np.delete(
            reskin_reading, [0, 4, 8, 12, 16], 1
        )  # eliminate temperatures
        reskin_reading = np.swapaxes(reskin_reading, 0, 1)
        # draw
        plt.plot(
            [
                "bx1",
                "by1",
                "bz1",
                "bx2",
                "by2",
                "bz2",
                "bx3",
                "by3",
                "bz3",
                "bx4",
                "by4",
                "bz4",
                "bx5",
                "by5",
                "bz5",
            ],
            reskin_reading[:, -1],
        )
        if reskin_reading.all == 0:
            print("hey no readings!!")
        plt.draw()
        plt.pause(0.01)


# https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def crop_center(img, cropx, cropy):
    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx, ...]


def get_target_images(path, type, size, reconstructed=False, target_masks=None):
    if type == "masks" and not reconstructed:
        if not any("masks" in s for s in path):
            path = [p + "masks" for p in path]
        return get_target_masks(path, size)
    elif (
        type == "processed_depth" or type == "masked_depth_processed"
    ) and not reconstructed:
        if not any("depth_processed" in s for s in path):
            path = [p + "depth_processed" for p in path]
        return get_depth_processed(path, size)
    elif type == "depth" and not reconstructed:
        if not any("depth" in s for s in path):
            path = [p + "depth" for p in path]
    else:
        pass
    regex = re.compile(f"experiment_.*_{type}_.*")
    images = []
    for p in path:
        for root, dirs, files in os.walk(p):
            for file in files:
                if regex.match(file):
                    images.append(p + "/" + file)
    images = np.asarray(natsort.natsorted(images))
    if type == "rgb":
        target_images = np.array(
            [
                cropND(
                    np.array(Image.open(fname), dtype=np.uint8), (size[0], size[1], 3)
                )
                for fname in images
            ],
            dtype=list,
        )
    else:
        target_images = np.array(
            [
                cropND(np.array(Image.open(fname), dtype=np.uint8), (size[0], size[1]))
                for fname in images
            ],
            dtype=list,
        )
    if type == "masked_depth_processed" or type == "masked_depth":
        target_images = target_masks * target_images
    return target_images


def get_depth_processed(path, size):
    regex = re.compile(f"depth_processed_.*")
    depth_processed = []
    for p in path:
        for root, dirs, files in os.walk(p):
            for file in files:
                if regex.match(file):
                    depth_processed.append(p + "/" + file)
    depth_processed = np.asarray(natsort.natsorted(depth_processed))
    depth_processed = np.array(
        [
            cropND(np.array(Image.open(fname), dtype=np.uint8), (size[0], size[1]))
            for fname in depth_processed
        ],
        dtype=list,
    )
    return depth_processed


def get_target_masks(path, size):
    regex = re.compile(f"masks_.*")
    masks = []
    for p in path:
        for root, dirs, files in os.walk(p):
            for file in files:
                if regex.match(file):
                    masks.append(p + "/" + file)
    masks = np.asarray(natsort.natsorted(masks))
    masks = np.array(
        [
            cropND(np.array(Image.open(fname), dtype=np.uint8), (size[0], size[1]))
            for fname in masks
        ],
        dtype=list,
    )
    return masks


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def custom_visualization_pcd(
    pcd,
    points,
    discretized_normal_array_negative,
    discretized_normal_array_positive,
    safe_one,
    safe_two_indeces,
    not_safe_two_indeces,
    sampled_point_index,
    objects_names,
    many_pcds=False,
    sampled_point=None,
):
    """
    marked visualization of a sampled point from a point cloud, with safety_one and safety_two conditions defined in this utils script
    """
    # color areas and draw geometries for visualization#
    # copy the point cloud
    pcd_copy = copy.deepcopy(pcd)
    colors = np.asarray(pcd_copy.colors)
    color_safe = np.zeros((safe_two_indeces[0].shape[0], 3)) + np.array(
        [0.2, 0.2, 0.2]
    )  # grey
    color_unsafe = np.zeros((not_safe_two_indeces[0].shape[0], 3)) + np.array(
        [1, 0, 0]
    )  # red
    colors[safe_two_indeces] = color_safe
    colors[not_safe_two_indeces] = color_unsafe
    if sampled_point in points and many_pcds:
        higher_zs = []
        for i, point in enumerate(np.asarray(pcd_copy.points)):
            if (
                (point[0] - sampled_point[0] < 0.002)
                and (point[1] - sampled_point[1] < 0.002)
                and (point[0] - sampled_point[0] > 0)
                and (point[1] - sampled_point[1] > 0)
            ):
                if point[2] >= sampled_point[2]:
                    higher_zs.append([point[2], i])
        higher_zs = np.asarray(higher_zs)
        if np.asarray(higher_zs).shape[0] > 0:
            for index in higher_zs[:, 1]:
                colors[int(index)] = np.array([1, 0.5, 0.5])
        # index where points is equal to sampled point using .all
        colors[np.where(points == sampled_point)[0][0]] = np.array(
            [1, 0.5, 0.5]
        )  # black, works only if the index was from the same PCD
    else:
        sampled_point = np.expand_dims(np.asarray(sampled_point), axis=0)
        # sampled_point_color=np.expand_dims(np.asarray([0,1,0]),axis=0)
        points = np.append(points, sampled_point, axis=0)
    points = np.append(points, discretized_normal_array_negative, axis=0)
    points = np.append(points, discretized_normal_array_positive, axis=0)
    if (
        many_pcds
    ):  # if several normals are provided, then it is a comparison, adapt colors
        # for every 10 points, there is a normal vector, choose another color for the normal vector
        i = 0
        j = 1
        for p in range(len(discretized_normal_array_negative)):
            # change the color of the normal vector every 10 points
            if i % 10 == 0 and i != 0:
                j += 1
                i = 0
            i += 1
            if j == 1:
                colors_normal_negative_true = np.zeros((1, 3)) + np.array([j, 0.5, 0])
            elif j == 2:
                colors_normal_negative_true = np.zeros((1, 3)) + np.array([0, j, 0.5])
            elif j == 3:
                colors_normal_negative_true = np.zeros((1, 3)) + np.array([0, 0.5, j])
            else:
                print("no option selected")
            colors = np.append(colors, colors_normal_negative_true, axis=0)
        if safe_one:
            i = 0
            j = 1
            for p in range(len(discretized_normal_array_positive)):
                # change the color of the normal vector every 10 points
                if i % 10 == 0 and i != 0:
                    j += 1
                    i = 0
                i += 1
                if j == 1:
                    colors_normal_positive_true = np.zeros((1, 3)) + np.array(
                        [j, 0.5, 0]
                    )
                elif j == 2:
                    colors_normal_positive_true = np.zeros((1, 3)) + np.array(
                        [0, j, 0.5]
                    )
                elif j == 3:
                    colors_normal_positive_true = np.zeros((1, 3)) + np.array(
                        [0, 0.5, j]
                    )
                else:
                    print("no option selected")
                colors = np.append(colors, colors_normal_positive_true, axis=0)
        else:
            for p in range(len(discretized_normal_array_positive)):
                # change the color of the normal vector every 10 points
                colors_normal_positive_true = np.zeros((1, 3)) + np.array([1, 0.5, 0.5])
                colors = np.append(colors, colors_normal_positive_true, axis=0)
    else:
        colors_normal_negative = np.zeros((10, 3))  # black
        if safe_one:
            colors_normal_positive = np.zeros((10, 3)) + np.array([0, 1, 1])  # indigo
        else:
            colors_normal_positive = np.zeros((10, 3)) + np.array([1, 0, 0])  # red
        colors = np.append(colors, colors_normal_negative, axis=0)
        colors = np.append(colors, colors_normal_positive, axis=0)
    pcd_viz = o3d.geometry.PointCloud()
    pcd_viz.points = o3d.utility.Vector3dVector(points)
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
    -orientation: orientation vector
    -tol: minimum cosine value between the up and the orientation vectors
    """
    z_in_world = np.array([0, 0, 1])
    condition_1 = orientation.dot(z_in_world) > tol
    # print("safety one "+str(condition_1))
    return condition_1


def safety_two(pcd, point, nnn, sl, z_tol):
    """
    Don't allow target points, that you will encounter obstacles on their path
    point: target 3d point
    -nnn: normalized negative normal, generally a linear path vector
    -sl: patch square side length around the target point
    -z_tol: maximum tolerance to higher points in z-direction in meters

    Returns not_safe_indices,safe_indices, and safety_boolean
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
    # print("safety two " +str(condition_2))
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
    # In our experimental setting, this condition will deem some safe points unsafe.
    # Specifically, when those points are away from the raised object location
    # in x,y coordinates but still lower than the location's z
    # A temporary solution for that could be to ensure the cosine angle threshold is large enough and ignore this condition
    # print("safety three " +str(condition_3))
    return condition_3


def is_string_present_in_filename(filename, strings):
    """
    Check if any of the strings are present in the given filename.

    Args:
        filename (str): The file name to check.
        strings (list): List of strings to search for.

    Returns:
        bool: True if any of the strings are present in the filename, False otherwise.
    """
    for string in strings:
        if string in filename:
            return string, True
    return None, False


def load_pcds(cfg, down_sampled=False, estimate_normals=False):
    """
    For every given object in the cfg load a combined view points pcd or a down sampled version of them if down_sampled is set.
    These pcds are stored in the cfg data_path directory, and collected by the save_pcds.py script.
    returns x,y maximum and minimum coordinates across all given objects.
    """
    # initialize variables and get object names
    min_xs = []
    min_ys = []
    max_xs = []
    max_ys = []
    pcd_array = []
    objects_names = cfg.objects_names.split(",")
    pcd_files = []
    repo_directory = search_folder("/", cfg.repo_directory)
    # configure the pcd loading
    if down_sampled:
        regex = re.compile(f".*_down.pcd")
    else:
        regex = re.compile(f".*_combined.pcd")
    for root, dirs, files in os.walk(f"{repo_directory}/{cfg.data_path}"):
        for file in files:
            print(file)
            if regex.match(file):
                (
                    matched_object_string,
                    matched_object_boolean,
                ) = is_string_present_in_filename(file, objects_names)
                if matched_object_boolean:
                    pcd_files.append(
                        f"{repo_directory}/{cfg.data_path}/{matched_object_string}/{file}"
                    )
    pcd_files = natsort.natsorted(pcd_files)
    # if downsampled pcds are not available, load the original ones
    if len(pcd_files) != len(objects_names) and down_sampled:
        print("downsampled pcds not found, loading original pcds")
        return load_pcds(cfg, down_sampled=False)
    elif len(pcd_files) != len(objects_names) and not down_sampled:
        print("not enough pcds found, please run save_pcds.py script first")
        raise ValueError
    for file in pcd_files:
        pcd = o3d.io.read_point_cloud(file)
        if estimate_normals:
            pcd.estimate_normals()
        # store pcds and their min/max coordinates
        pcd_array.append(pcd)
        xs, ys, _ = point_cloud_info(pcd)
        min_xs.append(np.min(xs))
        min_ys.append(np.min(ys))
        max_xs.append(np.max(xs))
        max_ys.append(np.max(ys))
    # convert pcds to numpy array
    pcd_array = np.asarray(pcd_array)
    print(f"loaded {pcd_array.shape[0]} pcds \n")
    return np.min(min_xs), np.max(max_xs), np.min(min_ys), np.max(max_ys), pcd_array


def load_points_indeces(cfg):
    points_array = []
    regex_points = re.compile(".*\_points.npy$")
    objects_names = cfg.objects_names.split(",")
    points_files = []
    for root, dirs, files in os.walk(f"{repo_directory}/{cfg.data_path}"):
        for file in files:
            if regex_points.match(file):
                (
                    matched_object_string,
                    matched_object_boolean,
                ) = is_string_present_in_filename(file, objects_names)
                if matched_object_boolean:
                    points_files.append(
                        f"{repo_directory}/{cfg.data_path}/{matched_object_string}/{file}"
                    )
    points_files = natsort.natsorted(points_files)
    if len(points_files) != len(objects_names):
        print("not enough safe points files found, please create them first")
        return None
    for points_file in points_files:
        points = np.load(points_file, allow_pickle=True)
        points_array.append(points.T)
    points_array = np.asarray(points_array, dtype=object)
    print(f"loaded {points_array.shape[0]} safe points arrays \n")
    return points_array


def load_normals(cfg):
    points_array = []
    regex_points = re.compile(".*\_normals.npy$")
    objects_names = cfg.objects_names.split(",")
    points_files = []
    for root, dirs, files in os.walk(f"{repo_directory}/{cfg.data_path}"):
        for file in files:
            if regex_points.match(file):
                (
                    matched_object_string,
                    matched_object_boolean,
                ) = is_string_present_in_filename(file, objects_names)
                if matched_object_boolean:
                    points_files.append(
                        f"{repo_directory}/{cfg.data_path}/{matched_object_string}/{file}"
                    )
    points_files = natsort.natsorted(points_files)
    if len(points_files) != len(objects_names):
        print("not enough normals files found, please create them first")
        raise ValueError
    for points_file in points_files:
        points = np.load(points_file, allow_pickle=True)
        points_array.append(points)
    points_array = np.asarray(points_array, dtype=object)
    print(f"loaded {points_array.shape[0]} normal arrays \n")
    return points_array


def filter_reskin(reskin_reading, multiple_samples=False, norm=False):
    # fix dtype
    reskin_reading = np.array(reskin_reading, dtype=object)
    if multiple_samples:
        # extract lists of magnetometers values and temperatures as array of lists
        reskin_reading = np.squeeze(reskin_reading)[:, 2]
    # convert to list of lists
    reskin_reading = list(reskin_reading)
    # convert to nd array
    reskin_reading = np.asarray(reskin_reading, dtype=object)
    # eliminate temperatures
    reskin_reading = np.delete(reskin_reading, [0, 4, 8, 12, 16], 1)
    if norm:
        reskin_reading = np.swapaxes(reskin_reading, 0, 1)
        current_reskin = np.linalg.norm(np.mean(reskin_reading, axis=1), 2)
        return current_reskin
    else:
        return np.mean(reskin_reading, axis=0)

    return reskin_norm
