# Specific modules
from src.imagine2touch.reskin_sensor.sensor_proc import ReSkinProcess, ReSkinSettings
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
from robot_io.utils.utils import pos_orn_to_matrix
from src.imagine2touch.utils.utils import (
    FIXED_ROBOT_ORN,
    eulertoquat,
    move_to_pt_with_v_safe,
    euler_from_vector,
    homog_vector_to_3d,
    threed_vector_to_homog,
    AWAY_POSE,
    create_rotation_from_normal,
    create_homog_transformation_from_rotation,
    point_cloud_info,
    plot_reskin,
    HOME_POSE,
    WORLD_IN_ROBOT,
    ROBOT_IN_WORLD,
)

# Standard useful libraries
import numpy as np
import open3d as o3d
import signal
import time
import sys
from PIL import Image
import rospy
import os
from argparse import ArgumentParser
import re


# custom utilities


def custom_visualization_pcd(
    pcd, points, discretized_normal_array_negative, discretized_normal_array_positive
):
    # color areas and draw geometries for visualization#
    points = np.append(points, discretized_normal_array_negative, axis=0)
    points = np.append(points, discretized_normal_array_positive, axis=0)
    colors = np.asarray(pcd.colors)
    colors_normal_negative = np.zeros((10, 3))  # black
    if condition_1:
        colors_normal_positive = np.zeros((10, 3)) + np.array([0, 1, 1])  # indigo
    else:
        colors_normal_positive = np.zeros((10, 3)) + np.array([1, 0, 0])  # red
    colors = np.append(colors, colors_normal_negative, axis=0)
    colors = np.append(colors, colors_normal_positive, axis=0)
    pcd_viz = o3d.geometry.PointCloud()
    pcd_viz.points = o3d.utility.Vector3dVector(points)
    color_safe = np.zeros((safe_two[0].shape[0], 3)) + np.array([0, 1, 0])  # green
    color_unsafe = np.zeros((not_safe_two[0].shape[0], 3)) + np.array([1, 0, 0])  # red
    colors[safe_two] = color_safe
    colors[not_safe_two] = color_unsafe
    colors[sampled_point_index] = np.array(
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
    # print("safety one "+str(condition_1))
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
    P1_IN_ROBOT = goal_tcp_in_robot.dot(np.array([x, y, -z, 1]))[:3]
    P2_IN_ROBOT = goal_tcp_in_robot.dot(np.array([-x, y, -z, 1]))[:3]
    P3_IN_ROBOT = goal_tcp_in_robot.dot(np.array([x, -y, -z, 1]))[:3]
    P4_IN_ROBOT = goal_tcp_in_robot.dot(np.array([-x, -y, -z, 1]))[:3]
    P = np.vstack((P1_IN_ROBOT, P2_IN_ROBOT, P3_IN_ROBOT, P4_IN_ROBOT))
    condition_3 = min(ROBOT_IN_WORLD.dot(P[:, 2])) >= threshold
    # print("safety three " +str(condition_3))
    return condition_3


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(description="Object recognition Script")
    parser.add_argument("object_name", help="object name")
    parser.add_argument("experiment_number", help="experiment number")
    args = parser.parse_args()

    # initialize devices
    robot = RobotClient("/robot_io_ros_server")
    sensor_settings = ReSkinSettings(
        num_mags=5, port="/dev/ttyACM0", baudrate=115200, burst_mode=True, device_id=1
    )
    sensor_process = ReSkinProcess(sensor_settings)
    sensor_process.start()
    time.sleep(1)

    # prepare recording
    reskin_recordings = []
    ambient_recordings = []
    ambient_recordings = np.array(ambient_recordings, dtype=object)
    prepare_poses = []
    final_poses = []
    rotation_list = []
    print("taking ambient reading")
    ambient_recordings = np.append(
        ambient_recordings,
        np.array(sensor_process.get_data(num_samples=10), dtype=object),
    )
    C = 0
    CONTACTS = 50
    N = 1
    count_ambient = 0
    count_contact = 0
    training_model_tcp_to_patch = 0.04

    # Robot reseting block
    start_pos = robot.get_state()["tcp_pos"]
    start_rot = robot.get_state()["tcp_orn"]
    start_pos_up = [start_pos[0], start_pos[1], 0.1]
    if start_pos[2] < 0.1:
        robot.move_cart_pos_abs_ptp(start_pos_up, eulertoquat(FIXED_ROBOT_ORN))
        print("raised robot")
    else:
        robot.move_cart_pos_abs_ptp(
            start_pos, eulertoquat(FIXED_ROBOT_ORN)
        )  # reset robot orientation
        start_rot = FIXED_ROBOT_ORN  # update start_rot
        print("oriented robot")
    time.sleep(1)
    start_pos = 1 * robot.get_state()["tcp_pos"][:3]  # update start pos
    print("robot_reset done")

    # Load PCDs
    pcd_array = []
    regex = re.compile(".*\.ply$")
    pcd_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if regex.match(file):
                pcd_files.append(dir_path + "/" + file)
    for file in pcd_files:
        pcd = o3d.io.read_point_cloud(file)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(200)
        pcd_array.append(pcd)
    pcd_array = np.asarray(pcd_array)
    print(f"loaded {pcd_array.shape[0]} pcds \n")

    # Start automation
    while C < CONTACTS:
        pcd_probabilities = np.ones(pcd_array.shape[0]) * (1 / pcd_array.shape[0])
        sampled_pcds_indeces = np.random.choice(
            np.arange(0, pcd_array.shape[0]), 1, p=pcd_probabilities, replace=False
        )
        sampled_pcd_index = sampled_pcds_indeces[0]
        pcd = pcd_array[sampled_pcd_index]
        original_points = np.asarray(pcd.points)
        original_normals = np.asarray(pcd.normals)
        C += 1
        # print("sampled a pcd")

        # Start point sampling
        for i in range(N):
            condition_1 = False
            condition_2 = False
            condition_3 = False

            while (not condition_1) or (not condition_2) or (not condition_3):
                # restore a good starting position
                current_pos = robot.get_state()["tcp_pos"]
                if i > 0:
                    if current_pos[2] > 0.2 or current_pos[2] < 0.1:
                        # print('moving away to restore a good position')
                        robot.move_cart_pos_abs_ptp(
                            HOME_POSE, eulertoquat(FIXED_ROBOT_ORN)
                        )

                # reset points and normals to original scene
                points = original_points
                normals = original_normals

                # get a sampled point from pcd
                elements = np.where(points[:, 2] >= -100)  # hack to get all indices
                probabilities = np.ones(elements[0].shape[0]) * (
                    1 / elements[0].shape[0]
                )
                sampled_points_indices = np.random.choice(
                    elements[0], 1, p=probabilities, replace=False
                )
                sampled_point_index = sampled_points_indices[0]
                sampled_points = points[sampled_points_indices]
                sampled_point = sampled_points[0]
                sampled_point_homog = threed_vector_to_homog(sampled_point)
                sampled_point_homog_in_robot = WORLD_IN_ROBOT.dot(sampled_point_homog)
                sampled_point_in_robot = homog_vector_to_3d(
                    sampled_point_homog_in_robot
                )
                # print(f'sampled point is {sampled_point_homog[:3]}')

                # get normal to sampled point
                sampled_normals = normals[sampled_points_indices]
                sampled_normal = sampled_normals[0]
                (
                    discretized_normal_array_negative,
                    discretized_normal_array_positive,
                ) = discretize_vector(sampled_point, sampled_normal, 0.005, 10)
                # print(f'sampled normal is {sampled_normal}')

                # get goal orientation from normal
                negative_normal = -1 * sampled_normal
                alignment_vector = negative_normal / np.linalg.norm(negative_normal, 2)
                goal_orientation = euler_from_vector(0, alignment_vector)

                # move along the alignment vector
                alpha = 0.08
                sampled_point_in_robot_prepare = (
                    sampled_point_in_robot - alpha * alignment_vector
                )

                # safety checks
                condition_1 = safety_one(sampled_normal, 0.25)
                not_safe_two, safe_two, condition_2 = safety_two(
                    pcd, sampled_point, alignment_vector, 0.0085, 0.001
                )
                condition_3 = safety_three(
                    sampled_point_in_robot[:3],
                    eulertoquat(goal_orientation),
                    [0.015, 0.09, 0.058],
                    0.01,
                )

                # color areas and draw geometries for visualization#
                pcd_viz = custom_visualization_pcd(
                    pcd,
                    points,
                    discretized_normal_array_negative,
                    discretized_normal_array_positive,
                )

                # stopping signal handler
                end = False

                def cb_int(*args):
                    global end
                    end = True

                signal.signal(signal.SIGINT, cb_int)

                # Move Robot
                if condition_1 and condition_2 and condition_3:
                    # visualize next contact
                    vis = o3d.visualization.Visualizer()
                    vis.create_window()
                    vis.add_geometry(pcd_viz)
                    vis.run()

                    # target positions
                    prepare_contact = sampled_point_in_robot_prepare
                    final_contact = sampled_point_in_robot

                    # prepare position for contact
                    try:
                        robot.move_cart_pos_abs_ptp(
                            prepare_contact, eulertoquat(goal_orientation)
                        )
                        # print("success in moving to prepare_contact")
                    except rospy.service.ServiceException as e:
                        print("failed to prepare for contact\n")
                        continue
                    time.sleep(0.1)

                    # make contact
                    rotation = robot.get_state()["tcp_orn"]
                    contact, final_pose = move_to_pt_with_v_safe(
                        robot,
                        goal_point=final_contact,
                        goal_rot=rotation,  # eulertoquat(goal_orientation),
                        p_i=0.005,
                        dt=0.1,
                        end=end,
                        f_z_max=-6,
                        direct=True,
                        direction=alignment_vector,
                        sensor_process=sensor_process,
                    )
                    if contact == -1:
                        print("point too far, failed contact")
                        continue
                    # print("sucess in contacting point")
                    time.sleep(1)

                    # record data
                    count_contact = count_contact + 1
                    reskin_recordings.append(sensor_process.get_data(num_samples=1))
                    plot_reskin(reskin_recordings)
                    if count_ambient == 10:
                        ambient_recordings = np.append(
                            ambient_recordings,
                            np.array(
                                sensor_process.get_data(num_samples=10), dtype=object
                            ),
                        )
                        count_ambient = 0
                    prepare_poses.append(
                        np.add(
                            prepare_contact,
                            (alpha - training_model_tcp_to_patch) * alignment_vector,
                        )
                    )
                    final_poses.append(final_pose["tcp_pos"])
                    rotation_list.append(final_pose["tcp_orn"])
                    reskin_recordings_np = np.array(reskin_recordings, dtype=object)
                    with open(
                        f"{dir_path}/data/{args.object_name}_tactile/{args.experiment_number}_reskin",
                        "wb",
                    ) as readings:
                        np.save(readings, reskin_recordings_np)
                    with open(
                        f"{dir_path}/data/{args.object_name}_tactile/{args.experiment_number}_reskin_ambient",
                        "wb",
                    ) as ambient_readings:
                        np.save(ambient_readings, ambient_recordings)
                    with open(
                        f"{dir_path}/data/{args.object_name}_poses/{args.experiment_number}_prepare_contact",
                        "wb",
                    ) as prepare:
                        np.save(prepare, prepare_poses)
                    with open(
                        f"{dir_path}/data/{args.object_name}_poses/{args.experiment_number}_final_contact",
                        "wb",
                    ) as final:
                        np.save(final, final_poses)
                    with open(
                        f"{dir_path}/data/{args.object_name}_poses/{args.experiment_number}_rotations",
                        "wb",
                    ) as rotation:
                        np.save(rotation, rotation_list)
                    # print("saved data")

                    # move back to prepare position
                    rotation = robot.get_state()["tcp_orn"]
                    away, _ = move_to_pt_with_v_safe(
                        robot,
                        goal_point=prepare_contact,
                        goal_rot=rotation,
                        p_i=0.005,
                        dt=0.1,
                        end=end,
                        f_z_max=-20,
                        direct=True,
                        direction=alignment_vector,
                    )
                    print("\n")
                    if away == 2:
                        robot.move_cart_pos_abs_ptp(
                            AWAY_POSE, eulertoquat(FIXED_ROBOT_ORN)
                        )
                        print("moved to away pose forcefully")
                    else:
                        pass
                        # print("success in moving away on the alignment vector")
                    time.sleep(1)
                    count_ambient = count_ambient + 1
                    if count_ambient == 10:
                        ambient_recordings = np.append(
                            ambient_recordings,
                            np.array(
                                sensor_process.get_data(num_samples=10), dtype=object
                            ),
                        )
                        count_ambient = 0
                else:
                    pass
                    # print("looking for a feasible point")

                if sensor_process.is_alive():
                    pass
                    # print('sensor is alive')
                else:
                    # early logging
                    print("sensor dead")
                    with open(
                        f"{dir_path}/data/{args.object_name}_tactile/{args.experiment_number}_reskin",
                        "wb",
                    ) as readings:
                        np.save(readings, reskin_recordings_np)
                    with open(
                        f"{dir_path}/data/{args.object_name}_tactile/{args.experiment_number}_reskin_ambient",
                        "wb",
                    ) as ambient_readings:
                        np.save(ambient_readings, ambient_recordings)
                    print("saved reskin data")
                    sys.exit()
    # logging
    print("Finished N contacts")
    with open(
        f"{dir_path}/data/{args.object_name}_tactile/{args.experiment_number}_reskin",
        "wb",
    ) as readings:
        np.save(readings, reskin_recordings_np)
    with open(
        f"{dir_path}/data/{args.object_name}_tactile/{args.experiment_number}_reskin_ambient",
        "wb",
    ) as ambient_readings:
        np.save(ambient_readings, ambient_recordings)
    print("saved reskin data")

    # move away
    robot.move_cart_pos_abs_ptp(AWAY_POSE, eulertoquat(FIXED_ROBOT_ORN))
    print("success in moving away fully")

    # turn off automation
    if sensor_process.is_alive:
        sensor_process.pause_streaming()
        sensor_process.join()
    sys.exit()
