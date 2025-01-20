# Specific modules
from src.imagine2touch.localizers.aruco_cam_robot_world_localizer import (
    FrameUtil,
)  # Used to get transforms in the environment setting
from test_point_cloud_filter import (
    convert_image_to_point_cloud,
    point_cloud_info,
    mask_rgb_and_dep,
)
from src.imagine2touch.reskin_sensor.sensor_proc import ReSkinProcess, ReSkinSettings
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
from robot_io.cams.realsense.realsense import Realsense
from src.imagine2touch.utils import (
    add_ee,
    FIXED_ROBOT_ORN,
    inverse_transform,
    eulertoquat,
    move_to_pt_with_v_safe,
    euler_from_vector,
    homog_vector_to_3d,
    WCAMERA_IN_TCP,
    threed_vector_to_homog,
    ROBOT_IN_WORLD,
    AWAY_POSE,
    create_rotation_from_normal,
    create_homog_transformation_from_rotation,
    show_rgb_and_depth,
    MASK_KINECT_1,
    MASK_KINECT_2,
    get_crop_indeces,
)
from robot_io.utils.utils import pos_orn_to_matrix


# Standard useful libraries
import numpy as np
import open3d as o3d
import signal
import time
import sys
from PIL import Image
import rospy


# custom utilities
def estimate_normal(points, center):
    center_point = points[center]
    cross_normals = []
    for i in range(points.shape[0]):
        v_i = np.divide(
            np.subtract(points[i], center_point),
            np.linalg.norm(np.subtract(points[i], center_point), 2),
        )
        for j in range(i + 1, points.shape[0]):
            v_j = np.divide(
                np.subtract(points[j], center_point),
                np.linalg.norm(np.subtract(points[j], center_point), 2),
            )
            if np.abs(v_i.dot(np.transpose(v_j))) < 0.1:
                cross = np.divide(
                    np.cross(v_i, v_j), np.linalg.norm(np.cross(v_i, v_j), 2)
                )
                cross_normals.append(cross[0])
    cross_normals = np.asarray(cross_normals)
    if cross_normals.shape[0] == 0:
        return None
    return np.mean(cross_normals, axis=0)


def discretize_vector(point, vector):
    discretized_normal_array_negative = np.zeros((10, 3))
    discretized_normal_array_positive = np.zeros((10, 3))
    next_point = point  # tail of the normal
    for i in range(10):
        next_point = next_point - 0.005 * vector
        discretized_normal_array_negative[i] = next_point
    next_point = point  # tail of the negative normal
    for i in range(10):
        next_point = next_point + 0.005 * vector
        discretized_normal_array_positive[i] = next_point
    return discretized_normal_array_negative, discretized_normal_array_positive


def safety_one(normal):
    z_in_world = np.array([0, 0, 1])
    normal_in_world = normal  # orientation vector
    condition_1 = normal_in_world.dot(z_in_world) > 0.25
    print("safety one " + str(condition_1))
    return condition_1


def safety_two(pcd, point, nnn):
    s_pcd = o3d.geometry.PointCloud(
        pcd
    )  # transformed point cloud for processing safety
    s_pcd.translate(
        -1 * point
    )  # translate from origin of world to origin of target patch
    s_pcd.transform(
        create_homog_transformation_from_rotation(create_rotation_from_normal(nnn))
    )
    xs, ys, zs = point_cloud_info(s_pcd)
    condition_x = (xs < 0.015) & (xs > -0.015)  # side length of intended patch (1.5 cm)
    condition_y = (ys < 0.015) & (ys > -0.015)  # side length of intended patch (1.5 cm)
    condition_z = zs > 0.006  # 6 mm tolerance
    counter_condition_z = zs <= 0.006
    not_safe_two = np.where(condition_x & condition_y & condition_z)
    safe_two = np.where(condition_x & condition_y & counter_condition_z)
    condition_2 = not_safe_two[0].shape[0] == 0
    print("safety two " + str(condition_2))
    return not_safe_two, safe_two, condition_2


def safety_three(goal_point, goal_rot):
    goal_tcp_in_robot = pos_orn_to_matrix(goal_point, goal_rot)
    P1_IN_ROBOT = add_ee(
        goal_tcp_in_robot.dot(np.array([0.015, 0.09, -0.025, 1]))[:3], goal_orientation
    )
    P2_IN_ROBOT = add_ee(
        goal_tcp_in_robot.dot(np.array([-0.015, 0.09, -0.025, 1]))[:3], goal_orientation
    )
    P3_IN_ROBOT = add_ee(
        goal_tcp_in_robot.dot(np.array([0.015, -0.09, -0.025, 1]))[:3], goal_orientation
    )
    P4_IN_ROBOT = add_ee(
        goal_tcp_in_robot.dot(np.array([-0.015, -0.09, -0.025, 1]))[:3],
        goal_orientation,
    )
    P = np.vstack(
        (np.vstack((P1_IN_ROBOT, P2_IN_ROBOT)), np.vstack((P3_IN_ROBOT, P4_IN_ROBOT)))
    )
    condition_3 = min(ROBOT_IN_WORLD.dot(P[:, 2])) >= 0.01
    print("safety three " + str(condition_3))
    return condition_3


def filter_wrist_points(pcd, c_in_w, w_in_c):
    wc_pcd = o3d.geometry.PointCloud(pcd)
    wc_pcd.translate(-1 * c_in_w[:3, 3])
    wc_pcd.transform(create_homog_transformation_from_rotation(w_in_c[:3, :3]))
    xs, ys, zs = point_cloud_info(wc_pcd, display=True)
    # between 6.5 and 8 cm away on the alignment vector
    condition_z = (zs > -WCAMERA_IN_TCP[:, 3][2] + 0.057 + 0.065) & (
        zs < -WCAMERA_IN_TCP[:, 3][2] + 0.057 + 0.08
    )
    indices = np.where(condition_z)
    pcd = pcd.select_by_index(indices[0], invert=False)
    print(f"no. of points in wrist crop {np.asarray(pcd.points).shape[0]}")

    return pcd, indices


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("pass name of experiment")
        exit()

    # initialize devices
    robot = RobotClient("/robot_io_ros_server")  # name of server node
    camera = Realsense(img_type="rgb_depth")
    projection_matrix = camera.get_projection_matrix()
    resolution = np.reshape(
        np.array([camera.get_intrinsics()["width"], camera.get_intrinsics()["height"]]),
        (-1, 1),
    )
    sensor_settings = ReSkinSettings(
        num_mags=5, port="/dev/ttyACM0", baudrate=115200, burst_mode=True, device_id=1
    )
    sensor_process = ReSkinProcess(sensor_settings)
    sensor_process.start()
    time.sleep(1)

    # prepare recording
    reskin_recordings = []
    ambient_recordings = []
    ambient_recordings.append(sensor_process.get_data(num_samples=10))
    N = 10
    count_ambient = 0
    count_contact = 0

    # Robot reseting block
    start_pos = robot.get_state()["tcp_pos"]
    start_rot = robot.get_state()["tcp_orn"]
    start_pos_up = [start_pos[0], start_pos[1], 0.1]
    if add_ee(start_pos, FIXED_ROBOT_ORN)[2] < 0.1:
        robot.move_cart_pos_abs_ptp(add_ee(start_pos_up), start_rot)
        print("raised robot")
    else:
        robot.move_cart_pos_abs_ptp(
            start_pos, eulertoquat(FIXED_ROBOT_ORN)
        )  # reset robot orientation
        start_rot = FIXED_ROBOT_ORN  # update start rot
        print("oriented robot")
    time.sleep(1)
    start_pos = 1 * robot.get_state()["tcp_pos"][:3]  # update start pos
    print("robot_reset done")

    # transforms and pcd from scene
    frame_util = FrameUtil(robot)
    frame_util.start()
    rgb, dep = frame_util.camera.get_image()
    while frame_util._T_camera_in_robot is None or frame_util._T_world_in_robot is None:
        print("transform is none")
        continue
    print("transform calculated")
    T_world_in_camera = inverse_transform(
        frame_util.T_robot_in_world.dot(frame_util._T_camera_in_robot)
    )
    WORLD_IN_ROBOT = frame_util.T_world_in_robot
    pcd = convert_image_to_point_cloud(
        "kinect",
        rgb,
        dep,
        T_world_in_camera,
        0.008,
        0.2,
        -0.06,
        MASK_KINECT_1,
        MASK_KINECT_2,
    )
    original_points = np.asarray(pcd.points)
    normals = pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    original_normals = np.asarray(pcd.normals)

    # Start automation
    for i in range(N):
        condition_1 = False
        condition_2 = False
        condition_3 = False

        while (not condition_1) or (not condition_2) or (not condition_3):
            # reset points and normals to original scene
            points = original_points
            normals = original_normals

            # get a sampled point from pcd
            elements = np.where(points[:, 2] >= -100)  # 100 meter
            probabilities = np.ones(elements[0].shape[0]) * (1 / elements[0].shape[0])
            sampled_points_indices = np.random.choice(
                elements[0], 1, p=probabilities, replace=False
            )
            sampled_point_index = sampled_points_indices[0]
            sampled_points = points[sampled_points_indices]
            sampled_point = sampled_points[0]
            sampled_point_homog = threed_vector_to_homog(sampled_point)
            sampled_point_homog_in_robot = WORLD_IN_ROBOT.dot(sampled_point_homog)
            sampled_point_in_robot = homog_vector_to_3d(sampled_point_homog_in_robot)
            print(f"sampled point is {sampled_point_homog[:3]}")

            # get normal to sampled point
            sampled_normals = normals[sampled_points_indices]
            sampled_normal = sampled_normals[0]
            (
                discretized_normal_array_negative,
                discretized_normal_array_positive,
            ) = discretize_vector(sampled_point, sampled_normal)
            print(f"sampled normal is {sampled_normal}")

            # get goal orientation from normal
            negative_normal = -1 * sampled_normal
            alignment_vector = negative_normal / np.linalg.norm(negative_normal, 2)
            goal_orientation = euler_from_vector(0, alignment_vector)

            # move along the alignment vector
            alpha = 0.08  # 8cm
            sampled_point_in_robot_prepare = (
                sampled_point_in_robot - alpha * alignment_vector
            )
            sampled_point_in_robot = (
                sampled_point_in_robot - 0.001 * alignment_vector
            )  # 1mm

            # safety checks
            condition_1 = safety_one(sampled_normal)
            not_safe_two, safe_two, condition_2 = safety_two(
                pcd, sampled_point, alignment_vector
            )
            condition_3 = safety_three(
                sampled_point_in_robot[:3], eulertoquat(goal_orientation)
            )

            # color areas and draw geometries for visualization#
            points = np.append(points, discretized_normal_array_negative, axis=0)
            points = np.append(points, discretized_normal_array_positive, axis=0)
            colors = np.asarray(pcd.colors)
            colors_normal_negative = np.zeros((10, 3))  # black
            if condition_1:
                colors_normal_positive = np.zeros((10, 3)) + np.array(
                    [0, 1, 1]
                )  # Indigo
            else:
                colors_normal_positive = np.zeros((10, 3)) + np.array([1, 0, 0])  # Red
            colors = np.append(colors, colors_normal_negative, axis=0)
            colors = np.append(colors, colors_normal_positive, axis=0)
            pcd_viz = o3d.geometry.PointCloud()
            pcd_viz.points = o3d.utility.Vector3dVector(points)
            intended_color_safe = np.zeros((safe_two[0].shape[0], 3)) + np.array(
                [0, 1, 0]
            )  # green
            intended_color_unsafe = np.zeros((not_safe_two[0].shape[0], 3)) + np.array(
                [1, 0, 0]
            )  # red
            colors[safe_two] = intended_color_safe
            colors[not_safe_two] = intended_color_unsafe
            colors[sampled_point_index] = np.array([1, 1, 0])  # yellow
            pcd_viz.colors = o3d.utility.Vector3dVector(colors)

            # stopping signal handler
            end = False

            def cb_int(*args):
                global end
                end = True

            signal.signal(signal.SIGINT, cb_int)

            # Move Robot
            if condition_1 and condition_2 and condition_3:
                # visualize movement
                # o3d.visualization.draw_geometries([pcd_viz],'window')

                # target positions
                prep = add_ee(sampled_point_in_robot_prepare, goal_orientation)
                final = add_ee(sampled_point_in_robot, goal_orientation)

                # move to prepared point in high z plane
                try:
                    robot.move_cart_pos_abs_ptp(
                        [prep[0], prep[1], 0.20], eulertoquat(FIXED_ROBOT_ORN)
                    )
                except rospy.service.ServiceException as e:
                    print("failed to move to fixed z plane\n")
                    continue
                print("success in moving to fixed z plane")
                time.sleep(0.1)

                # orient on the alignment vector and let the wrist cam see
                T_tcp_in_robot = pos_orn_to_matrix(prep, eulertoquat(goal_orientation))
                cam_view = np.array(
                    [-WCAMERA_IN_TCP[:, 3][0], -WCAMERA_IN_TCP[:, 3][1], 0, 1]
                )
                cam_view_in_robot = T_tcp_in_robot.dot(cam_view)
                try:
                    robot.move_cart_pos_abs_ptp(
                        cam_view_in_robot[:3], eulertoquat(goal_orientation)
                    )
                except rospy.service.ServiceException as e:
                    print("failed to let cam see\n")
                    continue
                print("success in orienting and letting cam see")
                time.sleep(0.1)

                # procedure correction
                rgb_w, depth_w = camera.get_image()
                T_tcp_in_robot = robot.get_tcp_pose()
                W_CAM_IN_WORLD = ROBOT_IN_WORLD.dot(T_tcp_in_robot.dot(WCAMERA_IN_TCP))
                extrinsic = inverse_transform(W_CAM_IN_WORLD)
                pcd_wrist = convert_image_to_point_cloud(
                    "realsense",
                    rgb_w,
                    depth_w,
                    extrinsic,
                    minz=0.008,
                    maxz=0.2,
                    minx=-0.06,
                )
                pcd_wc, indices = filter_wrist_points(
                    pcd_wrist, W_CAM_IN_WORLD, extrinsic
                )
                pcd_tree = o3d.geometry.KDTreeFlann(pcd_wc)
                try:
                    nearest_point_info = pcd_tree.search_knn_vector_3d(
                        sampled_point, 100
                    )
                except RuntimeError as e:
                    print("no nearest points, run time error\n")
                    continue
                nearest_point_index = nearest_point_info[1]
                pcd_nearest = pcd_wc.select_by_index(nearest_point_index, invert=False)
                pcd_nearest.estimate_normals()
                points = np.asarray(pcd_nearest.points)
                if points.shape[0] == 0:
                    print("no nearest points found\n")
                    continue
                try:
                    pcd_nearest.orient_normals_consistent_tangent_plane(10)
                except RuntimeError as e:
                    print("couldnt orient normals consistently\n")
                normals = np.asarray(pcd_nearest.normals)
                condition_1 = False
                condition_2 = False
                condition_3 = False
                count = 0
                while (
                    (not condition_1) or (not condition_2) or (not condition_3)
                ) and (count <= 100):
                    elements = np.where(points[:, 2] >= -100)
                    probabilities = np.ones(elements[0].shape[0]) * (
                        1 / elements[0].shape[0]
                    )
                    sampled_points_indices = np.random.choice(
                        elements[0], 1, p=probabilities, replace=False
                    )
                    sampled_point_index = sampled_points_indices[0]
                    sampled_points = points[sampled_points_indices]
                    sampled_point = sampled_points[0]
                    print(f"sampled point is {sampled_point}")
                    sampled_normal = normals[sampled_point_index]
                    if sampled_normal is None or (
                        np.linalg.norm(sampled_normal, 2) == 0
                    ):
                        count = count + 1
                        continue
                    print(f"sampled normal is {sampled_normal}")
                    sampled_point_homog = threed_vector_to_homog(sampled_point)
                    sampled_point_homog_in_robot = WORLD_IN_ROBOT.dot(
                        sampled_point_homog
                    )
                    sampled_point_in_robot = homog_vector_to_3d(
                        sampled_point_homog_in_robot
                    )
                    negative_normal = -1 * sampled_normal
                    alignment_vector = negative_normal / np.linalg.norm(
                        negative_normal, 2
                    )
                    goal_orientation = euler_from_vector(0, alignment_vector)
                    alpha = 0.08  # 8cm
                    sampled_point_in_robot_prepare = (
                        sampled_point_in_robot - alpha * alignment_vector
                    )
                    sampled_point_in_robot = (
                        sampled_point_in_robot - 0.001 * alignment_vector
                    )  # 1mm
                    condition_1 = safety_one(sampled_normal)
                    not_safe_two, safe_two, condition_2 = safety_two(
                        pcd_wrist, sampled_point, alignment_vector
                    )
                    condition_3 = safety_three(
                        sampled_point_in_robot[:3], eulertoquat(goal_orientation)
                    )
                    count = count + 1
                    # corrected target positions
                    prep = add_ee(sampled_point_in_robot_prepare, goal_orientation)
                    final = add_ee(sampled_point_in_robot, goal_orientation)
                if count == 101:  # failed 100 times
                    print("couldn't correct normal, correction failed\n")
                    continue

                # orient on corrected alignment vector
                T_tcp_in_robot = pos_orn_to_matrix(prep, eulertoquat(goal_orientation))
                cam_view = np.array(
                    [-WCAMERA_IN_TCP[:, 3][0], -WCAMERA_IN_TCP[:, 3][1], 0, 1]
                )
                cam_view_in_robot = T_tcp_in_robot.dot(cam_view)
                try:
                    robot.move_cart_pos_abs_ptp(
                        cam_view_in_robot, eulertoquat(goal_orientation)
                    )
                except rospy.service.ServiceException as e:
                    print("failed to orient on corrected alignment vector\n")
                    continue
                print("success in orienting on corrected alignment vector")
                time.sleep(0.1)

                # take image
                T_tcp_in_robot = robot.get_tcp_pose()
                W_CAM_IN_WORLD = ROBOT_IN_WORLD.dot(T_tcp_in_robot.dot(WCAMERA_IN_TCP))
                WORLD_IN_W_CAM = inverse_transform(W_CAM_IN_WORLD)
                hom_point_in_cam = WORLD_IN_W_CAM.dot(
                    threed_vector_to_homog(sampled_point)
                )
                crop_indeces = get_crop_indeces(
                    hom_point_in_cam,
                    inverse_transform(WCAMERA_IN_TCP),
                    np.divide(projection_matrix, 100),
                    resolution,
                )
                rgb, dep = camera.get_image()
                x_1 = min(crop_indeces[0, :].astype(int))
                x_2 = max(crop_indeces[0, :].astype(int))
                y_1 = min(crop_indeces[1, :].astype(int))
                y_2 = max(crop_indeces[1, :].astype(int))
                rgb = rgb[y_1:y_2, x_1:x_2]
                dep = dep[y_1:y_2, x_1:x_2]  # TODO::verify crop using mask
                print("took shot\n")

                # prepare position for contact
                try:
                    robot.move_cart_pos_abs_ptp(prep, eulertoquat(goal_orientation))
                    print("success in moving to prep")
                except rospy.service.ServiceException as e:
                    print("failed to prepare for contact\n")
                    continue
                time.sleep(0.1)

                # make contact
                rot = robot.get_state()["tcp_orn"]
                move_to_pt_with_v_safe(
                    robot,
                    goal_point=final,
                    goal_rot=rot,
                    p_i=0.01,
                    dt=0.2,
                    end=end,
                    f_z_max=-40,
                    direct=True,
                    direction=alignment_vector,
                )
                print("sucess in moving to point")
                time.sleep(0.1)

                # record data
                count_contact = count_contact + 1
                reskin_recordings.append(sensor_process.get_data(num_samples=1))
                count_ambient = count_ambient + 1
                if count_ambient == 10:
                    ambient_recordings.append(sensor_process.get_data(num_samples=10))
                    count_ambient = 0
                rgb_im = Image.fromarray(rgb)
                rgb_im.save(f"./cutter_images/{sys.argv[1]}_rgb_{count_contact}.jpeg")
                dep_im = (dep * 1000).astype(np.uint16)
                dep_im = Image.fromarray(dep_im)
                dep_im.save(f"./cutter_images/{sys.argv[1]}_depth_{count_contact}.tif")
                print("saved data")

                # move back to prepare position
                move_to_pt_with_v_safe(
                    robot,
                    goal_point=prep,
                    goal_rot=eulertoquat(goal_orientation),
                    p_i=0.01,
                    dt=0.2,
                    end=end,
                    f_z_max=-40,
                    direct=True,
                    direction=alignment_vector,
                )
                print("success in moving away on the alignment vector")

            else:
                print("looking for a feasible point")

            if sensor_process.is_alive():
                print("sensor is alive")
            else:
                # early logging
                print("sensor dead, will save")
                reskin_recordings = np.array(reskin_recordings)
                with open(f"./cutter_tactile/{sys.argv[1]}_reskin", "wb") as readings:
                    np.save(readings, reskin_recordings)
                with open(
                    f"./cutter_tactile/{sys.argv[1]}_reskin_ambient", "wb"
                ) as ambient_readings:
                    np.save(ambient_readings, ambient_recordings)
                sys.exit()
    # logging
    print("Finished N contacts")
    reskin_recordings = np.array(reskin_recordings)
    with open(f"./cutter_tactile/{sys.argv[1]}_reskin", "wb") as readings:
        np.save(readings, reskin_recordings)
    with open(
        f"./cutter_tactile/{sys.argv[1]}_reskin_ambient", "wb"
    ) as ambient_readings:
        np.save(ambient_readings, ambient_recordings)

    # move away
    robot.move_cart_pos_abs_ptp(AWAY_POSE, eulertoquat(FIXED_ROBOT_ORN))
    print("success in moving away fully")

    # turn off automation
    if sensor_process.is_alive:
        sensor_process.pause_streaming()
        sensor_process.join()
    sys.exit()
