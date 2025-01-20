# repo modules
from src.imagine2touch.localizers.wrist_camera_localizer import (
    generate_step_orientation,
    generate_step_pos,
)
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
from robot_io.cams.realsense.realsense import Realsense
from src.imagine2touch.utils.utils import (
    FIXED_ROBOT_ORN,
    ROBOT_IN_WORLD,
    WORLD_IN_ROBOT,
    WCAMERA_IN_TCP,
    inverse_transform,
    eulertoquat,
    convert_image_to_point_cloud,
    point_cloud_info,
    debounce_tcp_pose,
)

# Standard useful libraries
import numpy as np
import open3d as o3d
import time
from PIL import Image
import rospy
import os
import hydra
from omegaconf import OmegaConf
import sys


def wait_until_stable_joint_velocities(robot):
    # wait for joint velocity to be very close to 0
    while True:
        if np.all(np.abs(robot.get_state()["joint_velocities"]) < 1e-3):
            break
        time.sleep(0.1)
        print("waiting for robot to stop moving")
        print(robot.get_state()["joint_velocities"])


def capture_wristcam_image():
    """
    will work only for realsense cameras
    """
    camera = Realsense(img_type="rgb_depth")
    rgb_w, depth_w = camera.get_image()
    del camera
    time.sleep(0.1)  # ensure camera process terminated
    return rgb_w, depth_w


if __name__ == "__main__":
    # script configuration and constants
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("save_pcds.yaml")
    cfg.max_inclination = cfg.max_inclination * np.pi / 180
    # initialize devices
    robot = RobotClient("/robot_io_ros_server")
    camera = Realsense(img_type="rgb_depth")
    projection_matrix = camera.get_projection_matrix()
    resolution = np.reshape(
        np.array([camera.get_intrinsics()["width"], camera.get_intrinsics()["height"]]),
        (-1, 1),
    )
    del camera

    ### start devices
    time.sleep(1)
    ## robot reseting block
    start_pos = robot.get_state()["tcp_pos"]
    start_rot = robot.get_state()["tcp_orn"]
    start_pos_up = [start_pos[0], start_pos[1], cfg.minimum_robot_height]
    if start_pos[2] < cfg.minimum_robot_height:
        robot.move_cart_pos_abs_ptp(start_pos_up, eulertoquat(FIXED_ROBOT_ORN))
        print("raised robot")
    else:
        # reset robot orientation
        robot.move_cart_pos_abs_ptp(start_pos, eulertoquat(FIXED_ROBOT_ORN))
        # update start_rot
        start_rot = FIXED_ROBOT_ORN
        print("oriented robot")
    time.sleep(1)
    # update start pos
    start_pos = 1 * robot.get_state()["tcp_pos"][:3]
    print("robot_reset done")

    ## create combined views PCD with wrist cam
    pcd_views = []
    original_points = []
    i = 1
    pose_in_world = [float(num) for num in cfg.starting_corner_in_world.split(",")]
    orns = generate_step_orientation(
        cfg.fixed_roll, cfg.max_inclination, cfg.orientation_step / 2
    )
    poses = generate_step_pos(
        pose_in_world,
        cfg.r,
        cfg.initial_orientation,
        len(orns),
        cfg.orientation_step,
        cfg.translation_step,
    )
    orns = np.repeat(orns, 7, axis=0)
    views = zip(poses, orns)
    camera = Realsense(img_type="rgb_depth")

    for view, orn in views:
        # for view,orn in zip(VIEWS,ORNS):
        view = WORLD_IN_ROBOT.dot(view)[:3]
        try:
            robot.move_cart_pos_abs_ptp(view, eulertoquat(orn))
        except rospy.service.ServiceException as e:
            if (
                np.linalg.norm(robot.get_state()["tcp_pos"][:3] - view, 2)
                > cfg.pos_tolerance
            ):
                print("failed to get view")
                continue
            else:
                print("got view within the following tolerance: ")
                print(np.linalg.norm(robot.get_state()["tcp_pos"][:3] - view, 2))
                # ensure updated robot state
                time.sleep(1)
        # wait for robot to stop moving
        wait_until_stable_joint_velocities(robot)
        T_tcp_in_robot = debounce_tcp_pose(robot, delay=False)
        W_CAM_IN_WORLD = ROBOT_IN_WORLD.dot(T_tcp_in_robot.dot(WCAMERA_IN_TCP))
        # W_CAM_IN_WORLD = T_tcp_in_robot.dot(WCAMERA_IN_TCP)
        WORLD_IN_W_CAM = inverse_transform(W_CAM_IN_WORLD)
        rgb_w, depth_w = camera.get_image()

        pcd_view = convert_image_to_point_cloud(
            "realsense",
            rgb_w,
            depth_w,
            WORLD_IN_W_CAM,
            minz=cfg.crop.minz,
            maxz=cfg.crop.maxz,
            minx=cfg.crop.minx,
            maxx=cfg.crop.maxx,
            miny=cfg.crop.miny,
            maxy=cfg.crop.maxy,
            voxel=False,
            segment=True,
        )
        # display single view pcds information
        # if pcd is not empty
        if np.array(pcd_view.points).shape[0] > 0:
            point_cloud_info(pcd_view, True)
            pcd_views.append(pcd_view)
        # save single view pcds
        # o3d.io.write_point_cloud(f"{cfg.save_directory}/{cfg.object_name}_view_{i}.pcd", pcd_view)
        i += 1

    # combine pcds
    pcds = pcd_views
    pcd = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcd += pcds[point_id]
    pcd.estimate_normals()

    folder_path = f"{cfg.save_directory}/{cfg.object_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{cfg.object_name}' created.")
    else:
        print(f"Folder '{cfg.object_name}' already exists.")

    # save combined pcd
    o3d.io.write_point_cloud(
        f"{cfg.save_directory}/{cfg.object_name}/{cfg.object_name}_combined.pcd", pcd
    )
    pcd = o3d.io.read_point_cloud(
        f"{cfg.save_directory}/{cfg.object_name}/{cfg.object_name}_combined.pcd"
    )
    # visualize combined views pcd
    original_points = np.asarray(pcd.points)
    original_normals = np.asarray(pcd.normals)
    o3d.visualization.draw_geometries([pcd])
