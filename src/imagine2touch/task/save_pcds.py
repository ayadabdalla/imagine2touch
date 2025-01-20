# repo modules
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
from robot_io.cams.realsense.realsense import Realsense
from robot_io.utils.utils import pos_orn_to_matrix
from src.imagine2touch.reskin_sensor.sensor_proc import ReSkinProcess, ReSkinSettings
from src.imagine2touch.utils.utils import (
    FIXED_ROBOT_ORN,
    HOME_POSE,
    VIEWS,
    ORNS,
    ROBOT_IN_WORLD,
    WORLD_IN_ROBOT,
    WCAMERA_IN_TCP,
    AWAY_POSE,
    inverse_transform,
    eulertoquat,
    move_to_pt_with_v_safe,
    euler_from_vector,
    homog_vector_to_3d,
    threed_vector_to_homog,
    get_crop_indeces,
    convert_image_to_point_cloud,
    point_cloud_info,
    plot_reskin,
    segment_point_cloud,
    debounce_tcp_pose,
    filter_reskin,
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
import hydra
from omegaconf import OmegaConf
import sys


def capture_wristcam_image():
    """
    will work only for realsense cameras
    """
    camera = Realsense(img_type="rgb_depth")
    rgb_w, depth_w = camera.get_image()
    del camera
    time.sleep(1)  # ensure camera process terminated
    return rgb_w, depth_w


if __name__ == "__main__":
    # script configuration and constants
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("save_pcds.yaml")

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
    # del VIEWS[-2]
    # del ORNS[-2]
    for view, orn in zip(VIEWS, ORNS):
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
                time.sleep(3)
        T_tcp_in_robot = debounce_tcp_pose(robot)
        W_CAM_IN_WORLD = ROBOT_IN_WORLD.dot(T_tcp_in_robot.dot(WCAMERA_IN_TCP))
        WORLD_IN_W_CAM = inverse_transform(W_CAM_IN_WORLD)
        rgb_w, depth_w = capture_wristcam_image()
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

    # save combined pcd

    folder_path = f"{cfg.save_directory}/{cfg.object_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{cfg.object_name}' created.")
    else:
        print(f"Folder '{cfg.object_name}' already exists.")

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
