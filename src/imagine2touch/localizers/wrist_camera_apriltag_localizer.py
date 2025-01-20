# Import standard libraries
from pathlib import Path
import numpy as np
import cv2
from src.imagine2touch.task.save_pcds_extra_views import (
    custom_wait,
    wait_until_stable_joint_velocities,
)
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
import rospy
import hydra
from omegaconf import OmegaConf
import os
import time
import sys

# Import repo modules
from robot_io.cams.realsense.realsense import Realsense
from src.imagine2touch.utils.utils import (
    WORLD_IN_ROBOT,
    FIXED_ROBOT_ORN,
    eulertoquat,
    search_folder,
)
from robot_io.cams.realsense.realsense import Realsense
from robot_io.marker_detection.core.board_detector import BoardDetector
from robot_io.marker_detection.core.tag_pose_estimator import TagPoseEstimator
from src.imagine2touch.localizers.wrist_camera_localizer import (
    generate_step_pos,
    generate_step_orientation,
)


class ApriltagDetector:
    def __init__(self, cam, marker_description, min_tags):
        # set up detector and estimator
        self.cam = cam
        marker_description = (Path(__file__).parent / marker_description).as_posix()
        self.detector = BoardDetector(marker_description)
        self.min_tags = min_tags
        self.estimator = TagPoseEstimator(self.detector)
        self.K = cam.get_camera_matrix()
        self.dist_coeffs = np.zeros(12)

    def estimate_pose(self, rgb=None, visualize=True):
        if rgb is None:
            rgb, _ = self.cam.get_image()

        points2d, point_ids = self.detector.process_image_m(rgb)

        T_cam_marker = self._estimate_pose(points2d, point_ids)
        if T_cam_marker is None:
            print("No marker detected")
            return None
        if visualize:
            self.detector.draw_board(rgb, points2d, point_ids, show=False, linewidth=1)
            cv2.drawFrameAxes(
                rgb,
                self.K,
                self.dist_coeffs,
                T_cam_marker[:3, :3],
                T_cam_marker[:3, 3],
                0.1,
            )
            cv2.imshow("window", rgb[:, :, ::-1])
            cv2.waitKey(1)
        return T_cam_marker

    def _estimate_pose(self, p2d, pid):
        if p2d.shape[0] < self.min_tags * 4:
            return None
        ret = self.estimator.estimate_relative_cam_pose(
            self.K, self.dist_coeffs, p2d, pid
        )
        if ret is None:
            return None
        points3d_pred, rot, trans = ret
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = rot
        T_cam_marker[:3, 3:] = trans
        return T_cam_marker


if __name__ == "__main__":
    # script configurations
    repository_directory = search_folder("/", "imagine2touch")
    hydra.initialize("./cfg", version_base=None)
    cfg = hydra.compose("wrist.yaml")
    starting_corner_in_world = [
        float(num) for num in cfg.starting_corner_in_world.split(",")
    ]
    OmegaConf.register_new_resolver("rad_to_deg", lambda x: x * (np.pi / 180))

    # initialize devices
    robot = RobotClient("/robot_io_ros_server")  # name of node in server launch file
    start_pos = robot.get_state()["tcp_pos"]
    start_rot = robot.get_state()["tcp_orn"]
    camera = Realsense(img_type="rgb_depth")

    # initialize variables and storage
    cam_marker_transforms = []
    tcp_robot_transforms = []
    i = 0
    if not cfg.single:
        orns = generate_step_orientation(
            cfg.fixed_roll, cfg.max_inclination, cfg.orientation_step / 2
        )
        poses = generate_step_pos(
            starting_corner_in_world,
            cfg.r,
            0,
            len(orns),
            cfg.orientation_step,
            cfg.translation_step,
        )
        orns = np.repeat(orns, 7, axis=0)
        print(len(orns))
        print(len(poses))
        views = zip(poses, orns)
    else:
        views = zip([[0, 0, cfg.single_view_height, 1]], [FIXED_ROBOT_ORN])
    for pose, orn in views:
        world_pose_in_robot = WORLD_IN_ROBOT.dot(pose)

        try:
            robot.move_cart_pos_abs_ptp(world_pose_in_robot[:3], eulertoquat(orn))
        except rospy.service.ServiceException as e:
            if (
                np.linalg.norm(
                    robot.get_state()["tcp_pos"][:3] - world_pose_in_robot[:3], 2
                )
                > 0.01
            ):
                print("failed to get view, pos error is: ")
                print(
                    np.linalg.norm(
                        robot.get_state()["tcp_pos"][:3] - world_pose_in_robot[:3], 2
                    )
                )
                continue
            else:
                print("got view within the following tolerance: ")
                print(
                    np.linalg.norm(
                        robot.get_state()["tcp_pos"][:3] - world_pose_in_robot[:3], 2
                    )
                )
                pass
        wait_until_stable_joint_velocities(robot)
        marker_detector = ApriltagDetector(camera, cfg.marker_description, cfg.min_tags)
        T_tcp_in_robot = robot.get_tcp_pose()
        T_cam_in_marker = marker_detector.estimate_pose(visualize=True)
        print(T_tcp_in_robot)
        print(T_cam_in_marker)
        cam_marker_transforms.append(T_cam_in_marker)
        tcp_robot_transforms.append(T_tcp_in_robot)
        i = i + 1
        np.save(
            f"{repository_directory}/{cfg.save_directory}/wcamera_marker_transforms",
            cam_marker_transforms,
            allow_pickle=True,
        )
        np.save(
            f"{repository_directory}/{cfg.save_directory}/robot_tcp_transforms",
            tcp_robot_transforms,
            allow_pickle=True,
        )
    # save transforms
    np.save(
        f"{repository_directory}/{cfg.save_directory}/wcamera_marker_transforms",
        cam_marker_transforms,
        allow_pickle=True,
    )
    np.save(
        f"{repository_directory}/{cfg.save_directory}/robot_tcp_transforms",
        tcp_robot_transforms,
        allow_pickle=True,
    )
    print("saved transforms")
    sys.exit()
