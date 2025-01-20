# Standard programming and scientific libraries
import signal
from multiprocessing import RLock
from threading import Thread
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import cv2.aruco as aruco

# Robot_io interface modules
from robot_io.input_devices.space_mouse import SpaceMouse

# from robot_io.robot_interface.iiwa_interface  import IIWAInterface
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
from robot_io.cams.kinect4.kinect4 import Kinect4
from robot_io.cams.realsense.realsense import Realsense
from robot_io.marker_detection.aruco_detector import ArucoDetector
from robot_io.utils.utils import pos_orn_to_matrix, matrix_to_pos_orn
from src.imagine2touch.utils import utils


# Constants
ROBOT_MARKER_ID = 4  # hard coded to suit printed markers
WORLD_MARKER_ID = 5
MARKER_IDS = {ROBOT_MARKER_ID, WORLD_MARKER_ID}
WORLD_IN_MARKER = pos_orn_to_matrix([-0.35, -0.04, 0.035], [0, 0, 0.5 * np.pi])
ROBOT_IN_MARKER = pos_orn_to_matrix([-0.108, 0.145, 0], [0, 0, 0])


class FrameUtil(object):
    def __init__(self, robot, camera, marker_ids=None) -> None:
        super().__init__()
        self._lock = RLock()

        self.marker_aliases = {ROBOT_MARKER_ID: "robot", WORLD_MARKER_ID: "world"}
        if marker_ids is not None:
            self.marker_aliases = dict(
                sum([list(self.marker_aliases.items()), list(marker_ids.items())], [])
            )
        self.marker_ids = set(self.marker_aliases.keys())
        self.marker_aliases = {v: k for k, v in self.marker_aliases.items()}

        # Robot frame is world frame
        self._T_world_in_robot = None
        self._T_camera_in_robot = None
        self._T_camera_in_world = None
        self._T_camera_obs_series = {id: [] for id in self.marker_ids}

        self._shutdown = False
        self._obs_thread = None

        self.camera = camera
        self.detector = None
        self.robot = robot

    def start(self):
        while not self._shutdown:
            try:
                # test image
                # rgb, depth = self.camera.get_image()
                # cv2.imshow("image",rgb[:, :, ::-1])
                # cv2.waitKey(0)
                break
            except RuntimeError as e:
                # print(f'Failed to connect to Kinect: {e}')
                print(f"Failed to connect to realsense: {e}")

        self.detector = ArucoDetector(
            self.camera, 0.08, aruco.DICT_3X3_100, None, visualize=True
        )
        self._obs_thread = Thread(target=self._thread_observe_markers)
        self._obs_thread.start()

    def stop(self):
        self._shutdown = True

    def wait(self):
        """Util function for waiting for this component to end"""
        if self._obs_thread is not None:
            self._obs_thread.join()

    @property
    def T_world_in_robot(self):
        with self._lock:
            return self._T_world_in_robot

    @property
    def T_robot_in_world(self):
        return utils.inverse_transform(self._T_world_in_robot)

    @property
    def T_tcp_in_robot(self):
        return self.robot.get_tcp_pose()

    @property
    def T_tcp_in_world(self):
        if self._T_world_in_robot is not None:
            return utils.inverse_transform(self.T_world_in_robot).dot(
                self.robot.get_tcp_pose()
            )
        return None

    def T_in_camera(self, name):
        if name not in self.marker_aliases:
            raise Exception(f'Unknown pose "{name}"')
        Id = self.marker_aliases[name]
        with self._lock:
            pos, orn = self._T_camera_obs_series[Id][-1]
        return pos_orn_to_matrix(pos, orn)

    def _thread_observe_markers(self):
        print("Observation thread started")
        while not self._shutdown:
            rgb, _ = self.camera.get_image()
            self._process_image(rgb)

    def _process_image(self, rgb):
        poses = self.detector.estimate_poses(
            rgb, markers=self.marker_ids, show_window=False
        )

        for marker_id in self.marker_ids:
            if poses[marker_id] is not None:
                with self._lock:
                    self._T_camera_obs_series[marker_id].append(
                        matrix_to_pos_orn(poses[marker_id])
                    )
                    # Only consider the last 10 observations
                    self._T_camera_obs_series[marker_id] = self._T_camera_obs_series[
                        marker_id
                    ][-10:]
        if len(self._T_camera_obs_series[ROBOT_MARKER_ID]) > 0:
            pos, orns = zip(*self._T_camera_obs_series[ROBOT_MARKER_ID])
            mean_pos = np.mean(pos, axis=0)
            mean_orn = R.from_quat(orns).mean()

            T_camera_in_r_marker = utils.inverse_transform(
                pos_orn_to_matrix(mean_pos, mean_orn)
            )
            with self._lock:
                self._T_camera_in_robot = utils.inverse_transform(ROBOT_IN_MARKER).dot(
                    T_camera_in_r_marker
                )

        if len(self._T_camera_obs_series[WORLD_MARKER_ID]) > 0:
            pos, orns = zip(*self._T_camera_obs_series[WORLD_MARKER_ID])
            mean_pos = np.mean(pos, axis=0)
            mean_orn = R.from_quat(orns).mean()

            T_camera_in_s_marker = utils.inverse_transform(
                pos_orn_to_matrix(mean_pos, mean_orn)
            )
            with self._lock:
                self._T_camera_in_world = utils.inverse_transform(WORLD_IN_MARKER).dot(
                    T_camera_in_s_marker
                )

        if (
            self._T_camera_in_robot is not None
            and len(self._T_camera_obs_series[WORLD_MARKER_ID]) > 0
        ):
            pos, orns = zip(*self._T_camera_obs_series[WORLD_MARKER_ID])
            mean_pos = np.mean(pos, axis=0)
            mean_orn = R.from_quat(orns).mean()

            T_s_marker_in_camera = pos_orn_to_matrix(mean_pos, mean_orn)
            with self._lock:
                self._T_world_in_robot = self._T_camera_in_robot.dot(
                    T_s_marker_in_camera
                ).dot(WORLD_IN_MARKER)


if __name__ == "__main__":
    robot = RobotClient("/robot_io_ros_server")  # name of node in server launch file
    print(robot.get_tcp_pose())
    camera = Kinect4(0)
    frame_util = FrameUtil(robot, camera, {1: "check"})

    def signal_handler(signum, *args):
        print("Received SIGINT. Shutting down...")
        frame_util.stop()
        frame_util.wait()

    signal.signal(signal.SIGINT, signal_handler)

    frame_util.start()
    while frame_util._T_camera_in_robot is None or frame_util._T_world_in_robot is None:
        # print("transform not yet seen")
        continue
    print("transform seen")
    T_world_in_camera = utils.inverse_transform(
        frame_util.T_robot_in_world.dot(frame_util._T_camera_in_robot)
    )  # inverted for extrinsic camera
    T_world_in_robot = frame_util.T_world_in_robot
    T_camera_in_robot = frame_util._T_camera_in_robot
    np.save("world_camera_transform", T_world_in_camera, allow_pickle=True)
    np.save("world_robot_transform", T_world_in_robot, allow_pickle=True)
    np.save("camera_robot_transform", T_camera_in_robot, allow_pickle=True)
    print("transforms saved in the directory you ran the script from")

    frame_util.stop()
    frame_util.wait()
