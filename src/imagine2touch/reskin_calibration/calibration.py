import signal
import time
import numpy as np
import yaml
from src.imagine2touch.reskin_sensor.sensor_proc import ReSkinProcess, ReSkinSettings
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from robot_io.robot_interface.iiwa_interface import (
    IIWAInterface,
    TCP_RESKIN,
)  # Used to initialize robot
from robot_io.input_devices.space_mouse import SpaceMouse
import sys
import math
import src.imagine2touch.utils as utils


FIXED_ROBOT_ORN = [np.pi, 0, -2.32208689]  # nice orientation of TCP
ROBOT_HOME_POSE = [
    0.05730437,
    1.07472593,
    1.09051066,
    -1.71193305,
    -1.06877005,
    1.07252174,
    -1.65786651,
]  # nice joint angles (7)

##########################################################

if __name__ == "__main__":
    parser = ArgumentParser(description="calibration of reskin sensor")
    # TODO :: Add logic for checking which experiment is currently available in the repo and add the intuitive next in the sequence
    # parser.add_argument('dataset', help='Name of generated dataset file')
    parser.add_argument("-config", default=None, help="Location config for sensor")
    args = parser.parse_args()

    # initialize robot
    robot = IIWAInterface(
        workspace_limits=((-1, -1, -0.1), (1, 1, 0.4)),
        use_impedance=True,
        open_gripper=False,
    )

    # initialize robot gripper
    robot.close_gripper()
    # robot.open_gripper()

    # initialize sensor
    sensor_settings = ReSkinSettings(
        num_mags=5, port="/dev/ttyACM0", baudrate=115200, burst_mode=True, device_id=1
    )  # TODO :: take them as arguments
    sensor_process = ReSkinProcess(sensor_settings)
    sensor_process.start()
    time.sleep(0.1)

    # Add signal handler to be able to terminate using keyboard
    end = False

    def cb_int(*args):
        global end
        end = True

    signal.signal(signal.SIGINT, cb_int)

    # warm up with a move till nice joint angles configuration
    robot.move_joint_pos(ROBOT_HOME_POSE)

    # Initialize reset data, go to where you are with fixed orientation
    start_state = robot.get_state()["tcp_pose"]  # state has position and rotation
    start_rot = start_state[3:]
    goal_rot = start_rot  # unchanged
    reset_rot = start_rot  # unchanged
    start_pos = 1 * start_state[:3]
    robot.move_cart_pos_abs_ptp(start_pos, FIXED_ROBOT_ORN)

    # initialize space mouse
    space_mouse = SpaceMouse(dv=1, drot=0.05)
    space_mouse.clear_gripper_input()

    # time the experiment begins
    start_time = time.time()
    fig1 = plt.figure()

    while not end:
        action, record_info = space_mouse.get_action()
        (m_lin, m_ang, a_gripper) = action["motion"]
        if args.config is None:
            path = "./reskin/calibration/sensor_config.yaml"  # assuming we are in the hgigher workspace folder including the repo
            config = utils.main_record_config(robot, space_mouse, a_gripper, path)
            if config != "":
                with open(path, "r") as config:
                    config_dict = yaml.safe_load(config)
                if len(config_dict) == 2:  # 2 points recorded
                    print(utils.getTransform(config_dict))  # test recording
        else:
            with open(args.config, "r") as config:
                config_dict = yaml.safe_load(config)

        x_y_recording = []
        force_recording = []
        reskin_recording = []
        previous_point_x = 0
        ambient_readings = []
        square = utils.make_square()
        for i, point in enumerate(square):
            sensor_side_length = 0.017
            first_corner = (
                point[0] < 0.008 and point[1] < 0.008
            )  # 8 instead of 5 because of new indenters

            second_corner = point[0] > (sensor_side_length - 0.008) and point[1] < 0.008

            third_corner = point[0] > (sensor_side_length - 0.008) and point[1] > (
                sensor_side_length - 0.008
            )

            fourth_corner = point[0] < 0.008 and point[1] > (sensor_side_length - 0.008)

            if (
                (not first_corner)
                and (not second_corner)
                and (not third_corner)
                and not (fourth_corner)
                and point[0] > 0.001
            ):  # last condition because sensor origin recording has 1 mm tolerance
                if (
                    point[0] != previous_point_x
                ):  # get ambient reading every time you move one unit on the grid in the x direction including one before first x
                    # print ('getting ambient samples')
                    ambient_readings.append(sensor_process.get_data(num_samples=10))
                goal_point = np.squeeze(
                    np.asarray(
                        utils.transform_sensor_to_robot(
                            np.array([point[0], point[1], point[2]]),
                            utils.getTransform(config_dict),
                        )
                    )
                )
                reset_point = start_pos  # first safe pose
                utils.move_to_pt_with_v_safe(
                    robot,
                    goal_point[:3],
                    goal_rot,
                    reset_point,
                    reset_rot,
                    p_i=0.3,
                    dt=1 / 500,
                )  # move to goal point which is 1 cm away of the actual goal in z direction
                utils.move_to_pt_with_v_safe(
                    robot,
                    goal_point[:3] - np.array([0, 0, 0.01]),
                    goal_rot,
                    reset_point,
                    reset_rot,
                    p_i=0.1,
                    dt=1 / 500,
                )  # move this one cm with a slightly slower velocity
                for d in np.arange(
                    0.0103, 0.0115, 0.0003
                ):  # 4 depths of 0.3mm step ,not there is 1 cm added due to goal point being defined as 1 cm away of the actual goal in z direction
                    reset_point = goal_point[
                        :3
                    ]  # 1 cm in z away from sensor in correct x, y
                    print(d)
                    utils.move_to_pt_with_v_safe(
                        robot,
                        goal_point[:3] - np.array([0, 0, d]),
                        goal_rot,
                        reset_point,
                        reset_rot,
                        p_i=0.01,
                        dt=1 / 50,
                        accurate=True,
                    )  # move down to sensor level
                    # display force
                    print(
                        f'Position: {robot.get_state()["tcp_pose"]}\n  Force: {robot.get_state()["force_torque"][2]}'
                    )
                    # print ('getting sample per indentation')
                    reskin_recording.append(sensor_process.get_data(num_samples=1))
                    if len(reskin_recording) > 1:
                        reskin_reading = np.squeeze(reskin_recording)[
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
                        plt.draw()
                        plt.pause(0.01)
                    # print('got sample per indentation')
                    force_recording.append(robot.get_state()["force_torque"][2])
                    x_y_recording.append([point[0], point[1]])
                reset_point = start_pos  # reset to first safe pose
                utils.move_to_pt_with_v_safe(
                    robot,
                    goal_point[:3],
                    goal_rot,
                    reset_point,
                    reset_rot,
                    p_i=0.3,
                    dt=1 / 500,
                )  # move back up
                previous_point_x = point[0]
            if sensor_process.is_alive():
                print("sensor is alive")
            else:
                print("done with recording")
                reskin_recording = np.array(reskin_recording)
                print("done with storing reskin recordings as array")
                force_recording = np.array(force_recording)
                print("done with storing force as numpy array")
                x_y_recording = np.array(x_y_recording)
                print("done with storing x_y as numpy array")

                with open("./reskin/calibration/reskin_readings", "wb") as readings:
                    np.save(readings, reskin_recording)
                print("done with saving reskin readings")
                with open("./reskin/calibration/ambient_readings", "wb") as ambient:
                    np.save(ambient, ambient_readings)
                print("done with saving ambient readings")
                with open("./reskin/calibration/force", "wb") as force:
                    np.save(force, force_recording)
                print("done with saving force")
                with open("./reskin/calibration/pose", "wb") as pose:
                    np.save(pose, x_y_recording)
                print("done with saving pose")

                print("sensor stopped before finishing")
                # control robot with space mouse
                # step_move_robot_with_device(robot,m_lin,start_pos,start_rot)

                # display force
                # print(f'Position: {state["tcp_pose"]}\n  Force: {state["force_torque"][:3]}')
                print("should exit now")
                sys.exit()  # exit program

            print(f"finished " + str(i) + "th iteration")

        print("done with recording")
        reskin_recording = np.array(reskin_recording)
        print("done with storing reskin recordings as array")
        force_recording = np.array(force_recording)
        print("done with storing force as numpy array")
        x_y_recording = np.array(x_y_recording)
        print("done with storing x_y as numpy array")

        with open("./reskin/calibration/reskin_readings", "wb") as readings:
            np.save(readings, reskin_recording)
        print("done with saving reskin readings")
        with open("./reskin/calibration/ambient_readings", "wb") as ambient:
            np.save(ambient, ambient_readings)
        print("done with saving ambient readings")
        with open("./reskin/calibration/force", "wb") as force:
            np.save(force, force_recording)
        print("done with saving force")
        with open("./reskin/calibration/pose", "wb") as pose:
            np.save(pose, x_y_recording)
        print("done with saving pose")

        print("sensor stopped after finishing")
        # control robot with space mouse
        # step_move_robot_with_device(robot,m_lin,start_pos,start_rot)

        print("should exit now")

        if sensor_process.is_alive:
            sensor_process.pause_streaming()
            sensor_process.join()

        sys.exit()  # exit program
