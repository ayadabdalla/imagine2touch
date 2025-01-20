from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
import time
import os
import numpy as np

if __name__ == "__main__":
    robot = RobotClient("/robot_io_ros_server")
    time.sleep(2)
    force_array_1 = []
    force_array_2 = []
    for i in range(100):
        state = robot.get_state()
        force = state["force_torque"]
        force_array_1.append(force)

    print("waiting")
    # go push the robot before 10 seconds pass:)
    time.sleep(10)
    for j in range(100):
        state = robot.get_state()
        force = state["force_torque"]
        force_array_2.append(force)

    # rough measure for a relative external wrench
    print(
        np.mean(np.asarray(force_array_1), axis=0)
        - np.mean(np.asarray(force_array_2), axis=0)
    )

    # rough measure for ambient wrench
    # print(np.mean(np.asarray(force_array_1),axis=0))
