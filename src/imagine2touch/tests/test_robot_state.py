from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient

robot = RobotClient("/robot_io_ros_server")  # name of node in server launch file


def ping_robot_state_blocking():
    state = None
    while state is None:
        state = robot.get_state()
    print("waiting for state")


def log_robot_state():
    print(robot.get_state())


log_robot_state()
