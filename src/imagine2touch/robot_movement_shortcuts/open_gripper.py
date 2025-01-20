from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient

# initialize robot

robot = RobotClient("robot_io_ros_server")  # name of node in server launch file
robot.open_gripper(True)  # True for blocking
