"""
A robot, a touch sensor and a depth camera that exist only in memory, behind the interfaces of `interfaces.py`,
and a calibration board for the simulated camera to find.

  SimulatedScene     the scan of a real object on a table: what the robot and the camera meet
  SimulatedRobot     moves the sensor where it is told, except into the object
  SimulatedCamera    a depth camera at the wrist, looking at the scene
  SimulatedSensor    answers with the signal the real sensor gave at the nearest recorded touch
  SimulatedBoard     a calibration board on the table: says where it is instead of being found in a picture

Scene and sensor are built from the dataset (`of_session`), so a simulated robot touches the paper's
objects and feels what the real sensor felt there.  The arm has no joints, and the camera's colour
image is only its depth in grey.
"""

from .simulated_board import SimulatedBoard
from .simulated_camera import SimulatedCamera
from .simulated_robot import SimulatedRobot
from .simulated_scene import SimulatedScene
from .simulated_sensor import SimulatedSensor

__all__ = ["SimulatedBoard", "SimulatedCamera", "SimulatedRobot", "SimulatedScene", "SimulatedSensor"]
