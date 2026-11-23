"""
Imagine2touch: predicting the touch signal of a ReSkin sensor from depth images of the surface.

  datasets/         the two datasets of the paper; how they are made from the robot's recordings
  geometry/         poses, the robot's kinematics, an object's scan and the depth image it shows
  method/           what the paper proposes: preprocessing, the models, recognition by touch

  experiments/      the method on the recorded data: every experiment of the paper
  procedures/       the method while touching: a robot collects touches and recognises objects
  interfaces.py     Robot, TouchSensor, DepthCamera: what the procedures need, and what the next two provide
  reskin_reading.py the layout of a ReSkin reading, shared by the sensor's adapter and the recordings
  devices/          the real Franka arm, ReSkin and RealSense camera for the procedures
  simulation/       a simulated robot, sensor and camera for the procedures, on the real scans and recorded signals
  visualization/    the figures

The benchmark is run by the two scripts in the repository's `scripts/` folder.  Every other script is
in the `scripts/` folder of the package it drives.

`experiments/` and `procedures/` both use `method/`.  One experiment, object recognition, runs the
procedures on the simulation.
"""

__version__ = "2.0.0"
