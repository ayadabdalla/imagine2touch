"""
The real Franka arm, ReSkin and RealSense camera behind the interfaces of `interfaces.py`.  Not run on
hardware since the paper.  These adapters only translate calls; they hold no logic of
their own.  Each is imported on its own, because each needs its device's library:

    from imagine2touch.devices.franka_robot import FrankaRobot
    from imagine2touch.devices.reskin_sensor import ReSkinSensor
    from imagine2touch.devices.realsense_camera import RealSenseCamera
"""
