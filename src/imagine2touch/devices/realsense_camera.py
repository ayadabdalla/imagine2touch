"""
A RealSense depth camera behind the DepthCamera interface, through pyrealsense2.

Written from the scripts that recorded the paper's data (commit c2ea2c3 of this repository) and not run on hardware since:
check it against your setup before letting the robot move.
"""

import numpy as np

from imagine2touch.interfaces import DepthCamera


class RealSenseCamera(DepthCamera):
    def __init__(self, width=640, height=480):
        import pyrealsense2 as rs
        self.rs, self.pipeline = rs, rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])

    def capture(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        colour = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()) * self.depth_scale
        return colour, depth
