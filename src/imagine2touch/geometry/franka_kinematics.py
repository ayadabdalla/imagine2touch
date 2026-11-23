"""Where the Franka's flange is for given joint angles."""

import numpy as np

from .pose import Pose


class FrankaKinematics:
    """
    The forward kinematics of the Franka Emika Panda.  The recordings store joint angles, not the
    pose the sensor reached; this turns the first into the second.

    Each of the seven joints has a frame.  Going from the frame of one joint to the next is described
    by three fixed numbers and the joint's angle (Craig's "modified Denavit-Hartenberg" convention,
    with the values Franka publishes):

      LINK_LENGTH   distance between the two joint axes, along the previous frame's x [mm]
      LINK_TWIST    angle between the two joint axes, about the previous frame's x [rad]
      LINK_OFFSET   distance along the joint's own axis (its z) [mm]
      joint angle   rotation about the joint's own axis, the one number that changes [rad]
    """

    LINK_LENGTH = (0, 0, 0, 82.5, -82.5, 0, 88)                                                # [mm], one per joint
    LINK_TWIST = (0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2)       # [rad]
    LINK_OFFSET = (333, 0, 316, 0, 384, 0, 0)                                                  # [mm]
    FLANGE_OFFSET = 107         # [mm], from the frame of the last joint to the flange, along that joint's axis

    @staticmethod
    def joint_frame(length, twist, offset, angle):
        """
        Takes the three fixed numbers of a joint (length [mm], twist [rad], offset [mm]) and its angle [rad].
        Returns the 4x4 matrix of that joint's frame, as seen from the frame of the joint before it.
        """
        cos_twist, sin_twist, cos_angle, sin_angle = np.cos(twist), np.sin(twist), np.cos(angle), np.sin(angle)
        return np.array([[cos_angle, -sin_angle, 0, length],
                         [sin_angle * cos_twist, cos_angle * cos_twist, -sin_twist, -offset * sin_twist],
                         [sin_angle * sin_twist, cos_angle * sin_twist, cos_twist, offset * cos_twist],
                         [0, 0, 0, 1]])

    @classmethod
    def flange_pose(cls, joint_angles):
        """Takes the seven joint angles [rad].  Returns the pose of the flange in the frame of the robot's base, position in mm."""
        last_joint = np.eye(4)                      # the frame of the last joint reached so far, seen from the base
        for length, twist, offset, angle in zip(cls.LINK_LENGTH, cls.LINK_TWIST, cls.LINK_OFFSET, joint_angles):
            last_joint = last_joint @ cls.joint_frame(length, twist, offset, angle)
        rotation, position, joint_axis = last_joint[:3, :3], last_joint[:3, 3], last_joint[:3, 2]
        return Pose(rotation, position + cls.FLANGE_OFFSET * joint_axis)
