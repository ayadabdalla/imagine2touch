"""A rigid motion, used for every pose and change of frame."""

import numpy as np
from scipy.spatial.transform import Rotation


class Pose:
    """
    (R, t): maps a point x to R x + t.  As the pose of a body, t is its origin and the columns of R
    are its axes; the third is its `normal`, the direction it faces.
    """

    def __init__(self, R=None, t=None):
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.t = np.zeros(3) if t is None else np.asarray(t, float)

    @classmethod
    def from_matrix(cls, T):
        """Takes a 4x4 homogeneous matrix.  Returns the pose."""
        T = np.asarray(T, float)
        return cls(T[:3, :3], T[:3, 3])

    @classmethod
    def from_quaternion(cls, xyzw, t=None):
        """Takes a rotation as a quaternion (x, y, z, w), the order the recordings use, and a position t.  Returns the pose."""
        return cls(Rotation.from_quat(xyzw).as_matrix(), t)

    @classmethod
    def from_normal(cls, normal, t=None, first_axis=None):
        """
        Takes the direction to face, `normal` (a vector, not necessarily of length one), and a position t.
        Returns a pose at t whose third axis is that direction.  Its first axis points as near to
        `first_axis` as that allows; without `first_axis` the turn about the normal is chosen freely.
        """
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        if first_axis is None:
            helper = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
            e_u = np.cross(helper, n)
        else:
            e_u = np.asarray(first_axis, float) - (np.asarray(first_axis, float) @ n) * n
        e_u /= np.linalg.norm(e_u)
        return cls(np.stack([e_u, np.cross(n, e_u), n], axis=1), t)

    @property
    def normal(self):
        """Returns the direction the pose faces: its third axis, a unit vector."""
        return self.R[:, 2]

    @property
    def matrix(self):
        """Returns the pose as a 4x4 homogeneous matrix."""
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    @property
    def quaternion(self):
        """Returns the rotation as a quaternion (x, y, z, w)."""
        return Rotation.from_matrix(self.R).as_quat()

    def apply(self, x):
        """Takes points x (..., 3) in the coordinates of this pose.  Returns them in the outer frame: R x + t."""
        return np.asarray(x, float) @ self.R.T + self.t

    def to_local(self, x):
        """Takes points x (..., 3) in the outer frame.  Returns them in the coordinates of this pose; the third is the distance in front of it."""
        return (np.asarray(x, float) - self.t) @ self.R

    def inverse(self):
        """Returns the pose that undoes this one."""
        return Pose(self.R.T, -self.R.T @ self.t)

    def __matmul__(self, other):
        """a @ b: returns the pose that applies b first, then a."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def angle_to(self, other):
        """Takes another pose.  Returns the angle between the directions the two face [rad]."""
        return float(np.arccos(np.clip(self.normal @ other.normal, -1, 1)))
