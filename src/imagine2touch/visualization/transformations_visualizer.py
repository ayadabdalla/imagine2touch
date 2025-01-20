import os
import sys
import hydra
import numpy as np
from mayavi import mlab
from omegaconf import OmegaConf
from src.imagine2touch.utils.utils import inverse_transform

if __name__ == "__main__":
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("custom_path.yaml")
    # Create a Mayavi figure
    fig = mlab.figure()
    print("created figure")
    # Create an array of 3D transformations
    transforms = np.random.rand(
        10, 4, 4
    )  # 10 transformations, each represented by a 4x4 matrix
    T_wcamera_in_marker = np.load(
        f"{cfg.path}/wcamera_marker_transforms.npy", allow_pickle=True
    )
    T_tcp_in_robot = np.load(f"{cfg.path}/robot_tcp_transforms.npy", allow_pickle=True)
    T_wcamera_marker_filtered = T_wcamera_in_marker
    j = 0
    indeces = []
    for i, T in enumerate(T_wcamera_marker_filtered):
        if T is None:
            T_wcamera_marker_filtered = np.delete(
                T_wcamera_marker_filtered, i - j, axis=0
            )
            indeces.append(i)
            j += 1
    T_wcamera_marker_filtered = list(
        T_wcamera_marker_filtered
    )  # convert to list of lists then to nd array
    T_wcamera_marker_filtered = np.asarray(T_wcamera_marker_filtered, dtype=object)
    c_T_m = np.reshape(T_wcamera_marker_filtered, (-1, 4, 4))
    r_T_h = np.delete(T_tcp_in_robot, indeces, axis=0)
    # Visualize the transformations using arrows
    for transform in c_T_m:
        # Extract the translation vector and rotation matrix from the transformation matrix
        transform = inverse_transform(transform)
        translation = transform[:3, 3]
        rotation = transform[:3, :3]

        # Compute the arrow direction and magnitude based on the rotation matrix
        direction = rotation @ [0, 0, 1]
        magnitude = np.linalg.norm(direction)
        # Create an arrow to represent the transformation
        arrow = mlab.quiver3d(
            translation[0],
            translation[1],
            translation[2],
            direction[0],
            direction[1],
            direction[2],
            scale_factor=0.02,
            color=(0, 0, 1),
        )

    # Show the Mayavi figure
    print("Showing the Mayavi figure...")
    # Set up the axes and the coordinate system
    # mlab.axes(xlabel='x', ylabel='y', zlabel='z', ranges=[-1, 1, -1, 1, -1, 1])
    mlab.show()
