# import standard libraries
import numpy as np

z = np.array([0, 0, 1])  # up_axis in world frame and robot frame
normal = np.array(
    [0, 0.5, -0.2]
)  # example normal vector on a sample point from the point cloud (alignment/orientation vector)
condition = (
    normal.dot(z) > 0.2
)  # if angle between normal and z is small (smaller than 90), then okay
print(condition)
