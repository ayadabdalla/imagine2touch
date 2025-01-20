# import standard libraries
import open3d as o3d
import numpy as np
import math

from src.imagine2touch.utils.utils import (
    create_homog_transformation_from_rotation,
    create_rotation_from_normal,
)


def create_grid(n):
    """
    creates a cuboid of n 3D points, with 3 levels in z direction, sqrt(n/3) levels in x direction and sqrt(n/3) levels in y direction
    <n>: divisible by 3
    """
    points = np.zeros((n, 3))
    count = 0
    for i in range(3):
        for j in range(int(math.sqrt(n // 3))):
            for k in range(int(math.sqrt(n / 3))):
                points[count][0] = k
                points[count][1] = j
                points[count][2] = i
                count = count + 1
    return points


if __name__ == "__main__":
    point_cloud = (
        o3d.geometry.PointCloud()
    )  # create empty point cloud | replace it with your target pcd
    trasformed_point_cloud = (
        o3d.geometry.PointCloud()
    )  # empty point cloud to be transformed
    n = 243
    point_cloud.points = o3d.utility.Vector3dVector(
        create_grid(n)
    )  # fill it with dummy test data
    trasformed_point_cloud.points = o3d.utility.Vector3dVector(
        create_grid(n)
    )  # fill it with dummy test data
    point_cloud.estimate_normals()
    test_normal = np.array(
        [-1, 0, 0]
    )  # replace with negative normal on target point from point cloud (normal example: x-axis)
    test_point = -np.array(
        [3, 4, 1]
    )  # replace with target point from point cloud, don't remove the negative

    # apply transformation to temp pcd
    trasformed_point_cloud.translate(test_point)
    trasformed_point_cloud.transform(
        create_homog_transformation_from_rotation(
            create_rotation_from_normal(test_normal)
        )
    )

    # transformed test_points before filtering#
    points = np.asarray(trasformed_point_cloud.points)
    zs = points[:, 2]  # get z-coordinates of point cloud
    xs = points[:, 0]  # get x-coordinates of point cloud
    ys = points[:, 1]  # get y-coordinates of point cloud

    # safety condition test
    condition_x = (xs < 3) & (
        xs > -3
    )  # replace the value with side length of intended patch
    condition_y = (ys < 3) & (
        ys > -3
    )  # replace the value with side length of intended patch
    condition_z = zs > 0
    indices = np.where(condition_x & condition_y & condition_z)
    print(indices[0].shape)  # ensure that this equals 0 to be safe
    point_cloud = point_cloud.select_by_index(
        indices[0], invert=False
    )  # select and visualize points greater than min or smaller than max threshold, they should be zero if safe

    # untransformed test_points after filtering#
    points = np.asarray(point_cloud.points)
    zs = points[:, 2]  # get z-coordinates of point cloud
    xs = points[:, 0]  # get x-coordinates of point cloud
    ys = points[:, 1]  # get y-coordinates of point cloud

    # point cloud info | print information of untransformed point cloud after filtering using the transformed one
    print("zmin " + str(np.min(zs)))
    print("zmax " + str(np.max(zs)))

    print("xmin " + str(np.min(xs)))
    print("xmax " + str(np.max(xs)))

    print("ymin " + str(np.min(ys)))
    print("ymax " + str(np.max(ys)))

    o3d.visualization.draw_geometries(
        [point_cloud], point_show_normal=True
    )  # visualization test
