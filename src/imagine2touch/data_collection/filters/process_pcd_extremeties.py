import os
import open3d as o3d
import numpy as np
from src.imagine2touch.utils.utils import search_folder
from src.imagine2touch.task.process_pcd_extremeties import point_cloud_info

if __name__ == "__main__":
    # take object name from user input interactively
    object_name = input("Enter the name of the object: ")
    num = input("Enter the number of the pcd file you want to filter: ")
    num = int(num)
    # take the numer of the axis
    x = input("Enter the axis you want to filter: ")
    x = int(x)
    # take the lower limit
    a = input("Enter the lower limit on the positive side: ")
    a = float(a)
    # take the upper limit
    b = input("Enter the upper limit on the negative side: ")
    b = float(b)

    path = search_folder("/", "imagine2touch")
    point_cloud = o3d.io.read_point_cloud(
        f"{path}/src/imagine2touch/data_collection/data/{object_name}/pcds/experiment_{num}_combined.pcd"
    )
    point_cloud_info(point_cloud, display=True)

    # adhoc segmentation filter on point cloud
    # change x, a, b as per requirement to view the point cloud extremities and remove the unwanted points
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    red = np.asarray([1, 0, 0])
    green = np.asarray([0, 1, 0])
    red_points = points[(points[:, x] > a)]
    green_points = points[(points[:, x] < b)]
    red_colors = np.tile(red, (len(red_points), 1))
    green_colors = np.tile(green, (len(green_points), 1))
    colors[(points[:, x] > a)] = red_colors
    colors[(points[:, x] < b)] = green_colors
    original_colors = np.asarray(point_cloud.colors)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
    # reset the colors
    point_cloud.colors = o3d.utility.Vector3dVector(original_colors)
    # inquire which points to remove
    color = input("delete red or green points:")
    if color == "red":
        point_cloud = point_cloud.select_by_index(
            np.where(points[:, x] < a)[0], invert=False
        )
    elif color == "green":
        point_cloud = point_cloud.select_by_index(
            np.where(points[:, x] > b)[0], invert=False
        )
    o3d.visualization.draw_geometries([point_cloud])
    # prompt user to save the point cloud
    save = input("save the point cloud? (y/n):")
    if save == "y":
        # # delete the old file
        os.remove(
            f"{path}/src/imagine2touch/data_collection/data/{object_name}/pcds/experiment_{num}_combined.pcd"
        )
        # Save the point cloud as a .pcd file
        o3d.io.write_point_cloud(
            f"{path}/src/imagine2touch/data_collection/data/{object_name}/pcds/experiment_{num}_combined.pcd",
            point_cloud,
        )
    else:
        print("point cloud not saved")
