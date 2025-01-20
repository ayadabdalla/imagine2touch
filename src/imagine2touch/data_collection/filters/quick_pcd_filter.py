import os
import open3d as o3d
from src.imagine2touch.utils.utils import search_folder
from src.imagine2touch.utils.utils import point_cloud_info

if __name__ == "__main__":
    # take object name from user input interactively
    object_name = input("Enter the name of the object: ")
    # take number from user input interactively
    num = input("Enter the number of the pcd file you want to filter: ")
    num = int(num)

    # get the path of this file using os.path
    path = search_folder("/", "imagine2touch")

    while True:
        # Load point cloud from a file
        point_cloud = o3d.io.read_point_cloud(
            f"{path}/src/imagine2touch/data_collection/data/{object_name}/pcds/experiment_{num}_combined.pcd"
        )
        # radius filter on point cloud
        point_cloud.estimate_normals()
        # get point cloud info from function in ths repo
        point_cloud_info(point_cloud, display=True)

        # parameters for a dense point cloud
        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.0001)
        r = input("Enter the radius for radius outlier filter: ")
        r = float(r)
        point_cloud = point_cloud.remove_radius_outlier(nb_points=30, radius=r)[0]

        # visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])

        # # delete the old file
        os.remove(
            f"{path}/src/imagine2touch/data_collection/data/{object_name}/pcds/experiment_{num}_combined.pcd"
        )
        # Save the point cloud as a .pcd file
        o3d.io.write_point_cloud(
            f"{path}/src/imagine2touch/data_collection/data/{object_name}/pcds/experiment_{num}_combined.pcd",
            point_cloud,
        )
        # prompt user to continue or not
        cont = input("Do you want to continue? (y/n): ")
        if cont == "n":
            break
