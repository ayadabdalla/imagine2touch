import hydra
from omegaconf import OmegaConf
from src.imagine2touch.utils.utils import (
    get_target_images,
    convert_image_to_point_cloud,
    point_cloud_info,
    show_rgb_and_depth,
    threed_vector_to_homog,
    get_crop_indeces,
    ROBOT_IN_WORLD,
    WCAMERA_IN_TCP,
    inverse_transform,
    make_plane,
)
from argparse import ArgumentParser
import os
import numpy as np
import open3d as o3d
import sys


def get_crops(W_CAM_IN_WORLD, WCAMERA_IN_TCP, point_in_world, cx=None, cy=None):
    """
    This function calculates the "realsense D405 camera" principal point offset for a reskin image crop
    - point in world: image center point in world coordinates
    """
    WORLD_IN_W_CAM = inverse_transform(W_CAM_IN_WORLD)
    hom_point_in_cam = WORLD_IN_W_CAM.dot(threed_vector_to_homog(point_in_world))
    # realsense D405 camera intrinsic parameters
    if (cx is None) or (cy is None):
        cx = 314.7156
        cy = 183.8709
    projection_matrix = np.array(
        [[322.3749, 0, cx, 0], [0, 322.07534, cy, 0], [0, 0, 1, 0]]
    )
    resolution = np.reshape([640, 360], (-1, 1))
    crop_indeces = get_crop_indeces(
        hom_point_in_cam,
        inverse_transform(WCAMERA_IN_TCP),
        projection_matrix,
        resolution,
    )
    x_1 = min(crop_indeces[0, :].astype(int))
    y_1 = min(crop_indeces[1, :].astype(int))
    x_2 = max(crop_indeces[0, :].astype(int))
    y_2 = max(crop_indeces[1, :].astype(int))
    return x_1, y_1, x_2, y_2


if __name__ == "__main__":
    ## script configuration
    cam_z_to_patch = 0.04 + -WCAMERA_IN_TCP[:, 3][2]
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("reconstruction.yaml")
    cfg.image_size = [int(cfg.image_size), int(cfg.image_size)]

    ##rgb and depth images ground truths
    path_rgb = f"{cfg.experiment_dir}/{cfg.object}/{cfg.object}_images/rgb"
    path_rgb = path_rgb.split(",")
    path_target = f"{cfg.experiment_dir}/{cfg.object}/{cfg.object}_images/depth"
    path_target = path_target.split(",")
    rgb_images = get_target_images(path_rgb, "rgb", cfg.image_size)
    rgb_images = np.asarray(rgb_images, dtype=np.uint8)
    depth_images = get_target_images(path_target, "depth", cfg.image_size)
    depth_images = np.asarray(depth_images, dtype=float)
    depth_images = np.reshape(depth_images, (-1, cfg.image_size[0] * cfg.image_size[1]))
    for i, image in enumerate(depth_images):
        for j, pixel in enumerate(image):
            if np.abs(pixel - (cam_z_to_patch * 1000)) > 20:
                image[j] = np.nan
        depth_images[i] = image
    depth_images = np.reshape(depth_images, (-1, cfg.image_size[0], cfg.image_size[1]))

    ## proprioception information
    # cam world transform when the camera was looking at the target patch
    wcamera_world_transforms = np.load(
        f"{cfg.experiment_dir}/{cfg.object}/{cfg.object}_poses/{cfg.object}_camera_world_transforms.npy",
        allow_pickle=True,
    )
    # final contact
    target_points_in_robot = np.load(
        f"{cfg.experiment_dir}/{cfg.object}/{cfg.object}_poses/{cfg.object}_target_points_in_robot.npy",
        allow_pickle=True,
    )
    target_points_in_robot = np.column_stack(
        (target_points_in_robot, np.ones((target_points_in_robot.shape[0], 1)))
    )

    ## predicted depth images
    if cfg.model:
        # model depth images
        path_target_predicted = (
            f"{cfg.experiment_dir}/{cfg.object}/{cfg.object}_images_reconstructed/masks"
        )
        path_target_predicted = path_target_predicted.split(",")
        depth_images_predicted = get_target_images(
            path_target_predicted, "masks", cfg.image_size, reconstructed=True
        )
        depth_images_predicted = np.asarray(depth_images_predicted, dtype=float)
        depth_images_predicted[depth_images_predicted == 0] = np.nan
        depth_images_predicted[depth_images_predicted == 1] = cam_z_to_patch
    else:
        # baseline depth image
        baseline_planes = []
        for point in target_points_in_robot:
            plane_generator_number = (
                np.ceil(np.sqrt(np.square(cfg.image_size[0]) / 2)) - 1
            )  # -1: for 0 starting index,/2: for half the area
            baseline_planes.append(
                make_plane(plane_generator_number, cam_z_to_patch, cfg.points)
            )
        baseline_planes = np.asarray(baseline_planes)
        baseline_planes = np.reshape(
            baseline_planes[:, : cfg.image_size[0] * cfg.image_size[1], :],
            (
                baseline_planes.shape[0],
                cfg.image_size[0],
                cfg.image_size[1],
                baseline_planes.shape[2],
            ),
        )
        depth_images_predicted = baseline_planes[:, :, :, 2]
    # print(np.min(depth_images_predicted))
    # print(np.max(depth_images_predicted))
    # print(np.mean(depth_images_predicted))

    ## prepare reconstruction
    points = []
    colors = []
    predicted_points = []
    colors_predicted = []
    distances_per_touch_way_one = []
    distances_per_touch_way_two = []
    distances_per_touch_compare_way_one = []
    distances_per_touch_compare_way_two = []
    current_size = 0
    current_size_predicted = 0

    for rgb_im, depth_im, transform, point in zip(
        rgb_images,
        np.divide(depth_images, 1000),
        wcamera_world_transforms,
        target_points_in_robot,
    ):  # gt reconstruction loop
        ## process proprioception information
        wcam_in_world = transform
        world_in_cam = inverse_transform(transform)
        x, y = get_crops(wcam_in_world, WCAMERA_IN_TCP, ROBOT_IN_WORLD.dot(point)[:3])
        pcd = convert_image_to_point_cloud(
            "realsense",
            rgb_im,
            depth_im,
            world_in_cam,
            minz=-1000,
            maxz=1000,
            minx=-1000,
            cx_offset=x,
            cy_offset=y,
            reconstruct_patches=True,
        )  # reconstruct gt pcd
        current_size = current_size + np.asarray(pcd.points).shape[0]
        points = np.append(points, np.asarray(pcd.points))
        points = np.reshape(points, (current_size, 3))
        colors = np.append(colors, np.asarray(pcd.colors))
        colors = np.reshape(colors, (current_size, 3))
        # show_rgb_and_depth(rgb_im,depth_im,0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd = o3d.io.read_point_cloud(
        f"{cfg.experiment_dir}/{cfg.object}/pcds/experiment_1_combined.pcd"
    )
    pcd_compare = o3d.io.read_point_cloud(
        f"{cfg.experiment_dir}/{cfg.comparison_object}/pcds/experiment_1_combined.pcd"
    )

    i = 5
    ###reconstruction
    for depth_predicted, wcamera_world_transform, point_in_robot in zip(
        depth_images_predicted, wcamera_world_transforms, target_points_in_robot
    ):
        ## process proprioception information
        wcam_in_world = wcamera_world_transform
        world_in_cam = inverse_transform(wcamera_world_transform)
        ## crop positions in full resolution image
        x, y = get_crops(
            wcam_in_world, WCAMERA_IN_TCP, ROBOT_IN_WORLD.dot(point_in_robot)[:3]
        )
        ## reconstruct pcd
        predicted_pcd = convert_image_to_point_cloud(
            "realsense",
            None,
            depth_predicted,
            world_in_cam,
            minz=-1000,
            maxz=1000,
            minx=-1000,
            cx_offset=x,
            cy_offset=y,
            reconstruct_patches=True,
        )
        ## accumulate touch distances to gt pcd
        distances_per_touch_way_one = np.append(
            distances_per_touch_way_one,
            np.asarray(predicted_pcd.compute_point_cloud_distance(pcd)),
        )
        distances_per_touch_way_two = np.append(
            distances_per_touch_way_two,
            np.asarray(pcd.compute_point_cloud_distance(predicted_pcd)),
        )
        distances_per_touch_compare_way_one = np.append(
            distances_per_touch_compare_way_one,
            np.asarray(predicted_pcd.compute_point_cloud_distance(pcd_compare)),
        )
        distances_per_touch_compare_way_two = np.append(
            distances_per_touch_compare_way_two,
            np.asarray(pcd_compare.compute_point_cloud_distance(predicted_pcd)),
        )
        ## accumulate touch points
        current_size_predicted = (
            current_size_predicted + np.asarray(predicted_pcd.points).shape[0]
        )
        current_touch_points = np.asarray(predicted_pcd.points)
        predicted_points = np.append(predicted_points, current_touch_points)
        predicted_points = np.reshape(predicted_points, (current_size_predicted, 3))
        i -= 1
        if i == 0:
            break
        ## visualize each touch
        # predicted_pcd=o3d.geometry.PointCloud()
        # predicted_pcd.points=o3d.utility.Vector3dVector(current_touch_points)
        # o3d.visualization.draw_geometries([pcd,predicted_pcd])
        # point_cloud_info(predicted_pcd,True)

    ## visualize all touches
    predicted_pcd = o3d.geometry.PointCloud()
    predicted_pcd.points = o3d.utility.Vector3dVector(predicted_points)
    ## compute distances of reconstructed pcd to gt pcd as a whole
    distances_way_one = np.asarray(predicted_pcd.compute_point_cloud_distance(pcd))
    distances_way_two = np.asarray(pcd.compute_point_cloud_distance(predicted_pcd))
    distances_compare_way_one = np.asarray(
        predicted_pcd.compute_point_cloud_distance(pcd_compare)
    )
    distances_compare_way_two = np.asarray(
        pcd_compare.compute_point_cloud_distance(predicted_pcd)
    )
    ## display metrics
    print(
        "mean of distances to gt pcd per point per touch",
        np.mean(distances_per_touch_way_one),
    )  # +np.mean(distances_per_touch_way_two))
    print(
        "mean of distances to gt pcd per point for a reconstructed pcd",
        np.mean(distances_way_one),
    )  # +np.mean(distances_way_two))
    print(
        "mean of distances to compare pcd per point per touch",
        np.mean(distances_per_touch_compare_way_one),
    )  # +np.mean(distances_per_touch_compare_way_two))
    print(
        "mean of distances to compare pcd per point for a reconstructed pcd",
        np.mean(distances_compare_way_one),
    )  # +np.mean(distances_compare_way_two))
    print("model", cfg.model, "points", cfg.points)
    ## visualize pcds
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_compare)
    vis.add_geometry(predicted_pcd)
    # point_cloud_info(predicted_pcd,True)
    vis.run()


### old code
# for rgb_im,depth_im,depth_predicted,transform,point in \
#     zip(rgb_images,np.divide(depth_images,1000),np.divide(depth_images_predicted,1000),camera_world_transforms,target_points_in_robot): #gt reconstruction loop
#     predicted_pcd = convert_image_to_point_cloud("realsense",rgb_im,depth_predicted,world_in_cam,minz=-1000,maxz=1000,minx=-1000,cx_offset=x,cy_offset=y)
#     pcd = convert_image_to_point_cloud("realsense",rgb_im,depth_im,world_in_cam,minz=-1000,maxz=1000,minx=-1000,cx_offset=x,cy_offset=y) #reconstruct gt pcd
#     current_size=current_size+np.asarray(pcd.points).shape[0]
#     points=np.append(points,np.asarray(pcd.points))
#     points=np.reshape(points,(current_size,3))
#     colors=np.append(colors,np.asarray(pcd.colors))
#     colors=np.reshape(colors,(current_size,3))
#     show_rgb_and_depth(rgb_im,depth_im,0)
#     #color predicted pcd
#     colors_predicted=np.append(colors_predicted,np.asarray(predicted_pcd.colors))
#     colors_predicted=np.reshape(colors_predicted,(current_size_predicted,3))
##for reconstructing gt pcd
# pcd=o3d.geometry.PointCloud()
# pcd.points=o3d.utility.Vector3dVector(points)
# pcd.colors=o3d.utility.Vector3dVector(colors)
