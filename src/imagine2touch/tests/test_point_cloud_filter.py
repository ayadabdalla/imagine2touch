# README run in test_dummy_data directory
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
import src.imagine2touch.utils.utils as utils
from robot_io.cams.realsense.realsense import Realsense
from src.imagine2touch.localizers.aruco_cam_robot_world_localizer import (
    FrameUtil,
)  # Used to get transforms in the environment setting
from robot_io.cams.kinect4.kinect4 import Kinect4
import time

import sys


robot = RobotClient("/robot_io_ros_server")


def point_cloud_info(pcd, display=False):
    """
    Returns segregated arrays for each co-ordinate across all points
    <display>: print out pcd data
    """
    points = np.asarray(pcd.points)
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    if display:
        print("z min " + str(np.min(zs)))
        print("x min " + str(np.min(xs)))
        print("y min " + str(np.min(ys)))
        print("z max " + str(np.max(zs)))
        print("x max " + str(np.max(xs)))
        print("y max " + str(np.max(ys)))
    return xs, ys, zs


def segment_point_cloud(pcd, minz, maxz, minx):
    """
    filter out noisy points and distance threshold the pcd
    """
    xs, ys, zs = point_cloud_info(pcd)
    original_filter = (zs > minz) & (zs <= maxz) & (xs > minx)
    test_filter = True
    indices = np.where(original_filter & test_filter)
    pcd = pcd.select_by_index(indices[0], invert=False)
    # remove points that are on average farther away from their neighbors , if neighbours are 50
    pcd = pcd.remove_statistical_outlier(50, 0.2)
    pcd = pcd[0]  # get filtered pcd
    pcd = pcd.remove_radius_outlier(
        50, 0.01
    )  # remove points that have less than 50 neighbors in one centimeter
    pcd = pcd[0]  # get filtered pcd
    return pcd


def mask_rgb_and_dep(im, dep_im, mask_p1, mask_p2, show=False):
    """
    crop a target portion of the rgb and depth images w/o changing resolution
    """
    if show:
        cv2.rectangle(im, mask_p1, mask_p2, [0, 255, 0], 1)  # hardcoded from image
        cv2.imshow("image", im)  # show current image test
        cv2.waitKey(0)
    else:
        mask = np.zeros(im.shape[:2], dtype="uint8")
        cv2.rectangle(mask, mask_p1, mask_p2, 255, -1)  # hardcoded from image
        im = cv2.bitwise_and(im, im, mask=mask)
        dep_im = cv2.bitwise_and(dep_im * 1000, dep_im * 1000, mask=mask)
    return im, dep_im


def convert_image_to_point_cloud(
    camera,
    im,
    dep_im,
    extrinsic,
    minz,
    maxz,
    minx,
    mask_p1=None,
    mask_p2=None,
    display=False,
):
    if (mask_p1 is None) or (mask_p2 is None):
        dep_im = dep_im * 1000
    else:
        im, dep_im = mask_rgb_and_dep(im, dep_im, mask_p1, mask_p2)
    im = o3d.geometry.Image((im).astype(np.uint8))
    dep = o3d.geometry.Image(dep_im)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im, dep, convert_rgb_to_intensity=False
    )
    phc = o3d.camera.PinholeCameraIntrinsic()
    if camera == "kinect":
        phc.set_intrinsics(
            width=1920,
            height=1080,
            fx=912.103516,
            fy=911.885925,
            cx=953.576111,
            cy=555.760559,
        )
    else:
        phc.set_intrinsics(
            width=640,
            height=360,
            fx=322.37493896484375,
            fy=322.0753479003906,
            cx=314.71563720703125,
            cy=183.8709716796875,
        )
        print("intrinsics of realsense loaded")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, phc, extrinsic)
    pcd = segment_point_cloud(pcd, minz, maxz, minx)
    point_cloud_info(pcd, display)
    return pcd


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print('pass path to rgb and depth images')
    #     exit()

    # camera = Kinect4(0)
    # frame_util = FrameUtil(robot,camera)# here you initialized a program that will sit on top of the camera
    # frame_util.start() #change camera in this function to realsenses
    # rgb,dep = frame_util.camera.get_image() # here you get the latest image once
    # dep = dep * 1000

    # ##example image##
    # # file_rgb=sys.argv[1]
    # # file_dep=sys.argv[2]
    # # im_cv = cv2.imread(file_rgb)#[:,:,::-1]
    # # depth = Image.open(file_dep)
    # # depth = np.array(depth)
    # # utils.show_rgb_and_depth(rgb,dep,0)

    # ##adjust format for o3d##
    # img = o3d.geometry.Image((rgb).astype(np.uint8))
    # dep = o3d.geometry.Image(dep)
    # # print(np.unique(np.asarray(dep)))

    # ##conversion##
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, dep,convert_rgb_to_intensity=False)
    # phc = o3d.camera.PinholeCameraIntrinsic() # pinhole camera
    # phc.set_intrinsics(width=1920,height=1080,fx=912.103516,fy=911.885925,cx=953.576111,cy=555.760559) #kinect4
    # # phc.set_intrinsics(width=640,height=360,fx=322.37493896484375,fy=322.0753479003906,cx=314.71563720703125,cy=183.8709716796875)

    # extrinsic = utils.WORLD_IN_CAMERA #Kinect

    # # T_tcp_in_robot = robot.get_tcp_pose()
    # # extrinsic=utils.inverse_transform(utils.ROBOT_IN_WORLD.dot(
    # #                                     T_tcp_in_robot.dot(
    # #                                     utils.WCAMERA_IN_TCP))) #realsense
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, phc,extrinsic=extrinsic)#pcd object

    # ##processing and filteration##
    # # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) #invert y-axis

    # #threshold with distance
    # points=np.asarray(pcd.points)
    # zs=points[:,2]
    # ys=points[:,1]
    # xs=points[:,0]
    # indices=np.where((zs>0.01)&(zs<=0.2)&(xs>-0.1)&(xs<0.1)) # max threshold 20 cm z, min 0.5 cm z, and min -5 cm x
    # #print(indices[0].shape) #no. of points that passed the threshold filter from the pcd
    # pcd=pcd.select_by_index(indices[0],invert=False) #select points greater than min or smaller than max threshold

    # #other filters
    # pcd=pcd.remove_statistical_outlier(50, 0.2) #remove points that are on average farther away from their neighbors , if neighbours are one hundred
    # pcd=pcd[0] #get filtered pcd
    # pcd=pcd.remove_radius_outlier(50, 0.01) #remove points that have less than 50 neighbors in one centimeter
    # pcd=pcd[0] #get filtered pcd

    # ##visualization and logging##
    # pcd.estimate_normals()

    T_tcp_in_robot = robot.get_tcp_pose()
    W_CAM_IN_WORLD = utils.ROBOT_IN_WORLD.dot(T_tcp_in_robot.dot(utils.WCAMERA_IN_TCP))
    WORLD_IN_W_CAM = utils.inverse_transform(W_CAM_IN_WORLD)

    camera = Realsense(img_type="rgb_depth")
    print(camera.get_intrinsics())
    projection_matrix = camera.get_projection_matrix()
    resolution = np.reshape(
        np.array([camera.get_intrinsics()["width"], camera.get_intrinsics()["height"]]),
        (-1, 1),
    )
    rgb_w, depth_w = camera.get_image()
    del camera
    time.sleep(1)
    pcd = utils.convert_image_to_point_cloud(
        "realsense", rgb_w, depth_w, WORLD_IN_W_CAM, minz=0.005, maxz=0.2, minx=-0.06
    )
    pcd = pcd.translate([-0.0045, -0.013, -0.005])
    o3d.visualization.draw_geometries(
        [pcd], "window", point_show_normal=False
    )  # click and hold with mouse and rotate
    # print(np.asarray(pcd.points).shape) #total no. of points w/o filtering
    point_cloud_info(pcd, display=True)
