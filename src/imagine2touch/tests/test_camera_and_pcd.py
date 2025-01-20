from robot_io.cams.kinect4.kinect4 import Kinect4
import cv2
import open3d as o3d
import numpy as np
from PIL import Image
from robot_io.cams.realsense.realsense import Realsense
from sklearn.preprocessing import MinMaxScaler
from src.imagine2touch.utils.utils import (
    inverse_transform,
    ROBOT_IN_WORLD,
    WCAMERA_IN_TCP,
    point_cloud_info,
)
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient


robot = RobotClient("/robot_io_ros_server")
# camera =Kinect4(0)#get camera
camera = Realsense(img_type="rgb_depth")  # uncomment this line for gripper cam
scaler_images = MinMaxScaler()


print(camera.get_intrinsics())  # current kinect 4b camera parameters
while True:
    rgb, depth = camera.get_image()  # get live image
    cv2.imshow("image", rgb[:, :, ::-1])
    cv2.imshow("depth", depth)
    cv2.waitKey(0)
    # Test mask co-ordinates on live image
    # mask = np.zeros(rgb.shape[:2], dtype="uint8")
    # cv2.rectangle(mask,(850,150),(1130,500),255,-1)
    # rgb = cv2.bitwise_and(rgb,rgb,mask=mask)
    # depth = cv2.bitwise_and(depth,depth,mask=mask)
    # utils.show_rgb_and_depth(rgb,depth)

    # save masked image
    rgb_im = Image.fromarray(rgb)
    rgb_im.save("test.jpeg")
    print("saved rgb")

    # depth image masked, convert to mm then save
    dep_im = (depth * 1000).astype(np.uint16)
    dep_im = Image.fromarray(dep_im)
    dep_im.save("test_dep.tif")
    print("saved depth")
    # correct format for open 3d
    img = o3d.geometry.Image(rgb.astype(np.uint8))
    dep = o3d.geometry.Image((depth * 1000).astype(np.uint16))

    # conversion rgbd to point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        img, dep, convert_rgb_to_intensity=False
    )  # merge rgb and depth
    phc = o3d.camera.PinholeCameraIntrinsic()  # pinhole camera
    # phc.set_intrinsics(width=1920,height=1080,fx=912.103516,fy=911.885925,cx=953.576111,cy=555.760559)#phc intrinsic parameters of kinect4
    phc.set_intrinsics(
        width=640,
        height=360,
        fx=322.37493896484375,
        fy=322.0753479003906,
        cx=314.71563720703125,
        cy=183.8709716796875,
    )
    # get extrinsic#
    T_tcp_in_robot = robot.get_tcp_pose()
    W_CAM_IN_WORLD = ROBOT_IN_WORLD.dot(T_tcp_in_robot.dot(WCAMERA_IN_TCP))
    WORLD_IN_W_CAM = inverse_transform(W_CAM_IN_WORLD)
    ###############
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, phc, WORLD_IN_W_CAM, project_valid_depth_only=True
    )  # pcd object
    xs, ys, zs = point_cloud_info(pcd)
    original_filter = (
        (zs > 0) & (zs <= 0.2) & (xs > -0.01) & (xs < 0.01) & (ys > -0.01) & (ys < 0.01)
    )
    test_filter = True
    indices = np.where(original_filter & test_filter)
    pcd = pcd.select_by_index(indices[0], invert=False)

    # pcd = pcd.voxel_down_sample(voxel_size=0.005)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries(
        [pcd], "window"
    )  # click and hold with mouse and rotate
