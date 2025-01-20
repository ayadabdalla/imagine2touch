# repo modules
if False:
    import rospy
    from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
    from robot_io.cams.realsense.realsense import Realsense
    from robot_io.utils.utils import pos_orn_to_matrix
    from src.imagine2touch.task.save_pcds_extra_views import (
        wait_until_stable_joint_velocities,
    )
    from src.imagine2touch.localizers.wrist_camera_localizer import (
        generate_step_orientation,
        generate_step_pos,
    )
    from src.imagine2touch.task.save_pcds_extra_views import (
        wait_until_stable_joint_velocities,
    )

from src.imagine2touch.reskin_sensor.reskin_sensor.sensor_proc import ReSkinProcess
from src.imagine2touch.utils.utils import (
    FIXED_ROBOT_ORN,
    ROBOT_IN_WORLD,
    WORLD_IN_ROBOT,
    WCAMERA_IN_TCP,
    AWAY_POSE,
    inverse_transform,
    eulertoquat,
    move_to_pt_with_v_safe,
    euler_from_vector,
    homog_vector_to_3d,
    threed_vector_to_homog,
    get_crop_indeces,
    convert_image_to_point_cloud,
    point_cloud_info,
    plot_reskin,
    segment_point_cloud,
    debounce_tcp_pose,
    filter_reskin,
    search_folder,
)
from src.imagine2touch.utils.data_utils import (
    safety_one,
    safety_two,
    safety_three,
    custom_visualization_pcd,
    discretize_vector,
    set_up_directory,
    capture_wristcam_image,
    log_experiment_meta_data,
)

# Standard useful libraries
import numpy as np
import open3d as o3d
import signal
import sys
from PIL import Image
import os
import hydra
from omegaconf import OmegaConf
import sys

repo_path = search_folder("/", "imagine2touch")


def meta_script_init():
    # script meta data, configuration and constants
    OmegaConf.register_new_resolver("quarter_pi", lambda x: np.pi / 4)
    # remove leading slash from string
    hydra.initialize("../src/imagine2touch/data_collection/cfg", version_base=None)
    cfg = hydra.compose("collection.yaml")
    robot_flange_in_tcp = [float(num) for num in cfg.robot_flange_in_tcp.split(",")]
    N = cfg.N - 1  # counting from 0
    reskin_recordings = []
    ambient_recordings = []
    wrench_recordings = []
    contact_returns = []
    prepare_poses = []
    final_poses = []
    rotation_list = []
    joints_state_list = []
    reskin_contact_magnitudes = []
    count_contact = 0
    previous_reskin = 0
    set_up_directory(cfg)
    log_experiment_meta_data(cfg)
    np.save(
        f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/experiment_{cfg.experiment_number}_wcamera_in_tcp.npy",
        WCAMERA_IN_TCP,
        allow_pickle=True,
    )
    np.save(
        f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/experiment_{cfg.experiment_number}_robot_in_world.npy",
        ROBOT_IN_WORLD,
        allow_pickle=True,
    )
    return (  # return the variables to be used in the main function
        cfg,
        robot_flange_in_tcp,
        N,
        reskin_recordings,
        ambient_recordings,
        wrench_recordings,
        contact_returns,
        prepare_poses,
        final_poses,
        rotation_list,
        joints_state_list,
        reskin_contact_magnitudes,
        count_contact,
        previous_reskin,
    )


def init_devices():
    # initialize devices
    robot = RobotClient("/robot_io_ros_server")
    sensor_settings = ReSkinSettings(
        num_mags=5, port="/dev/ttyACM0", baudrate=115200, burst_mode=True, device_id=1
    )
    sensor_process = ReSkinProcess(sensor_settings)
    camera = Realsense(img_type="rgb_depth")
    projection_matrix = camera.get_projection_matrix()
    resolution = np.reshape(
        np.array([camera.get_intrinsics()["width"], camera.get_intrinsics()["height"]]),
        (-1, 1),
    )
    ## start devices
    ### ReSkin tactile sensor
    sensor_process.start()
    wait_until_stable_joint_velocities(robot)
    ### robot
    start_pos = robot.get_state()["tcp_pos"]
    start_rot = robot.get_state()["tcp_orn"]
    start_pos_up = [start_pos[0], start_pos[1], cfg.minimum_robot_height]
    if start_pos[2] < cfg.minimum_robot_height:
        robot.move_cart_pos_abs_ptp(start_pos_up, eulertoquat(FIXED_ROBOT_ORN))
        print("raised robot")
    else:
        robot.move_cart_pos_abs_ptp(
            start_pos, eulertoquat(FIXED_ROBOT_ORN)
        )  # reset robot orientation

        start_rot = FIXED_ROBOT_ORN  # update start_rot
        print("oriented robot")
    wait_until_stable_joint_velocities(robot)
    start_pos = 1 * robot.get_state()["tcp_pos"][:3]  # update start pos
    print("robot_reset done")
    return (
        robot,
        sensor_process,
        camera,
        projection_matrix,
        resolution,
        start_pos,
        start_rot,
        start_pos_up,
    )


class ExperimentStrategy:
    def get_pcd_file(self):
        file = f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/experiment_{cfg.experiment_number}_combined.pcd"
        return file

    def finalize_rendering(self):
        pcd = o3d.io.read_point_cloud(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/experiment_{cfg.experiment_number}_combined.pcd"
        )
        o3d.visualization.draw_geometries([pcd])
        if (
            input(
                "Are you satisfied with the point cloud, if not exit and filter it manually? (y/n)"
            )
            == "n"
        ):
            sys.exit()
        else:
            print("continuing")
        return pcd

    def get_points_normals(self, pcd):
        pcd.estimate_normals()
        original_points = np.asarray(pcd.points)
        original_normals = np.asarray(pcd.normals)
        return original_points, original_normals

    def finalize_strategy(self):
        if input("Do you want to proceed with data collection? (y/n)") == "n":
            sys.exit()
        else:
            print("continuing")

    def load_safe_points(self, cfg):
        safe_points_indeces = np.load(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_combined_safe_points_{cfg.experiment_number}.npy",
            allow_pickle=True,
        )
        safe_points_indeces = safe_points_indeces[-1]
        safe_points_indeces = np.int64(safe_points_indeces)
        print("safe points loaded")
        return safe_points_indeces


class NewExperimentStrategy(ExperimentStrategy):
    def render(self, cfg, robot, camera, camera_type):
        print("pcds dont exist or creating new pcds")
        pcd_views = []
        i = 1
        pose_in_world = [float(num) for num in cfg.starting_corner_in_world.split(",")]
        orns = generate_step_orientation(
            cfg.fixed_roll, cfg.max_inclination, cfg.orientation_step / 2
        )
        poses = generate_step_pos(
            pose_in_world,
            cfg.r,
            cfg.initial_orientation,
            len(orns),
            cfg.orientation_step,
            cfg.translation_step,
        )
        orns = np.repeat(orns, 7, axis=0)
        views = zip(poses, orns)
        for view, orn in views:
            view = WORLD_IN_ROBOT.dot(view)[:3]
            try:
                robot.move_cart_pos_abs_ptp(view, eulertoquat(orn))
            except rospy.service.ServiceException as e:
                if (
                    np.linalg.norm(robot.get_state()["tcp_pos"][:3] - view, 2)
                    > cfg.pos_tolerance
                ):
                    print("failed to get view")
                    continue
                else:
                    print("got view within the following tolerance: ")
                    print(np.linalg.norm(robot.get_state()["tcp_pos"][:3] - view, 2))
            T_tcp_in_robot = debounce_tcp_pose(
                robot, delay=False
            )  # ensure updated robot state
            W_CAM_IN_WORLD = ROBOT_IN_WORLD.dot(T_tcp_in_robot.dot(WCAMERA_IN_TCP))
            WORLD_IN_W_CAM = inverse_transform(W_CAM_IN_WORLD)
            rgb_w, depth_w = camera.get_image()
            pcd_view = convert_image_to_point_cloud(
                camera_type,
                rgb_w,
                depth_w,
                WORLD_IN_W_CAM,
                minz=cfg.crop.minz,
                maxz=cfg.crop.maxz,
                minx=cfg.crop.minx,
                maxx=cfg.crop.maxx,
                miny=cfg.crop.miny,
                maxy=cfg.crop.maxy,
                voxel=False,
                radius_filter=True,
                segment=True,
            )
            point_cloud_info(pcd_view, True)
            pcd_views.append(pcd_view)
            i += 1
        pcds = pcd_views
        pcd = o3d.geometry.PointCloud()
        for point_id in range(len(pcds)):
            pcd += pcds[point_id]
        pcd = segment_point_cloud(
            pcd,
            minz=-1000,
            maxz=1000,
            minx=-1000,
            maxx=1000,
            miny=-1000,
            maxy=1000,
            statistical_filter=False,
            voxel=True,
            voxel_size=0.0001,
            radius_filter=True,
            radius=0.001,
            nb_points=30,
        )
        o3d.io.write_point_cloud(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/experiment_{cfg.experiment_number}_combined.pcd",
            pcd,
        )
        pcd = super().finalize_rendering()
        return pcd

    def voxelize(self, pcd):
        print(f"downsampling pcd {cfg.object_name}")
        pcd = pcd.voxel_down_sample(voxel_size=0.0005)
        o3d.io.write_point_cloud(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_{cfg.experiment_number}_down.pcd",
            pcd,
        )
        print(f"saved downsampled pcd {cfg.object_name}")
        original_points, original_normals = super().get_points_normals(pcd)
        return original_points, original_normals

    def create_safe_points(self, cfg, pcd, original_points, original_normals):
        print("creating new safe points")
        safe_points_indeces = []
        print(f"creating safe pcd for {cfg.object_name}")
        j = 0
        for point, normal in zip(original_points, original_normals):
            ###get point in robot
            point_homog = threed_vector_to_homog(point)
            point_homog_in_robot = WORLD_IN_ROBOT.dot(point_homog)
            point_in_robot = homog_vector_to_3d(point_homog_in_robot)
            ###get goal orientation from normal
            negative_normal = -1 * normal
            if np.linalg.norm(negative_normal, 2) == 0:
                continue
            alignment_vector = negative_normal / np.linalg.norm(negative_normal, 2)
            goal_orientation = euler_from_vector(0, alignment_vector)
            ###safety checks
            condition_1 = safety_one(normal, cfg.safety.one)
            _, _, condition_2 = safety_two(
                pcd,
                point,
                alignment_vector,
                cfg.reskin_side_length,
                cfg.safety.two,
            )
            condition_3 = safety_three(
                point_in_robot[:3],
                eulertoquat(goal_orientation),
                robot_flange_in_tcp,
                cfg.safety.three,
            )
            if condition_1 and condition_2 and condition_3:
                safe_points_indeces.append(j)
            j += 1
            print(f"point {j} out of {len(original_points)}")
        ### save safe points and normals
        safe_points_indeces.append(np.asarray(safe_points_indeces, dtype=object))
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_combined_safe_points_{cfg.experiment_number}.npy",
            "wb",
        ) as safe_indeces_file:
            np.save(safe_indeces_file, safe_points_indeces)
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_original_normals_{cfg.experiment_number}.npy",
            "wb",
        ) as normals_file:
            np.save(normals_file, original_normals)
        print("saved safe points indeces and normals \n")
        safe_points_indeces = super().load_safe_points(cfg)
        return safe_points_indeces

    def run_strategy(self, cfg):
        pcd = self.render(cfg, robot)
        original_points, original_normals = self.voxelize(pcd)
        safe_points_indeces = self.create_safe_points(
            cfg, pcd, original_points, original_normals
        )
        return pcd, original_points, original_normals, safe_points_indeces


class PastExperimentStrategy(ExperimentStrategy):
    def render(self, cfg):
        print("copying last experiment pcds")
        file = f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/experiment_{cfg.experiment_number - 1}_combined.pcd"
        pcd = o3d.io.read_point_cloud(file)
        o3d.io.write_point_cloud(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/experiment_{cfg.experiment_number}_combined.pcd",
            pcd,
        )
        pcd = super().finalize_rendering()
        return pcd

    def voxelize(self):
        print("copying last experiment downsampeled pcd")
        file = f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_{cfg.experiment_number - 1}_down.pcd"
        pcd = o3d.io.read_point_cloud(file)
        o3d.io.write_point_cloud(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_{cfg.experiment_number}_down.pcd",
            pcd,
        )
        original_points, original_normals = super().get_points_normals(pcd)
        return original_points, original_normals

    def create_safe_points(self, cfg):
        #### copy safe points indeces and normals of previous experiment
        print("copying safe points indeces and normals")
        safe_points_indeces = np.load(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_combined_safe_points_{cfg.experiment_number - 1}.npy",
            allow_pickle=True,
        )
        original_normals = np.load(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_original_normals_{cfg.experiment_number - 1}.npy",
            allow_pickle=True,
        )
        print("copied safe points indeces and normals \n")
        #### write the loaded files to the current experiment
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_combined_safe_points_{cfg.experiment_number}.npy",
            "wb",
        ) as safe_indeces_file:
            np.save(safe_indeces_file, safe_points_indeces)
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_original_normals_{cfg.experiment_number}.npy",
            "wb",
        ) as normals_file:
            np.save(normals_file, original_normals)
        safe_points_indeces = super().load_safe_points(cfg)
        return safe_points_indeces

    def run_strategy(self, cfg):
        pcd = self.render(cfg)
        original_points, original_normals = self.voxelize()
        safe_points_indeces = self.create_safe_points(cfg)
        return pcd, original_points, original_normals, safe_points_indeces


class RepeatExperimentStrategy(ExperimentStrategy):
    def render(self):
        print("pcds exist, not creating new pcds or copying old pcds")
        pcd = super().finalize_rendering()
        return pcd

    def voxelize(self):
        file = f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_{cfg.experiment_number}_down.pcd"
        print(f"down sampeled pcd for {cfg.object_name} exists")
        pcd = o3d.io.read_point_cloud(file)
        original_points, original_normals = super().get_points_normals(pcd)
        return original_points, original_normals

    def create_safe_points(self, cfg):
        safe_points_indeces = super().load_safe_points(cfg)
        return safe_points_indeces

    def run_strategy(self, cfg):
        pcd = self.render()
        original_points, original_normals = self.voxelize()
        safe_points_indeces = self.create_safe_points(cfg)
        return pcd, original_points, original_normals, safe_points_indeces


if __name__ == "__main__":
    (
        cfg,
        robot_flange_in_tcp,
        N,
        reskin_recordings,
        ambient_recordings,
        wrench_recordings,
        contact_returns,
        prepare_poses,
        final_poses,
        rotation_list,
        joints_state_list,
        reskin_contact_magnitudes,
        count_contact,
        previous_reskin,
    ) = meta_script_init()
    if cfg.copy_experiment and cfg.new_experiment:
        raise Exception(
            "copying pcds and creating new pcds are mutually exclusive, please choose one"
        )
    # (
    #     robot,
    #     sensor_process,
    #     camera,
    #     projection_matrix,
    #     resolution,
    #     start_pos,
    #     start_rot,
    #     start_pos_up,
    # ) = init_devices()
    if cfg.new_experiment:
        pcd_strategy = NewExperimentStrategy()
    elif cfg.copy_experiment:
        pcd_strategy = PastExperimentStrategy()
    else:
        file_pcd = f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/experiment_{cfg.experiment_number}_combined.pcd"
        file_pcd_down = f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_{cfg.experiment_number}_down.pcd"
        file_safe_points = f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_combined_safe_points_{cfg.experiment_number}.npy"
        file_normals = f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/pcds/{cfg.object_name}_original_normals_{cfg.experiment_number}.npy"
        # check if all files exist
        if (
            not os.path.isfile(file_pcd)
            and not os.path.isfile(file_pcd_down)
            and not os.path.isfile(file_safe_points)
            and not os.path.isfile(file_normals)
        ):
            raise Exception(
                "required files dont exist, please set new_experiment or copy_experiment to True"
            )
        pcd_strategy = RepeatExperimentStrategy()
    (
        pcd,
        original_points,
        original_normals,
        safe_point_indices,
    ) = pcd_strategy.run_strategy(cfg=cfg)
    sys.exit()
    # start automation
    while count_contact <= N:
        ## restore a good starting position
        current_pos = robot.get_state()["tcp_pos"]
        current_rot = robot.get_state()["tcp_orn"]
        current_pos_homog = threed_vector_to_homog(current_pos)
        current_pos_homog_in_world = ROBOT_IN_WORLD.dot(current_pos_homog)
        current_pos_in_world = homog_vector_to_3d(current_pos_homog_in_world)
        if count_contact >= 0:
            if current_pos_in_world[2] > 0.2 or current_pos_in_world[2] < 0.08:
                current_pos_in_world[2] = 0.1
                current_pos_homog_in_robot_up = WORLD_IN_ROBOT.dot(
                    threed_vector_to_homog(current_pos_in_world)
                )
                current_pos_in_robot_up = homog_vector_to_3d(
                    current_pos_homog_in_robot_up
                )
                try:
                    print("attempting to restore to a higher position")
                    robot.move_cart_pos_abs_ptp(current_pos_in_robot_up, current_rot)
                except:
                    print("ptp failed with same orientation, attempting away pose")
                    robot.move_cart_pos_abs_ptp(AWAY_POSE, eulertoquat(FIXED_ROBOT_ORN))

        ## get a sampled point from pcd
        points = original_points[safe_points_indeces]
        normals = original_normals[safe_points_indeces]
        elements = np.where(points[:, 2] >= -100)  # heurestic to get all indices
        probabilities = np.ones(elements[0].shape[0]) * (1 / elements[0].shape[0])
        sampled_points_indices = np.random.choice(
            elements[0], 1, p=probabilities, replace=False
        )
        sampled_point_index = sampled_points_indices[0]
        sampled_points = points[sampled_points_indices]
        sampled_point = sampled_points[0]
        sampled_point_homog = threed_vector_to_homog(sampled_point)
        sampled_point_homog_in_robot = WORLD_IN_ROBOT.dot(sampled_point_homog)
        sampled_point_in_robot = homog_vector_to_3d(sampled_point_homog_in_robot)
        sampled_normals = normals[sampled_points_indices]  # get normal to sampled point
        sampled_normal = sampled_normals[0]
        ### get goal orientation from normal
        negative_normal = -1 * sampled_normal
        alignment_vector = negative_normal / np.linalg.norm(negative_normal, 2)
        goal_orientation = euler_from_vector(0, alignment_vector)
        ### get usable positions from sampled point
        sampled_point_in_robot_prepare = (
            sampled_point_in_robot - cfg.normal_looking_distance * alignment_vector
        )
        sampled_point_in_robot_away = (
            sampled_point_in_robot
            - (cfg.normal_looking_distance + 0.02) * alignment_vector
        )
        ### visualization
        (
            discretized_normal_array_negative,
            discretized_normal_array_positive,
        ) = discretize_vector(sampled_point, sampled_normal, 0.005, 10)
        not_safe_two, safe_two, condition_2 = safety_two(
            pcd,
            sampled_point,
            alignment_vector,
            cfg.reskin_side_length,
            cfg.safety.two,
        )
        (
            discretized_normal_array_negative,
            discretized_normal_array_positive,
        ) = discretize_vector(sampled_point, sampled_normal, 0.005, 10)
        pcd_viz = custom_visualization_pcd(
            pcd,
            original_points,
            discretized_normal_array_negative,
            discretized_normal_array_positive,
            True,
            safe_two,
            not_safe_two,
            safe_patch_center=sampled_point_index,
            safe_points=points,
        )

        ## moving the robot
        prepare_contact = sampled_point_in_robot_prepare
        away_from_contact = sampled_point_in_robot_away
        final_contact = sampled_point_in_robot
        end = False  # stopping signal handler

        def cb_int(*args):
            global end
            end = True

        signal.signal(signal.SIGINT, cb_int)
        ### orient the wrist camera with the alignment vector
        e_prepare_tcp_in_robot = pos_orn_to_matrix(
            prepare_contact, eulertoquat(goal_orientation)
        )  # expected tcp to robot transform after aligning the tcp with the alignment vector
        OFFSET_TCP_FROM_WCAM_IN_XY = np.array(
            [-WCAMERA_IN_TCP[:, 3][0], -WCAMERA_IN_TCP[:, 3][1], 0, 1]
        )  # calculate the tcp offset to get the wrist camera in the former place (x,y), -ve sign to reverse the offset
        cam_looking_tcp_in_robot_pos = e_prepare_tcp_in_robot.dot(
            OFFSET_TCP_FROM_WCAM_IN_XY
        )  # tcp to robot transform after offsetting the tcp
        if (
            ROBOT_IN_WORLD.dot(threed_vector_to_homog(cam_looking_tcp_in_robot_pos))[2]
            < cfg.camera_looking_safety_tolerance_in_world
        ):
            print("rejected wrist camera alignment, will hit while looking")
            continue
        current_tcp_pos = robot.get_state()["tcp_pos"]
        current_tcp_orn = robot.get_state()["tcp_orn"]
        try:
            robot.move_cart_pos_abs_ptp(
                (cam_looking_tcp_in_robot_pos[:3]),
                eulertoquat(goal_orientation),
            )
        except rospy.service.ServiceException as e:
            error = np.linalg.norm(
                robot.get_state()["tcp_pos"][:3] - cam_looking_tcp_in_robot_pos[:3],
                2,
            )
            if error > cfg.franka_exception_pos_tolerance:
                print(
                    "failed to orient the wrist camera with the alignment vector, error is:",
                    error,
                )
                robot.move_cart_pos_abs_ptp(start_pos_up, eulertoquat(FIXED_ROBOT_ORN))
                continue
            else:
                print(
                    "oriented the wrist camera with the alignment vector within the following tolerance",
                    np.linalg.norm(
                        robot.get_state()["tcp_pos"][:3]
                        - cam_looking_tcp_in_robot_pos[:3],
                        2,
                    ),
                )
                wait_until_stable_joint_velocities(robot)  # ensure updated robot state
        print("success in orienting the wrist camera with the alignment vector")
        wait_until_stable_joint_velocities(robot)

        ### collect vision data
        T_tcp_in_robot = robot.get_tcp_pose()
        W_CAM_IN_WORLD = ROBOT_IN_WORLD.dot(T_tcp_in_robot.dot(WCAMERA_IN_TCP))
        WORLD_IN_W_CAM = inverse_transform(W_CAM_IN_WORLD)
        hom_point_in_cam = WORLD_IN_W_CAM.dot(threed_vector_to_homog(sampled_point))
        crop_indeces = get_crop_indeces(
            hom_point_in_cam,
            inverse_transform(WCAMERA_IN_TCP),
            projection_matrix,
            resolution,
        )
        rgb, dep = capture_wristcam_image()
        x_1 = min(crop_indeces[0, :].astype(int))
        x_2 = max(crop_indeces[0, :].astype(int))
        y_1 = min(crop_indeces[1, :].astype(int))
        y_2 = max(crop_indeces[1, :].astype(int))
        rgb = rgb[y_1:y_2, x_1:x_2]
        dep = dep[y_1:y_2, x_1:x_2]
        print("took shot\n")

        ### prepare tcp position for contact
        try:
            robot.move_cart_pos_abs_ptp(prepare_contact, eulertoquat(goal_orientation))
            print("success in moving to prepare_contact")
        except rospy.service.ServiceException as e:
            wait_until_stable_joint_velocities(robot)
            if (
                np.linalg.norm(robot.get_state()["tcp_pos"][:3] - prepare_contact, 2)
                > cfg.franka_exception_pos_tolerance
            ):
                print(
                    "failed to prepare_contact, error is:",
                    np.linalg.norm(
                        robot.get_state()["tcp_pos"][:3] - prepare_contact, 2
                    ),
                )
                robot.move_cart_pos_abs_ptp(start_pos_up, eulertoquat(FIXED_ROBOT_ORN))
                continue
            else:
                print(
                    "prepared contact within the following tolerance: ",
                    np.linalg.norm(
                        robot.get_state()["tcp_pos"][:3] - prepare_contact, 2
                    ),
                )
                wait_until_stable_joint_velocities(robot)  # ensure updated robot state
        wait_until_stable_joint_velocities(robot)

        ### make contact
        rotation = robot.get_state()["tcp_orn"]
        contact = move_to_pt_with_v_safe(
            robot,
            goal_point=final_contact,
            goal_rot=rotation,
            sensor_process=sensor_process,
            p_i=cfg.movement_wrapper_step_size,
            dt=cfg.movement_wrapper_dt,
            end=end,
            f_max=cfg.f_max_contact,
            T_max=cfg.T_max_contact,
            direction=alignment_vector,
            contact=True,
            reskin_threshold=cfg.reskin_threshold,
            reskin_ambient_recording_distance_to_contact_threshold=cfg.reskin_ambient_recording_distance_to_contact_threshold,
        )
        print(f"moving to contact point function returned with {contact[0]}")
        wait_until_stable_joint_velocities(robot)
        if contact[3] is not None:
            current_reskin = filter_reskin(
                sensor_process.get_data(num_samples=2),
                multiple_samples=True,
                norm=True,
            )
            ambient_recording = contact[3]
        else:
            print("no ambient recording, skipping")
            wait_until_stable_joint_velocities(robot)
            rotation = robot.get_state()["tcp_orn"]
            state = move_to_pt_with_v_safe(
                robot,
                goal_point=away_from_contact,
                goal_rot=rotation,
                p_i=cfg.movement_wrapper_step_size,
                dt=cfg.movement_wrapper_dt,
                end=end,
                f_max=cfg.f_max_away,
                T_max=cfg.T_max_away,
                direction=alignment_vector,
                contact=False,
                goal_distance_tol=0.02,
            )
            if state[0] == -1:
                print("failed to move away from contact on the alignment vector")
                continue
            print("success in moving away from contact on the alignment vector")
            wait_until_stable_joint_velocities(robot)
            continue

        ### record data
        count_contact = count_contact + 1
        if previous_reskin == current_reskin or current_reskin == 0:
            raise Exception("sensor data corrupted, check cables")
        reskin_recordings.append(sensor_process.get_data(num_samples=1))
        ambient_recordings.append(ambient_recording)
        reskin_recordings_np = np.array(reskin_recordings, dtype=object)
        ambient_recordings_np = np.array(ambient_recordings, dtype=object)
        plot_reskin(reskin_recordings)  # visualize reskin data
        #### accumulate robot proprioception data
        prepare_poses.append(prepare_contact)
        final_poses.append(final_contact)
        rotation_list.append(eulertoquat(goal_orientation))
        joints_state_list.append(robot.get_state()["joint_positions"])
        #### get images
        rgb_im = Image.fromarray(rgb)
        dep_im = (dep * 1000).astype(np.uint16)
        dep_im = Image.fromarray(dep_im)
        #### get robot wrench and reskin norm gradient while in contact
        state = contact[1]
        wrench = state["force_torque"]
        wrench_recordings.append(wrench)
        contact_returns.append(contact[0])
        if contact[0] == 2:
            reskin_contact_magnitudes.append(contact[2])
        else:
            reskin_contact_magnitudes.append(-1)
        #### save data
        rgb_im.save(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_images/rgb/experiment_{cfg.experiment_number}_rgb_{count_contact}.png"
        )  # save rgb image crop
        dep_im.save(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_images/depth/experiment_{cfg.experiment_number}_depth_{count_contact}.tif"
        )  # save depth image crop
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_tactile/experiment_{cfg.experiment_number}_reskin",
            "wb",
        ) as readings:
            np.save(
                readings, reskin_recordings_np
            )  # append reskin tactile reading to saved reskin readings
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_tactile/experiment_{cfg.experiment_number}_reskin_ambient",
            "wb",
        ) as ambient_readings:
            np.save(
                ambient_readings, ambient_recordings_np
            )  # append ambient reskin tactile reading to saved ambient reskin readings
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_poses/experiment_{cfg.experiment_number}_prepare_contact",
            "wb",
        ) as prepare:
            np.save(
                prepare, prepare_poses
            )  # append prepare contact position to saved prepare contact positions
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_poses/experiment_{cfg.experiment_number}_final_contact",
            "wb",
        ) as final:
            np.save(
                final, final_poses
            )  # append final contact position to saved final contact positions
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_poses/experiment_{cfg.experiment_number}_rotations",
            "wb",
        ) as rotation:
            np.save(
                rotation, rotation_list
            )  # append rotation to saved rotations; rotation is similar for prepare and final contact
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_poses/experiment_{cfg.experiment_number}_joints_state",
            "wb",
        ) as joints:
            np.save(
                joints, joints_state_list
            )  # append joint states to saved joint states
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_forces/experiment_{cfg.experiment_number}_forces_torques",
            "wb",
        ) as wrenches:
            np.save(wrenches, wrench_recordings)  # append wrench to saved wrenches
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_forces/experiment_{cfg.experiment_number}_contact_returns",
            "wb",
        ) as contact_return:
            np.save(
                contact_return, contact_returns
            )  # append contact return state to saved contact return states
        with open(
            f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_forces/experiment_{cfg.experiment_number}_reskin_contact_magnitudes",
            "wb",
        ) as reskin_mags:
            np.save(
                reskin_mags, reskin_contact_magnitudes
            )  # append reskin contact magnitude to saved reskin contact magnitudes
        print("saved data")

        ### move away to tcp prepare position after contact
        rotation = robot.get_state()["tcp_orn"]
        state = move_to_pt_with_v_safe(
            robot,
            goal_point=away_from_contact,
            goal_rot=rotation,
            p_i=cfg.movement_wrapper_step_size,
            dt=cfg.movement_wrapper_dt,
            end=end,
            f_max=cfg.f_max_away,
            T_max=cfg.T_max_away,
            direction=alignment_vector,
            contact=False,
            goal_distance_tol=0.02,
        )
        if state[0] == -1:
            print("failed to move away on the alignment vector")
        else:
            print("success in moving away on the alignment vector")
        wait_until_stable_joint_velocities(robot)

        ### ReSkin sensor death check
        previous_reskin = current_reskin  # reset reskin to check if sensor is corrupted
        if sensor_process.is_alive():
            print("sensor is alive")
        else:
            print("sensor dead")  # early logging checkpoint
            with open(
                f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_tactile/experiment_{cfg.experiment_number}_reskin",
                "wb",
            ) as readings:
                np.save(readings, reskin_recordings_np)
            with open(
                f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_tactile/experiment_{cfg.experiment_number}_reskin_ambient",
                "wb",
            ) as ambient_readings:
                np.save(ambient_readings, ambient_recordings)
            print("saved reskin data checkpoint")
            sys.exit()

    # finalize collection
    print("Finished N contact trials")
    with open(
        f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_tactile/experiment_{cfg.experiment_number}_reskin",
        "wb",
    ) as readings:
        np.save(readings, reskin_recordings_np)
    with open(
        f"{repo_path}/{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_tactile/experiment_{cfg.experiment_number}_reskin_ambient",
        "wb",
    ) as ambient_readings:
        np.save(ambient_readings, ambient_recordings)
    print("saved reskin data after N contact trials")
    ## final robot pose reset
    robot.move_cart_pos_abs_ptp(AWAY_POSE, eulertoquat(FIXED_ROBOT_ORN))
    print("success in moving away fully")
    ## turn off devices
    if sensor_process.is_alive:
        sensor_process.pause_streaming()
        sensor_process.join()
    sys.exit()
