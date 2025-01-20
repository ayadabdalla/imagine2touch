if False:
    from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
    from robot_io.utils.utils import pos_orn_to_matrix
    from robot_io.cams.realsense.realsense import Realsense
    import rospy


# repo modules
import argparse
from scripts.collect_data import init_devices
from src.imagine2touch.utils.task_utils import (
    mask_from_depth_mesh,
    normalize_p,
    update_po,
)
from src.imagine2touch.utils.utils import (
    load_normals,
    safety_one,
    safety_two,
    safety_three,
    eulertoquat,
    FIXED_ROBOT_ORN,
    load_pcds,
    HOME_POSE,
    threed_vector_to_homog,
    homog_vector_to_3d,
    WORLD_IN_ROBOT,
    move_to_pt_with_v_safe,
    euler_from_vector,
    AWAY_POSE,
    filter_reskin,
    load_points_indeces,
    ROBOT_IN_WORLD,
    WCAMERA_IN_TCP,
    inverse_transform,
)
from src.imagine2touch.task.reconstruct_pcd import get_crops
from src.imagine2touch.reskin_sensor.reskin_sensor.sensor_proc import ReSkinProcess

# standard libraries
import time
import numpy as np
import signal
import os
from pickle import load
import torch
import open3d as o3d
import natsort
import hydra
from omegaconf import OmegaConf
import cv2
import matplotlib.pyplot as plt


def meta_script_init():
    # script variables and counters/flags initializations
    C = 0  # contacts counter
    p_o_graph = []
    votes_graph = []
    touches_distances_cum = np.zeros(len(objects_names))
    vote = np.zeros(len(objects_names))  # pmf
    touches_distances_log = []
    p_o_update = np.zeros(len(objects_names))
    p_o = np.ones(len(objects_names))  # pdf
    p_o_probability = np.zeros(len(objects_names))  # [0,1]
    condition_1 = False
    condition_2 = False
    condition_3 = False
    condition_4 = False
    unreachable_target_point = False
    ambient_recording = []
    reskin_recording = []
    pcd_trees = []
    pcd_safe_trees = []
    pcd_safe_array = []
    return (
        C,
        p_o_graph,
        votes_graph,
        touches_distances_cum,
        vote,
        touches_distances_log,
        p_o_update,
        p_o,
        p_o_probability,
        condition_1,
        condition_2,
        condition_3,
        condition_4,
        unreachable_target_point,
        ambient_recording,
        reskin_recording,
        pcd_trees,
        pcd_safe_trees,
        pcd_safe_array,
    )


if __name__ == "__main__":
    # script configurations
    hydra.initialize("../src/imagine2touch/task/cfg", version_base=None)
    cfg_model = hydra.compose("model.yaml")
    cfg = hydra.compose("online.yaml")
    cfg.image_size = [int(cfg.image_size), int(cfg.image_size)]
    objects_names = cfg.objects_names.split(",")
    objects_names = natsort.natsorted(objects_names)
    robot_flange_in_tcp = [float(num) for num in cfg.robot_flange_in_tcp.split(",")]

    parser = argparse.ArgumentParser()
    # add parser argument for model.encode_image
    args = parser.parse_args()

    (
        C,
        p_o_graph,
        votes_graph,
        touches_distances_cum,
        vote,
        touches_distances_log,
        p_o_update,
        p_o,
        p_o_probability,
        condition_1,
        condition_2,
        condition_3,
        condition_4,
        unreachable_target_point,
        ambient_recording,
        reskin_recording,
        pcd_trees,
        pcd_safe_trees,
        pcd_safe_array,
    ) = meta_script_init()
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

    # process task pcds
    ## loading pcds
    print("loading down sampled pcds")
    min_x, max_x, min_y, max_y, pcd_array = load_pcds(cfg, down_sampled=True)
    print("loading safety checked pcds")
    all_objects_safe_points_indeces_array = load_points_indeces(cfg)
    if all_objects_safe_points_indeces_array is None:
        pass
    else:
        print("loading surface normals of down sampled pcds")
        all_objects_normals_array = load_normals(cfg)
    ## create new safe pcds if needed
    if cfg.new_safe_points:
        print("creating new safe points")
        all_objects_safe_points_indeces_array = []
        all_objects_normals_array = []
        min_x, max_x, min_y, max_y, pcd_array = load_pcds(cfg, down_sampled=False)
        for i, pcd in enumerate(pcd_array):
            ### downsample pcd and estimate normals
            print(f"downsampling pcd {objects_names[i]}")
            pcd_array[i] = pcd.voxel_down_sample(voxel_size=0.0005)
            pcd = pcd_array[i]
            o3d.io.write_point_cloud(
                f"{cfg.data_path}/{objects_names[i]}/{objects_names[i]}_down.pcd", pcd
            )
            print(f"saved downsampled pcd {objects_names[i]}")
            safe_points_indeces = []
            original_points = np.asarray(pcd.points)
            pcd.estimate_normals()
            original_normals = np.asarray(pcd.normals)
            print("estimated normals")
            all_objects_normals_array.append(original_normals)
            print(f"creating safe pcd for {objects_names[i]}")
            j = 0
            ### create safe pcds
            for point, normal in zip(original_points, original_normals):
                ####get point in robot
                point_homog = threed_vector_to_homog(point)
                point_homog_in_robot = WORLD_IN_ROBOT.dot(point_homog)
                point_in_robot = homog_vector_to_3d(point_homog_in_robot)
                ####get goal orientation from normal
                negative_normal = -1 * normal
                if np.linalg.norm(negative_normal, 2) == 0:
                    continue
                alignment_vector = negative_normal / np.linalg.norm(negative_normal, 2)
                goal_orientation = euler_from_vector(0, alignment_vector)
                ####safety checks
                condition_1 = safety_one(normal, cfg.robot_tool_cosine_tolerance)
                not_safe_two, safe_two, condition_2 = safety_two(
                    pcd,
                    point,
                    alignment_vector,
                    cfg.reskin_side_length,
                    cfg.convexity_tolerance,
                )
                condition_3 = safety_three(
                    point_in_robot[:3],
                    eulertoquat(goal_orientation),
                    robot_flange_in_tcp,
                    cfg.robot_flange_z_tolerance,
                )
                if condition_1 and condition_2 and condition_3:
                    safe_points_indeces.append(j)
                j += 1
            ### save safe pcds
            all_objects_safe_points_indeces_array.append(
                np.asarray(safe_points_indeces, dtype=object)
            )
            with open(
                f"{cfg.data_path}/{objects_names[i]}/{objects_names[i]}_combined_safe_points.npy",
                "wb",
            ) as safe_indeces_file:
                np.save(safe_indeces_file, safe_points_indeces)
            with open(
                f"{cfg.data_path}/{objects_names[i]}/{objects_names[i]}_original_normals.npy",
                "wb",
            ) as normals_file:
                np.save(normals_file, original_normals)
            print("saved safe points indeces and normals \n")
        print("reloading safety checked points")
        ### reload safety checked pcds
        all_objects_safe_points_indeces_array = load_points_indeces(cfg)
        all_objects_safe_points_indeces_array = np.asarray(
            all_objects_safe_points_indeces_array, dtype=object
        )
    ## create pcd trees
    for i, pcd in enumerate(pcd_array):
        pcd_trees.append(o3d.geometry.KDTreeFlann(pcd))
    pcd_trees = np.asarray(pcd_trees)
    for i, pcd in enumerate(pcd_array):
        pcd_safe_array.append(
            pcd.select_by_index(all_objects_safe_points_indeces_array[i])
        )
    pcd_safe_trees = np.asarray(
        [o3d.geometry.KDTreeFlann(pcd_safe_array[i]) for i in range(len(pcd_array))]
    )
    # start task
    while C < cfg.n_contacts:
        ## initialize contact reading
        print(f"taking ambient reading, for contact number {C}")
        ambient_recording = np.array(
            sensor_process.get_data(num_samples=10), dtype=object
        )
        ambient_recording = filter_reskin(ambient_recording, multiple_samples=True)

        ## sample an object pcd based on current weights
        if C == 0:
            sampled_pcds_indeces = np.random.choice(
                np.arange(0, pcd_array.shape[0]),
                size=1,
                p=np.ones(p_o.shape[0]) * (1 / p_o.shape[0]),
                replace=False,
            )
            o_hat = sampled_pcds_indeces[0]
        else:
            o_hat = np.argmax(p_o)
        ### process sampled object pcd
        o_hat_pcd = pcd_array[o_hat]
        o_hat_points = np.asarray(o_hat_pcd.points)
        safe_o_hat_points_indeces = all_objects_safe_points_indeces_array[o_hat]
        safe_o_hat_points = np.squeeze(o_hat_points[safe_o_hat_points_indeces])
        o_hat_normals = all_objects_normals_array[o_hat]
        safe_o_hat_normals = np.squeeze(o_hat_normals[safe_o_hat_points_indeces])
        if (
            unreachable_target_point and C > 0
        ):  # delete hardware unreachable target points to avoid infinite loop, for the first contact restart
            safe_o_hat_points = np.delete(safe_o_hat_points, target_point_index, axis=0)

        ## choose next contact point from current object that maximizes a metric compared to all other objects
        distances_to_non_targets = []
        non_target_indices = []
        non_target_distances_non_agg = []
        non_target_normals = []
        for point, normal in zip(safe_o_hat_points, safe_o_hat_normals):
            indeces_of_non_target_objects = []
            distances_of_non_target_objects_indiv = []
            non_target_normals_indiv = []
            point_homog = threed_vector_to_homog(point)
            ### get goal orientation in o_hat from normal
            negative_normal = -1 * normal
            alignment_unit_vector = negative_normal / np.linalg.norm(negative_normal, 2)
            ### compute the metric for each non-target object
            sum_distance_to_non_target_objects = 0
            undefined_flag = False
            for i in range(len(pcd_safe_array)):
                if i == o_hat:
                    continue
                else:
                    try:
                        k_neighbours = 20
                        nearest_point_info = pcd_safe_trees[i].search_knn_vector_3d(
                            point, k_neighbours
                        )
                    except RuntimeError as e:
                        print("no nearest points, run time error\n")
                        continue
                    nearest_point_index = nearest_point_info[1][0]
                    #### store nearest point index in object i in an array of shape (safe_o_hat_points.shape[0],non o_hat objects length)
                    indeces_of_non_target_objects.append(nearest_point_index)
                    #### get normal to next contact point in object i
                    normal_in_object_i = all_objects_normals_array[i][
                        all_objects_safe_points_indeces_array[i]
                    ][nearest_point_index]
                    normal_in_object_i_unit = normal_in_object_i / np.linalg.norm(
                        normal_in_object_i, 2
                    )
                    non_target_normals_indiv.append(normal_in_object_i_unit)
                    #### get alignment vector from normal in object i
                    negative_normal_in_object_i = -1 * normal_in_object_i
                    alignment_unit_vector_object_i = (
                        negative_normal_in_object_i
                        / np.linalg.norm(negative_normal_in_object_i, 2)
                    )
                    #### metric computation
                    if (
                        np.dot(alignment_unit_vector_object_i, alignment_unit_vector)
                        > 0.2
                        and np.dot(
                            alignment_unit_vector_object_i, alignment_unit_vector
                        )
                        < 1
                    ):
                        next_contact_metric = -np.dot(
                            alignment_unit_vector_object_i, alignment_unit_vector
                        )
                        if not undefined_flag:
                            sum_distance_to_non_target_objects += next_contact_metric
                    else:
                        next_contact_metric = -np.dot(
                            alignment_unit_vector_object_i, alignment_unit_vector
                        )
                        sum_distance_to_non_target_objects = 0
                        undefined_flag = True
                    distances_of_non_target_objects_indiv.append(next_contact_metric)
            ### store metric for each point in o_hat and its associated indeces of non-target objects
            distances_to_non_targets.append(sum_distance_to_non_target_objects)
            non_target_indices.append(indeces_of_non_target_objects)
            non_target_distances_non_agg.append(distances_of_non_target_objects_indiv)
            non_target_normals.append(non_target_normals_indiv)
        distances_to_non_targets = np.asarray(distances_to_non_targets)
        non_target_indices = np.asarray(non_target_indices)
        ### index of next contact point from o_hat
        target_point_maximums = np.where(
            (distances_to_non_targets > -1.3) & (distances_to_non_targets < 0)
        )[0]
        all_safe_o_hat_indices = np.arange(target_point_maximums.shape[0])
        o_hat_points_probabilities = np.ones(
            all_safe_o_hat_indices.shape
        ) * normalize_p(distances_to_non_targets[target_point_maximums])
        sampled_points_indices = np.random.choice(
            np.arange(target_point_maximums.shape[0]),
            1,
            p=o_hat_points_probabilities,
            replace=False,
        )
        target_point_maximum = target_point_maximums[sampled_points_indices[0]]
        target_point_maximum_value = distances_to_non_targets[target_point_maximum]

        ### sample the next contact point from the current object
        #### sample the next contact point from the current object with a bit of randomness
        target_point_index = target_point_maximum
        #### get next contact point that maximizes the normal metric
        sampled_point = safe_o_hat_points[target_point_index]
        sampled_point_homog = threed_vector_to_homog(sampled_point)
        sampled_point_homog_in_robot = WORLD_IN_ROBOT.dot(sampled_point_homog)
        sampled_point_in_robot = homog_vector_to_3d(sampled_point_homog_in_robot)
        #### get normal to next contact point that maximizes the normal metric
        sampled_normal = safe_o_hat_normals[target_point_index]
        sampled_normal = sampled_normal / np.linalg.norm(sampled_normal, 2)
        #### get goal orientation from normal
        negative_normal = -1 * sampled_normal
        alignment_vector = negative_normal / np.linalg.norm(negative_normal, 2)
        goal_orientation = euler_from_vector(0, alignment_vector)
        #### move along the alignment vector
        task_clearance_target_contact_to_tcp = cfg.normal_looking_distance
        sampled_point_in_robot_prepare = (
            sampled_point_in_robot
            - task_clearance_target_contact_to_tcp * alignment_vector
        )

        ##Move Robot##
        # stopping signal handler
        end = False

        def cb_int(*cfg):
            global end
            end = True

        signal.signal(signal.SIGINT, cb_int)

        # restore a good starting position
        current_pos = robot.get_state()["tcp_pos"]
        if C > 0:
            if current_pos[2] > 0.2 or current_pos[2] < 0.1:
                print("moving away to restore a good position")
                robot.move_cart_pos_abs_ptp(HOME_POSE, eulertoquat(FIXED_ROBOT_ORN))

        # target positions
        prepare_contact = sampled_point_in_robot_prepare
        final_planned_contact = sampled_point_in_robot

        # prepare position for contact
        try:
            robot.move_cart_pos_abs_ptp(prepare_contact, eulertoquat(goal_orientation))
            print("success in moving to prepare_contact")
            unreachable_target_point = False
        except rospy.service.ServiceException as e:
            unreachable_target_point = True
            print("failed to prepare for contact\n")
            print(
                np.linalg.norm(robot.get_state()["tcp_pos"], 2)
                - np.linalg.norm(prepare_contact, 2)
            )
            if (
                np.linalg.norm(robot.get_state()["tcp_pos"], 2)
                - np.linalg.norm(prepare_contact, 2)
                < cfg.pos_tolerance
            ):
                pass
            else:
                continue
        time.sleep(0.1)

        # make contact
        normal_aligned_rotation = robot.get_state()["tcp_orn"]
        contact, final_state, reskin_norm_diff, ambient = move_to_pt_with_v_safe(
            robot,
            goal_point=final_planned_contact,
            goal_rot=normal_aligned_rotation,
            p_i=0.002,
            dt=0.1,
            end=end,
            f_max=7,
            direction=alignment_vector,
            sensor_process=sensor_process,
            contact=True,
            reskin_threshold=25,
            goal_distance_tol=0.005,
        )
        print(f"contact returned with {contact}")

        # take reskin recording
        reskin_recording = np.array(
            sensor_process.get_data(num_samples=10), dtype=object
        )
        time.sleep(1)

        # move away from contact
        normal_aligned_rotation = robot.get_state()["tcp_orn"]
        away, _, _, _ = move_to_pt_with_v_safe(
            robot,
            goal_point=prepare_contact,
            goal_rot=normal_aligned_rotation,
            p_i=0.005,
            dt=0.1,
            end=end,
            f_max=20,
            direction=alignment_vector,
            goal_distance_tol=0.02,
        )
        if away == 3 or away == 2:
            try:
                robot.move_cart_pos_abs_ptp(AWAY_POSE, eulertoquat(FIXED_ROBOT_ORN))
            except rospy.service.ServiceException as e:
                print("couldnot move away forcefully or gracefully")
            print("moved to away pose forcefully")
        elif away == -1:
            print("slipped but moved away")
        else:
            print("moved to away pose gracefully")
        # take ambient every contact when possible
        if ambient is not None:
            ambient_recording = ambient
        else:
            print("failed to get ambient recording")
            continue
        if not (contact == 2 or contact == 3 or contact == 4):
            print("failed contact")
            continue
        else:
            print("success in contact")
            C = C + 1

        ### get images from prior pcds
        ## get camera world transforms (orient the wrist camera virtually with the alignment vector)
        e_T_tcp_in_robot = pos_orn_to_matrix(
            prepare_contact, eulertoquat(goal_orientation)
        )
        # calculate the tcp offset to get the wrist camera in its place (x,y)
        OFFSET_TCP = np.array(
            [-WCAMERA_IN_TCP[:, 3][0], -WCAMERA_IN_TCP[:, 3][1], 0, 1]
        )
        cam_looking_at_patch_tcp_in_robot_pos = e_T_tcp_in_robot.dot(
            OFFSET_TCP
        )  # tcp to robot transform after offsetting the tcp
        cam_looking_at_patch_tcp_in_robot = pos_orn_to_matrix(
            cam_looking_at_patch_tcp_in_robot_pos[:3], eulertoquat(goal_orientation)
        )
        W_CAM_IN_WORLD = ROBOT_IN_WORLD.dot(
            cam_looking_at_patch_tcp_in_robot.dot(WCAMERA_IN_TCP)
        )
        WORLD_IN_W_CAM = inverse_transform(W_CAM_IN_WORLD)
        contacted_point_in_robot = final_state["tcp_pos"]
        final_state_robot_in_tcp = inverse_transform(
            pos_orn_to_matrix(final_state["tcp_pos"], final_state["tcp_orn"])
        )
        contacted_point_in_robot_homog = threed_vector_to_homog(
            contacted_point_in_robot
        )
        masks_from_pcd_numpy = []
        masks_from_pcds = []
        masks_from_pcds_numpy = []
        depth_images_from_pcds = []
        contacted_point_in_world = ROBOT_IN_WORLD.dot(
            threed_vector_to_homog(contacted_point_in_robot)
        )[:3]
        for i in range(len(pcd_array)):
            masks_from_pcd = []
            depth_images_from_pcd = []
            point_homog = threed_vector_to_homog(contacted_point_in_world)
            x_1, y_1, x_2, y_2 = get_crops(
                W_CAM_IN_WORLD, WCAMERA_IN_TCP, point_homog[:3]
            )
            ## imagine object
            # create scene
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=640, height=360)
            vis.add_geometry(pcd_array[i])
            camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
            # set camera
            camera.extrinsic = WORLD_IN_W_CAM
            fx = 322.37493896484375  # Focal length in x-direction
            fy = 322.0753479003906  # Focal length in y-direction
            camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=640, height=360, fx=fx, fy=fy, cx=640 / 2 - 0.5, cy=360 / 2 - 0.5
            )
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera)
            # capture depth and rgb images
            vis.poll_events()
            vis.update_renderer()
            # vis.run()
            depth = vis.capture_depth_float_buffer()
            depth = np.asarray(depth)
            rgb = vis.capture_screen_float_buffer(do_render=True)
            rgb = np.asarray(rgb)
            dep_im = (depth * 1000).astype(np.uint8)
            # crop depth and rgb images
            dep_im = dep_im[y_1:y_2, x_1:x_2]
            rgb = rgb[y_1:y_2, x_1:x_2]
            dep_im = cv2.resize(dep_im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            # get mask
            target_masks, target_images, target_mask_view = mask_from_depth_mesh(
                cfg, None, dep_im, cfg.image_size, view=True
            )
            target_mask = target_masks[0]
            target_image = target_images[0]
            target_mask = target_mask.astype(np.uint8)
            mask_tensor = (
                torch.FloatTensor(target_mask)
                .view(1, cfg.image_size[0], cfg.image_size[1])
                .to(device)
            )
            target_mask_view = target_mask_view[0].astype(np.uint8)
            target_mask_view = target_mask_view * 255

            ## visualize images
            # cv2.imshow("depth_unequalized", dep_im)
            # cv2.waitKey(0)
            # dep_im = cv2.equalizeHist(dep_im)
            # cv2.imshow("target_mask", target_mask_view)
            # # cv2.waitKey(0)
            # cv2.imshow("depth", dep_im)
            # # cv2.waitKey(0)
            # cv2.imshow("rgb", rgb[:, :, ::-1])
            # cv2.waitKey(0)

            ## store images
            # extra packaging in case we add a number of images per pcd
            masks_from_pcd.append(mask_tensor)
            masks_from_pcd_numpy.append(target_mask)
            depth_images_from_pcd.append(target_image)
            # add data per pcd
            masks_from_pcds.append(masks_from_pcd)
            masks_from_pcds_numpy.append(masks_from_pcd_numpy)
            depth_images_from_pcds.append(depth_images_from_pcd)
            vis.destroy_window()

        # distribution update
        #############################################################################################################################
        # calculate updates
        touches_distances = []
        if cfg.mode == "baseline":
            for i in range(pcd_array.shape[0]):
                try:
                    k_neighbours = 1
                    nearest_point_info = pcd_trees[i].search_knn_vector_3d(
                        ROBOT_IN_WORLD.dot(
                            threed_vector_to_homog(contacted_point_in_robot)
                        )[:3],
                        k_neighbours,
                    )
                except RuntimeError as e:
                    print("no nearest points, run time error\n")
                    continue
                # negate distance error to be later exponentiated, keep datatype as DoubleVector
                nearest_point_distance = o3d.utility.DoubleVector(
                    [elem * -1 for elem in nearest_point_info[2]]
                )
                touches_distances = np.append(touches_distances, nearest_point_distance)
        else:
            # preprocess reskin reading
            reskin_recording = filter_reskin(reskin_recording, multiple_samples=True)
            reading = np.zeros(reskin_recording.shape[0])
            for dim in range(reading.shape[0]):
                if reading[dim] >= 0:
                    reading[dim] = reskin_recording[dim] - ambient_recording[dim]
                else:
                    reading[dim] = -(reskin_recording[dim] - ambient_recording[dim])
            reading = (reading - reskin_mean) / reskin_std
            # reading = reskin_quantile.transform(np.reshape(reading, (1, -1)))
            reading = torch.FloatTensor(
                np.reshape(np.asarray(reading, dtype=float), (1, -1))
            ).to(device)
            for masks in masks_from_pcds:
                sum_errors = 0
                for mask_i in masks:
                    with torch.no_grad():
                        mask_i = torch.unsqueeze(mask_i, 0)
                        mask_i = mask_i.permute(0, 1, 2, 3)
                        tactile, dep_image, mask_image, _, _ = model(reading, mask_i)
                        dep_image = dep_image.cpu().numpy()
                        mask_image = mask_image.cpu().numpy()
                        print(f"predicted error {torch.norm(tactile - reading)}")
                        sum_errors += torch.norm(tactile - reading)
                touches_distances.append(-(sum_errors / len(masks)).cpu().numpy())
        touches_distances_log.append(touches_distances)

        ## apply updates
        touches_distances = np.exp(
            touches_distances
        )  # turn to positive probability but maintain relations
        touches_distances_cum = np.add(touches_distances_cum, touches_distances)
        p_o = update_po(p_o, touches_distances)
        # voting
        touches_distances_max = np.argmax(touches_distances)
        touches_distances_votes = np.zeros(len(objects_names))
        touches_distances_votes[touches_distances_max] = 1 / cfg.n_contacts
        vote = np.add(vote, touches_distances_votes)
        print(p_o)
        print(vote)
        p_o_graph.append(p_o)
        votes_graph.append(vote)

    ## visualize the winner
    print(f"final probability {normalize_p(p_o)}")
    print(f"final voting {vote}")
    o_hat = np.argmax(p_o)
    o3d.visualization.draw_geometries([pcd_array[o_hat]])
    for i, column in enumerate(np.asarray(votes_graph).T):
        plt.plot(column, label=objects_names[i])
        plt.title(f"{cfg.true_object} {cfg.mode} experiment")
        plt.legend()
        plt.show()

    ## save results
    with open(
        f"{cfg.task_path}/results/{cfg.mode}_distribution_{cfg.experiment_name}.npy",
        "wb",
    ) as file:
        np.save(file, p_o_graph)
    with open(
        f"{cfg.task_path}/results/{cfg.mode}_distances_{cfg.experiment_name}.npy", "wb"
    ) as file:
        np.save(file, touches_distances_log)
    with open(
        f"{cfg.task_path}/results/{cfg.mode}_votes_{cfg.experiment_name}.npy", "wb"
    ) as file:
        np.save(file, votes_graph)
    with open(
        f"{cfg.task_path}/results/{cfg.mode}_cumulative_{cfg.experiment_name}.npy", "wb"
    ) as file:
        np.save(file, touches_distances_cum)
