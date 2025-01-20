# repo modules
from src.imagine2touch.utils.task_utils.utils import normalize_p, viz_PC
from src.imagine2touch.utils.utils import (
    load_normals,
    safety_one,
    safety_two,
    safety_three,
    custom_visualization_pcd,
    discretize_vector,
    eulertoquat,
    load_pcds,
    threed_vector_to_homog,
    homog_vector_to_3d,
    WORLD_IN_ROBOT,
    euler_from_vector,
    load_points_indeces,
)

# standard libraries
import numpy as np
import os

# import rospy
import open3d as o3d
import natsort
import hydra
from omegaconf import OmegaConf


def display():
    normal_set_of_sets_negative = []
    normal_set_of_sets_positive = []
    for (
        data
    ) in (
        viz_data_all_objects
    ):  ###### for each object sampled point, compute the normals on the surface contacts of all objects passing through it.
        sampled_point = data[11]
        normal_sets_negative = []
        normal_sets_positive = []
        for data in viz_data_all_objects:
            normal = data[10]
            (
                discretized_normal_array_negative,
                discretized_normal_array_positive,
            ) = discretize_vector(sampled_point, normal, 0.005, 10)
            normal_sets_negative.append(discretized_normal_array_negative)
            normal_sets_positive.append(discretized_normal_array_positive)
        normal_set_of_sets_negative.append(normal_sets_negative)
        normal_set_of_sets_positive.append(normal_sets_positive)
    for i in range(len(viz_data_all_objects)):
        viz_data_all_objects[i][2] = np.asarray(normal_set_of_sets_negative[i]).reshape(
            -1, 3
        )
        viz_data_all_objects[i][3] = np.asarray(normal_set_of_sets_positive[i]).reshape(
            -1, 3
        )
    for i, data in enumerate(viz_data_all_objects):
        pcd_viz = custom_visualization_pcd(
            data[0],
            data[1],
            data[2],
            data[3],
            safe_one=data[4],
            safe_two_indeces=data[5],
            not_safe_two_indeces=data[6],
            sampled_point_index=data[7],
            objects_names=data[8],
            many_pcds=True,
            sampled_point=data[11],
        )
        PC = [pcd_viz]
        vis = viz_PC(PC, data[11], data[10])


# script configurations
if __name__ == "__main__":
    # load configurations
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("online.yaml")
    cfg.image_size = [int(cfg.image_size), int(cfg.image_size)]
    objects_names = cfg.objects_names.split(",")
    objects_names = natsort.natsorted(objects_names)
    robot_flange_in_tcp = [float(num) for num in cfg.robot_flange_in_tcp.split(",")]

    # script variables and counters/flags initializations
    C = 0  # contacts counter
    p_o_graph = []
    votes_graph = []
    touches_distances_cum = np.zeros(len(objects_names))
    vote = np.zeros(len(objects_names))  # pmf
    touches_distances_log = []
    p_o_update = np.zeros(len(objects_names))
    p_o = np.zeros(len(objects_names))  # pdf
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

    # process task pcds
    ##loading
    print("loading down sampled pcds")
    min_x, max_x, min_y, max_y, pcd_array = load_pcds(cfg, down_sampled=True)
    print("loading safety checked pcds")
    all_objects_safe_points_indeces_array = load_points_indeces(cfg)
    print("loading normals")
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
        #### delete hardware unreachable target points to avoid infinite loop, for the first contact restart
        if unreachable_target_point and C > 0:
            safe_o_hat_points = np.delete(safe_o_hat_points, target_point_index, axis=0)

        ## choose next contact point from current object that maximizes a metric to all other objects
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
        target_point_minimum = np.where(
            distances_to_non_targets
            == np.min(distances_to_non_targets[np.nonzero(distances_to_non_targets)])
        )[0][0]
        target_point_minimum_value = distances_to_non_targets[target_point_minimum]
        target_point_maximum_test = np.where(
            distances_to_non_targets
            == np.max(distances_to_non_targets[np.nonzero(distances_to_non_targets)])
        )[0][0]
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

        ### Next contact point visualizations
        #### visualize the next contact point in all other objects with a minimized normal metric
        all_normals_min_neg = np.empty((0, 3))
        all_normals_min_pos = np.empty((0, 3))
        all_normals_max_neg = np.empty((0, 3))
        all_normals_max_pos = np.empty((0, 3))
        j = 0
        viz_data_all_objects = []
        for i in range(len(objects_names)):
            if i == o_hat:
                continue
            index = non_target_indices[target_point_minimum][j]
            sampled_point_object_i = np.squeeze(
                np.asarray(pcd_array[i].points)[
                    all_objects_safe_points_indeces_array[i]
                ]
            )[index]
            normal_in_object_i = all_objects_normals_array[i][
                all_objects_safe_points_indeces_array[i]
            ][index]
            normal_in_object_i_unit = normal_in_object_i / np.linalg.norm(
                normal_in_object_i, 2
            )
            negative_normal_in_object_i = -1 * normal_in_object_i
            alignment_unit_vector_object_i = (
                negative_normal_in_object_i
                / np.linalg.norm(negative_normal_in_object_i, 2)
            )
            not_safe_two, safe_two, condition_2 = safety_two(
                pcd_array[i],
                sampled_point_object_i,
                alignment_unit_vector_object_i,
                cfg.reskin_side_length,
                cfg.convexity_tolerance,
            )
            condition_1 = safety_one(
                normal_in_object_i_unit, cfg.robot_tool_cosine_tolerance
            )
            (
                discretized_normal_array_negative,
                discretized_normal_array_positive,
            ) = discretize_vector(
                sampled_point_object_i, normal_in_object_i_unit, 0.005, 10
            )
            all_normals_min_neg = np.append(
                all_normals_min_neg, discretized_normal_array_negative, axis=0
            )
            all_normals_min_pos = np.append(
                all_normals_min_pos, discretized_normal_array_positive, axis=0
            )
            viz_data_all_objects.append(
                [
                    pcd_array[i],
                    np.asarray(pcd_array[i].points),
                    None,
                    None,
                    condition_1,
                    safe_two,
                    not_safe_two,
                    index,
                    objects_names,
                    True,
                    normal_in_object_i,
                    sampled_point_object_i,
                    pcd_array[i],
                ]
            )
            j += 1
        #### visualize the next contact point in the sampled object with a minimized normal metric
        target_point_index = target_point_minimum  # deterministic alternative
        ##### get next contact point that minimizes the metric
        sampled_point = safe_o_hat_points[target_point_index]
        ##### get normal to next contact point that minimizes the metric
        sampled_normal = safe_o_hat_normals[target_point_index]
        sampled_normal = sampled_normal / np.linalg.norm(sampled_normal, 2)
        (
            discretized_normal_array_negative,
            discretized_normal_array_positive,
        ) = discretize_vector(sampled_point, sampled_normal, 0.005, 10)
        all_normals_min_neg = np.append(
            all_normals_min_neg, discretized_normal_array_negative, axis=0
        )
        all_normals_min_pos = np.append(
            all_normals_min_pos, discretized_normal_array_positive, axis=0
        )
        negative_normal = -1 * sampled_normal
        alignment_vector = negative_normal / np.linalg.norm(negative_normal, 2)
        o_hat_safe_pcd = o_hat_pcd.select_by_index(safe_o_hat_points_indeces)
        not_safe_two, safe_two, condition_2 = safety_two(
            o_hat_pcd,
            sampled_point,
            alignment_vector,
            cfg.reskin_side_length,
            cfg.convexity_tolerance,
        )
        condition_1 = safety_one(sampled_normal, cfg.robot_tool_cosine_tolerance)
        viz_data_all_objects.append(
            [
                pcd_array[o_hat],
                np.asarray(pcd_array[o_hat].points),
                None,
                None,
                condition_1,
                safe_two,
                not_safe_two,
                target_point_index,
                objects_names,
                True,
                sampled_normal,
                sampled_point,
                pcd_array[o_hat],
            ]
        )
        print("min next contact metric: ", target_point_minimum_value)
        display()

        viz_data_all_objects = []
        #### visualize the next contact point in all other objects with a maximized normal metric
        j = 0
        for i in range(len(objects_names)):
            if i == o_hat:
                continue
            index = non_target_indices[target_point_maximum][j]
            sampled_point_object_i = np.squeeze(
                np.asarray(pcd_array[i].points)[
                    all_objects_safe_points_indeces_array[i]
                ]
            )[index]
            normal_in_object_i = all_objects_normals_array[i][
                all_objects_safe_points_indeces_array[i]
            ][index]
            negative_normal_in_object_i = -1 * normal_in_object_i
            alignment_unit_vector_object_i = (
                negative_normal_in_object_i
                / np.linalg.norm(negative_normal_in_object_i, 2)
            )
            not_safe_two, safe_two, condition_2 = safety_two(
                pcd_array[i],
                sampled_point_object_i,
                alignment_unit_vector_object_i,
                cfg.reskin_side_length,
                cfg.convexity_tolerance,
            )
            condition_1 = safety_one(
                normal_in_object_i, cfg.robot_tool_cosine_tolerance
            )
            normal_in_object_i = normal_in_object_i / np.linalg.norm(
                normal_in_object_i, 2
            )
            viz_data_all_objects.append(
                [
                    pcd_array[i],
                    np.asarray(pcd_array[i].points),
                    None,
                    None,
                    condition_1,
                    safe_two,
                    not_safe_two,
                    index,
                    objects_names,
                    True,
                    normal_in_object_i,
                    sampled_point_object_i,
                    pcd_array[i],
                ]
            )
            j += 1

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
        ####visualization
        o_hat_safe_pcd = o_hat_pcd.select_by_index(safe_o_hat_points_indeces)
        not_safe_two, safe_two, condition_2 = safety_two(
            o_hat_pcd,
            sampled_point,
            alignment_vector,
            cfg.reskin_side_length,
            cfg.convexity_tolerance,
        )
        condition_1 = safety_one(sampled_normal, cfg.robot_tool_cosine_tolerance)
        viz_data_all_objects.append(
            [
                pcd_array[o_hat],
                np.asarray(pcd_array[o_hat].points),
                None,
                None,
                condition_1,
                safe_two,
                not_safe_two,
                target_point_index,
                objects_names,
                True,
                sampled_normal,
                sampled_point,
                pcd_array[o_hat],
            ]
        )
        print("max next contact metric: ", target_point_maximum_value)
        display()
