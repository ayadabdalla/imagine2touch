# standard modules
import hydra
import numpy as np
import os
from omegaconf import OmegaConf

# repo modules
from src.imagine2touch.utils.utils import WORLD_IN_ROBOT, WCAMERA_IN_TCP, search_folder
from src.imagine2touch.utils.data_utils import (
    get_object_transforms_camera_in_world,
    get_spatial_info,
)

if __name__ == "__main__":
    # script meta data, configuration and constants
    start_path = "/"
    hydra.initialize("../cfg", version_base=None)
    cfg = hydra.compose("collection.yaml")
    cfg.repository_directory = search_folder(start_path, cfg.repository_directory)
    cfg.experiment_directory = f"{cfg.repository_directory}/{cfg.experiment_directory}"
    original_clearance_patch_to_tcp = 0.04
    task_clearance_patch_to_tcp = 0.08
    cam_z_to_reskin = -WCAMERA_IN_TCP[:, 3][2]
    cam_z_to_patch = original_clearance_patch_to_tcp + cam_z_to_reskin

    rotations = get_spatial_info(
        "rotation", cfg.experiment_directory, cfg.object_name, "irrelevant"
    )
    # tcp positions in robot before making the controlled contacts
    prepare_poses = get_spatial_info(
        "position", cfg.experiment_directory, cfg.object_name, "prepare_contact"
    )
    # tcp positions in robot when making the controlled contacts
    final_poses = get_spatial_info(
        "position", cfg.experiment_directory, cfg.object_name, "final_contact"
    )
    alignment_vectors = (final_poses - prepare_poses) / np.reshape(
        np.linalg.norm(final_poses - prepare_poses, 2, 1), (prepare_poses.shape[0], -1)
    )
    # when tcp is hypothetically looking at the target patch; for task data
    task_prepare_poses = (
        final_poses - original_clearance_patch_to_tcp * alignment_vectors
    )
    camera_world_transforms = get_object_transforms_camera_in_world(
        prepare_poses, rotations, WORLD_IN_ROBOT, WCAMERA_IN_TCP
    )
    np.save(
        f"{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_poses/{cfg.object_name}_camera_world_transforms",
        camera_world_transforms,
        allow_pickle=True,
    )
    np.save(
        f"{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_poses/{cfg.object_name}_target_points_in_robot",
        final_poses,
        allow_pickle=True,
    )

    # sanity check
    print(
        np.load(
            f"{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_poses/{cfg.object_name}_camera_world_transforms.npy",
            allow_pickle=True,
        ).shape
    )
    print(
        np.load(
            f"{cfg.experiment_directory}/{cfg.object_name}/{cfg.object_name}_poses/{cfg.object_name}_target_points_in_robot.npy",
            allow_pickle=True,
        ).shape
    )
