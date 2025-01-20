import os
import hydra
from omegaconf import OmegaConf
import numpy as np
import cv2
from src.imagine2touch.utils.utils import (
    get_target_images,
    get_depth_processed,
    get_target_masks,
)
from src.imagine2touch.reskin_calibration import dataset
from src.imagine2touch.visualizations.reskin_image_visualizer import (
    visualize_reskin_image,
)
import sys


if __name__ == "__main__":
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("dataset.yaml")

    file_dep = [
        cfg.data_path + "/" + cfg.object_name + "/" + cfg.object_name + "_images/depth"
    ]
    file_dep_processed = [
        cfg.data_path
        + "/"
        + cfg.object_name
        + "/"
        + cfg.object_name
        + "_images/depth_processed"
    ]
    file_rgb = [
        cfg.data_path + "/" + cfg.object_name + "/" + cfg.object_name + "_images/rgb"
    ]
    file_mask = [
        cfg.data_path + "/" + cfg.object_name + "/" + cfg.object_name + "_images/masks"
    ]
    file_tactile = [
        cfg.data_path + "/" + cfg.object_name + "/" + cfg.object_name + "_tactile"
    ]

    (
        tactile_input,
        tactile_targets,
        mean_reskin,
        std_reskin,
    ) = dataset.prepare_reskin_data(
        file_tactile,
        cfg.binary,
        ambient_every_reading=cfg.ambient_every_contact,
        ambient_aggregated=cfg.aggregated_ambient,
        standardize=False,
    )  # normalized differential tactile data
    morphed_tactile_input = visualize_reskin_image(tactile_input)
    rgb_images = get_target_images(file_rgb, "rgb", [cfg.image_size, cfg.image_size])
    rgb_images = np.asarray(rgb_images, dtype=np.uint8)
    target_images = get_target_images(
        file_dep, "depth", [cfg.image_size, cfg.image_size]
    )
    target_images = np.asarray(target_images, dtype=np.uint8)
    target_images_processed = get_depth_processed(
        file_dep_processed, [cfg.image_size, cfg.image_size]
    )
    target_images_processed = np.asarray(target_images_processed, dtype=np.uint8)
    target_masks = get_target_masks(file_mask, [cfg.image_size, cfg.image_size])
    target_masks = np.asarray(target_masks, dtype=np.uint8)
    if cfg.return_reskin_filter:
        return_state = np.load(
            cfg.data_path
            + "/"
            + cfg.object_name
            + "/"
            + cfg.object_name
            + "_forces"
            + "/"
            + "experiment_1_contact_returns"
        )
        indeces = np.where(return_state == 2)
        target_images = target_images[indeces]
        target_images_processed = target_images_processed[indeces]
        target_masks = target_masks[indeces]
        rgb_images = rgb_images[indeces]
        morphed_tactile_input = morphed_tactile_input[indeces]
        tactile_input = tactile_input[indeces]
    i = 0
    for depth, depth_processed, mask, rgb, tactile, tactile_raw in zip(
        target_images,
        target_images_processed,
        target_masks,
        rgb_images,
        morphed_tactile_input,
        tactile_input,
    ):
        depth = cv2.equalizeHist(depth)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = cv2.bitwise_not(depth)
        depth_rgb = np.stack([depth] * 3, axis=-1)

        depth_processed = cv2.equalizeHist(depth_processed)
        depth_processed = cv2.rotate(depth_processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_processed = cv2.bitwise_not(depth_processed)
        depth_processed_rgb = np.stack([depth_processed] * 3, axis=-1)

        mask = cv2.equalizeHist(mask)
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask_rgb = np.stack([mask] * 3, axis=-1)
        if np.sum(mask_rgb.flatten()) == 48 * 48 * 3:
            mask_rgb = mask_rgb * 255
            print("mask is all white")
        # normalize rgb image

        rgb = rgb[..., ::-1]
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

        tactile[0, 0] = 0
        tactile[0, 2] = 0
        tactile[2, 0] = 0
        tactile[2, 2] = 0
        print(tactile)
        tactile_upscaled_image = np.kron(tactile, np.ones((16, 16), dtype=np.uint8))
        tactile_rgb = np.stack([tactile_upscaled_image] * 3, axis=-1)
        # add color map of plasma
        # set the zero pixels to nan considering that it is not an integer
        tactile_rgb = np.where(tactile_rgb == 0, np.nan, tactile_rgb)
        # normalize the tactile data to a max of 255 and min of 0
        tactile_rgb = (tactile_rgb - np.nanmin(tactile_rgb)) / (
            np.nanmax(tactile_rgb) - np.nanmin(tactile_rgb)
        )
        # multiply by 255 to get the values between 0 and 255
        tactile_rgb = tactile_rgb * 255
        # convert to uint8
        tactile_rgb = tactile_rgb.astype(np.uint8)
        tactile_rgb = cv2.applyColorMap(
            tactile_rgb.astype(np.uint8), cv2.COLORMAP_PLASMA
        )
        # make all four corners of 16 pixels in a 48 * 48 image white
        tactile_rgb[0:16, 0:16] = 255
        tactile_rgb[0:16, 32:48] = 255
        tactile_rgb[32:48, 0:16] = 255
        tactile_rgb[32:48, 32:48] = 255

        width = 48
        height = 48
        plasma_colormap = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_PLASMA
        )
        plasma_bar = cv2.resize(plasma_colormap, (width, height))
        # make the plasma bar vertical
        plasma_bar = cv2.rotate(plasma_bar, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # make first 5 columns and last 38 columns white
        plasma_bar[:, 0:5] = 255
        plasma_bar[:, 25:48] = 255
        # rotate tactile_rgb
        concatenated_image = np.concatenate(
            (rgb, depth_rgb, depth_processed_rgb, mask_rgb, tactile_rgb, plasma_bar),
            axis=1,
        )

        cv2.imshow("images", concatenated_image)

        print(i)
        i += 1
        cv2.waitKey(0)
