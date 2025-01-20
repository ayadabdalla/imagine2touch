# standard library imports
import os
import cv2
import hydra
from matplotlib import patches
from omegaconf import OmegaConf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# repo modules
from src.imagine2touch.utils.model_utils import preprocess_object_data


def is_image_informative(cfg, image, n_pixels=100):
    numpy_image = np.array(image)
    # Calculate the mean pixel value across all channels
    mean_pixel_value = np.mean(numpy_image)
    return mean_pixel_value > (
        n_pixels / (cfg.image_size[0] * cfg.image_size[1])
    ) and mean_pixel_value < (
        ((cfg.image_size[0] * cfg.image_size[1]) - n_pixels)
        / (cfg.image_size[0] * cfg.image_size[1])
    )


def log_mask_filter(cfg, images, type="training"):
    added_indeces = []
    for i, image in enumerate(images):
        if is_image_informative(cfg, image):
            added_indeces.append(i)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)

    added_indeces = np.sort(added_indeces)
    with open(
        f"{cfg.experiment_dir}/{type}_masks_filter.npy", "wb"
    ) as contrast_indeces_file:
        np.save(contrast_indeces_file, added_indeces)
    print(added_indeces.shape)


if __name__ == "__main__":
    ## configurations from trainae.yaml to use trainae preprocess
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    OmegaConf.register_new_resolver(
        "image_factor", lambda x, y: 1 if y else 3 if x == "rgb" else 1
    )
    OmegaConf.register_new_resolver("rgb_gray_factor", lambda x: 1 if x else 3)
    hydra.initialize("../models/conf", version_base=None)
    cfg = hydra.compose("trainae.yaml")
    cfg.image_size = [int(cfg.image_size), int(cfg.image_size)]
    (
        scaler_images,
        scaler_rgb,
        scaler_reskin,
        tactile_targets,
        target_images,
        tactile_input,
        rgb_images,
        _,
        mean_images,
        std_images,
        mean_reskin,
        std_reskin,
        target_masks,
    ) = preprocess_object_data(cfg, cfg.objects_names)

    target_masks = (
        target_masks.cpu().numpy().reshape(-1, cfg.image_size[0], cfg.image_size[1])
    )

    # TODO
    # change the type depending on the dataset you want to use in trainae.yaml
    # The dataset is defined by cfg.object_names in train.yaml file
    # comment the shuffling in the preprocess_object_data function to generate the masks_filters
    log_mask_filter(cfg, target_masks, type="training")
