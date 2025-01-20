# standard libraries
import copy
import sys
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import hydra
from omegaconf import OmegaConf
import os
from PIL import Image

# repo modules
from src.imagine2touch.utils.utils import WCAMERA_IN_TCP, search_folder

# relative modules
from src.imagine2touch.models.depth_correction_utils import (
    get_rgb_depth,
    apply_depth_correction,
)


# utilities
def save_gt(cfg, target_images, type):
    if type not in ["masks", "depth_processed"]:
        raise ValueError('please enter either "masks" or "depth_processed"')
    folder = f"{cfg.experiment_dir}/{cfg.object_name}/{cfg.object_name}_images/{type}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for index, target_image in enumerate(target_images):
        target_image = target_image.astype(np.uint16)
        target_image = Image.fromarray(target_image)
        target_image.save(
            f"{cfg.experiment_dir}/{cfg.object_name}/{cfg.object_name}_images/{type}/{type}_{index+1}.tif"
        )


def display(target_images, masks, rgb_images, number):
    plt.gray()
    for index in range(number):
        if target_images is not None:
            depth_processed = target_images[index]
            depth_processed = np.array(depth_processed, dtype=np.uint8).reshape(
                cfg.image_size[0], cfg.image_size[1]
            )
            rgb_image = rgb_images[index]
            rgb_image = np.array(rgb_image, dtype=np.uint8).reshape(
                cfg.image_size[0], cfg.image_size[1], 3
            )
            mask = masks[index]
            mask = np.array(mask, dtype=np.uint8).reshape(
                cfg.image_size[0], cfg.image_size[1]
            )

            ax = plt.subplot(3, number, index + 1)
            plt.imshow(depth_processed)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, number, index + 1 + number)
            plt.imshow(mask)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, number, index + 1 + 2 * number)
            plt.imshow(rgb_image)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            print("\n")
            print(np.min(depth_processed))
            print(np.max(depth_processed))
            print(np.mean(depth_processed))
            print("\n")

    plt.show()


def mask_far_pixels(cfg, cam_z_distance, images, pc=False, view=False):
    if pc:
        for i, image in enumerate(images):
            image = np.asarray(image, dtype=np.float64)

            # Apply the Laplacian filter
            laplacian = cv2.Laplacian(image, cv2.CV_64F)

            # Apply the sharpening operation using Silver's method
            sharpened = cv2.add(image, laplacian, dtype=cv2.CV_8U)
            # sharpened = np.asarray(image, dtype=np.uint8)

            # Apply Canny edge detection
            edges = cv2.Canny(sharpened, 5, 15)

            # Create a mask of the edges
            edges_mask = np.zeros_like(edges)

            # Set the edges as white in the mask
            edges_mask[edges != 0] = 255
            # cv2.imshow('edges',edges_mask)
            # cv2.waitKey(0)

            # Set the edges to 0 in the original image
            output = image.copy()
            output[edges_mask == 255] = 0
            images[i] = output
    if view:
        temp = cfg.image_size
        cfg.image_size = (images[0].shape[0], images[0].shape[1])
    images = np.array(np.reshape(images, (-1, cfg.image_size[0] * cfg.image_size[1])))
    for i, image in enumerate(images):
        if cfg.masks.use_min_depth:
            if pc:
                non_zero_pixels = image[image > 60]
                if len(non_zero_pixels) == 0:
                    cam_z_distance = -(cfg.masks.tolerance_in_mm + 1)
                    print("no pixels > 60")
                else:
                    non_zero_pixels = np.sort(non_zero_pixels)
                    unique_values, occurrences = np.unique(
                        non_zero_pixels, return_counts=True
                    )
                    flag = False
                    for value, count in zip(unique_values, occurrences):
                        if count > cfg.masks.min_occurences:
                            cam_z_distance = value
                            flag = True
                            break
                    if not flag:
                        print(f"occurences smaller than {cfg.masks.min_occurences}")
                        cam_z_distance = -(cfg.masks.tolerance_in_mm + 1)
            else:
                original_z_distance = cam_z_distance
                cam_z_distance = np.min(image)
        print(cam_z_distance, "minimum depth")
        if cam_z_distance == 0:
            cam_z_distance = original_z_distance
        for j, pixel in enumerate(image):
            pixel = np.uint16(pixel)
            cam_z_distance = np.uint16(cam_z_distance)
            compare = np.abs(pixel - cam_z_distance)
            if compare > cfg.masks.tolerance_in_mm:
                image[j] = 0
            else:
                image[j] = 1
        if cfg.masks.blob_filter:
            image = np.reshape(image, (cfg.image_size[0], cfg.image_size[1]))
            # Perform connected component analysis and compute the size of each component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                image, connectivity=cfg.masks.blob_connectivity
            )
            sizes = stats[1:, -1]

            # Get the label of the largest connected component
            max_label = 1 + np.argmax(sizes)

            # Create a new binary image with only the largest connected component
            new_img = np.zeros_like(image)
            new_img[labels == max_label] = 1
            images[i] = np.reshape(new_img, (cfg.image_size[0] * cfg.image_size[1]))
        if cfg.masks.erode:
            image = np.reshape(image, (cfg.image_size[0], cfg.image_size[1]))
            kernel = np.ones(
                (cfg.masks.erode_kernel_size, cfg.masks.erode_kernel_size), np.uint8
            )
            image = cv2.erode(image, kernel, iterations=1)
            images[i] = np.reshape(image, (cfg.image_size[0] * cfg.image_size[1]))
    images = np.reshape(images, (-1, cfg.image_size[0], cfg.image_size[0]))
    if view:
        cfg.image_size = temp
    return images


def main(cfg, cam_z_distance):
    target_images, rgb_images = get_rgb_depth(cfg, cfg.object_name)
    target_images, rgbs_quantized = apply_depth_correction(
        cfg,
        dups_threshold=cam_z_distance - cfg.masks.dups_threshold,
        target_images=target_images,
        rgb_images=rgb_images,
    )
    target_masks = mask_far_pixels(
        cfg, cam_z_distance=cam_z_distance, images=target_images
    )
    save_gt(cfg, target_images, "depth_processed")
    # display(target_images,target_masks,rgb_images,number=10)
    save_gt(cfg, target_masks, "masks")
    print(f"saved {cfg.object_name} masks")
    print(f"saved {cfg.object_name} depth processed")


if __name__ == "__main__":
    # configuration
    hydra.initialize("./cfg", version_base=None)
    cfg = hydra.compose("generate_masks.yaml")
    cfg.repository_directory = search_folder("/", "imagine2touch")
    cfg.image_size = [int(cfg.image_size), int(cfg.image_size)]
    cam_z_distance = (
        cfg.masks.tcp_z_distance + -cfg.masks.wcamera_in_tcp_z
    ) * 1000  # add camera to tcp distance on their z-axis and convert to mm
    objects_array = cfg.object_name.split(",")
    for object_name in objects_array:
        cfg.object_name = object_name
        cfg_copy = copy.deepcopy(cfg)
        t = threading.Thread(
            target=main,
            args=(
                cfg_copy,
                cam_z_distance,
            ),
        )
        t.start()
        time.sleep(2)
