from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
import os
import hydra
from omegaconf import OmegaConf
from src.imagine2touch.reskin_calibration import dataset
import seaborn as sns
from numpy.ma import masked_array


def normalize_image(image):
    # Convert the image to float32 data type
    image = image.astype(np.float32)

    # Normalize the image by subtracting the minimum value and dividing by the range
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)

    # Scale the pixel values to the range of 0-255 (for uint8 images)
    normalized_image = (normalized_image * 255).astype(np.uint8)

    return normalized_image


def visualize_reskin_image(tactile_input, get_data=False):
    reskin_images = []
    # normalize per magnetometer if not yet normalized
    if tactile_input.shape[1] == 15:
        centers = np.expand_dims(np.expand_dims(tactile_input[:, 0:3], 1), 2)
        # centers = np.linalg.norm(centers, 2, axis=3)
        centers = (
            0 * centers[:, :, :, 0] + 0 * centers[:, :, :, 1] + 1 * centers[:, :, :, 2]
        )
        tops = np.expand_dims(np.expand_dims(tactile_input[:, 3:6], 1), 2)
        # tops = np.linalg.norm(tops, axis=3)
        tops = 0 * tops[:, :, :, 0] + 0 * tops[:, :, :, 1] + 1 * tops[:, :, :, 2]
        rights = np.expand_dims(np.expand_dims(tactile_input[:, 6:9], 1), 2)
        # rights = np.linalg.norm(rights, axis=3)
        rights = (
            0 * rights[:, :, :, 0] + 0 * rights[:, :, :, 1] + 1 * rights[:, :, :, 2]
        )
        bottoms = np.expand_dims(np.expand_dims(tactile_input[:, 9:12], 1), 2)
        # bottoms = np.linalg.norm(bottoms, axis=3)
        bottoms = (
            0 * bottoms[:, :, :, 0] + 0 * bottoms[:, :, :, 1] + 1 * bottoms[:, :, :, 2]
        )
        lefts = np.expand_dims(np.expand_dims(tactile_input[:, 12:15], 1), 2)
        lefts = 0 * lefts[:, :, :, 0] + 0 * lefts[:, :, :, 1] + 1 * lefts[:, :, :, 2]
        # lefts = np.linalg.norm(lefts, axis=3)
        # normalize per tactile image
        i = 0
        for center, top, left, right, bottom in zip(
            centers, tops, lefts, rights, bottoms
        ):
            # get maximum value of center, top, left, right, bottom
            maximum = np.max([center, top, left, right, bottom])
            if maximum == 0:
                maximum = 0.0001
            centers[i] = center / maximum
            tops[i] = top / maximum
            lefts[i] = left / maximum
            rights[i] = right / maximum
            bottoms[i] = bottom / maximum
            i += 1
    else:
        centers = np.reshape(tactile_input[:, 0], (-1, 1))
        tops = np.reshape(tactile_input[:, 1], (-1, 1))
        rights = np.reshape(tactile_input[:, 2], (-1, 1))
        bottoms = np.reshape(tactile_input[:, 3], (-1, 1))
        lefts = np.reshape(tactile_input[:, 4], (-1, 1))

    # create reskin image
    zeros = np.zeros((1, 1))
    for center, top, right, bottom, left in zip(centers, tops, rights, bottoms, lefts):
        center = np.reshape(center, (1, 1))
        top = np.reshape(top, (1, 1))
        right = np.reshape(right, (1, 1))
        bottom = np.reshape(bottom, (1, 1))
        left = np.reshape(left, (1, 1))
        top_row = np.hstack((zeros, top))
        top_row = np.hstack((top_row, zeros))
        middle_row = np.hstack((left, center))
        middle_row = np.hstack((middle_row, right))
        bottom_row = np.hstack((zeros, bottom))
        bottom_row = np.hstack((bottom_row, zeros))
        current_image = np.vstack((top_row, middle_row, bottom_row))
        reskin_images.append(current_image)

    # retrieve data or data and images
    if not get_data:
        return np.array(reskin_images)
    else:
        five_d_data = np.squeeze(
            np.stack((centers, tops, rights, bottoms, lefts), axis=3)
        )
        return np.array(reskin_images), five_d_data


def split_rgb(reskin_images, display=True):
    # Load the RGB image

    for image in reskin_images:
        # Split the image into separate channels
        image = normalize_image(image)
        b, g, r = cv2.split(image)
        # Create an empty canvas to stack the channels
        height, width = image.shape[:2]
        canvas = np.zeros((height, width * 3), dtype=np.uint8)

        # Place the grayscale channels onto the canvas
        canvas[:, :width] = b
        canvas[:, width : 2 * width] = g
        canvas[:, 2 * width : 3 * width] = r

        # Display the stacked grayscale channels
        if display:
            cv2.imshow("Stacked Channels", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            return canvas


def plot_combined_heatmap(
    images_array,
    figsize=(10, 6),
    cmap="magma",
    rotate=False,
    draw=False,
    set_bad=False,
    title="Reskin",
    display=False,
    per_image=False,
):
    """
    Plots a combined heatmap over grayscale versions of RGB images.

    Parameters:
        images_array (ndarray): A 4-dimensional array of images with shape (N, height, width, 1).
        figsize (tuple): Figure size. Default is (10, 6).
        cmap (str): Colormap for the heatmap. Default is 'viridis'.
        rotate (bool): Whether to rotate the heatmap 90 degrees counterclockwise. Default is False.
        show (bool): Whether to display the plot. Default is True.
        per_image (bool): Whether to plot a heatmap per image or calculate a combined heatmap. Default is False.
    """

    # Calculate the combined heatmap from the grayscale images
    if set_bad:
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color="white", alpha=0.2)
        mask_indices_list = [(0, 0), (0, 2), (2, 0), (2, 2)]
        if per_image:
            images_array[mask_indices_list[0][0], mask_indices_list[0][1]] = np.nan
            images_array[mask_indices_list[1][0], mask_indices_list[1][1]] = np.nan
            images_array[mask_indices_list[2][0], mask_indices_list[2][1]] = np.nan
            images_array[mask_indices_list[3][0], mask_indices_list[3][1]] = np.nan
            combined_heatmap = images_array
        else:
            images_array[:, mask_indices_list[0][0], mask_indices_list[0][1]] = np.nan
            images_array[:, mask_indices_list[1][0], mask_indices_list[1][1]] = np.nan
            images_array[:, mask_indices_list[2][0], mask_indices_list[2][1]] = np.nan
            images_array[:, mask_indices_list[3][0], mask_indices_list[3][1]] = np.nan
            combined_heatmap = np.nanmean(images_array, axis=0)
    else:
        if per_image:
            combined_heatmap = images_array
        else:
            combined_heatmap = np.mean(images_array, axis=0)
    if rotate:
        combined_heatmap = np.rot90(combined_heatmap, k=1)
    if draw:
        plt.figure(figsize=figsize)
        plt.imshow(combined_heatmap, cmap=cmap)
        plt.colorbar()
        plt.title(f"{title}")
        plt.axis("off")  # Turn off axis for cleaner display
        plt.tight_layout()
    if display:
        plt.show()
    return combined_heatmap


if __name__ == "__main__":
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("dataset.yaml")

    path_reskin = [
        cfg.data_path + "/" + cfg.object_name + "/" + cfg.object_name + "_tactile"
    ]
    (
        tactile_input,
        tactile_targets,
        mean_reskin,
        std_reskin,
    ) = dataset.prepare_reskin_data(
        path_reskin, cfg.binary, raw=False, mean=0, std=1
    )  # reskin tactile data
    tactile_input = visualize_reskin_image(tactile_input)
    for i, tactile in enumerate(tactile_input):
        plot_combined_heatmap(
            tactile_input[i],
            draw=True,
            display=True,
            title="Reskin",
            per_image=True,  # not combined
            set_bad=True,
            cmap="copper",
        )
        print(i)
    # split_rgb(visualize_reskin_image(tactile_input))
