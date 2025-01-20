import sys
import numpy as np
import cv2
from src.imagine2touch.utils.utils import get_target_images, rgb2gray
from tqdm import tqdm


def get_dups(dep_im, threshold):
    DUPS_indeces_i = np.array([])
    DUPS_indeces_j = np.array([])
    DUPS_keys = np.array([])
    for i, row in enumerate(dep_im):
        for j, pixel in enumerate(row):
            if pixel < threshold:
                DUPS_indeces_i = np.append(DUPS_indeces_i, i)
                DUPS_indeces_j = np.append(DUPS_indeces_j, j)
                DUPS_keys = np.append(DUPS_keys, i + j)
    # sort dups by an adjacency measure
    DUPS_indeces_i = np.reshape(DUPS_indeces_i, (-1, 1))
    DUPS_indeces_j = np.reshape(DUPS_indeces_j, (-1, 1))
    DUPS_keys = np.reshape(DUPS_keys, (-1, 1))
    stacked = np.hstack((DUPS_indeces_i, DUPS_indeces_j, DUPS_keys))
    stacked = stacked[stacked[:, 2].argsort()]
    DUPS_indeces_i = stacked[:, 0]
    DUPS_indeces_j = stacked[:, 1]
    return DUPS_indeces_i, DUPS_indeces_j


def get_rgb_depth(cfg, object_names):
    # get data
    object_names = object_names.split(",")
    path_target = []
    path_rgb = []
    for p in object_names:
        path_target.append(
            f"{cfg.repository_directory}/{cfg.experiment_dir}/{p}/{p}_images/depth"
        )
        path_rgb.append(
            f"{cfg.repository_directory}/{cfg.experiment_dir}/{p}/{p}_images/rgb"
        )
    target_images = get_target_images(path_target, "depth", cfg.image_size)
    target_images = np.asarray(target_images, dtype=np.uint8)
    target_images_length = target_images.shape[0]
    target_images = np.reshape(
        target_images, (target_images_length, cfg.image_size[0], cfg.image_size[1])
    )
    rgb_images = get_target_images(path_rgb, "rgb", cfg.image_size)
    rgb_images = np.reshape(
        rgb_images, (target_images_length, cfg.image_size[0], cfg.image_size[1], 3)
    )
    return target_images, rgb_images


def gray_dup_helper(cfg, rgb_images):
    grays_quantized = np.zeros(rgb_images.shape[:3])
    rgbs_quantized = np.zeros(rgb_images.shape)
    for i, rgb_image in enumerate(rgb_images):
        rgb_image = cv2.edgePreservingFilter(rgb_image)
        # quantize colors
        Z = rgb_image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            cfg.masks.k_means_max_iter,
            cfg.masks.k_means_accuracy,
        )
        ret, label, center = cv2.kmeans(
            Z,
            cfg.masks.dups_k,
            None,
            criteria,
            cfg.masks.k_means_n_init,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((rgb_image.shape))
        gray_image = rgb2gray(res2)
        rgbs_quantized[i] = res2
        grays_quantized[i] = gray_image
    return np.array(grays_quantized), np.array(rgbs_quantized)


def estimate_dups(cfg, dep_im, gray_image, DUPS_indeces_i, DUPS_indeces_j, threshold):
    output_image = np.array(dep_im)
    for DUP_index_i, DUP_index_j in zip(DUPS_indeces_i, DUPS_indeces_j):
        color = gray_image[int(DUP_index_i), int(DUP_index_j)]
        # calculate value for one DUP
        # equation 1
        temp_image = np.zeros(gray_image.shape)
        for i, row in enumerate(gray_image):
            for j, pixel in enumerate(row):
                if pixel == color:
                    temp_image[i, j] = dep_im[i, j]
                else:
                    temp_image[i, j] = 0
        # equation 2
        count = 0
        sum = 0
        sum, count = grass_fire_dup_estimation(
            cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
        )

        if count == 0:
            average = np.median(np.unique(temp_image.flatten()))
        else:
            average = sum / count
        output_image[int(DUP_index_i), int(DUP_index_j)] = average
    return output_image


def grass_fire_dup_estimation(
    cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
):
    sum, count = grass_fire_north(
        cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
    )
    sum, count = grass_fire_north_east(
        cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
    )
    sum, count = grass_fire_north_west(
        cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
    )
    sum, count = grass_fire_south(
        cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
    )
    sum, count = grass_fire_south_east(
        cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
    )
    sum, count = grass_fire_south_west(
        cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
    )
    sum, count = grass_fire_east(
        cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
    )
    sum, count = grass_fire_west(
        cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
    )
    return sum, count


def grass_fire_north(cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold):
    north = 1
    while int(DUP_index_i) + north < cfg.image_size[0]:
        if temp_image[int(DUP_index_i) + north, int(DUP_index_j)] < threshold:
            north = north + 1
        else:
            count += 1
            sum = sum + temp_image[int(DUP_index_i) + north, int(DUP_index_j)]
            break
    return sum, count


def grass_fire_south(cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold):
    south = -1
    while int(DUP_index_i) + south >= 0:
        if temp_image[int(DUP_index_i) + south, int(DUP_index_j)] < threshold:
            south = south - 1
        else:
            count += 1
            sum = sum + temp_image[int(DUP_index_i) + south, int(DUP_index_j)]
            break
    return sum, count


def grass_fire_north_east(
    cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
):
    north_east_north = 1
    north_east_east = 1
    while (
        int(DUP_index_i) + north_east_north < cfg.image_size[0]
        and int(DUP_index_j) + north_east_east < cfg.image_size[0]
    ):
        if (
            temp_image[
                int(DUP_index_i) + north_east_north, int(DUP_index_j) + north_east_east
            ]
            < threshold
        ):
            north_east_north = north_east_north + 1
            north_east_east = north_east_east + 1
        else:
            count += 1
            sum = (
                sum
                + temp_image[
                    int(DUP_index_i) + north_east_north,
                    int(DUP_index_j) + north_east_east,
                ]
            )
            break
    return sum, count


def grass_fire_north_west(
    cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
):
    north_west_west = -1
    north_west_north = 1
    while (
        int(DUP_index_i) + north_west_north < cfg.image_size[0]
        and int(DUP_index_j) + north_west_west >= 0
    ):
        if (
            temp_image[
                int(DUP_index_i) + north_west_north, int(DUP_index_j) + north_west_west
            ]
            < threshold
        ):
            north_west_north = north_west_north + 1
            north_west_west = north_west_west - 1
        else:
            count += 1
            sum = (
                sum
                + temp_image[
                    int(DUP_index_i) + north_west_north,
                    int(DUP_index_j) + north_west_west,
                ]
            )
            break
    return sum, count


def grass_fire_south_east(
    cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
):
    south_east_south = -1
    south_east_east = 1
    while (
        int(DUP_index_i) + south_east_south >= 0
        and int(DUP_index_j) + south_east_east < cfg.image_size[0]
    ):
        if (
            temp_image[
                int(DUP_index_i) + south_east_south, int(DUP_index_j) + south_east_east
            ]
            < threshold
        ):
            south_east_south = south_east_south - 1
            south_east_east = south_east_east + 1
        else:
            count += 1
            sum = (
                sum
                + temp_image[
                    int(DUP_index_i) + south_east_south,
                    int(DUP_index_j) + south_east_east,
                ]
            )
            break
    return sum, count


def grass_fire_south_west(
    cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold
):
    south_west_south = -1
    south_west_west = -1
    while (
        int(DUP_index_i) + south_west_south >= 0
        and int(DUP_index_j) + south_west_west >= 0
    ):
        if (
            temp_image[
                int(DUP_index_i) + south_west_south, int(DUP_index_j) + south_west_west
            ]
            < threshold
        ):
            south_west_south = south_west_south - 1
            south_west_west = south_west_west - 1
        else:
            count += 1
            sum = (
                sum
                + temp_image[
                    int(DUP_index_i) + south_west_south,
                    int(DUP_index_j) + south_west_west,
                ]
            )
            break
    return sum, count


def grass_fire_east(cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold):
    east = 1
    while int(DUP_index_j) + east < cfg.image_size[0]:
        if temp_image[int(DUP_index_i), int(DUP_index_j) + east] < threshold:
            east = east + 1
        else:
            count += 1
            sum = sum + temp_image[int(DUP_index_i), int(DUP_index_j) + east]
            break
    return sum, count


def grass_fire_west(cfg, sum, count, temp_image, DUP_index_i, DUP_index_j, threshold):
    west = -1
    while int(DUP_index_j) + west >= 0:
        if temp_image[int(DUP_index_i), int(DUP_index_j) + west] < threshold:
            west = west - 1
        else:
            count += 1
            sum = sum + temp_image[int(DUP_index_i), int(DUP_index_j) + west]
            break
    return sum, count


def apply_depth_correction(cfg, dups_threshold, target_images, rgb_images):
    """
    dups_threshold: minimum distance in mm, smaller than which is considered an unmeasured pixel
    """
    rgb_images = np.asarray(rgb_images, dtype=np.uint8)
    target_images = np.asarray(target_images, dtype=np.uint8)
    print("processing dups")
    dups = np.array(
        [get_dups(target_image, dups_threshold) for target_image in target_images],
        dtype=object,
    )
    print("done getting dups")
    grays_quantized, rgbs_quantized = gray_dup_helper(cfg, rgb_images)
    print("done preparing estimation tools")
    # Create an empty list to store the results
    results = []
    # Iterate through the zipped arrays with tqdm to log progress
    for target_image, gray_image, dups_i, dups_j in tqdm(
        zip(target_images, grays_quantized, dups[:, 0], dups[:, 1])
    ):
        result = estimate_dups(
            cfg, target_image, gray_image, dups_i, dups_j, dups_threshold
        )
        results.append(result)
    # Convert the results list to a numpy array
    target_images = np.array(results)
    print("done processing dups")
    return target_images, rgbs_quantized
