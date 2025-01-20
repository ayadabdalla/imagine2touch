# standard library imports
import cv2
from matplotlib.colorbar import ColorbarBase
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# from sklearn.manifold import TSNE
from openTSNE import TSNE  # alternative tsne implementation
import matplotlib.pyplot as plt
import os
import hydra
from matplotlib import patches
from omegaconf import OmegaConf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import OneHotEncoder
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib import colors as normcolors

# repo modules
from src.imagine2touch.utils.model_utils import preprocess_object_data
from src.imagine2touch.visualizations.reskin_image_visualizer import (
    plot_combined_heatmap,
    split_rgb,
    visualize_reskin_image,
)


def find_center_point(points):
    """
    Find the center point of a list of 2D points."""
    total_x = 0
    total_y = 0

    for x, y in points:
        total_x += x
        total_y += y

    center_x = total_x / len(points)
    center_y = total_y / len(points)

    # Find the point closest to the calculated center
    center_point = min(
        points,
        key=lambda point: (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2,
    )

    return center_point


def plot_heat_map_wrapper(
    clustered_target_images,
    clustered_tactile_data,
    draw=False,
    display=False,
    rotate=False,
    tactile_shape=15,
):
    """
    Plot heatmaps for each cluster in the dataset. The dataset is composed of images and touches.
    -draw: if True, create plots for each cluster, necessary for -display
    -display: if True, show each cluster touch and image mean as a heatmap in individual plots
    """
    mask_clusters = []
    tactile_clusters = []
    tactile_raws = []
    for label in range(np.unique(clustered_target_images[:, -1]).shape[0]):
        clustered_target_image = clustered_target_images[
            clustered_target_images[:, -1] == label
        ]
        clustered_target_image = clustered_target_image[:, :-1]
        clustered_target_image = clustered_target_image.reshape(
            -1, cfg.image_size[0], cfg.image_size[1]
        )
        clustered_tactile_data_ = clustered_tactile_data[
            clustered_tactile_data[:, -1] == label
        ]
        clustered_tactile_data_ = clustered_tactile_data_[:, :-1]
        mask = plot_combined_heatmap(
            clustered_target_image,
            rotate=False,
            draw=draw,
            title="masks_cluster_" + str(label),
            set_bad=False,
        )
        if label == np.unique(clustered_target_images[:, -1]).shape[0] - 1:
            tactile_heatmap = plot_combined_heatmap(
                visualize_reskin_image(clustered_tactile_data_[:, :tactile_shape]),
                draw=draw,
                set_bad=True,
                title="tactile_cluster_" + str(label),
                display=display,
            )
        else:
            tactile_heatmap = plot_combined_heatmap(
                visualize_reskin_image(clustered_tactile_data_[:, :tactile_shape]),
                draw=draw,
                set_bad=True,
                title="tactile_cluster_" + str(label),
            )
        mask_clusters.append(mask)
        tactile_clusters.append(tactile_heatmap)
        tactile_raw = np.mean(clustered_tactile_data_[:, :tactile_shape], axis=0)
        tactile_raws.append(tactile_raw)
    return mask_clusters, tactile_clusters, tactile_raws


def threshold_cluster_points(X_tsne, threshold_x=3, threshold_y=1, means=None):
    """
    pick points that are at least threshold units away from each other in a cluster
    """
    cluster_points = X_tsne
    if len(cluster_points) > 0:
        if means is not None:
            selected_points = [means]
        else:
            selected_points = [cluster_points[0]]
        selected_points = np.array(selected_points)
        selected_points = selected_points.reshape(-1, 2)
        for point in cluster_points[1:]:
            distances_x = np.abs(point[0] - selected_points[:, 0])
            distances_y = np.abs(point[1] - selected_points[:, 1])
            if np.all(distances_x >= threshold_x) and np.all(
                distances_y >= threshold_y
            ):
                selected_points = np.vstack((selected_points, point))
    return selected_points


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
    for i, _ in enumerate(target_masks):
        target_masks[i] = np.rot90(target_masks[i], k=1)
    # target_masks = np.asarray(
    #     [
    #         cv2.resize(
    #             target_masks[i],
    #             (3, 3),
    #             interpolation=cv2.INTER_NEAREST,
    #         )
    #         for i in range(len(target_masks))
    #     ]
    # )
    # cfg.image_size = [3, 3]

    target_masks = target_masks.reshape(-1, cfg.image_size[0] * cfg.image_size[1])

    # rescale target_images
    target_images = target_images.cpu().numpy()
    target_images = target_images.reshape(-1, cfg.image_size[0], cfg.image_size[1])
    for i, _ in enumerate(target_images):
        target_images[i] = np.rot90(target_images[i], k=1)
    target_images = target_images.reshape(-1, cfg.image_size[0] * cfg.image_size[1])
    target_images = target_images * std_images + mean_images

    tactile_input = visualize_reskin_image(tactile_input, get_data=True)[1]
    tactile_images = visualize_reskin_image(tactile_targets)

    print("tactile_input.shape", tactile_input.shape)
    print("target_masks.shape", target_masks.shape)
    # X = target_masks
    X = tactile_input
    # Perform clustering with KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)
    clustered_target_images = np.column_stack((target_masks, labels))
    clustered_depth_images = np.column_stack((target_images, labels))
    clustered_tactile_data = np.column_stack((tactile_input, labels))
    clustered_rgb_images = np.column_stack((rgb_images, labels))

    # # Visualize images in one cluster
    # for images in clustered_target_images[clustered_target_images[:, -1] == 0]:
    #     plt.imshow(images[:-1].reshape(cfg.image_size[0], cfg.image_size[1]))
    #     plt.show()
    # Visualize tac in one cluster
    # tac_images = tactile_images[clustered_tactile_data[:, -1] == 1]
    # for image in tac_images:
    #     plt.imshow(image)
    #     plt.show()

    # Perform TSNE with 2 components
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne_fit = tsne.fit(X)
    X_tsne = X_tsne_fit.transform(X)

    # Visualize tsne results
    fig, ax = plt.subplots()
    means = []
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10")
    for j in range(np.unique(labels).shape[0]):
        masks_heatmaps, tactile_heatmaps, raws = plot_heat_map_wrapper(
            clustered_target_images, clustered_tactile_data, tactile_shape=5
        )
        mask_heatmap = masks_heatmaps[j]
        tactile_heatmap = tactile_heatmaps[j]
        raw = raws[j]
        tsne_projection = X_tsne_fit.transform(raw.reshape(1, -1))
        x_m, y_m = tsne_projection[0]
        X_tsne = np.vstack((X_tsne, [x_m, y_m]))
        labels = np.append(labels, j)
        means.append([x_m, y_m])
    for j in range(np.unique(labels).shape[0]):
        # plot last point in cluster as an x
        cluster_points = X_tsne[labels == j]
        x, y = cluster_points[-1]
        # get colors of scatter plot
        colors = scatter.to_rgba(labels[labels == j])
        plt.scatter(x, y, marker="x", color="red", s=400)

    selected_points = threshold_cluster_points(
        X_tsne, threshold_y=0.5, threshold_x=0.8, means=means
    )
    for j in range(np.unique(labels).shape[0]):
        cluster_points = X_tsne[labels == j]
        mask_heatmap = masks_heatmaps[j]
        tactile_heatmap = tactile_heatmaps[j]
        # plot individual points
        for i, (x, y) in enumerate(cluster_points):
            if any(np.array_equal([x, y], arr) for arr in selected_points) or (
                i == cluster_points.shape[0] - 1
            ):
                if i != cluster_points.shape[0] - 1:
                    depth = clustered_depth_images[clustered_depth_images[:, -1] == j][
                        i, : cfg.image_size[0] * cfg.image_size[1]
                    ]
                    mask = clustered_target_images[clustered_target_images[:, -1] == j][
                        i, : cfg.image_size[0] * cfg.image_size[1]
                    ]
                    mask = mask.reshape(cfg.image_size[0], cfg.image_size[1])
                    # mask = np.kron(mask, np.ones((16, 16), dtype=np.uint8))
                    rgb = clustered_rgb_images[clustered_rgb_images[:, -1] == j][
                        i, : 48 * 48 * 3
                    ]
                    rgb = rgb.reshape(48, 48, 3).astype(np.uint8)
                    tactile_image = visualize_reskin_image(
                        np.asarray(
                            clustered_tactile_data[clustered_tactile_data[:, -1] == j][
                                i, :5
                            ]
                        ).reshape(1, 5)
                    )
                    tactile_image = np.squeeze(tactile_image)  # remove extra dimension
                    tactile_image = np.kron(
                        tactile_image, np.ones((16, 16), dtype=np.uint8)
                    )  # upsample to match mask size
                    plt.imsave(f"./{j}_{i}_tac.png", tactile_image)

                    tactile_image[tactile_image == 0] = np.nan  # set corners to nan
                    rgb = np.rot90(rgb, k=1)
                    cmap = plt.cm.get_cmap("copper")
                    zoom = 0.3
                    spacing = 0
                    rgb_box = OffsetImage(
                        rgb, zoom=zoom
                    )  # Adjust the zoom factor as needed
                    ef = AnnotationBbox(
                        rgb_box,
                        (x, y),
                        xybox=(x, y - 6 - spacing),
                        frameon=False,
                    )
                    ef = plt.gca().add_artist(ef)
                    # save as tactile as png image
                    # save mask as png image
                    plt.imsave(f"./{j}_{i}_mask.png", mask)
                    # save rgb as png image
                    plt.imsave(f"./{j}_{i}_rgb.png", rgb)
                    # save depth as png image usin PIL
                    depth = depth.reshape(48, 48)
                    depth = 255 - depth
                    # invert depth using bitwise not
                    plt.imsave(f"./{j}_{i}_depth.png", depth)
                else:
                    mask = mask_heatmap
                    # mask = np.kron(mask, np.ones((16, 16), dtype=np.uint8))
                    tactile_image = tactile_heatmap
                    tactile_image = np.kron(
                        tactile_image, np.ones((16, 16), dtype=np.uint8)
                    )  # upsample to match mask size
                    cmap = plt.cm.get_cmap("copper")
                    zoom = 0.6
                    spacing = 2

                cmap.set_bad(color="#e6f2ff", alpha=0.2)
                # cmap.set_bad(color='white',alpha=0.2)
                mask_box = OffsetImage(
                    mask,
                    zoom=zoom,
                    cmap=cmap,
                )
                # Adjust the zoom factor as needed
                tactile_imagebox = OffsetImage(
                    tactile_image, zoom=zoom, cmap=cmap
                )  # Adjust the zoom factor as needed
                ab = AnnotationBbox(
                    tactile_imagebox,
                    (x, y),
                    xybox=(x, y - 4 - spacing),
                    frameon=False,
                )
                cd = AnnotationBbox(
                    mask_box,
                    (x, y),
                    xybox=(x, y - 3),
                    frameon=False,
                    arrowprops=dict(arrowstyle="->", lw=1.5),
                )
                cd = plt.gca().add_artist(cd)
                ab = plt.gca().add_artist(ab)
                # if artist outside the frame, remove it
                if y - np.min(X_tsne[:, 1]) < (
                    np.max(X_tsne[:, 1]) - np.min(X_tsne[:, 1])
                ) * (5 / 100):
                    print("removing artist")
                    ab.remove()
                    cd.remove()
                    if i != cluster_points.shape[0] - 1:
                        ef.remove()
                plt.gca().set_facecolor("#e6f2ff")

    min_value = 0
    max_value = 1
    cmap = plt.cm.copper  # You can choose any colormap you like
    norm = Normalize(vmin=min_value, vmax=max_value)
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0 + pos.width + 0.02, pos.y0, 0.02, pos.height])
    cbar = ColorbarBase(ax=cbar_ax, cmap=cmap, norm=norm)
    cbar.set_label("Data heatmap", rotation=270, labelpad=5)
    plt.savefig("tsne.png", dpi=300)
    plt.show()
