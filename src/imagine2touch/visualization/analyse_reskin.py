import matplotlib.cm as cm
import numpy as np
from src.imagine2touch.reskin_calibration.dataset import (
    get_ambient_data,
    get_reskin_reading,
    subtract_ambient,
)
import numpy as np
import re
import os
import natsort
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


if __name__ == "__main__":
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("dataset.yaml")
    file_tactile = []
    object_names = cfg.object_names.split(",")
    object_names_string = ",".join(object_names)
    for object_name in object_names:
        file_tactile.append(
            cfg.data_path + "/" + object_name + "/" + object_name + "_tactile"
        )

    tactile_input = get_reskin_reading(
        file_tactile,
        binary=True,
        differential_signal=False,
        raw_ambient=cfg.aggregated_ambient,
        ambient_every_reading=cfg.ambient_every_contact,
    )

    # plot configuration initialization
    num_samples, num_dimensions = tactile_input.shape
    minimum_data = tactile_input.min()
    maximum_data = tactile_input.max()
    z_grid = np.linspace(minimum_data, maximum_data, num_samples)
    colormap = cm.get_cmap("tab10", num_dimensions)
    direction_mappings = {1: "Center", 2: "Top", 3: "Left", 4: "Bottom", 5: "Right"}
    legend_labels = [
        f"{direction_mappings[j]}" for j in range(1, len(direction_mappings) + 1)
    ]
    # Create separate plots for each axis
    for axis in ["x", "y", "z"]:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=45 + 180)  # Set consistent viewing angles
        # Plot the KDEs
        j = 1
        l = 1
        k = 1
        m = 1
        for i in range(num_dimensions):
            # Create a meshgrid for x (dimensions) and z (range of values)
            if i % 3 == 0 and i != 0:
                j += 1
            z = tactile_input[:, i]
            mean_z = np.mean(z)
            if axis == "x":
                if i in {0, 3, 6, 9, 12}:
                    kde = gaussian_kde(z)
                    y_estimate = kde.evaluate(z_grid)
                    color = colormap(l / 5)
                    ax.plot3D(
                        np.ones_like(z_grid) * l,
                        z_grid,
                        y_estimate,
                        label=f"{direction_mappings[j]} b_{axis}",
                        color=color,
                    )
                    ax.plot3D(
                        np.linspace(1, 5, 5),
                        np.zeros(5),
                        np.zeros(5),
                        color="black",
                        linewidth=2,
                    )
                    peak_idx = np.argmax(y_estimate)
                    ax.scatter(l, z_grid[peak_idx], c="red", marker="o", s=50)
                    # Create vertices for the translucent polygonal surface
                    verts = [
                        (l, z_val, y_estimate[idx]) for idx, z_val in enumerate(z_grid)
                    ]
                    verts.append((l, z_grid[-1], 0))  # Vertex at the end of the curve
                    verts.append(
                        (l, z_grid[0], 0)
                    )  # Vertex at the beginning of the curve
                    # Plot the polygonal surface with the unique color
                    poly = Poly3DCollection([verts], color=color, alpha=0.3)
                    ax.add_collection3d(poly)
                    ax.plot3D(
                        [l, l],
                        [z_grid[peak_idx], z_grid[peak_idx]],
                        [y_estimate[peak_idx], 0],
                        linestyle="dotted",
                        color="black",
                    )
                    # Set the x-axis tick positions
                    ax.set_xticks(range(1, len(legend_labels) + 1))
                    # Set the x-axis tick labels using the legend labels
                    ax.set_xticklabels(legend_labels)
                    l += 1
            elif axis == "y":
                if i in {1, 4, 7, 10, 13}:
                    kde = gaussian_kde(z)
                    y_estimate = kde.evaluate(z_grid)
                    color = colormap(k / 5)
                    ax.plot3D(
                        np.ones_like(z_grid) * k,
                        z_grid,
                        y_estimate,
                        label=f"{direction_mappings[j]} b_{axis}",
                        color=color,
                    )
                    ax.plot3D(
                        np.linspace(1, 5, 5),
                        np.zeros(5),
                        np.zeros(5),
                        color="black",
                        linewidth=2,
                    )
                    peak_idx = np.argmax(y_estimate)
                    ax.scatter(k, z_grid[peak_idx], c="red", marker="o", s=50)
                    # Create vertices for the translucent polygonal surface
                    verts = [
                        (k, z_val, y_estimate[idx]) for idx, z_val in enumerate(z_grid)
                    ]
                    verts.append((k, z_grid[-1], 0))  # Vertex at the end of the curve
                    verts.append(
                        (k, z_grid[0], 0)
                    )  # Vertex at the beginning of the curve
                    # Plot the polygonal surface with the unique color
                    poly = Poly3DCollection([verts], color=color, alpha=0.3)
                    ax.add_collection3d(poly)
                    ax.plot3D(
                        [k, k],
                        [z_grid[peak_idx], z_grid[peak_idx]],
                        [y_estimate[peak_idx], 0],
                        linestyle="dotted",
                        color="black",
                    )
                    # Set the x-axis tick positions
                    ax.set_xticks(range(1, len(legend_labels) + 1))
                    # Set the x-axis tick labels using the legend labels
                    ax.set_xticklabels(legend_labels)
                    k += 1
            elif axis == "z":
                if i in {2, 5, 8, 11, 14}:
                    kde = gaussian_kde(z)
                    y_estimate = kde.evaluate(z_grid)
                    color = colormap(m / 5)
                    ax.plot3D(
                        np.ones_like(z_grid) * m,
                        z_grid,
                        y_estimate,
                        label=f"{direction_mappings[j]} b_{axis}",
                        color=color,
                    )
                    ax.plot3D(
                        np.linspace(1, 5, 5),
                        np.zeros(5),
                        np.zeros(5),
                        color="black",
                        linewidth=2,
                    )
                    peak_idx = np.argmax(y_estimate)
                    ax.scatter(m, z_grid[peak_idx], c="red", marker="o", s=50)
                    # Create vertices for the translucent polygonal surface
                    verts = [
                        (m, z_val, y_estimate[idx]) for idx, z_val in enumerate(z_grid)
                    ]
                    verts.append((m, z_grid[-1], 0))  # Vertex at the end of the curve
                    verts.append(
                        (m, z_grid[0], 0)
                    )  # Vertex at the beginning of the curve
                    # Plot the polygonal surface with the unique color
                    poly = Poly3DCollection([verts], color=color, alpha=0.3)
                    ax.add_collection3d(poly)
                    peak_idx = np.argmax(y_estimate)
                    ax.plot3D(
                        [m, m],
                        [z_grid[peak_idx], z_grid[peak_idx]],
                        [y_estimate[peak_idx], 0],
                        linestyle="dotted",
                        color="black",
                    )
                    # Set the x-axis tick positions
                    ax.set_xticks(range(1, len(legend_labels) + 1))
                    # Set the x-axis tick labels using the legend labels
                    ax.set_xticklabels(legend_labels)
                    m += 1
        if axis == "x":
            ax.set_ylabel("Range of Values")
            ax.set_zlabel("Probability Estimate")
        elif axis == "y":
            ax.set_ylabel("Range of Values")
            ax.set_zlabel("Probability Estimate")
        elif axis == "z":
            ax.set_ylabel("Range of Values")
            ax.set_zlabel("Probability Estimate")
        ax.set_title(f"Stacked 3D Probability Estimates (KDE) - Axis: {axis}")
        # Set consistent aspect ratio and plot limits
        # ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([0, 5])
        ax.set_ylim([np.min(tactile_input), np.max(tactile_input)])
        ax.set_zlim([0, 0.06])

        #  Export the plot to an image file (e.g., PNG)
        # plt.savefig(
        #     f"{object_names_string}_b_{axis}.png", dpi=2000, bbox_inches="tight"
        # )
        ax.legend()
        plt.tight_layout()  # Ensures the layout is adjusted correctly
        plt.show()
