# standard library imports
import os
import hydra
from matplotlib import patches
from omegaconf import OmegaConf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# repo modules
from src.imagine2touch.utils.model_utils import preprocess_object_data

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
    # reskin raw data with ambient subtracted
    tactile_input = tactile_input.cpu().numpy().reshape(-1, 15)
    tactile_input = scaler_reskin.inverse_transform(tactile_input)
    tactile_input = (tactile_input * std_reskin) + mean_reskin

    def log_contrast_filter(tactile_input, type="training"):
        ## algorithm to increase variance of data
        # initialize variables
        data = []
        readings_weights = []
        added_indeces = []
        prev_mean = 0
        # set convergence threshold
        convergence_threshold = 0.05
        # loop until convergence
        while True:
            readings_weights = []
            data_temp = []
            for i, reading in enumerate(tactile_input):
                if i not in (added_indeces) and np.linalg.norm(reading, 2) < 1000:
                    new_mean = np.linalg.norm(reading, 2)
                    readings_weights.append(np.abs(new_mean - prev_mean))
                else:
                    readings_weights.append(-1000)
            index = np.argmax(readings_weights, 0)
            added_indeces.append(index)
            data.append(tactile_input[index])
            new_mean = np.linalg.norm(np.mean(data, 0), 2)
            if (np.abs(new_mean - prev_mean)) < convergence_threshold:
                break
            prev_mean = new_mean
        removed_indeces = np.setdiff1d(np.arange(0, len(tactile_input)), added_indeces)
        added_indeces = np.sort(added_indeces)
        with open(
            f"{cfg.experiment_dir}/{type}_contrast_filter.npy", "wb"
        ) as contrast_indeces_file:
            np.save(contrast_indeces_file, added_indeces)
        # print(np.linalg.norm(np.std(tactile_input,0)))
        # print(np.linalg.norm(np.std(data,0)))
        # print(np.linalg.norm(np.mean(tactile_input,0)))
        # print(np.linalg.norm(np.mean(data,0)))
        data = np.array(data)
        # print(data.shape)
        # print(tactile_input.shape)
        # print(removed_indeces.shape)
        # print(added_indeces.shape)
        return data

    ## plot processed and original data
    data = log_contrast_filter(tactile_input, "training")
    # create a figure and set of subplots to draw the plots on
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    # loop over each dimension of the data and plot the kernel density estimate
    artists = []
    labels = []
    for i in range(15):
        row = i // 4
        col = i % 4
        a = sns.kdeplot(
            data=tactile_input[:, i],
            fill=True,
            ax=axes[row, col],
            label="original data",
            color="orange",
        )
        b = sns.kdeplot(
            data=data[:, i],
            fill=True,
            ax=axes[row, col],
            label="processed data",
            color="blue",
        )
        axes[row, col].set_title("Dimension {}".format(i + 1), fontdict={"fontsize": 8})
        axes[row, col].set_ylabel("pdf")
        axes[row, col].set_xlim([data.min(), data.max()])
    c = sns.kdeplot(
        data=np.linalg.norm(tactile_input, 2, 1),
        fill=True,
        ax=axes[3, 3],
        label="original data norm",
        color="red",
    )
    d = sns.kdeplot(
        data=np.linalg.norm(data, 2, 1),
        fill=True,
        ax=axes[3, 3],
        label="processed data norm",
        color="green",
    )
    axes[3, 3].set_xlim(
        [np.min(np.linalg.norm(data, 2, 1)), np.max(np.linalg.norm(data, 2, 1))]
    )
    # update the legend with the new labels and colors
    labels = [
        "original data",
        "processed data",
        "original data norm",
        "processed data norm",
    ]
    legend_labels = {
        "original data": ["orange"],
        "processed data": ["blue"],
        "original data norm": ["red"],
        "processed data norm": ["green"],
    }
    handles = []
    for label in legend_labels:
        colors = legend_labels[label]
        for color in colors:
            handle = patches.Patch(color=color, label=label)
            handles.append(handle)
    plt.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6)
    plt.subplots_adjust(hspace=1, wspace=0.5)
    plt.show()
