# standard libraries
import os
import sys
from typing import MutableMapping
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, RobustScaler
import sklearn.utils.validation
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
from PIL import Image

# from pyemd import emd
from skimage.metrics import structural_similarity as ssim
import torchvision


# repo modules
from src.imagine2touch.utils.utils import (
    NotAdaptedError,
    get_target_images,
    rgb2gray,
    get_target_masks,
    get_depth_processed,
)
from src.imagine2touch.reskin_calibration import dataset


def save_tactile(
    tactile_reconstructions, scaler_reskin, mean_reskin, std_reskin, tactile_path
):
    tactile_reconstructions = tactile_reconstructions.cpu().numpy()
    for i in range(tactile_reconstructions.shape[0]):
        tactile_reconstructions[i] = tactile_reconstructions[i].reshape(1, -1)
    # remove standardization
    tactile_reconstructions = tactile_reconstructions * std_reskin + mean_reskin
    # save tactile data
    np.save(f"{tactile_path}/tactile_reconstructed.npy", tactile_reconstructions)
    print("Tactile data saved")


def custom_masked_mse_loss(predictions, ground_truths, masks_pred, masks_gt):
    batch_loss_images = torch.mean(
        nn.MSELoss(reduction="none")(predictions, ground_truths)
        * nn.MSELoss(reduction="none")(masks_pred, masks_gt)
    )
    return batch_loss_images


def accuracy(predictions, groud_truths):
    """calculate the accuracy of the predictions"""
    batch_loss_masks_acc = np.sum(np.where(predictions == groud_truths, 1, 0)) / (
        predictions.shape[0] * predictions.shape[1]
    )
    return batch_loss_masks_acc


def focal_loss(predictions, ground_truths):
    batch_loss_masks = torchvision.ops.focal_loss.sigmoid_focal_loss(
        predictions, ground_truths, reduction="mean"
    )
    return batch_loss_masks


def custom_emd(predictions, ground_truths):
    """calculate the earth mover's distance between two torch images"""
    histograms1 = []
    histograms2 = []
    for image1, image2 in zip(
        predictions.detach()
        .cpu()
        .numpy()
        .reshape(
            -1, int(np.sqrt(predictions.shape[1])), int(np.sqrt(predictions.shape[1]))
        ),
        ground_truths.detach()
        .cpu()
        .numpy()
        .reshape(
            -1, int(np.sqrt(predictions.shape[1])), int(np.sqrt(predictions.shape[1]))
        ),
    ):
        # Compute the histograms of the images
        histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

        # Normalize the histograms
        histogram1 = cv2.normalize(histogram1, histogram1).flatten()
        histogram2 = cv2.normalize(histogram2, histogram2).flatten()
        histograms1.append(histogram1)
        histograms2.append(histogram2)

    # Calculate the Earth Mover's Distance (EMD)
    cost_matrix = np.zeros((256, 256), dtype=np.float32)
    for i in range(256):
        for j in range(256):
            cost_matrix[i, j] = abs(i - j)
    emd_distances = []
    for histogram1, histogram2 in zip(histograms1, histograms2):
        emd_distance = emd(
            np.asarray(histogram1, dtype=np.float64),
            np.asarray(histogram2, dtype=np.float64),
            np.asarray(cost_matrix, dtype=np.float64),
        )
        emd_distances.append(emd_distance)
    batch_loss_masks = (
        torch.mean(torch.from_numpy(np.asarray(emd_distances.copy()))) / 256
    )
    return batch_loss_masks


def custom_ssim(predictions, ground_truths):
    """calculate the structural similarity between two torch images"""
    ssims = []
    for image1, image2 in zip(
        predictions.detach()
        .cpu()
        .numpy()
        .reshape(
            -1, int(np.sqrt(predictions.shape[1])), int(np.sqrt(predictions.shape[1]))
        ),
        ground_truths.detach()
        .cpu()
        .numpy()
        .reshape(
            -1, int(np.sqrt(predictions.shape[1])), int(np.sqrt(predictions.shape[1]))
        ),
    ):
        ssim_loss = (
            1
            - (1 + ssim(image2, image1, data_range=1.0, gaussian_weights=True, sigma=5))
            / 2
        )
        ssims.append(ssim_loss)
    batch_loss_masks = torch.mean(torch.from_numpy(np.asarray(ssims.copy())))
    return batch_loss_masks


def morph_tactile_image(tactile_reading):
    """morph tactile readings into numpy images"""
    centers = np.expand_dims(np.expand_dims(tactile_reading[:, 0:3], 1), 2)
    tops = np.expand_dims(np.expand_dims(tactile_reading[:, 3:6], 1), 2)
    rights = np.expand_dims(np.expand_dims(tactile_reading[:, 6:9], 1), 2)
    bottoms = np.expand_dims(np.expand_dims(tactile_reading[:, 9:12], 1), 2)
    lefts = np.expand_dims(np.expand_dims(tactile_reading[:, 12:15], 1), 2)
    zeros = np.zeros((1, 1, 3))
    reskin_images = []
    for center, top, right, bottom, left in zip(centers, tops, rights, bottoms, lefts):
        top_row = np.hstack((zeros, top))
        top_row = np.hstack((top_row, zeros))
        middle_row = np.hstack((left, center))
        middle_row = np.hstack((middle_row, right))
        bottom_row = np.hstack((zeros, bottom))
        bottom_row = np.hstack((bottom_row, zeros))
        current_image = np.vstack((top_row, middle_row, bottom_row))
        reskin_images.append(current_image)
    return np.array(reskin_images)


def morph_tactile_image_tensor(tactile_reading):
    """morph tactile readings into torch images"""
    centers = torch.unsqueeze(torch.unsqueeze(tactile_reading[:, 0:3], 1), 2)
    tops = torch.unsqueeze(torch.unsqueeze(tactile_reading[:, 3:6], 1), 2)
    rights = torch.unsqueeze(torch.unsqueeze(tactile_reading[:, 6:9], 1), 2)
    bottoms = torch.unsqueeze(torch.unsqueeze(tactile_reading[:, 9:12], 1), 2)
    lefts = torch.unsqueeze(torch.unsqueeze(tactile_reading[:, 12:15], 1), 2)
    zeros = torch.zeros((1, 1, 3))
    reskin_images = []
    for center, top, right, bottom, left in zip(centers, tops, rights, bottoms, lefts):
        top_row = torch.cat((zeros, top), dim=1)
        top_row = torch.cat((top_row, zeros), dim=1)
        middle_row = torch.cat((left, center), dim=1)
        middle_row = torch.cat((middle_row, right), dim=1)
        bottom_row = torch.cat((zeros, bottom), dim=1)
        bottom_row = torch.cat((bottom_row, zeros), dim=1)
        current_image = torch.cat((top_row, middle_row, bottom_row), dim=0)
        reskin_images.append(current_image)
    return torch.stack(reskin_images)


def set_max_to_1(arr, threshold=0.8):
    """create a mask to set the maximum value, greater than threshold to 1 and the rest to 0"""
    mask = torch.zeros_like(arr)
    for i in range(arr.shape[0]):
        if torch.max(arr[i]) > threshold:
            mask[i, torch.argmax(arr[i])] = 1
        else:
            mask[i] = arr[i]
    result = torch.mul(mask, arr)
    return result


def no_loss(*cfg):
    return 0


losses = {
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
    "kld": nn.KLDivLoss(log_target=True, reduction="batchmean"),
    "bce": nn.BCELoss(),
    "ce": nn.CrossEntropyLoss(),
    "no_loss": no_loss,
    "emd": custom_emd,
    "ssim": custom_ssim,
    "focal_loss": focal_loss,
    "acc": accuracy,
    "masked_mse_loss": custom_masked_mse_loss,
}
optimizers = {"adam": optim.Adam, "sgd": optim.SGD, "rmsprop": optim.RMSprop}


def infer(
    model,
    cfg,
    device,
    test_loader,
    original_tactile_list=[],
    original_images_list=[],
    original_rgb_list=[],
    original_masks_list=[],
    loader_indeces_list=[],
    masks_reconstructions=[],
    tactile_reconstructions=[],
):
    for (
        batch_tactile_features,
        _,
        batch_target_images,
        batch_rgb_images,
        batch_target_masks,
        loader_indeces,
    ) in test_loader:
        original_tactile = batch_tactile_features.to(device)
        if cfg.model.cnn_images_encoder:
            original_target_images = batch_target_images.view(
                -1, cfg.image_size[0], cfg.image_size[1], 1
            ).to(device)
            original_target_masks = batch_target_masks.view(
                -1, cfg.image_size[0], cfg.image_size[1], 1
            ).to(device)
            original_target_images = original_target_images.permute(0, 3, 1, 2)
            original_target_masks = original_target_masks.permute(0, 3, 1, 2)
        else:
            original_target_images = batch_target_images.view(
                -1, cfg.image_size[0] * cfg.image_size[1] * 1
            ).to(device)
            if not cfg.classification:
                original_target_masks = batch_target_masks.view(
                    -1, cfg.image_size[0] * cfg.image_size[1]
                ).to(device)
            else:
                original_target_masks = batch_target_masks.to(device)
        original_rgb_images = batch_rgb_images.view(
            -1, cfg.image_size[0] * cfg.image_size[1] * 3
        ).to(device)
        original_images_list.append(original_target_images)
        original_rgb_list.append(original_rgb_images)
        original_masks_list.append(original_target_masks)
        original_tactile_list.append(original_tactile)
        (
            tactile_reconstruction,
            mask_reconstruction,
            _,
        ) = model(original_target_masks)
        loader_indeces_list.append(loader_indeces)
        masks_reconstructions.append(mask_reconstruction)
        tactile_reconstructions.append(tactile_reconstruction)
    return (
        original_tactile_list,
        original_rgb_list,
        original_images_list,
        original_masks_list,
        tactile_reconstructions,
        masks_reconstructions,
        loader_indeces_list,
    )


def reorder_shuffled(array, *args):
    array = torch.cat(array, dim=0)
    for arg in args:
        array = array[arg]
    return array


def fetch_data(cfg, repo_path, train=True):
    if train:
        objects_names = cfg.objects_names.split(",")
    else:
        objects_names = cfg.test_objects_names.split(",")
    paths_rgbs = []
    paths_target_images = []
    paths_reskin = []
    paths_masks = []
    for o in objects_names:
        paths_rgbs.append(f"{repo_path}/{cfg.experiment_dir}/{o}/{o}_images/rgb")
        paths_target_images.append(f"{repo_path}/{cfg.experiment_dir}/{o}/{o}_images/")
        paths_reskin.append(f"{repo_path}/{cfg.experiment_dir}/{o}/{o}_tactile")
        paths_masks.append(f"{repo_path}/{cfg.experiment_dir}/{o}/{o}_images/masks")
    rgb_images = get_target_images(paths_rgbs, "rgb", cfg.image_size)
    rgb_images = np.asarray(rgb_images, dtype=np.uint8)
    target_images = get_target_images(
        paths_target_images, cfg.image_type, cfg.image_size
    )
    target_images = np.asarray(target_images, dtype=np.uint8)
    target_masks = get_target_masks(paths_masks, cfg.image_size)
    rgb_images = np.reshape(
        rgb_images,
        (rgb_images.shape[0], cfg.image_size[0] * cfg.image_size[1] * 3),
    )
    target_images = np.reshape(
        target_images,
        (target_images.shape[0], cfg.image_size[0] * cfg.image_size[1]),
    )
    target_masks = np.array(
        np.reshape(
            target_masks, (target_masks.shape[0], cfg.image_size[0] * cfg.image_size[1])
        ),
        dtype=np.uint8,
    )
    return (rgb_images, target_images, target_masks, paths_reskin)

    # probably not needed
    if cfg.contrast_filter and cfg.masks_filter:
        raise ValueError("Cannot use both contrast and masks filters in this version.")


def scale_data(cfg, data, scaler):
    pass


def preprocess_object_data(
    cfg,
    repo_path,
    mean_images=None,
    std_images=None,
    mean_reskin=None,
    std_reskin=None,
    scaler_images=None,
    scaler_rgb=None,
    scaler_reskin=None,
    train=True,
):
    # get data.
    (rgb_images, target_images, target_masks, path_reskin) = fetch_data(
        cfg, repo_path, train
    )

    # scale data
    simple_standardize = True
    if train:
        ## Calculate data statistics
        mean_images = np.mean(target_images)
        std_images = np.std(target_images)
        target_images = np.divide(target_images - mean_images, std_images)
        (
            tactile_input,
            tactile_targets,
            mean_reskin,
            std_reskin,
        ) = dataset.prepare_reskin_data(
            path_reskin,
            cfg.binary,
            differential_signal=True,
            standardize=simple_standardize,
            ambient_aggregated=False,
            ambient_every_reading=False,
        )
        scaler_images = QuantileTransformer(
            n_quantiles=target_images.shape[0], output_distribution="uniform"
        )
        scaler_rgb = QuantileTransformer(
            n_quantiles=rgb_images.shape[0], output_distribution="uniform"
        )
        scaler_reskin = RobustScaler()
        tactile_input = (tactile_input - np.min(tactile_input)) / (
            np.max(tactile_input) - np.min(tactile_input)
        )
        tactile_targets = tactile_input
    else:
        ## Apply passed Z-standard normalization on test target images and test reskin data
        tactile_input, tactile_targets, _, _ = dataset.prepare_reskin_data(
            path_reskin,
            cfg.binary,
            mean_reskin,
            std_reskin,
            differential_signal=True,
            standardize=simple_standardize,
            ambient_aggregated=False,
            ambient_every_reading=False,
        )
        # min max scale tactile_input between 0 and 1
        tactile_input = (tactile_input - np.min(tactile_input)) / (
            np.max(tactile_input) - np.min(tactile_input)
        )
        tactile_targets = tactile_input
        target_images = np.divide(target_images - mean_images, std_images)

    # Stack data
    data = np.hstack(
        (tactile_input, tactile_targets, target_images, rgb_images, target_masks)
    )
    data = np.asarray(data)
    # Preprocess test data
    if not train:
        if not cfg.inference:
            if cfg.split_training_set:
                _, data = train_test_split(data, cfg.training_split_ratio)
            # Use a portion of the test set for validation and shuffle it.
            test_data, validation_data = train_test_split(
                data, cfg.validation_split_ratio
            )
            test_data = list(enumerate(test_data))
            validation_data = list(enumerate(validation_data))
            np.random.shuffle(validation_data)
            validation_indices, validation_data = zip(*validation_data)
            validation_data = np.asarray(validation_data, dtype=np.float32)
            # Separate shuffled validation data into tensors.
            validation_tactile_input = torch.FloatTensor(
                validation_data[:, : cfg.model.tactile_input_shape]
            )
            validation_tactile_targets = torch.FloatTensor(
                validation_data[
                    :, cfg.model.tactile_input_shape : cfg.model.tactile_input_shape * 2
                ]
            )
            validation_target_images_stop_index = (
                cfg.image_size[0] * cfg.image_size[1]
            ) + cfg.model.tactile_input_shape * 2
            validation_target_images = torch.FloatTensor(
                validation_data[
                    :,
                    cfg.model.tactile_input_shape
                    * 2 : validation_target_images_stop_index,
                ]
            )
            validation_rgb_images_stop_index = (
                cfg.image_size[0] * cfg.image_size[1] * 3
            ) + validation_target_images_stop_index
            validation_rgb_images = torch.FloatTensor(
                validation_data[
                    :,
                    validation_target_images_stop_index:validation_rgb_images_stop_index,
                ]
            )
            validation_target_masks = torch.FloatTensor(
                validation_data[:, validation_rgb_images_stop_index:]
            )
        else:
            test_data = list(enumerate(data))
        # Shuffle test data.
        np.random.shuffle(test_data)
        test_indices, test_data = zip(*test_data)
        test_indices = np.squeeze(test_indices)
        test_data = np.asarray(test_data, dtype=np.float32)
        # Separate shuffled test data into tensors.
        test_tactile_input = torch.FloatTensor(
            test_data[:, : cfg.model.tactile_input_shape]
        )
        test_tactile_targets = torch.FloatTensor(
            test_data[
                :, cfg.model.tactile_input_shape : cfg.model.tactile_input_shape * 2
            ]
        )
        test_target_images_stop_index = (
            cfg.image_size[0] * cfg.image_size[1] * 1
        ) + cfg.model.tactile_input_shape * 2
        test_target_images = torch.FloatTensor(
            test_data[
                :, cfg.model.tactile_input_shape * 2 : test_target_images_stop_index
            ]
        )
        test_rgb_images_stop_index = (
            cfg.image_size[0] * cfg.image_size[1] * 3
        ) + test_target_images_stop_index
        test_rgb_images = torch.FloatTensor(
            test_data[:, test_target_images_stop_index:test_rgb_images_stop_index]
        )
        test_target_masks = torch.FloatTensor(test_data[:, test_rgb_images_stop_index:])
        if cfg.inference:
            return (
                test_tactile_targets,
                test_target_images,
                test_tactile_input,
                test_rgb_images,
                test_target_masks,
                test_indices,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            return (
                test_tactile_targets,
                test_target_images,
                test_tactile_input,
                test_rgb_images,
                test_target_masks,
                test_indices,
                validation_tactile_targets,
                validation_target_images,
                validation_tactile_input,
                validation_rgb_images,
                validation_target_masks,
                validation_indices,
            )
    # Preprocess training data
    else:
        if cfg.split_training_set:
            data, _ = train_test_split(data, cfg.training_split_ratio)
        # Shuffle training data
        data = list(enumerate(data))
        np.random.shuffle(data)
        indices, data = zip(*data)
        indices = np.squeeze(indices)
        data = np.array(data, dtype=np.float32)
        data = torch.FloatTensor(data)
        # Separate shuffled training data into tensors.
        tactile_input = torch.FloatTensor(data[:, : cfg.model.tactile_input_shape])
        tactile_targets = torch.FloatTensor(
            data[:, cfg.model.tactile_input_shape : cfg.model.tactile_input_shape * 2]
        )
        target_images_stop_index = (
            cfg.image_size[0] * cfg.image_size[1]
        ) + cfg.model.tactile_input_shape * 2
        target_images = torch.FloatTensor(
            data[:, cfg.model.tactile_input_shape * 2 : target_images_stop_index]
        )
        rgb_images_stop_index = (
            cfg.image_size[0] * cfg.image_size[1] * 3
        ) + target_images_stop_index
        rgb_images = torch.FloatTensor(
            data[:, target_images_stop_index:rgb_images_stop_index]
        )
        target_masks = torch.FloatTensor(data[:, rgb_images_stop_index:])
        return (
            scaler_images,
            scaler_rgb,
            scaler_reskin,
            tactile_targets,
            target_images,
            tactile_input,
            rgb_images,
            indices,
            mean_images,
            std_images,
            mean_reskin,
            std_reskin,
            target_masks,
        )


def _validating(loader, cfg, device, model, full_validation_loss_per_dimension=[]):
    epoch_loss_per_dimension = []
    criterion_sensor = losses[cfg.parameters.l_sensor]
    criterion_camera = losses[cfg.parameters.l_image]
    criterion_masks = losses[cfg.parameters.l_masks]
    criterion_masks_2 = losses[cfg.parameters.l_masks_2]
    criterion_code = losses[cfg.parameters.l_code]
    # initialize variables
    tactile_output = None
    loss = 0
    loss_images = 0
    loss_masks = 0
    loss_masks_acc = 0
    loss_tactile = 0
    for (
        batch_tactile_features,
        batch_tactile_targets,
        batch_target_images,
        batch_rgb_images,
        batch_target_masks,
        _,
    ) in loader:
        batch_tactile_targets = batch_tactile_targets.view(
            -1, cfg.model.tactile_input_shape
        ).to(device)
        batch_tactile_features = batch_tactile_features.view(
            -1, cfg.model.tactile_input_shape
        ).to(device)
        if cfg.model.cnn_images_encoder:
            batch_target_images = batch_target_images.view(
                -1, cfg.image_size[0], cfg.image_size[1], 1
            ).to(device)
            batch_rgb_images = batch_rgb_images.view(
                -1, cfg.image_size[0], cfg.image_size[1], 3
            ).to(device)
            batch_target_masks = batch_target_masks.view(
                -1, cfg.image_size[0], cfg.image_size[1], 1
            ).to(device)
            batch_target_images = batch_target_images.permute(0, 3, 1, 2)
            batch_target_masks = batch_target_masks.permute(0, 3, 1, 2)
        else:
            batch_target_images = batch_target_images.view(
                -1, cfg.image_size[0] * cfg.image_size[1] * 1
            ).to(device)
            batch_rgb_images = batch_rgb_images.view(
                -1, cfg.image_size[0] * cfg.image_size[1] * 3
            ).to(device)
            if not cfg.classification:
                batch_target_masks = batch_target_masks.view(
                    -1, cfg.image_size[0] * cfg.image_size[1]
                ).to(device)

        # infer model reconstructions and evaluate their losses
        ## initialize batch losses
        batch_loss_rgb = 0
        batch_loss_tactile = 0
        batch_loss_masks_acc = 0
        batch_loss_images = 0
        batch_loss = 0
        tactile_output, images_output, code_images = model(batch_target_masks)
        if cfg.constant_tactile_prediction:
            mean_tactile_reading = np.array(
                [
                    0.17954793,
                    -0.03041726,
                    0.18764104,
                    -0.09909063,
                    -0.05257506,
                    0.04022784,
                    -0.07302847,
                    -0.06282117,
                    0.17289482,
                    0.0272471,
                    -0.0390836,
                    -0.02754876,
                    -0.04583777,
                    -0.09837113,
                    -0.07878488,
                ]
            )
            bs = tactile_output.shape[0]
            tensor = np.tile(mean_tactile_reading, (bs, 1))
            tensor = tensor.reshape(bs, cfg.model.tactile_input_shape)
            tactile_output = torch.FloatTensor(tensor).to(device)
        batch_loss_tactile = (
            criterion_sensor(tactile_output, batch_tactile_targets)
            * cfg.tactile_loss_scale
        )

        # Calculate the number of dimensions (columns) in the tensors (num_features)
        num_dimensions = tactile_output.shape[1]
        # Instantiate the Mean Squared Error (MSE) loss
        to_be_stored_criterion = nn.MSELoss(reduction="none")
        # Calculate the loss separately for each dimension (feature) for each sample
        batch_loss_per_dimension = []
        for dim in range(num_dimensions):
            input_slice = tactile_output[
                :, dim
            ]  # Slice the input tensor along the dimension (feature)
            target_slice = batch_tactile_targets[
                :, dim
            ]  # Slice the target tensor along the dimension (feature)
            dimension_loss = (
                to_be_stored_criterion(input_slice, target_slice).cpu().numpy()
            )
            batch_loss_per_dimension.append(dimension_loss)
            mean_batch_loss_per_dimension = np.mean(batch_loss_per_dimension, axis=1)

        # postprocess outputs
        images_output_thresholded = torch.where(
            images_output > cfg.masks_threshold, 1.0, 0
        )
        # compute output configuration invariant losses
        if cfg.parameters.l_image == "masked_mse_loss":
            batch_loss_images = criterion_camera(
                images_output, batch_target_images, batch_target_masks
            )
        else:
            # flatten the output and target images
            images_output = images_output.view(
                -1, cfg.image_size[0] * cfg.image_size[1] * 1
            )
            batch_target_images = batch_target_images.view(
                -1, cfg.image_size[0] * cfg.image_size[1] * 1
            )
            batch_loss_images = criterion_camera(images_output, batch_target_images)
        if criterion_masks is no_loss:
            batch_loss_masks_acc = 0
        else:
            batch_loss_masks_acc = accuracy(
                images_output_thresholded.detach().cpu().numpy(),
                batch_target_masks.detach().cpu().numpy(),
            )

        # compute total batch loss
        batch_loss = batch_loss_tactile + batch_loss_images + batch_loss_rgb
        # compute accumulated gradients
        if not torch.isnan(batch_loss):
            # add the batch loss to epoch loss
            loss += batch_loss
            loss_images += batch_loss_images
            loss_masks_acc += batch_loss_masks_acc
            loss_tactile += batch_loss_tactile

        epoch_loss_per_dimension.append(mean_batch_loss_per_dimension)
    mean_epoch_loss_per_dimension = np.mean(epoch_loss_per_dimension, axis=0)
    full_validation_loss_per_dimension.append(mean_epoch_loss_per_dimension)
    # compute the epoch training loss
    if not torch.isnan(loss):
        loss = loss / len(loader)
        loss_images = loss_images / len(loader)
        loss_masks = loss_masks / len(loader)
        loss_tactile = loss_tactile / len(loader)
        loss_masks_acc = loss_masks_acc / len(loader)
    return (
        loss,
        loss_images,
        loss_masks_acc,
        loss_tactile,
        images_output,
        batch_rgb_images,
        batch_target_images,
        batch_target_masks,
        full_validation_loss_per_dimension,
    )


# write a function to make the value of each key in a dictionary either a string or a number, add the necessary new keys
def flatten_dict(d, parent_key="", sep="_"):
    # initialize the output dictionary
    items = []
    # loop through the dictionary
    for k, v in d.items():
        # initialize a new key
        new_key = parent_key + sep + k if parent_key else k
        # check if the value is a dictionary
        if isinstance(v, MutableMapping):
            # if it is, call the function recursively
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # if not, add it to the output dictionary
            items.append((new_key, v))
    # return the output dictionary
    return dict(items)


def train_touch_to_image(
    model,
    scaler_images,
    mean_images,
    std_images,
    scaler_rgb,
    train_loader,
    validation_loader,
    cfg,
    cfg_masks,
    device,
    parameters=None,
    autoML=False,
    model_folder=None,
):
    # initialize wandb
    config_dict = flatten_dict(cfg)
    masks_config_dict = flatten_dict(cfg_masks)
    combined_dict = config_dict.copy()  # Create a copy of the first dictionary
    combined_dict.update(masks_config_dict)  # Update it with the second dictionary
    wandb.init(project="touch-to-image", config=combined_dict)
    # tune cfg parameters
    test_objects_array = cfg.test_objects_names.split(",")
    test_objects_string = ""
    for i, object in enumerate(test_objects_array):
        if i < len(test_objects_array) - 1:
            test_objects_string += object + "_"
        else:
            test_objects_string += object
    # loss functions
    criterion_sensor = losses[cfg.parameters.l_sensor]
    criterion_camera = losses[cfg.parameters.l_image]
    criterion_masks = losses[cfg.parameters.l_masks]
    criterion_masks_2 = losses[cfg.parameters.l_masks_2]
    criterion_regularization = losses[cfg.parameters.l_regularization]
    criterion_code = losses[cfg.parameters.l_code]

    # create an optimizer object
    if not autoML:
        optimizer = optimizers[cfg.parameters.optimizer](
            model.parameters(), lr=cfg.parameters.lr
        )
    else:
        optimizer = optimizers[cfg.parameters.optimizer](
            model.parameters(), lr=parameters["lr"]
        )
        print(parameters["lr"])
    # output lists
    rgb_output = None
    tactile_output = None
    # unused training losses
    train_kl_loss_rgb = 0
    train_loss_rgb = 0
    for epoch in range(cfg.parameters.epochs):
        # epoch training losses
        loss = 0
        loss_images = 0
        loss_masks = 0
        loss_tactile = 0
        kl_loss_tactile = 0
        kl_loss_images = 0
        # training the model
        for (
            batch_tactile_features,
            batch_tactile_targets,
            batch_target_images,
            batch_rgb_images,
            batch_target_masks,
            _,
        ) in train_loader:
            batch_tactile_targets = batch_tactile_targets.view(
                -1, cfg.model.tactile_input_shape
            ).to(device)
            batch_tactile_features = batch_tactile_features.view(
                -1, cfg.model.tactile_input_shape
            ).to(device)
            if not cfg.model.cnn_images_encoder:
                batch_target_images = batch_target_images.view(
                    -1, cfg.image_size[0] * cfg.image_size[1] * 1
                ).to(device)
                batch_rgb_images = batch_rgb_images.view(
                    -1, cfg.image_size[0] * cfg.image_size[1] * 3
                ).to(device)
                if not cfg.classification:
                    batch_target_masks = batch_target_masks.view(
                        -1, cfg.image_size[0] * cfg.image_size[1]
                    ).to(device)
            else:
                batch_target_images = batch_target_images.view(
                    -1, cfg.image_size[0], cfg.image_size[1], 1
                ).to(device)
                batch_rgb_images = batch_rgb_images.view(
                    -1, cfg.image_size[0], cfg.image_size[1], 3
                ).to(device)
                batch_target_masks = batch_target_masks.view(
                    -1, cfg.image_size[0], cfg.image_size[1], 1
                ).to(device)
                batch_target_images = batch_target_images.permute(0, 3, 1, 2)
                batch_target_masks = batch_target_masks.permute(0, 3, 1, 2)

            # reset the gradients back to zero
            optimizer.zero_grad()
            (
                tactile_output,
                images_output,
                code_images,
            ) = model(batch_target_masks)
            # process the auxilary loss; tactile reconstruction loss
            train_loss_tactile = (
                criterion_sensor(tactile_output, batch_tactile_targets)
                * cfg.tactile_loss_scale
            )
            if cfg.parameters.kl_sensor:
                tactile_output_p = torch.softmax(tactile_output, -1)
                batch_tactile_targets_p = torch.softmax(batch_tactile_targets, -1)
                train_kl_loss_tactile = criterion_regularization(
                    torch.log(tactile_output_p), torch.log(batch_tactile_targets_p)
                )
            else:
                train_kl_loss_tactile = 0

            # compute the depth reconstruction loss and mask reconstruction loss
            if cfg.parameters.l_image == "masked_mse_loss":
                train_loss_images = criterion_camera(
                    images_output, batch_target_images, batch_target_masks
                )
            else:
                # flatten the output and target images
                images_output = images_output.view(
                    -1, cfg.image_size[0] * cfg.image_size[1] * 1
                )
                batch_target_images = batch_target_images.view(
                    -1, cfg.image_size[0] * cfg.image_size[1] * 1
                )
                train_loss_images = criterion_camera(images_output, batch_target_images)
            if cfg.parameters.kl_image:
                images_output_p = torch.softmax(images_output, -1)
                batch_target_images_p = torch.softmax(batch_target_images, -1)
                train_kl_loss_images = criterion_regularization(
                    torch.log(images_output_p), torch.log(batch_target_images_p)
                )
            else:
                train_kl_loss_images = 0

            # compute training and regularization losses
            train_kl_loss = (
                train_kl_loss_images + train_kl_loss_tactile + train_kl_loss_rgb
            )
            train_kl_loss = train_kl_loss * cfg.kl_loss_scale
            train_loss = train_loss_tactile + train_loss_images + train_loss_rgb
            # compute accumulated gradients
            if not torch.isnan(train_loss):
                if cfg.regularize:
                    train_loss += train_kl_loss
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the batch training loss to epoch loss
                loss += train_loss.item()
                loss_images += train_loss_images
                loss_tactile += train_loss_tactile
                if cfg.regularize:
                    kl_loss_images += train_kl_loss_images * cfg.kl_loss_scale
                    kl_loss_tactile += train_kl_loss_tactile * cfg.kl_loss_scale

        # compute the epoch training loss
        if not torch.isnan(train_loss):
            loss = loss / len(train_loader)
            loss_images = loss_images / len(train_loader)
            loss_tactile = loss_tactile / len(train_loader)
            kl_loss_images = kl_loss_images / len(train_loader)
            kl_loss_tactile = kl_loss_tactile / len(train_loader)

        # post process outputs
        images_output_thresholded = torch.where(
            images_output > cfg.masks_threshold, 1.0, 0
        )

        # display the epoch training loss
        print(
            "epoch : {}/{}, loss = {:.6f}".format(
                epoch + 1, cfg.parameters.epochs, loss
            )
        )
        if epoch % cfg.validation_epochs_step == 0:
            # model.eval()  # Set model to evaluation mode
            if epoch == 0:
                full_validation_loss_per_dimension = []
            with torch.no_grad():
                (
                    v_loss,
                    v_loss_images,
                    v_loss_masks_acc,
                    v_loss_tactile,
                    validation_images_output,
                    validation_rgb_targets,
                    validation_images_targets,
                    validation_masks_targets,
                    full_validation_loss_per_dimension,
                ) = _validating(
                    validation_loader,
                    cfg,
                    device,
                    model,
                    full_validation_loss_per_dimension,
                )
            # check if the loss is decreasing if not decrease the learning rate
            if epoch > 100:
                cfg.parameters.l_image = "no_loss"
                if loss_tactile - previous_tactile_loss > 0.01:
                    lr = lr * np.float(cfg.parameters.lr_decay_rate)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    # format print to display only 10 digits after the decimal point
                    print(
                        "loss increased from {:.10f} to {:.10f}".format(
                            previous_tactile_loss, loss_tactile
                        )
                    )
                    print("learning rate decreased to {:.10f}".format(lr))

                previous_tactile_loss = loss_tactile
            else:
                lr = cfg.parameters.lr
                previous_tactile_loss = loss_tactile
            print(epoch, "validation epoch number")
            validation_images_output = torch.where(
                validation_images_output > cfg.masks_threshold, 1.0, 0
            )
            # fig=plot_touch_to_image(cfg,None,validation_images_targets,validation_rgb_targets,validation_masks_targets,
            #     validation_images_output,None,None,validation_masks_output
            #     ,scaler_images,std_images,mean_images,scaler_rgb,validation_masks_output.shape[0])
            model.train()  # Set model back to training mode
            wandb.log(
                {
                    "total validation loss": v_loss,
                    "depth validation loss": v_loss_images,
                    "masks validation accuracy": v_loss_masks_acc,
                    "tactile validation loss": v_loss_tactile,
                    "total loss": loss,
                    "depth loss": loss_images,
                    "tactile loss": loss_tactile,
                    "masks loss": loss_masks,
                    "kl loss tactile": kl_loss_tactile,
                    "kl loss images": kl_loss_images,
                },
                step=epoch,
            )
            #    "current predictions":fig},step=epoch)

        else:
            wandb.log(
                {
                    "total loss": loss,
                    "depth loss": loss_images,
                    "tactile loss": loss_tactile,
                    "masks loss": loss_masks,
                    "kl loss tactile": kl_loss_tactile,
                    "kl loss images": kl_loss_images,
                },
                step=epoch,
            )
            # pass
    print(f"Final loss: {loss}")
    print(f"Final tactile validation loss: {v_loss_tactile}")
    np.save(
        f"{model_folder}/tactile_error_array.npy",
        np.asarray(full_validation_loss_per_dimension),
    )
    wandb.finish()
    return (
        model,
        batch_tactile_features,
        batch_target_images,
        batch_rgb_images,
        batch_target_masks,
        images_output,
        tactile_output,
        rgb_output,
        images_output_thresholded,
        v_loss_tactile,
    )


def plot_touch_to_image(
    cfg,
    tactile_test_examples,
    original_target_images,
    original_rgb_images,
    original_masks,
    image_reconstructions,
    rgb_reconstructions,
    tactile_reconstructions,
    masks_reconstructions,
    scaler_images,
    std_images,
    mean_images,
    scaler_rgb,
    N=5,
    show=False,
):
    with torch.no_grad():
        number = N
        fig = plt.figure(figsize=(20, 4))
        plt.gray()
        for index in range(number):
            # display original image data / uncomment for masked
            ax = plt.subplot(9, number, index + 1 + 2 * number)
            image_original = original_target_images[index].cpu().numpy().reshape(1, -1)
            try:
                image_original = scaler_images.inverse_transform(image_original)
            except:
                pass
            image_original = np.array(image_original, dtype=np.uint8).reshape(
                cfg.image_size[0], cfg.image_size[1]
            )
            image_original = cv2.equalizeHist(image_original)

            plt.imshow(image_original)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if index == number - 1:
                ax.text(60, 20, "original feed data")

            # display original mask
            ax = plt.subplot(9, number, index + 1 + 3 * number)
            mask_original = original_masks[index].cpu().numpy().reshape(1, -1)
            mask_original = np.array(mask_original, dtype=np.uint8).reshape(
                cfg.image_size[0], cfg.image_size[1]
            )
            plt.imshow(mask_original, vmin=0, vmax=1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if index == number - 1:
                ax.text(60, 20, "original mask data")

            ## display reconstructed images (masks or depth or masked depth)
            ax = plt.subplot(9, number, index + 1 + 4 * number)
            if not cfg.image_type == "masks":
                image_reconstruct = (
                    image_reconstructions[index].cpu().numpy().reshape(1, -1)
                )
                image_reconstruct = scaler_images.inverse_transform(image_reconstruct)
                image_reconstruct = (image_reconstruct * std_images) + mean_images
                image_reconstruct = np.array(image_reconstruct, dtype=np.uint8).reshape(
                    cfg.image_size[0], cfg.image_size[1]
                )
                image_reconstruct = cv2.equalizeHist(image_reconstruct)
                plt.imshow(image_reconstruct)
            else:
                mask_reconstruct = (
                    masks_reconstructions[index].cpu().numpy().reshape(1, -1)
                )
                mask_reconstruct = np.array(mask_reconstruct, dtype=np.uint8).reshape(
                    cfg.image_size[0], cfg.image_size[1]
                )
                plt.imshow(mask_reconstruct, vmin=0, vmax=1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if index == number - 1:
                ax.text(60, 20, "reconstructed feed data")

            ## display original rgb
            if original_rgb_images is not None:
                ax = plt.subplot(9, number, index + 1 + 7 * number)
                rgb_original = original_rgb_images[index].cpu().numpy().reshape(1, -1)
                rgb_original = scaler_rgb.inverse_transform(rgb_original)
                rgb_original = np.array(rgb_original, dtype=np.uint8).reshape(
                    cfg.image_size[0], cfg.image_size[1], 3
                )
                plt.imshow(rgb_original)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if index == number - 1:
                    ax.text(60, 20, "original rgb data")
        if show:
            plt.show()
        else:
            plt.close()
        return fig


def train_test_split(data, percentage):
    # train-test split
    data_train = data[: int(len(data) * percentage)]
    data_test = data[int(len(data) * percentage) :]
    return data_train, data_test


def evaluate(net, data_loader, device, infer, cfg):
    with torch.no_grad():
        (
            original_tactiles,
            original_rgb_images,
            original_target_images,
            original_target_masks,
            tactile_reconstructions,
            depth_reconstructions,
            rgb_reconstructions,
            masks_reconstructions,
            loader_indeces_list,
        ) = infer(net, cfg, device, data_loader)
    criterion = losses["mse"]
    criterion_images = losses["mse_reduced"]
    criterion_masks = losses["bce"]
    error = 0
    for (
        original_tactile_batch,
        original_target_image_batch,
        original_target_mask_batch,
        tactile_reconstruction_batch,
        depth_reconstruction_batch,
        masks_reconstruction_batch,
    ) in zip(
        original_tactiles,
        original_target_images,
        original_target_masks,
        tactile_reconstructions,
        depth_reconstructions,
        masks_reconstructions,
    ):
        with torch.no_grad():
            loss_images = criterion_images(
                depth_reconstruction_batch, original_target_image_batch
            )
            loss_masks = criterion_masks(
                masks_reconstruction_batch, original_target_mask_batch
            )
            error += (
                criterion(tactile_reconstruction_batch, original_tactile_batch)
                + torch.mean(loss_images * masks_reconstruction_batch)
                + torch.mean(loss_masks) / 5
            )
    error = np.float(error / len(data_loader))
    return error


def save_images(
    cfg,
    images,
    images_path,
    scaler_images=None,
    mean_images=None,
    std_images=None,
    type="depth",
):
    for index, image_reconstruction in enumerate(images):
        with torch.no_grad():
            image_reconstruction = image_reconstruction.cpu().numpy().reshape(1, -1)
            if type == "depth":
                # reversing the transform with training data statistics
                image_reconstruction = scaler_images.inverse_transform(
                    image_reconstruction
                )
                image_reconstruction = (image_reconstruction * std_images) + mean_images
            image_reconstruction = np.array(
                image_reconstruction, dtype=np.uint8
            ).reshape(cfg.image_size[0], cfg.image_size[1])
            image_reconstruction = Image.fromarray(
                image_reconstruction.astype(np.uint16)
            )
            image_reconstruction.save(
                f"{images_path}experiment_test_{type}_{index}.tif"
            )


def rename_model_folder(folder_path):
    directories = [
        dir
        for dir in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, dir))
    ]
    directories_int_list = [int(num) for num in directories]
    new_model_id = max(directories_int_list) + 1
    model_folder = f"{folder_path}/{new_model_id}"
    return model_folder, new_model_id
