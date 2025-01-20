import numpy as np
from torch.utils.data import Dataset
import numpy as np
import re
import sys
import os
import natsort


class Touch2imageSet(Dataset):
    def __init__(
        self, data, targets, target_images=None, rgb=None, masks=None, task=False
    ):
        self.data = data
        self.targets = targets
        self.target_images = target_images
        self.rgb = rgb
        self.task = task
        self.masks = masks

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        m = self.masks[index]
        if not self.task:
            z = self.target_images[index]
            l = self.rgb[index]
            return x, y, z, l, m, index
        return x, y, m, index

    def __len__(self):
        return len(self.data)


# TODO: write a function for deining true label of shape

# def get_force(path):
#     regex = re.compile('.*_experiment_*')
#     experiments=[]
#     for root, dirs, files in os.walk(path):
#         for dir in dirs:
#             if regex.match(dir):
#                 experiments.append(dir)
#     force = np.load(path+'/'+experiments[0]+'/force', mmap_mode='r')
#     if len(experiments) == 1:
#         return force * -1
#     for counter,experiment in enumerate(experiments):
#         if counter == 0:
#             continue
#         else:
#             force_i = np.load(path+'/'+experiment+'/force', mmap_mode='r')
#             force = np.hstack((force,force_i))
#     force = force.reshape((force.shape[0],1)) * -1
#     return force


# def get_pose(path):
#     regex = re.compile('.*_experiment_*')
#     experiments=[]
#     for root, dirs, files in os.walk(path):
#         for dir in dirs:
#             if regex.match(dir):
#                 experiments.append(dir)
#     pose = np.load(path+'/'+experiments[0]+'/pose', mmap_mode='r')
#     #for shapes experiment
#     zeros = np.zeros((pose.shape[0],1))
#     ones = np.ones((pose.shape[0],1))
#     if(experiments[0][0]=='T'):
#         pose = np.hstack((ones,zeros,zeros,zeros,zeros))
#     elif (experiments[0][0]=='C'):
#         pose = np.hstack((zeros,ones,zeros,zeros,zeros))
#     elif (experiments[0][0]=='V'):
#         pose = np.hstack((zeros,zeros,ones,zeros,zeros))
#     elif (experiments[0][0]=='M'):
#         pose = np.hstack((zeros,zeros,zeros,ones,zeros))
#     else:
#         pose = np.hstack((zeros,zeros,zeros,zeros,ones))
#     ##################################
#     if len(experiments) == 1:
#         return pose #*1000
#     for counter,experiment in enumerate(experiments):
#         if counter == 0:
#             continue
#         else:
#             pose_i = np.load(path+'/'+experiment+'/pose', mmap_mode='r')
#             zeros_i = np.zeros((pose_i.shape[0],1)) #for shapes experiment
#             ones_i = np.ones((pose_i.shape[0],1)) #for shapes experiment
#             #for shapes experiment
#             if(experiment[0]=='T'):
#                 pose_i = np.hstack((ones_i,zeros_i,zeros_i,zeros_i,zeros_i))
#             elif (experiment[0]=='C'):
#                 pose_i = np.hstack((zeros_i,ones_i,zeros_i,zeros_i,zeros_i))
#             elif (experiment[0]=='V'):
#                 pose_i = np.hstack((zeros_i,zeros_i,ones_i,zeros_i,zeros_i))
#             elif (experiment[0]=='M'):
#                 pose_i = np.hstack((zeros_i,zeros_i,zeros_i,ones_i,zeros_i))
#             else:
#                 pose_i = np.hstack((zeros_i,zeros_i,zeros_i,zeros_i,ones_i))
#             #########################
#             pose = np.vstack((pose,pose_i))
#     return pose #*1000


def get_reskin_reading(
    path,
    binary,
    differential_signal=False,
    ambient_every_reading=False,
    ambient_aggregated=False,
):
    regex = re.compile("experiment_.*_reskin$")
    experiments = []
    for p in path:
        for root, dirs, files in os.walk(p):
            for file in files:
                if regex.match(file):
                    experiments.append(p + "/" + file)
    first_path = path[0]
    experiments = np.asarray(natsort.natsorted(experiments))

    # store and pre-process first experiment
    reskin_reading = np.load(experiments[0], allow_pickle=True)
    reskin_reading = np.squeeze(reskin_reading)[
        :, 2
    ]  # extract lists of magnetometers values and temperatures as array of lists
    reskin_reading = list(reskin_reading)  # convert to list of lists then to nd array
    reskin_reading = np.asarray(reskin_reading)
    if binary == "False":
        reskin_reading = np.delete(
            reskin_reading, [3, 7, 11, 15, 19], 1
        )  # eliminate temperatures
    else:
        reskin_reading = np.delete(reskin_reading, [0, 4, 8, 12, 16], 1)
    if differential_signal:
        reskin_reading = subtract_ambient(
            reskin_reading,
            get_ambient_data(
                first_path, binary, experiments[0] + "_ambient", ambient_aggregated
            ),
            ambient_every_contact=ambient_every_reading,
        )
    else:
        pass

    if len(experiments) == 1:
        return reskin_reading

    for counter, experiment in enumerate(experiments):
        if counter == 0:
            continue
        else:
            reskin_reading_i = np.load(experiment, allow_pickle=True)
            if reskin_reading_i.shape[0] > 1:
                reskin_reading_i = np.squeeze(reskin_reading_i)[
                    :, 2
                ]  # extract lists of magnetometers values and temperatures as array of lists
            else:
                reskin_reading_i = np.squeeze(reskin_reading_i)[2]
            reskin_reading_i = list(
                reskin_reading_i
            )  # convert to list of lists then to nd array
            reskin_reading_i = np.asarray(reskin_reading_i).reshape(-1, 20)
            if binary == "False":
                reskin_reading_i = np.delete(
                    reskin_reading_i, [3, 7, 11, 15, 19], 1
                )  # eliminate temperatures
            else:
                reskin_reading_i = np.delete(
                    reskin_reading_i, [0, 4, 8, 12, 16], 1
                )  # eliminate temperatures
            if differential_signal:
                reskin_reading_i = subtract_ambient(
                    reskin_reading_i,
                    get_ambient_data(
                        path, binary, experiment + "_ambient", ambient_aggregated
                    ),
                    ambient_every_contact=ambient_every_reading,
                )
            else:
                pass
            reskin_reading = np.vstack((reskin_reading, reskin_reading_i))
    return reskin_reading


def get_ambient_data(path, binary, exp=None, aggregated=False):
    experiments = []
    regex = re.compile("experiment_.*_reskin_ambient")
    if exp is None:
        for p in path:
            for root, dirs, files in os.walk(p):
                for file in files:
                    if regex.match(file):
                        experiments.append(p + "/" + file)
        experiments = np.asarray(natsort.natsorted(experiments))
    else:
        experiments.append(exp)
    # store and pre-process first experiment
    ambient_reading = np.load(experiments[0], allow_pickle=True)
    processed_ambient_readings = []
    if ambient_reading.shape[1] == 15:
        aggregated = True
    else:
        aggregated = False
    if not aggregated:
        for (
            readings
        ) in (
            ambient_reading
        ):  # average sets of ambient readings if several measurements taken at once
            readings = np.squeeze(np.array(readings, dtype=object))[
                :, 2
            ]  # extract lists of magnetometers values and temperatures as array of lists
            readings = list(readings)  # convert to list of lists then to nd array
            readings = np.asarray(readings).reshape(-1, 20)
            if binary == "False":
                readings = np.delete(
                    readings, [3, 7, 11, 15, 19], 1
                )  # eliminate temperatures
            else:
                readings = np.delete(
                    readings, [0, 4, 8, 12, 16], 1
                )  # eliminate temperatures
            processed_ambient_readings.append(np.mean(readings, axis=0))
    else:
        processed_ambient_readings = ambient_reading
    processed_ambient_readings = np.asarray(processed_ambient_readings)
    # processed_ambient_readings = np.delete(processed_ambient_readings, (0), axis=0) #because first reading is irrelevant

    if len(experiments) == 1:
        return processed_ambient_readings
    for counter, experiment in enumerate(experiments):
        if counter == 0:
            continue
        else:
            ambient_reading_i = np.load(experiment, allow_pickle=True)
            processed_ambient_readings_i = []
            if not aggregated:
                for readings_i in ambient_reading_i:  # average sets of ambient readings
                    readings_i = np.squeeze(readings_i)[
                        :, 2
                    ]  # extract lists of magnetometers values and temperatures as array of lists
                    readings_i = list(
                        readings_i
                    )  # convert to list of lists then to nd array
                    readings_i = np.asarray(readings_i)
                    if binary == "False":
                        readings_i = np.delete(
                            readings_i, [3, 7, 11, 15, 19], 1
                        )  # eliminate temperatures
                    else:
                        readings_i = np.delete(
                            readings_i, [0, 4, 8, 12, 16], 1
                        )  # eliminate temperatures
                    processed_ambient_readings_i.append(np.mean(readings_i, axis=0))
            else:
                processed_ambient_readings_i = ambient_reading_i
            processed_ambient_readings_i = np.asarray(processed_ambient_readings_i)
            processed_ambient_readings = np.vstack(
                (processed_ambient_readings, processed_ambient_readings_i)
            )
    return processed_ambient_readings


def subtract_ambient(
    contact, ambient, ambient_every_contact=False, handle_negative=False
):
    """subtract ambient readings either for each batch of 10 readings or for each reading
    ambient: array of ambient readings
    contact: array of contact readings"""
    if ambient.shape[0] == contact.shape[0]:
        ambient_every_contact = True
    else:
        ambient_every_contact = False
    if not ambient_every_contact:
        i = 0
        k = 0
        for j, reading in enumerate(contact):
            if handle_negative:
                contact[j] = np.abs(reading - ambient[i])
            else:
                contact[j] = reading - ambient[i]
            if j - k == 10 - 1:
                if i < ambient.shape[0] - 1:
                    i = i + 1  # move to next ambient reading after 10 contacts
                    k = j
    else:
        for j, reading in enumerate(contact):
            if handle_negative:
                contact[j] = np.abs(reading - ambient[j])
            else:
                contact[j] = reading - ambient[j]
    return contact


def prepare_reskin_data(
    path,
    binary,
    mean=None,
    std=None,
    differential_signal=False,
    ambient_every_reading=False,
    ambient_aggregated=False,
    standardize=True,
):
    """
    prepare and scale/normalize reskin data
    """
    reskin_reading = get_reskin_reading(
        path,
        binary,
        differential_signal=differential_signal,
        ambient_every_reading=ambient_every_reading,
        ambient_aggregated=ambient_aggregated,
    )
    input = reskin_reading
    if mean is not None or std is not None:
        pass
    else:
        mean = np.mean(reskin_reading)
        std = np.std(input)
    if standardize:
        input = reskin_reading - mean
        input = input / std

    # targets = np.concatenate((np.reshape(pose,(-1,2)),np.reshape(force,(-1,1))),axis =1)
    targets = input

    return input, targets, mean, std
    # return torch.FloatTensor(input) , torch.FloatTensor(targets)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("pass path to dir experiments")
        exit()
    experiments_path = sys.argv[1]
    input, targets = prepare_reskin_data(experiments_path, True)
    print(input.shape)
