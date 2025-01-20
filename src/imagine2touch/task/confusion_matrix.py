import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import hydra
from omegaconf import OmegaConf
import natsort


# load touchtoimage distances parallelogram
# create an enum to map the class names to their corresponding index

dir_path = os.path.dirname(os.path.realpath(__file__))
OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
hydra.initialize("./conf", version_base=None)
cfg = hydra.compose("results.yaml")
objects_names = cfg.objects_names.split(",")
objects_names = natsort.natsorted(objects_names)

map = dict()
for i, object_name in enumerate(objects_names):
    map[i] = object_name
num_classes = len(map)


predicted_labels = np.array([])
for label in map.values():
    predicted_label = np.load(
        f"{cfg.results_directory}/{cfg.data}_distances_{label}_1.npy"
    )
    predicted_label = np.argmax(predicted_label, axis=1)
    predicted_labels = np.concatenate((predicted_labels, predicted_label), axis=0)


# create an array of the true labels with 10 elements for each index
true_labels = np.zeros(predicted_labels.shape[0])

true_labels[0 : int(predicted_labels.shape[0] / num_classes)] = 0
true_labels[
    int(predicted_labels.shape[0] / num_classes) : int(
        2 * predicted_labels.shape[0] / num_classes
    )
] = 1
true_labels[
    int(2 * predicted_labels.shape[0] / num_classes) : int(
        3 * predicted_labels.shape[0] / num_classes
    )
] = 2

print(predicted_labels)
print(true_labels)


# Create confusion matrix
class_true_labels = [label for label in true_labels]
class_predicted_labels = [label for label in predicted_labels]
print(class_true_labels)
print(class_predicted_labels)
cm = confusion_matrix(class_true_labels, class_predicted_labels)
print(cm.shape)

# Create a heatmap for the overall confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[f"{map[class_id]}" for class_id in range(num_classes)],
    yticklabels=[f"{map[class_id]}" for class_id in range(num_classes)],
    cbar_kws={"format": "%d", "ticks": range(0, np.max(cm) + 1, 1)},
)
# put the x-ticks at the top
plt.gca().xaxis.tick_top()
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title(f"{cfg.dataset} Confusion Matrix - {cfg.title}")
plt.show()
