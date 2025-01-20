from pyparsing import original_text_for
from torchvision.io.image import read_image
from torchvision.models.segmentation import (
    fcn_resnet50,
    deeplabv3_mobilenet_v3_large,
    FCN_ResNet50_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)
from torchvision.transforms.functional import to_pil_image
import torchvision
import numpy as np
import cv2
import torch


def post_process_mask(mask, img):
    "converts a segmentation network output mask to an rgb numpy array | img: reference image"
    segmentation_mask = to_pil_image(mask)
    segmentation_mask = torchvision.transforms.functional.resize(
        segmentation_mask, (img.size()[1], img.size()[2])
    )
    segmentation_mask = np.array(segmentation_mask)
    segmentation_mask = np.expand_dims(np.array(segmentation_mask), 0)
    segmentation_mask1 = np.vstack((segmentation_mask, segmentation_mask))
    segmentation_mask2 = np.vstack((segmentation_mask1, segmentation_mask))
    segmentation_mask2 = np.moveaxis(segmentation_mask2, 0, -1)
    return segmentation_mask2


img = read_image("../bus.jpeg")
img = img[:, 432:690, 715:1115]
# print (img.size())
# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
# weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
# weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
# model = deeplabv3_mobilenet_v3_large(weights=weights)
# model=lraspp_mobilenet_v3_large(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()


# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
print(class_to_idx)
# print(normalized_masks.shape)
# mask = normalized_masks[0, class_to_idx["diningtable"]]
mask2 = normalized_masks[0, class_to_idx["__background__"]]

# segmentation_mask=post_process_mask(mask,img)
mask2 = 255 - mask2
segmentation_mask2 = post_process_mask(mask2, img)

img = torch.permute(img, (1, 2, 0))
original = np.array(img)
# print(original.shape)
# rgb = cv2.bitwise_and(original,segmentation_mask)

print(original.shape)
print(segmentation_mask2.shape)

# (bla,binary)=cv2.threshold(segmentation_mask2,60,255,cv2.THRESH_BINARY)
rgb = cv2.bitwise_and(original, segmentation_mask2)
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
(bla, binary) = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
cv2.imshow("mask applied grayscale", gray)
cv2.imshow("mask applied thresholded", binary)
# cv2.imshow("mask thresholded",binary)
cv2.imshow("original", original)
cv2.imshow("mask", segmentation_mask2)
cv2.waitKey(0)
