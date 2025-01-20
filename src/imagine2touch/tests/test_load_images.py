import numpy as np
from PIL import Image
import re
import os
import sys
import operator


# https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("pass path to dir experiments")
        exit()

    path = sys.argv[1]
    regex = re.compile("experiment_._depth_.*")
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if regex.match(file):
                images.append(path + "/" + file)
    target_images = np.array(
        [
            cropND(np.array(Image.open(fname), dtype=float), (48, 48, 3))
            for fname in images
        ],
        dtype=list,
    )
    print(target_images.shape)
