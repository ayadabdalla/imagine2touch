import sys
import numpy as np
from PIL import Image
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
outlier_rgb = np.ones((48, 48, 3), np.uint8) * 255
outlier_rgb_im = Image.fromarray(outlier_rgb)
outlier_rgb_im.save(f"{dir_path}/outlier.png")

outlier_dep = np.ones((48, 48), np.uint16)
outlier_dep_im = Image.fromarray(outlier_dep)
outlier_dep_im.save(f"{dir_path}/outlier.tif")
