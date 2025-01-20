from PIL import Image
import numpy as np
from argparse import ArgumentParser
import cv2

if __name__ == "__main__":
    "In this script dark is always closer to the camera"
    parser = ArgumentParser(description=".tiff images visualizer for depth and mask")
    parser.add_argument("images_path", help="path to the images directory")
    parser.add_argument("n_images")
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    for i in range(1, int(args.n_images)):
        # modify file names according to directory cconvention
        file_name = str(args.experiment_name) + "_" + str(i) + ".tif"
        file_name = args.images_path + "/" + file_name
        img = Image.open(file_name)
        img = np.array(img, dtype=np.uint8)
        img = cv2.equalizeHist(img)
        # display ones image as white; heurestic
        if np.sum(img.flatten()) == 48 * 48:
            img = np.ones((48, 48, 3)) * 255
        # invert masks heurestic
        if len(np.unique(img)) <= 2:
            img = cv2.bitwise_not(img)
        cv2.imshow("img", img)
        cv2.waitKey(0)
