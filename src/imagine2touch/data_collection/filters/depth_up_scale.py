from PIL import Image
import os
import cv2
import hydra
from src.imagine2touch.utils.utils import search_folder

module_dir = os.path.dirname(__file__)  # get current directory
files = os.listdir(module_dir)

if __name__ == "__main__":
    # script meta data, configuration and constants
    start_path = "/"
    hydra.initialize("../cfg", version_base=None)
    cfg = hydra.compose("collection.yaml")
    cfg.repository_directory = search_folder(start_path, cfg.repository_directory)
    cfg.experiment_directory = f"{cfg.repository_directory}/{cfg.experiment_directory}"
    objects_array = cfg.object_name.split(",")
    i = 0
    for object_name in objects_array:
        path = (
            cfg.experiment_directory
            + "/"
            + object_name
            + "/"
            + object_name
            + "_images"
            + "/depth"
        )
        files = os.listdir(path)
        for file in files:
            if file.endswith(".tif"):
                image = cv2.imread(path + "/" + file, cv2.IMREAD_UNCHANGED)
                if image.shape[0] < 48 or image.shape[1] < 48:
                    while image.shape[0] < 48 or image.shape[1] < 48:
                        image = cv2.resize(
                            image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                        )
                    cv2.imwrite(path + "/" + file, image)
                    i += 1
        print(f"Upscaled {i} images")
