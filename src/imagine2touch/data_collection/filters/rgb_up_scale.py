from PIL import Image
import os
import hydra
from src.imagine2touch.utils.utils import search_folder

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
            + "/rgb"
        )
        # get all files in path
        files = os.listdir(path)
        for file in files:
            if file.endswith(".png"):
                image = Image.open(path + "/" + file)
                if image.size[0] < 48 or image.size[1] < 48:
                    upscaled_image = image.resize((60, 60), resample=Image.BICUBIC)
                    upscaled_image.save(path + "/" + file)
                    i += 1
            print(f"Upscaled {i} images")
