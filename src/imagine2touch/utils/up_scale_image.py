# rgb
from PIL import Image

# # Open the image
# TODO::add configuration for path
image = Image.open(
    "/export/home/ayada/reskin_ws/src/imagine2touch/touch2image/reskin/data_collection/data/square_v_2/square_v_2_images/rgb/experiment_8_rgb_23.png"
)

# Upscale the image by a factor of 2
upscaled_image = image.resize((60, 60), resample=Image.BICUBIC)

# Save the upscaled image
# TODO::add configuration for save
upscaled_image.save(
    "/export/home/ayada/reskin_ws/src/imagine2touch/touch2image/reskin/data_collection/data/square_v_2/square_v_2_images/rgb/experiment_8_rgb_23.png"
)

# depth
import cv2

# # Load the depth image
# depth_image = cv2.imread(
#     "/export/home/ayada/reskin_ws/src/imagine2touch/touch2image/reskin/data_collection/data/square_v_2/square_v_2_images/depth/experiment_8_depth_23.tif",
#     cv2.IMREAD_UNCHANGED,
# )

# # Upscale the image using bicubic interpolation
# upscaled_depth_image = cv2.resize(
#     depth_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
# )

# # Save the upscaled image as a new .tif file
# cv2.imwrite(
#     "/export/home/ayada/reskin_ws/src/imagine2touch/touch2image/reskin/data_collection/data/square_v_2/square_v_2_images/depth/experiment_8_depth_23.tif",
#     upscaled_depth_image,
# )
print("done")
