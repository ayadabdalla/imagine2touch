import numpy as np


def get_crop_indeces(hom_point_in_cam, tcp_in_c, intrinsic, resolution):
    sr = 0.0085  # sensor_radius
    first_corner = np.array([sr, sr, 0, 1])
    second_corner = np.array([sr, -sr, 0, 1])
    third_corner = np.array([-sr, sr, 0, 1])
    fourth_corner = np.array([-sr, -sr, 0, 1])
    S_in_tcp = np.vstack(
        (
            np.vstack((first_corner, second_corner)),
            np.vstack((third_corner, fourth_corner)),
        )
    )  # tcp space
    tcp_in_c[:, 3] = [0, 0, 0, 1]
    S_in_cam = tcp_in_c.dot(S_in_tcp.T)
    hom_point_in_cam[3] = 0
    addition = S_in_cam.T + hom_point_in_cam  # camera space
    crop = intrinsic.dot(
        addition.T
    )  # image space (Transpose to apply the transformation to multiple column vectors)
    print(f"crop before resolution {crop}")
    crop[1, :] = 1 - crop[1, :]
    crop = np.multiply(
        resolution, crop[:2, :]
    )  # 2*1 multiplies 2*4, broadcast for all columns
    return crop, addition.T, S_in_cam


if __name__ == "__main__":
    tcp_in_c = np.eye(4, 4)
    z = 1
    fx = 1
    fy = 1
    cx = 1
    cy = 1
    intrinsic = np.array([[fx / z, 0, cx, 0], [0, fy / z, cy, 0], [0, 0, 1, 0]])
    point = [0.3, 0.1, 0.1, 1]
    print(
        get_crop_indeces(
            point, tcp_in_c, intrinsic, np.reshape(np.array([100, 200]), (-1, 1))
        )[0]
    )
    print(
        get_crop_indeces(
            point, tcp_in_c, intrinsic, np.reshape(np.array([100, 200]), (-1, 1))
        )[1]
    )
    print(
        get_crop_indeces(
            point, tcp_in_c, intrinsic, np.reshape(np.array([100, 200]), (-1, 1))
        )[2]
    )
