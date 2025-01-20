import sys
from scipy.optimize import minimize, rosen, rosen_der
from robot_io.utils.utils import pos_orn_to_matrix, matrix_to_pos_orn, matrix_to_orn
import numpy as np
from src.imagine2touch.utils.utils import (
    inverse_transform,
    WORLD_IN_ROBOT,
    search_folder,
)
from omegaconf import OmegaConf
import hydra
import os

# script configurations
OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
hydra.initialize("./cfg", version_base=None)
repository_directory = search_folder("/", "imagine2touch")
cfg = hydra.compose("wrist.yaml")
starting_corner_in_world = [
    float(num) for num in cfg.starting_corner_in_world.split(",")
]
OmegaConf.register_new_resolver("quarter_pi", lambda x: np.pi / 4)

T_wcamera_in_marker = np.load(
    f"{repository_directory}/src/imagine2touch/utils/utils_data/wcamera_marker_transforms.npy",
    allow_pickle=True,
)
T_tcp_in_robot = np.load(
    f"{repository_directory}/src/imagine2touch/utils/utils_data/robot_tcp_transforms.npy",
    allow_pickle=True,
)
T_wcamera_marker_filtered = T_wcamera_in_marker
j = 0
indeces = []
for i, T in enumerate(T_wcamera_in_marker):
    if T is None:
        indeces.append(i)
c_T_ms = np.delete(T_wcamera_marker_filtered, indeces, axis=0)
r_T_hs = np.delete(T_tcp_in_robot, indeces, axis=0)


def pose_loss(a, b):
    a_T_b = inverse_transform(a).dot(b)
    rel_pos, rel_orn = matrix_to_pos_orn(a_T_b)
    return np.sqrt((rel_pos**2).sum()) * 50 + np.sqrt((rel_orn**2).sum())


def fun(x):
    # rot_matrix = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # cam_orn = matrix_to_orn(rot_matrix)
    # convert from rotation matrix to quaternion
    h_T_c = pos_orn_to_matrix(x[:3], x[3:7])
    r_T_m = pos_orn_to_matrix(x[7:10], x[10:14])
    r_T_ms = [r_T_h.dot(h_T_c).dot(c_T_m) for r_T_h, c_T_m in zip(r_T_hs, c_T_ms)]
    loss = np.mean([pose_loss(r_T_ms[i], r_T_m) for i in range(len(r_T_ms))])
    return loss


bnds = np.ones((14, 2))
bnds[:, 0] = -1
print(bnds)


init = np.ones(14) * 0.001
r_T_m_pose, r_T_m_orn = matrix_to_pos_orn(WORLD_IN_ROBOT)
init[7:10] = r_T_m_pose
init[10:14] = r_T_m_orn
# rot_matrix = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# cam_orn = matrix_to_orn(rot_matrix)
print(f"func(init)={fun(init)}")
res = minimize(
    fun, init, method="SLSQP", bounds=bnds, tol=1e-8, options={"maxiter": 10000}
)

wcamera_in_tcp = pos_orn_to_matrix(res.x[:3], res.x[3:7])
print(pos_orn_to_matrix(res.x[7:10], res.x[10:14]), "robot in marker")
print(wcamera_in_tcp, "wrist camera in tcp")
np.save(
    f"{cfg.save_directory}/wcamera_tcp_transform", wcamera_in_tcp, allow_pickle=True
)
# save pos_orn_
