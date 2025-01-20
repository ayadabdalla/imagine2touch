import os
import open3d as o3d
from argparse import ArgumentParser
import hydra
from omegaconf import OmegaConf

from src.imagine2touch.utils.utils import segment_point_cloud

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("save_pcds.yaml")
    pcd_old = o3d.io.read_point_cloud(
        f"{cfg.save_directory}/{cfg.object_name}/{cfg.object_name}_combined.pcd"
    )
    pcd = segment_point_cloud(
        pcd_old,
        minz=-1000,
        maxz=1000,
        minx=-1000,
        maxx=1000,
        miny=-1000,
        maxy=1000,
        statistical_filter=False,
        voxel=False,
        radius_filter=True,
    )
    if cfg.filter_overwrite:
        o3d.io.write_point_cloud(
            f"{cfg.save_directory}/{cfg.object_name}/{cfg.object_name}_combined.pcd",
            pcd,
        )
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_old)
    vis.run()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
