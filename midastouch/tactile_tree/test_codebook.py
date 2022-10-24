# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Loads object codebook and performs nearest neighbor queries
"""

from os import path as osp
import dill as pickle
import torch

from midastouch.viz.helpers import viz_query_target_poses_on_mesh
import trimesh
from midastouch.modules.mesh import sample_poses_on_mesh
from midastouch.modules.misc import DIRS
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    expt_cfg, tcn_cfg, tdn_cfg = cfg.expt, cfg.tcn, cfg.tdn
    obj_model = expt_cfg.obj_model
    codebook_path = osp.join(DIRS["trees"], obj_model, "codebook.pkl")
    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")

    if osp.exists(codebook_path):
        with open(codebook_path, "rb") as pickle_file:
            codebook = pickle.load(pickle_file)

    mesh = trimesh.load(obj_path)

    num_pose = 5
    query_poses = sample_poses_on_mesh(mesh=mesh, num_samples=num_pose, edges=False)
    query_poses = torch.from_numpy(query_poses)

    target_poses, _, _ = codebook.SE3_NN(query_poses)
    viz_query_target_poses_on_mesh(
        mesh_path=obj_path, query_pose=query_poses, target_poses=target_poses
    )


if __name__ == "__main__":
    main()
