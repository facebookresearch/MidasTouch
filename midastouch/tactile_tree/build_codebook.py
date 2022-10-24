# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Generates codebook from scratch by randomly sampling object meshes 
Run: python midastouch/tactile_tree/build_codebook.py expt.obj_model=005_tomato_soup_can expt.codebook_size=50000
"""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
from os import path as osp
import numpy as np

from midastouch.contrib.tdn_fcrn.tdn import TDN
from midastouch.contrib.tcn_minkloc.tcn import TCN
import hydra
from omegaconf import DictConfig
from midastouch.modules.misc import DIRS, get_device
from midastouch.modules.mesh import sample_poses_on_mesh
from midastouch.render.digit_renderer import digit_renderer
from tqdm import tqdm
import trimesh
import torch

from midastouch.tactile_tree.tactile_tree import tactile_tree
import dill as pickle


@hydra.main(config_path="../config", config_name="config")
def build_codebook(cfg: DictConfig, image_embedding=False):
    expt_cfg, tcn_cfg, tdn_cfg = cfg.expt, cfg.tcn, cfg.tdn

    num_samples = expt_cfg.codebook_size
    obj_model = expt_cfg.obj_model

    print(
        f"object: {obj_model}, codebook size: {num_samples}, image embedding: {image_embedding}"
    )

    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")

    if not image_embedding:
        tree_path = osp.join(DIRS["trees"], obj_model, "codebook.pkl")
    else:
        tree_path = osp.join(DIRS["trees"], obj_model, "image_codebook.pkl")

    tac_render = digit_renderer(cfg=tdn_cfg.render, obj_path=obj_path, randomize=True)
    digit_tcn = TCN(tcn_cfg)
    digit_tdn = TDN(tdn_cfg, bg=tac_render.get_background(frame="gel"))

    device = get_device(cpu=False)

    mesh = trimesh.load(obj_path)

    # Generate tree samples
    print("Generating {} samples".format(num_samples))
    samples = sample_poses_on_mesh(mesh=mesh, num_samples=num_samples, edges=False)

    """Get multimodal embeddings"""
    #####################################
    batch_size = 100
    num_batches = num_samples // batch_size
    num_batches = 1 if num_batches == 0 else num_batches

    pbar = tqdm(total=num_batches)
    gelposes, camposes, embeddings = (
        torch.zeros((num_samples, 4, 4)),
        torch.zeros((num_samples, 4, 4)),
        torch.zeros(
            (num_samples, digit_tcn.params.model_params.output_dim), dtype=torch.double
        ),
    )
    # heightmaps, masks, images = [None] * num_samples, [None] * num_samples, [None] * num_samples

    # heightmap_rmse, contact_mask_iou = [], []
    for i in range(num_batches):
        i_range = (
            np.array(range(i * batch_size, num_samples))
            if (i == num_batches - 1)
            else np.array(range(i * batch_size, (i + 1) * batch_size))
        )
        h, cm, tactileImages, campose, gelpose = tac_render.render_sensor_poses(
            samples[i_range, :, :], num_depths=1
        )
        gelposes[i_range, :] = torch.from_numpy(gelpose).float()
        camposes[i_range, :] = torch.from_numpy(campose).float()

        est_heightmaps, est_masks = [], []

        if not image_embedding:
            for j, image in enumerate(tactileImages):
                est_h = digit_tdn.image2heightmap(image)  # expensive
                est_c = digit_tdn.heightmap2mask(est_h)
                est_heightmaps.append(est_h)
                est_masks.append(est_c)

                # gt_h, gt_c = h[j], cm[j]
                # error_heightmap = np.abs(est_h - gt_h) * pixmm             # Get pixelwise RMSE in mm, and IoU of the contact masks
                # heightmap_rmse.append(np.sqrt(np.mean(error_heightmap**2)))
                # intersection = np.sum(np.logical_and(gt_c, est_c))
                # contact_mask_iou.append(intersection/(np.sum(est_c) + np.sum(gt_c) - intersection))
            tactile_code = digit_tcn.cloud_to_tactile_code(
                tac_render, est_heightmaps, est_masks
            )
        else:
            tactile_code = torch.zeros(
                (len(tactileImages), digit_tcn.params.model_params.output_dim),
                dtype=torch.double,
            )
            for j, image in enumerate(tactileImages):
                tactile_code[j, :] = digit_tdn.image2embedding(image)

        embeddings[i_range, :] = tactile_code.cpu()
        pbar.update(1)
    pbar.close()

    # heightmap RMSE (mm), contact mask IoU [0 - 1]
    # heightmap_rmse = [x for x in heightmap_rmse if str(x) != 'nan']
    # contact_mask_iou = [x for x in contact_mask_iou if str(x) != 'nan']
    # heightmap_rmse = sum(heightmap_rmse) / len(heightmap_rmse)
    # contact_mask_iou = sum(contact_mask_iou) / len(contact_mask_iou)
    # error_file = open(osp.join(tree_path,'{}_error.txt'.format(method)),'w')
    # error_file.write(str(heightmap_rmse) + "," + str(contact_mask_iou) + "\n")
    # error_file.close()

    #####################################
    codebook = tactile_tree(
        poses=gelposes,
        cam_poses=camposes,
        embeddings=embeddings,
    )
    print("Saving data to path: {}".format(tree_path))
    with open(tree_path, "wb") as file:
        pickle.dump(codebook, file)
    return


if __name__ == "__main__":
    build_codebook()
