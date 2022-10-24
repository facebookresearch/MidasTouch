# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Compute estimated average contact area per-object
"""

import os
from os import path as osp

from midastouch.render.digit_renderer import digit_renderer
from midastouch.contrib.tdn_fcrn import TDN
from midastouch.modules.misc import DIRS, load_images
import tqdm as tqdm
import numpy as np

import hydra
from omegaconf import DictConfig


def compute_contact_area(cfg: DictConfig, real=False):
    expt_cfg, tdn_cfg = cfg.expt, cfg.tdn
    obj_model = expt_cfg.obj_model

    # make paths
    if real:
        data_path = osp.join(DIRS["data"], "sim", obj_model)
    else:
        data_path = osp.join(DIRS["data"], "real", obj_model)

    print(f"compute_contact_area \n Object: {obj_model}\n")

    all_datasets = sorted(os.listdir(data_path))
    print(f"datasets: {all_datasets}")
    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")

    tac_render = digit_renderer(cfg=tdn_cfg.render, obj_path=obj_path)
    digit_tdn = TDN(tdn_cfg, bg=tac_render.get_background(frame="gel"))

    for dataset in all_datasets:
        if dataset == "bg" or not osp.isdir(osp.join(data_path, dataset)):
            continue

        dataset_path = osp.join(data_path, dataset)
        if real:
            image_path = osp.join(dataset_path, "frames")
        else:
            image_path = osp.join(dataset_path, "tactile_images")

        images = load_images(image_path)

        traj_sz = len(images)

        digit_area_cm_sq = 0.02 * 0.03 * (10**4)
        pbar = tqdm(total=traj_sz)
        areas = []
        for j, image in enumerate(images):
            est_h = digit_tdn.image2heightmap(image)
            est_c = digit_tdn.heightmap2mask(est_h)
            ratio = est_c.sum() / est_c.size
            areas.append(digit_area_cm_sq * ratio)
            pbar.update(1)
        pbar.close()

        avg_area = np.vstack(areas).mean()
        print(f"avg_contact_area: {avg_area}")
        np.save(osp.join(dataset_path, "avg_contact_area.npy"), avg_area)
    return


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    compute_contact_area(cfg=cfg)


if __name__ == "__main__":
    main()
