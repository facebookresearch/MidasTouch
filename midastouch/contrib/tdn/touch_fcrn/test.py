# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Loads real/sim test dataset and evaluates TDN accuracy 
"""

import torch
import torch.utils.data
from .data_loader import real_data_loader, data_loader
from .tdn import TDN
import numpy as np
import os
from os import path as osp
from torch.autograd import Variable
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plot
from tqdm import tqdm
from midastouch.render.digit_renderer import digit_renderer, pixmm
from midastouch.modules.misc import DIRS
import hydra
from omegaconf import DictConfig

dtype = torch.cuda.FloatTensor


@hydra.main(config_path="config", config_name="test")
def test(cfg: DictConfig):
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    tac_render = digit_renderer(obj_path=None)
    digit_tdn = TDN(cfg.tdn, bg=tac_render.get_background(frame="gel"))
    results_path = osp.join(DIRS["debug"], "tdn_test")
    if not osp.exists(results_path):
        os.makedirs(results_path)

    if cfg.real:
        test_file = osp.join("data", "test_data_real.txt")
        test_loader = torch.utils.data.DataLoader(
            real_data_loader(test_file), batch_size=50, shuffle=False, drop_last=True
        )

        # test on real dataset
        print("Testing on real data")
        with torch.no_grad():
            count = 0
            pbar = tqdm(total=len(test_loader))
            for input in test_loader:
                input_var = Variable(input.type(dtype))
                for i in range(len(input_var)):
                    input_rgb_image = (
                        input_var[i]
                        .data.permute(1, 2, 0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    est_h = digit_tdn.image2heightmap(input_rgb_image)
                    est_c = digit_tdn.heightmap2mask(est_h)
                    # pred_image /= np.max(pred_image)
                    plot.imsave(
                        osp.join(results_path, f"{count}_input.png"), input_rgb_image
                    )
                    plot.imsave(
                        osp.join(results_path, f"{count}_pred_heightmap.png"),
                        est_h,
                        cmap="viridis",
                    )
                    plot.imsave(osp.join(results_path, f"{count}_pred_mask.png"), est_c)
                    count += 1
                pbar.update(1)
            pbar.close()
        return
    else:
        test_file = osp.join("data", "test_data.txt")
        label_file = osp.join("data", "test_label.txt")
        test_loader = torch.utils.data.DataLoader(
            data_loader(test_file, label_file),
            batch_size=50,
            shuffle=False,
            drop_last=True,
        )

        heightmap_rmse, contact_mask_iou = [], []

        # test on real dataset
        print("Testing on sim data")
        with torch.no_grad():
            count = 0
            pbar = tqdm(total=len(test_loader))
            for input, depth in test_loader:
                input_var = Variable(input.type(dtype))
                gt_var = Variable(depth.type(dtype))

                for i in range(len(input_var)):
                    input_rgb_image = (
                        input_var[i]
                        .data.permute(1, 2, 0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    gt_c = gt_var[i].data.squeeze().cpu().numpy().astype(np.float32)

                    est_h = digit_tdn.image2heightmap(input_rgb_image)
                    est_c = digit_tdn.heightmap2mask(est_h)

                    error_heightmap = np.abs(est_h - gt_c) * pixmm
                    heightmap_rmse.append(np.sqrt(np.mean(error_heightmap**2)))
                    intersection = np.sum(np.logical_and(gt_c, est_c))
                    contact_mask_iou.append(
                        intersection / (np.sum(est_c) + np.sum(gt_c) - intersection)
                    )
                    count += 1
                pbar.update(1)
            pbar.close()

            heightmap_rmse = [x for x in heightmap_rmse if str(x) != "nan"]
            contact_mask_iou = [x for x in contact_mask_iou if str(x) != "nan"]
            heightmap_rmse = sum(heightmap_rmse) / len(heightmap_rmse)
            contact_mask_iou = sum(contact_mask_iou) / len(contact_mask_iou)
            error_file = open(osp.join(results_path, "tdn_error.txt"), "w")
            error_file.write(str(heightmap_rmse) + "," + str(contact_mask_iou) + "\n")
            error_file.close()


if __name__ == "__main__":
    test()
