# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Main script for particle filtering with tactile embeddings"""

from os import path as osp
import torch
from midastouch.modules.particle_filter import particle_filter
from midastouch.modules.misc import (
    DIRS,
    get_device,
)
import dill as pickle

import hydra
import sys

from omegaconf import DictConfig

from midastouch.viz.demo_visualizer import Viz
from midastouch.render.digit_renderer import digit_renderer
from midastouch.contrib.tdn.TactileDepth import TactileDepth

from midastouch.contrib.tcn_minkloc.tcn import TCN
from midastouch.modules.objects import ycb_test
import time
from tqdm import tqdm
from digit_interface import Digit, DigitHandler

import threading

"""Initialize the DIGIT capture"""


def connectDigit(resolution="QVGA"):
    try:
        connected_digit = DigitHandler.list_digits()[0]
    except:
        print("No DIGIT found!")
        sys.exit(1)
    digit = Digit(connected_digit["serial"])  # Unique serial number
    digit.connect()
    digit.set_resolution(Digit.STREAMS[resolution])
    digit.set_fps(30)
    print(digit.info())
    # print("Collecting data from DIGIT {}".format(digit.serial))
    return digit


def live_demo(cfg: DictConfig, viz: Viz) -> None:
    """
    Filtering pose for simulated data
    """

    digit = connectDigit()
    expt_cfg, tcn_cfg, tdn_cfg = cfg.expt, cfg.tcn, cfg.tdn

    device = get_device(cpu=False)  # get GPU

    # print('\n----------------------------------------\n')
    # print(OmegaConf.to_yaml(cfg))
    # print('----------------------------------------\n')

    obj_model = expt_cfg.obj_model
    small_parts = False  # if obj_model in ycb_test else True

    tree_path = osp.join(DIRS["trees"], obj_model, "codebook.pkl")
    print(f"Loading {tree_path}")
    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")

    pf = particle_filter(cfg, obj_path, 1.0, real=True)
    tac_render = digit_renderer(cfg=tdn_cfg.render, obj_path=obj_path)

    digit_tcn = TCN(tcn_cfg)
    digit_tdn = TactileDepth(depth_mode="vit", real=True)

    codebook = pickle.load(open(tree_path, "rb"))
    codebook.to_device(device)
    heatmap_poses, _ = codebook.get_poses()
    heatmap_embeddings = codebook.get_embeddings()

    viz.init_variables(mesh_path=obj_path)

    count = 0
    for _ in tqdm(range(10)):
        # grab a few frames for stability (10 secs)
        time.sleep(0.1)
        digit.get_frame()

    while True:
        image = digit.get_frame()
        image = image[:, :, ::-1]  # BGR -> RGB
        if count == 0:
            for _ in range(20):
                digit_tdn.bg = digit_tdn.image2heightmap(image).to(device)

        ### 1. TDN + TCN: convert image to heightmap and compress to tactile_code
        heightmap = digit_tdn.image2heightmap(image)  # expensive
        mask = digit_tdn.heightmap2mask(heightmap.to(device), small_parts=small_parts)
        # view_subplots([image/255.0, heightmap.detach().cpu().numpy(), mask.detach().cpu().numpy()], [["image", "heightmap", "mask"]])

        tactile_code = digit_tcn.cloud_to_tactile_code(
            tac_render, heightmap.to(device), mask
        )

        cluster_poses, cluster_stds = None, None
        if not torch.sum(mask):
            heatmap_weights = torch.zeros(heatmap_embeddings.shape[0])
        else:
            heatmap_weights = pf.get_similarity(
                tactile_code, heatmap_embeddings, softmax=True
            )

        viz.update(
            heatmap_poses,
            heatmap_weights,
            cluster_poses,
            cluster_stds,
            image,
            heightmap,
            mask,
            count,
        )

        count += 1
    return


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig, viz=None):
    if cfg.expt.render:
        viz = Viz(off_screen=cfg.expt.off_screen, zoom=1.0)
    t = threading.Thread(name="live_demo", target=live_demo, args=(cfg, viz))
    t.start()
    if viz:
        viz.plotter.app.exec_()
    t.join()


if __name__ == "__main__":
    main()
