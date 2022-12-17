# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Main script for particle filtering with tactile embeddings"""

from os import path as osp
import torch
from midastouch.modules.misc import (
    DIRS,
    get_device,
)
import dill as pickle

import hydra
import sys
import yappi

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

from torch.nn.functional import cosine_similarity
from torch import nn
def get_similarity(
    queries: torch.Tensor, targets: torch.Tensor, softmax=True
) -> torch.Tensor:
    """
    computing embedding similarity weights based on cosine score
    """
    weights = cosine_similarity(
        torch.atleast_2d(queries), torch.atleast_2d(targets)
    ).squeeze()
    # weights = np.random.randn(*weights.shape) # random weights
    if (
        not torch.isclose(
            weights.max() - weights.min(),
            torch.tensor([0.0], device=weights.device, dtype=weights.dtype),
        )
        and softmax
    ):
        weights = nn.Softmax(dim=0)(
            weights
        )  # softmax: torch.exp(weights) / torch.sum(torch.exp(weights))
    return weights

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

    obj_model = ['025_mug', '048_hammer']
    small_parts = False  # if obj_model in ycb_test else True

    obj_path = [None] *2
    obj_path[0] = osp.join(DIRS["obj_models"], obj_model[0], "nontextured.stl")
    obj_path[1] = osp.join(DIRS["obj_models"], obj_model[1], "nontextured.stl")

    viz.init_variables(mesh_path=obj_path)

    tac_render = digit_renderer(cfg=tdn_cfg.render, obj_path=None)

    digit_tcn = TCN(tcn_cfg)
    digit_tdn = TactileDepth(depth_mode="fcrn", real=True)

    tree_path = [None] *2

    tree_path[0] = osp.join(DIRS["trees"], obj_model[0], "codebook.pkl")
    tree_path[1] = osp.join(DIRS["trees"], obj_model[1], "codebook.pkl")

    print(f"Loading {tree_path}")
    codebook = [None] * 2
    codebook[0] = pickle.load(open(tree_path[0], "rb"))
    codebook[0].to_device(device)
    codebook[1] = pickle.load(open(tree_path[1], "rb"))
    codebook[1].to_device(device)

    heatmap_poses, heatmap_embeddings = [None] * 2, [None] * 2
    heatmap_poses[0], _ = codebook[0].get_poses()
    heatmap_embeddings[0] = codebook[0].get_embeddings()
    heatmap_poses[1], _ = codebook[1].get_poses()
    heatmap_embeddings[1] = codebook[1].get_embeddings()

    count = 0
    for _ in tqdm(range(10)):
        # grab a few frames for stability (10 secs)
        time.sleep(0.1)
        digit.get_frame()

    while True:
        mesh_number = viz.mesh_number
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
            heatmap_weights = torch.zeros(heatmap_embeddings[mesh_number].shape[0])
        else:
            heatmap_weights = get_similarity(
                tactile_code, heatmap_embeddings[mesh_number], softmax=True
            )

        viz.update(
            heatmap_poses[mesh_number],
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
def main(cfg: DictConfig, viz=None, profile=False):
    if cfg.expt.render:
        viz = Viz(off_screen=cfg.expt.off_screen, zoom=1.0)
    t = threading.Thread(name="live_demo", target=live_demo, args=(cfg, viz))
    t.start()
    if viz:
        viz.plotter.app.exec_()
    t.join()

if __name__ == "__main__":
    main()
