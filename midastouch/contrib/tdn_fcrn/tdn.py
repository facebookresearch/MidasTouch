# Copyright (c) Meta Platforms, Inc. and affiliates.


"""
Tactile depth network that converts tactile images to heightmaps/masks via a fully convolutional residual networks [Laina et. al. 2016]
"""

import torch
import torch
import torch.nn.functional
from .fcrn import FCRN_net
import numpy as np

from midastouch.render.digit_renderer import digit_renderer
from midastouch.viz.visualizer import Viz

from PIL import Image
import collections
from midastouch.modules.misc import view_subplots, DIRS, get_device
from midastouch.modules.pose import transform_pc, extract_poses_sim
import cv2
import os
from os import path as osp
import hydra
from omegaconf import DictConfig


class TDN:
    def __init__(
        self,
        cfg: DictConfig,
        bg: np.ndarray = None,
        bottleneck: bool = False,
        real: bool = False,
    ):

        tdn_weights = osp.join(DIRS["weights"], cfg.tdn_weights)

        fcrn_config = cfg.fcrn.real if real else cfg.fcrn.sim
        self.b, self.r, self.clip = (
            fcrn_config.border,
            fcrn_config.ratio,
            fcrn_config.clip,
        )
        self.batch_size = fcrn_config.batch_size
        self.params = {"batch_size": self.batch_size, "shuffle": False}

        self.model = FCRN_net(self.batch_size, bottleneck=bottleneck)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(tdn_weights, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.blend_sz = fcrn_config.blend_sz
        self.heightmap_window = collections.deque([])
        if bg is not None:
            self.bg = torch.Tensor(bg).to(self.device)

    def blend_heightmaps(self, heightmap: torch.Tensor) -> torch.Tensor:
        """Exponentially weighted heightmap blending.

        Args:
            heightmap: input heightmap

        Returns:
            blended_heightmap: output heightmap blended over self.heightmap_window

        """

        if not self.blend_sz:
            return heightmap

        if len(self.heightmap_window) >= self.blend_sz:
            self.heightmap_window.popleft()

        self.heightmap_window.append(heightmap)
        n = len(self.heightmap_window)

        weights = torch.tensor(
            [x / n for x in range(1, n + 1)], device=heightmap.device
        )  # exponentially weighted time series costs

        weights = torch.exp(weights) / torch.sum(torch.exp(weights))

        all_heightmaps = torch.stack(list(self.heightmap_window))
        blended_heightmap = torch.sum(
            (all_heightmaps * weights[:, None, None]) / weights.sum(), dim=0
        )  # weighted average

        # view_subplots([heightmap, blended_heightmap], [["heightmap", "blended_heightmap"]])
        return blended_heightmap

    def image2heightmap(self, image: np.ndarray) -> torch.Tensor:
        """Passes tactile image through FCRN and returns (blended) heightmap

        Args:
            image: single tactile image

        Returns:
            blended_output: resulting heightmap from FCRN + blending

        """

        assert (
            self.model.bottleneck is False
        ), "Bottleneck feature is enabled, can't carry out image2heightmap"
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        with torch.no_grad():
            image = torch.from_numpy(image).permute(2, 0, 1).to(self.device).float()
            output = self.model(image[None, :])[
                0
            ].squeeze()  # .data.cpu().squeeze().numpy()
            blended_output = self.blend_heightmaps(output)
            return blended_output

    def image2embedding(self, image: np.ndarray) -> torch.Tensor:
        """Passes tactile image through FCRN and returns bottleneck embedding of size 10 * 8 * 1024

        Args:
            image: single tactile image

        Returns:
            feature: feature tensor (10 * 8 * 1024, 1)

        """

        if self.model.bottleneck is False:
            print("Bottleneck feature extraction not enabled, switching")
            self.model.bottleneck = True
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        with torch.no_grad():
            image = torch.from_numpy(image).permute(2, 0, 1).to(self.device).float()
            output = self.model(image[None, :])[0].squeeze()
            feature = output.reshape((-1, 10 * 8 * 1024))
            feature = feature / torch.norm(feature, axis=1).reshape(-1, 1)
            return feature

    def heightmap2mask(
        self, heightmap: torch.tensor, small_parts: bool = False
    ) -> torch.Tensor:
        """Thresholds heightmap to return binary contact mask

        Args:
            heightmap: single tactile image

        Returns:
            padded_contact_mask: contact mask [True: is_contact, False: no_contact]

        """
        heightmap = heightmap[self.b : -self.b, self.b : -self.b]
        init_height = self.bg[self.b : -self.b, self.b : -self.b]
        diff_heights = heightmap - init_height
        diff_heights[diff_heights < self.clip] = 0

        contact_mask = diff_heights > torch.quantile(diff_heights, 0.8) * self.r
        padded_contact_mask = torch.zeros_like(self.bg, dtype=bool)

        total_area = contact_mask.shape[0] * contact_mask.shape[1]
        atleast_area = 0.01 * total_area if small_parts else 0.1 * total_area

        if torch.count_nonzero(contact_mask) < atleast_area:
            return padded_contact_mask
        padded_contact_mask[self.b : -self.b, self.b : -self.b] = contact_mask
        return padded_contact_mask


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    expt_cfg, tcn_cfg, tdn_cfg = cfg.expt, cfg.tcn, cfg.tdn
    device = get_device(cpu=False)  # get GPU

    obj_model = expt_cfg.obj_model
    log_id = str(expt_cfg.log_id).zfill(2)

    data_path = osp.join(DIRS["data"], "sim", obj_model, log_id)
    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")

    image_path, pose_path = osp.join(data_path, "tactile_images"), osp.join(
        data_path, "tactile_data.pkl"
    )
    heightmap_path, contactmask_path = osp.join(data_path, "gt_heightmaps"), osp.join(
        data_path, "gt_contactmasks"
    )

    viz = Viz(off_screen=False)

    tac_render = digit_renderer(cfg=tdn_cfg.render, obj_path=obj_path)
    digit_tdn = TDN(tdn_cfg, bg=tac_render.get_background(frame="gel"))

    # load images and ground truth depthmaps
    images = sorted(os.listdir(image_path), key=lambda y: int(y.split(".")[0]))
    heightmaps = sorted(os.listdir(heightmap_path), key=lambda y: int(y.split(".")[0]))
    contact_masks = sorted(
        os.listdir(contactmask_path), key=lambda y: int(y.split(".")[0])
    )

    # poses
    camposes, gelposes, _ = extract_poses_sim(
        osp.join(data_path, "tactile_data.pkl"), device=device
    )  # poses : (N , 4, 4)

    N = len(images)
    for i in range(N):
        # Open images
        image = np.array(Image.open(osp.join(image_path, images[i])))
        gt_heightmap = np.array(
            Image.open(osp.join(heightmap_path, heightmaps[i]))
        ).astype(np.int64)
        contactmask = np.array(
            Image.open(osp.join(contactmask_path, contact_masks[i]))
        ).astype(bool)

        # Convert image to heightmap via lookup
        est_heightmap = digit_tdn.image2heightmap(image)
        est_contactmask = digit_tdn.heightmap2mask(est_heightmap)
        # Get pixelwise RMSE in mm, and IoU of the contact masks
        error_heightmap = np.abs(est_heightmap - gt_heightmap) * tac_render.pixmm
        heightmap_rmse = np.sqrt(np.mean(error_heightmap**2))
        intersection = np.sum(np.logical_and(contactmask, est_contactmask))
        contact_mask_iou = intersection / (
            np.sum(contactmask) + np.sum(est_contactmask) - intersection
        )

        # Visualize heightmaps
        print(
            "Heightmap RMSE: {:.4f} mm, Contact mask IoU: {:.4f}".format(
                heightmap_rmse, contact_mask_iou
            )
        )
        view_subplots(
            [
                image / 255.0,
                gt_heightmap,
                contactmask,
                est_heightmap,
                est_contactmask,
                error_heightmap,
            ],
            [
                ["Tactile image", "GT heightmap", "GT contact mask"],
                ["Est. heightmap", "Est. contact mask", "Heightmap Error (mm"],
            ],
        )

        # Convert heightmaps to 3D
        gt_cloud = tac_render.heightmap2Pointcloud(gt_heightmap, contactmask)
        gt_cloud_w = transform_pc(gt_cloud.copy(), camposes[i])
        est_cloud = tac_render.heightmap2Pointcloud(est_heightmap, est_contactmask)
        est_cloud_w = transform_pc(est_cloud.copy(), camposes[i])


if __name__ == "__main__":
    main()
