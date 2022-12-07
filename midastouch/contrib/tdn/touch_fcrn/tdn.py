# Copyright (c) Meta Platforms, Inc. and affiliates.


"""
Tactile depth network that converts tactile images to heightmaps/masks via a fully convolutional residual networks [Laina et. al. 2016]
"""

import torch
import torch
import torch.nn.functional
from midastouch.contrib.tdn.touch_fcrn.fcrn import FCRN_net
from midastouch.modules.misc import DIRS
import numpy as np
import os

import cv2
from omegaconf import DictConfig
from hydra import compose
from hydra.utils import to_absolute_path


class TDN:
    def __init__(
        self,
        cfg: DictConfig,
        bg: np.ndarray = None,
        bottleneck: bool = False,
        real: bool = False,
    ):
        super(TDN, self).__init__()
        tdn_weights = os.path.join(DIRS["weights"], cfg.tdn_weights)

        fcrn_config = cfg.settings.real if real else cfg.settings.sim
        self.batch_size = fcrn_config.batch_size
        self.params = {"batch_size": self.batch_size, "shuffle": False}

        self.model = FCRN_net(self.batch_size, bottleneck=bottleneck)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(tdn_weights, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.to(self.device)

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
            # blended_output = self.correct_image_height_map(
            #     blended_output, output_frame="cam"
            # )

            return output

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
