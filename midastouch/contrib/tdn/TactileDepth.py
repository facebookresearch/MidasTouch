from midastouch.contrib.tdn.touch_vit.TouchVIT import TouchVIT
from midastouch.contrib.tdn.touch_fcrn.tdn import TDN
from hydra import compose
import os.path as osp
import numpy as np
import torch
import collections

dname = osp.dirname(osp.abspath(__file__))


class TactileDepth:
    def __init__(self, depth_mode, real=False):
        super(TactileDepth, self).__init__()

        cfg = compose(config_name=f"touch_depth/{depth_mode}").touch_depth

        if depth_mode == "gt":
            self.model = None
            return
        if depth_mode == "vit":
            # print("Loading ViT depth model----")
            self.model = TouchVIT(cfg=cfg)
        elif depth_mode == "fcrn":
            # print("Loading FCRN depth model----")
            self.model = TDN(cfg=cfg)
        else:
            raise NotImplementedError(f"Mode not implemented: {cfg.mode}")
        # print("done")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        settings_config = cfg.settings.real if real else cfg.settings.sim
        self.b, self.r, self.clip = (
            settings_config.border,
            settings_config.ratio,
            settings_config.clip,
        )

        self.blend_sz = settings_config.blend_sz
        self.heightmap_window = collections.deque([])

        self.bg = np.load(osp.join(dname, "bg.npy"))
        self.bg = torch.Tensor(self.bg).to(self.device)

    def image2heightmap(self, image: np.ndarray):
        heightmap = self.model.image2heightmap(image)
        return self.blend_heightmaps(heightmap)

    def heightmap2mask(
        self, heightmap: torch.tensor, small_parts: bool = False
    ) -> torch.Tensor:
        """Thresholds heightmap to return binary contact mask

        Args:
            heightmap: single tactile image

        Returns:
            padded_contact_mask: contact mask [True: is_contact, False: no_contact]

        """
        heightmap = heightmap.squeeze().to(self.device)
        heightmap = heightmap[self.b : -self.b, self.b : -self.b]
        init_height = self.bg[self.b : -self.b, self.b : -self.b]
        diff_heights = heightmap - init_height
        diff_heights[diff_heights < self.clip] = 0

        contact_mask = diff_heights > torch.quantile(diff_heights, 0.8) * self.r
        padded_contact_mask = torch.zeros_like(self.bg, dtype=bool)

        total_area = contact_mask.shape[0] * contact_mask.shape[1]
        atleast_area = 0.01 * total_area if small_parts else 0.05 * total_area

        if torch.count_nonzero(contact_mask) < atleast_area:
            return padded_contact_mask
        padded_contact_mask[self.b : -self.b, self.b : -self.b] = contact_mask
        return padded_contact_mask

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
