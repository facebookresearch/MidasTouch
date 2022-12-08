# Author: Jacek Komorowski
# Warsaw University of Technology
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

# Original source: https://github.com/jac99/MinkLoc3D
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from os import path as osp

import torch
import MinkowskiEngine as ME
from .utils import MinkLocParams
from .minkloc import MinkLoc
from omegaconf import DictConfig
from midastouch.modules.misc import DIRS, get_device


class TCN:
    def __init__(self, cfg: DictConfig):
        in_channels = 1

        self.params = MinkLocParams(cfg)  # Load MinkLoc3d params
        self.batch_size = cfg.model.batch_size
        if "MinkFPN" in self.params.model_params.model:
            self.model = MinkLoc(
                in_channels=in_channels,
                feature_size=self.params.model_params.feature_size,
                output_dim=self.params.model_params.output_dim,
                planes=self.params.model_params.planes,
                layers=self.params.model_params.layers,
                num_top_down=self.params.model_params.num_top_down,
                conv0_kernel_size=self.params.model_params.conv0_kernel_size,
            )
        else:
            raise NotImplementedError(
                "Model not implemented: {}".format(self.params.model_params.model)
            )

        tcn_weights = osp.join(DIRS["weights"], cfg.model.tcn_weights)
        self.load_weights(tcn_weights)

    def load_weights(self, weights):
        device = get_device(cpu=False, verbose=False)
        # Load MinkLoc weights
        weights = torch.load(weights, map_location=device)
        if type(weights) is dict:
            self.model.load_state_dict(weights["state_dict"])
        else:
            self.model.load_state_dict(weights)
        self.model.to(device)

    def cloud_to_tactile_code(self, tac_render, heightmaps, masks):
        # Adapted from original PointNetVLAD code
        self.model.eval()

        if type(heightmaps) is not list:
            heightmaps = [heightmaps]
            masks = [masks]

        device = next(self.model.parameters()).device

        embeddings_l = [None] * len(heightmaps)

        with torch.no_grad():
            numSamples = len(heightmaps)
            num_batches = numSamples // self.batch_size

            num_batches = 1 if num_batches == 0 else num_batches

            for i in range(num_batches):
                i_range = (
                    torch.IntTensor(range(i * self.batch_size, numSamples))
                    if (i == num_batches - 1)
                    else torch.IntTensor(
                        range(i * self.batch_size, (i + 1) * self.batch_size)
                    )
                )
                batch_clouds = [None] * len(i_range)
                for j, (h, c) in enumerate(
                    zip(
                        heightmaps[i_range[0] : i_range[-1] + 1],
                        masks[i_range[0] : i_range[-1] + 1],
                    )
                ):
                    batch_clouds[j] = tac_render.heightmap2Pointcloud(h, c)

                n_points = self.params.num_points
                for j, batch_cloud in enumerate(batch_clouds):
                    if batch_cloud.shape[0] == 0:
                        batch_cloud = torch.repeat_interleave(
                            torch.Tensor([[0, 0, 0]]).to(batch_cloud.device),
                            n_points,
                            dim=0,
                        )
                    else:
                        idxs = torch.arange(
                            batch_cloud.shape[0],
                            device=batch_cloud.device,
                            dtype=torch.float,
                        )
                        if n_points > batch_cloud.shape[0]:
                            downsampleIDs = torch.multinomial(
                                idxs, num_samples=n_points, replacement=True
                            )
                        else:
                            downsampleIDs = torch.multinomial(
                                idxs, num_samples=n_points, replacement=False
                            )
                        batch_cloud = batch_cloud[downsampleIDs, :]

                    # batch_clouds[j] = batch_cloud
                    # minv, _ = torch.min(batch_cloud, dim=0)
                    # maxv, _ = torch.max(batch_cloud, dim=0)
                    # batch_clouds[j] = 2.0 * (batch_cloud - minv) / (maxv - minv) - 1
                    batch_clouds[j] = (
                        2.0
                        * (batch_cloud - torch.min(batch_cloud))
                        / (torch.max(batch_cloud) - torch.min(batch_cloud))
                        - 1
                    )  # scale [-1, 1]

                batch_clouds = torch.stack(
                    batch_clouds, dim=0
                )  # Produces (batch_size, n_points, 3) tensor
                batch = {}

                # coords are (n_clouds, num_points, channels) tensor
                coords = [
                    ME.utils.sparse_quantize(
                        coordinates=e,
                        quantization_size=self.params.model_params.mink_quantization_size,
                    )
                    for e in batch_clouds
                ]
                coords = ME.utils.batched_coordinates(coords, device=device)
                # Assign a dummy feature equal to 1 to each point
                feats = torch.ones(
                    (coords.shape[0], 1), dtype=torch.float32, device=device
                )
                batch["coords"], batch["features"] = coords, feats

                embedding = self.model(batch)

                if self.params.normalize_embeddings:
                    embedding = torch.nn.functional.normalize(
                        embedding, p=2, dim=1
                    )  # Normalize embeddings
                # embedding = embedding.detach().cpu().numpy()
                embeddings_l[i_range[0] : i_range[-1] + 1] = embedding

        embeddings_l = torch.vstack(embeddings_l)  # list to array (set_sz, output_dim)
        return embeddings_l.double()  # double for precision
