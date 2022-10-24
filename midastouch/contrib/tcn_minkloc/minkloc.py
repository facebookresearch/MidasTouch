# Author: Jacek Komorowski
# Warsaw University of Technology

# Original source: https://github.com/jac99/MinkLoc3D
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch.nn as nn
import MinkowskiEngine as ME

from .minkfpn import MinkFPN
import torch
import torch.nn as nn


class MinkLoc(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_size,
        output_dim,
        planes,
        layers,
        num_top_down,
        conv0_kernel_size,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size  # Size of local features produced by local feature extraction block
        self.output_dim = output_dim  # Dimensionality of the global descriptor produced by pooling layer
        self.backbone = MinkFPN(
            in_channels=in_channels,
            out_channels=self.feature_size,
            num_top_down=num_top_down,
            conv0_kernel_size=conv0_kernel_size,
            layers=layers,
            planes=planes,
        )
        self.n_backbone_features = output_dim
        assert (
            self.feature_size == self.output_dim
        ), "output_dim must be the same as feature_size"
        self.pooling = GeM()

    def forward(self, batch):
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        x = ME.SparseTensor(batch["features"], coordinates=batch["coords"])
        x = self.backbone(x)

        # x is (num_points, n_features) tensor
        assert (
            x.shape[1] == self.feature_size
        ), "Backbone output tensor has: {} channels. Expected: {}".format(
            x.shape[1], self.feature_size
        )
        x = self.pooling(x)
        assert (
            x.dim() == 2
        ), "Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.".format(
            x.dim()
        )
        assert (
            x.shape[1] == self.output_dim
        ), "Output tensor has: {} channels. Expected: {}".format(
            x.shape[1], self.output_dim
        )
        # x is (batch_size, output_dim) tensor
        return x

    def print_info(self):
        print("Model class: MinkLoc")
        n_params = sum([param.nelement() for param in self.parameters()])
        print("Total parameters: {}".format(n_params))
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print("Backbone parameters: {}".format(n_params))
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print("Aggregation parameters: {}".format(n_params))
        if hasattr(self.backbone, "print_info"):
            self.backbone.print_info()
        if hasattr(self.pooling, "print_info"):
            self.pooling.print_info()


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = self.f(temp)  # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1.0 / self.p)  # Return (batch_size, n_features) tensor
