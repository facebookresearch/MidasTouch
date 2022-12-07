# Original source: https://github.com/XPFly1989/FCRN
# A Pytorch implementation of Laina, Iro, et al. "Deeper depth prediction with fully convolutional residual networks."
# 3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.

# Copyright (c) 2016, Iro Laina
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import torch.nn as nn
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UpProject(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size):
        super(UpProject, self).__init__()
        self.batch_size = batch_size

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))
        # out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 0, 1)))#right interleaving padding
        out1_2 = self.conv1_2(
            nn.functional.pad(x, (1, 1, 1, 0))
        )  # author's interleaving pading in github
        # out1_3 = self.conv1_3(nn.functional.pad(x, (0, 1, 1, 1)))#right interleaving padding
        out1_3 = self.conv1_3(
            nn.functional.pad(x, (1, 0, 1, 1))
        )  # author's interleaving pading in github
        # out1_4 = self.conv1_4(nn.functional.pad(x, (0, 1, 0, 1)))#right interleaving padding
        out1_4 = self.conv1_4(
            nn.functional.pad(x, (1, 0, 1, 0))
        )  # author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))
        # out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 0, 1)))#right interleaving padding
        out2_2 = self.conv2_2(
            nn.functional.pad(x, (1, 1, 1, 0))
        )  # author's interleaving pading in github
        # out2_3 = self.conv2_3(nn.functional.pad(x, (0, 1, 1, 1)))#right interleaving padding
        out2_3 = self.conv2_3(
            nn.functional.pad(x, (1, 0, 1, 1))
        )  # author's interleaving pading in github
        # out2_4 = self.conv2_4(nn.functional.pad(x, (0, 1, 0, 1)))#right interleaving padding
        out2_4 = self.conv2_4(
            nn.functional.pad(x, (1, 0, 1, 0))
        )  # author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = (
            torch.stack((out1_1, out1_2), dim=-3)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(self.batch_size, -1, height, width * 2)
        )
        out1_3_4 = (
            torch.stack((out1_3, out1_4), dim=-3)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(self.batch_size, -1, height, width * 2)
        )

        out1_1234 = (
            torch.stack((out1_1_2, out1_3_4), dim=-3)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(self.batch_size, -1, height * 2, width * 2)
        )

        out2_1_2 = (
            torch.stack((out2_1, out2_2), dim=-3)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(self.batch_size, -1, height, width * 2)
        )
        out2_3_4 = (
            torch.stack((out2_3, out2_4), dim=-3)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(self.batch_size, -1, height, width * 2)
        )

        out2_1234 = (
            torch.stack((out2_1_2, out2_3_4), dim=-3)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(self.batch_size, -1, height * 2, width * 2)
        )

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out


import torch.jit as jit


class FCRN_net(jit.ScriptModule):
    # class FCRN_net(nn.Module):

    def __init__(self, batch_size, bottleneck=False):
        super(FCRN_net, self).__init__()
        self.inplanes = 64
        self.batch_size = batch_size
        self.bottleneck = bottleneck

        # ResNet with out avrgpool & fc
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Up-Conv layers
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)

        self.up1 = self._make_upproj_layer(UpProject, 1024, 512, self.batch_size)
        self.up2 = self._make_upproj_layer(UpProject, 512, 256, self.batch_size)
        self.up3 = self._make_upproj_layer(UpProject, 256, 128, self.batch_size)
        self.up4 = self._make_upproj_layer(UpProject, 128, 64, self.batch_size)

        self.drop = nn.Dropout2d()

        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)

        self.upsample = nn.Upsample((320, 240), mode="bilinear", align_corners=False)

        # initialize
        if True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_upproj_layer(self, block, in_channels, out_channels, batch_size):
        return block(in_channels, out_channels, batch_size)

    @jit.script_method
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.bottleneck:
            return x  # feature vector

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.drop(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.upsample(x)
        return x
