# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os import path as osp
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import sys

sys.path.append("..")
from . import flow_transforms
import cv2
import warnings

warnings.filterwarnings("ignore")


class data_loader(data.Dataset):
    def __init__(self, data_path, label_path):
        self.data_path = data_path  # to txt file
        self.label_path = label_path  # to txt file

        self.data = open(osp.join(data_path), "r")
        self.label = open(osp.join(label_path), "r")

        self.data_content = self.data.read()
        self.data_list = self.data_content.split("\n")

        self.label_content = self.label.read()
        self.label_list = self.label_content.split("\n")

    def __getitem__(self, index):
        img_path = self.data_list[index].split(",")[1]
        depth_path = self.label_list[index].split(",")[1]

        image, depth = None, None
        with Image.open(img_path) as im:
            image = np.asarray(im)
            image = cv2.normalize(
                image, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX
            )
            # image = image[:,:,::-1] # RGB -> BGR
        with Image.open(depth_path) as dp:
            depth = np.asarray(dp).astype(np.int64)

        input_transform = transforms.Compose(
            [flow_transforms.Scale(240), flow_transforms.ArrayToTensor()]
        )
        target_depth_transform = transforms.Compose(
            [flow_transforms.Scale_Single(240), flow_transforms.ArrayToTensor()]
        )
        image = input_transform(image)
        depth = target_depth_transform(depth)

        return image, depth

    def __len__(self):
        return len(self.label_list) - 1


class real_data_loader(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path  # to txt file

        self.data = open(osp.join(data_path), "r")
        self.data_content = self.data.read()
        self.data_list = self.data_content.split("\n")

    def __getitem__(self, index):
        img_path = self.data_list[index].split(",")[1]

        image = None
        with Image.open(img_path) as im:
            image = np.asarray(im)
            image = cv2.normalize(
                image, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX
            )
            # image = image[:,:,::-1] # RGB -> BGR

        input_transform = transforms.Compose(
            [flow_transforms.Scale(240), flow_transforms.ArrayToTensor()]
        )
        image = input_transform(image)
        return image

    def __len__(self):
        return len(self.data_list) - 1
