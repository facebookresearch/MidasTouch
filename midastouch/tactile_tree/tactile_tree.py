# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pynanoflann
from midastouch.modules.pose import get_logmap_from_matrix
import torch
from torch import nn


class tactile_tree(nn.Module):
    def __init__(self, poses, cam_poses, embeddings):
        super(tactile_tree, self).__init__()
        self.poses = poses.float()
        self.logmap_pose = R3_SE3(self.poses.clone())
        self.cam_poses, self.embeddings = cam_poses.float(), embeddings
        self.tree, self.tree_size = None, 0
        self.init_tree()

    def __len__(self):
        return self.tree_size

    def __repr__(self):
        return "tactile Tree of size: {}".format(self.__len__)

    def to_device(self, device):
        self.poses = self.poses.to(device)
        self.logmap_pose = self.logmap_pose.to(device)
        self.cam_poses = self.cam_poses.to(device)
        self.embeddings = self.embeddings.to(device)

    def init_tree(self):
        range = np.max(
            np.ptp(self.logmap_pose.cpu().numpy(), axis=0)
        )  # maximum range of KDTree data
        self.tree = pynanoflann.KDTree(metric="L2", radius=range)
        self.tree.fit(self.logmap_pose.cpu().numpy())
        self.tree_size = self.poses.shape[0]
        return

    def SE3_NN(self, _query, nn=1):
        query = _query.clone()
        """
            Get best SE3 match based on R3 and logmap_SO3 distances from a set of candidates
        """
        query = torch.atleast_3d(query)
        query = R3_SE3(query)
        _, indices_p = self.tree.kneighbors(
            query.cpu().numpy(), n_neighbors=nn, n_jobs=16
        )
        indices_p = torch.tensor(indices_p.squeeze().astype(dtype=np.int64))
        return (
            self.poses[indices_p, :, :],
            self.cam_poses[indices_p, :, :],
            self.embeddings[indices_p, :],
        )

    def get_poses(self):
        return self.poses, self.cam_poses

    def get_pose(self, idx):
        return self.poses[idx, :]

    def get_embeddings(self):
        return self.embeddings

    def get_embedding(self, idx):
        return self.embeddings[idx, :]


def R3_SE3(poses, w=0.01):
    return torch.cat(
        ((1.0 - w) * poses[:, :3, 3], w * get_logmap_from_matrix(poses[:, :3, :3])),
        axis=1,
    )
