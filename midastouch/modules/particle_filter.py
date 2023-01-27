# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    Particle filtering class 
"""

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.nn.functional import cosine_similarity
from midastouch.modules.pose import (
    tf_to_xyzquat,
    wrap_angles,
    rot2euler,
    euler_angles_to_matrix,
    xyz_quat_averaged,
    log_map_averaged,
)
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import copy
from omegaconf import DictConfig
import torch
from torch import nn
import theseus as th
from torch.utils.data import WeightedRandomSampler
from typing import List, Tuple, Union


class Particles:
    """
    Particles class : [poses, weights, cluster labels]
    """

    poses = None
    weights = None
    labels = None

    def __init__(
        self,
        poses: torch.Tensor,
        weights: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        self.poses = poses
        if weights is not None:
            self.weights = weights
        else:
            self.weights = torch.ones(self.poses.shape[0], device=poses.device)

        if labels is not None:
            self.labels = labels
        else:
            self.labels = torch.zeros(self.poses.shape[0], device=poses.device)

    def __len__(self):
        return self.poses.shape[0]

    def remove(self, idxs: torch.Tensor) -> None:
        """
        Remove particles from indices idxs
        """
        self.poses = torch_delete(self.poses, idxs, dim=0)
        self.weights = torch_delete(self.weights, idxs)
        self.labels = torch_delete(self.labels, idxs)

    def add(
        self, poses: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """
        Remove particles from indices idxs
        """
        self.poses = torch.cat((self.poses, poses), dim=0)
        self.weights = torch.cat((self.weights, weights))
        self.labels = torch.cat((self.labels, labels))


def torch_delete(arr: torch.Tensor, idxs: int, dim: int = 0) -> torch.Tensor:
    """
    np.delete equivalent for torch
    """
    if idxs.nelement():
        full = torch.arange(0, arr.size(dim), device=idxs.device)
        complement = full[(full[:, None] != idxs).all(dim=1)]
        return arr.__getitem__(complement)
    else:
        return arr


class particle_filter:
    """
    particle filter class for update and propagation of SE(3) Particles on mesh
    """

    def __init__(
        self,
        cfg: DictConfig,
        mesh_path: str,
        noise: float = 1.0,
        real: bool = False,
        downsample: int = 10,
    ):
        self.pen_max = cfg.tdn.render.pen.max

        self.mesh = trimesh.load(mesh_path)
        mesh_vertices = np.array(self.mesh.vertices)[::downsample, :]
        self.mesh_kdtree = KDTree(mesh_vertices)
        if real:
            self.motion_noise = {
                "mu": 0,
                "sig_r": cfg.expt.params.noise_r.real,  # degrees
                "sig_t": cfg.expt.params.noise_t.real,  # m
            }
        else:
            self.motion_noise = {
                "mu": 0,
                "sig_r": cfg.expt.params.noise_r.sim,  # degrees
                "sig_t": cfg.expt.params.noise_t.sim,  # m
            }
        self.particle_var = torch.tensor([float("inf")])
        self.init_noise = [
            self.mesh_diagonal() / 3.0 * noise,
            180.0 / 3.0 * noise,
        ]  # 3σ = 0.5*diag, 3σ = max rot. (180°)

    def init_filter(
        self, gt_pose: torch.Tensor = torch.eye(4), N: int = 10000
    ) -> Particles:
        """
        Initialize particles with starting pose, number of particles, and uncertainty
        """
        gt_pose = torch.repeat_interleave(gt_pose[None, :, :], N, dim=0)

        tn = torch.normal(mean=0.0, std=self.init_noise[0], size=(N, 3))
        rotNoise = torch.normal(mean=0.0, std=self.init_noise[1], size=(N, 3))
        Rn = torch.tensor(
            R.from_euler("zyx", rotNoise, degrees=True).as_matrix()
        )  # (N, 3, 3)
        Tn = torch.zeros((N, 4, 4), dtype=gt_pose.dtype, device=gt_pose.device)
        Tn[:, :3, :3], Tn[:, :3, 3], Tn[:, 3, 3] = Rn, tn, 1
        initPoses = gt_pose @ Tn
        return Particles(initPoses)

    def mesh_diagonal(self):
        """
        Return volume diagonal of the mesh
        """
        return self.mesh.scale

    def get_cluster_centers(
        self, _particles: Particles, method: str = "logmap"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pose centers from labelled particles
        """
        particles = copy.copy(_particles)
        poses = particles.poses
        weights = particles.weights.float()
        labels = particles.labels

        unique_cluster_labels = torch.unique(labels)
        cluster_stds = torch.zeros(
            (unique_cluster_labels.shape[0], 3), device=unique_cluster_labels.device
        )
        cluster_poses = torch.zeros(
            (unique_cluster_labels.shape[0], 4, 4), device=unique_cluster_labels.device
        )

        for i, label in enumerate(unique_cluster_labels):
            target_idx = labels == label
            target_particles, target_weights = (
                poses[target_idx, :, :],
                weights[target_idx],
            )
            if torch.isclose(
                target_weights.max() - target_weights.min(),
                torch.tensor(
                    [0.0], device=target_weights.device, dtype=target_weights.dtype
                ),
            ):
                target_weights = torch.ones_like(target_weights)

            if method == "logmap":
                cluster_poses[i, :, :] = log_map_averaged(
                    target_particles, target_weights
                )
            elif method == "quat_avg":
                cluster_poses[i, :, :] = xyz_quat_averaged(
                    target_particles, target_weights
                )

            cluster_stds[i, :] = torch.sqrt(
                torch.sum(
                    (
                        (target_particles[:, :3, 3] - cluster_poses[i, :3, 3]) ** 2
                        * target_weights[:, None]
                    )
                    / target_weights.sum(),
                    dim=0,
                )
            )

        return cluster_poses, cluster_stds

    def cluster_particles(
        self, _particles: Particles, method: str = "euclidean", eps: float = 1e-2
    ) -> Particles:
        particles = copy.copy(_particles)
        min_samples = int(len(particles) / 5)

        if method == "euclidean":
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(
                particles.poses[:, :3, 3].cpu().numpy()
            )
        elif method == "logmap":
            T_th = th.SE3(tensor=particles.poses[:, :3, :])
            log_map = T_th.log_map()
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(
                log_map.cpu().numpy()
            )

        particles.labels = torch.tensor(
            clustering.labels_, device=particles.labels.device
        )
        return particles

    def resampler(
        self, _particles: Particles, resample: str = "weighted_random"
    ) -> Particles:
        particles = copy.copy(_particles)

        nSamples = len(particles)

        norm_weights = particles.weights / torch.sum(
            particles.weights
        )  # normalize weights
        if torch.all((norm_weights == 0)) or torch.any(torch.isnan(norm_weights)):
            return particles

        if resample == "weighted_random":
            # weighted random sample : fast and accurate
            idxs = list(WeightedRandomSampler(norm_weights, nSamples, replacement=True))
            resampledParticlePoses = particles.poses[idxs, :, :]
            resampledWeights = particles.weights[idxs]
            resampledLabels = particles.labels[idxs]
            return Particles(resampledParticlePoses, resampledWeights, resampledLabels)

        # cumulative sum - sample from positions along this
        weightsSum = torch.cumsum(norm_weights, dim=0, dtype=torch.float64)
        # locations to sample
        sampleLocs = (
            torch.tensor(
                range(0, nSamples, 1), dtype=torch.float64, device=weightsSum.device
            )
            / nSamples
        )  # float64 needed for precision
        offset = torch.rand(1, device=sampleLocs.device) / nSamples  # [0 - 1/N]
        sampleLocs = torch.remainder(sampleLocs + offset, 1)

        if resample == "low_var_batch":
            # batch low variance resampling : large space complexity
            sampleLocs = sampleLocs.repeat(nSamples, 1)  # (N, N)
            diff = weightsSum[:, None] - sampleLocs
            greater_than = (
                diff > 0.0
            )  # find all weightsum elements greater than the samplelocs
            greater_than_count = torch.sum(
                greater_than, dim=1, dtype=torch.short
            )  # number of elements > given element
            fwd_diff = (greater_than_count[1:] - greater_than_count[:-1].clone()).type(
                torch.long
            )
            idxs_arange = torch.arange(
                0, nSamples, device=fwd_diff.device, dtype=torch.long
            )
            fwd_diff = torch.cat((fwd_diff[:1], fwd_diff))  # Add the first element
            idxs = torch.repeat_interleave(
                idxs_arange, fwd_diff
            )  # number of times to sample each row

            resampledParticlePoses = particles.poses[idxs, :, :]
            resampledWeights = particles.weights[idxs]
            resampledLabels = particles.labels[idxs]
            return Particles(resampledParticlePoses, resampledWeights, resampledLabels)
        elif resample == "low_var":
            # low variance resampling : large tiem complexity
            _resampledParticlePoses = torch.zeros_like(particles.poses)
            _resampledWeights = torch.zeros_like(particles.weights)
            _resampledLabels = torch.zeros_like(particles.labels)
            currentLoc = 0
            _idxs = []
            for i in range(0, nSamples):
                while (currentLoc < nSamples) and (
                    sampleLocs[currentLoc] < weightsSum[i]
                ):
                    _resampledParticlePoses[currentLoc, :] = particles.poses[i, :, :]
                    _resampledWeights[currentLoc] = particles.weights[i]
                    _resampledLabels[currentLoc] = particles.labels[i]
                    currentLoc += 1
                    _idxs.append(i)
            _idxs = torch.tensor(_idxs, device=sampleLocs.device)
            return Particles(
                _resampledParticlePoses, _resampledWeights, _resampledLabels
            )

    def difference_intersection(self, a: list, b: list) -> List[Union[list, list]]:
        """
        Compares elements of two lists
        """
        combined = torch.cat((a, b))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        intersection = uniques[counts > 1]
        return difference, intersection

    def add_noise_to_odom(self, odom: torch.Tensor, mul: float = 1.0) -> torch.Tensor:
        """
        Takes in odom: (4 x 4 x N) and adds rotational and translational noise
        Return noisy_odom (4 x 4 x N)
        """
        N = odom.shape[0]

        tn = torch.normal(
            mean=self.motion_noise["mu"],
            std=float(mul) * self.motion_noise["sig_t"],
            size=(N, 3),
        ).to(odom.device)
        rotNoise = torch.normal(
            mean=self.motion_noise["mu"],
            std=float(mul) * self.motion_noise["sig_r"],
            size=(N, 3),
        ).to(odom.device)
        Rn = euler_angles_to_matrix(torch.deg2rad(rotNoise), "ZYX")

        ## slower
        # tn = torch.normal(mean = self.motion_noise["mu"], std = self.motion_noise["sig_t"], size=(N, 3))
        # rotNoise = torch.normal(mean = self.motion_noise["mu"], std = self.motion_noise["sig_r"], size=(N, 3))
        # Rn = torch.tensor(R.from_euler('zyx', rotNoise, degrees=True).as_matrix()) # (N, 3, 3)

        Tn = torch.zeros_like(odom)
        Tn[:, :3, :3], Tn[:, :3, 3], Tn[:, 3, 3] = Rn, tn, 1
        return odom @ Tn

    def check_quats(self, particles: Particles) -> Particles:
        """
        Check for norm zero quaternions and prune
        """
        # hack TODO: fix this
        quats = tf_to_xyzquat(particles.poses)
        pose_norm = torch.norm(quats[:, 3:], dim=1)
        invalid_idxs = torch.logical_or(pose_norm == 0, torch.isnan(pose_norm))
        invalid_idxs = invalid_idxs.nonzero()
        particles.remove(invalid_idxs)
        return particles

    def motionModel(
        self, _particles: Particles, odom: torch.Tensor, multiplier: float = 1.0
    ) -> Particles:
        """
        Applies odometry update to current particle distribution
        """
        if multiplier < 1.0:
            multiplier = 1.0
        particles = copy.copy(_particles)
        particlePoses = particles.poses

        odom = torch.repeat_interleave(odom[None, :, :], particlePoses.shape[0], dim=0)
        noisyOdom = self.add_noise_to_odom(odom, mul=multiplier)

        # SE(3) compose: R_ * T.R_, t_ + R_ * T.t_)
        particlePoses = particlePoses @ noisyOdom
        particles.poses = particlePoses
        particles = self.check_quats(particles)
        return particles

    def remove_invalid_particles(
        self, _particles: Particles, invalid_dist: bool = None
    ) -> Tuple[Particles, bool]:
        """
        Sets particle weights = 0 for particles that drift far from the surface
        """
        particles = copy.copy(_particles)
        di = self.mesh_kdtree.query(
            np.atleast_2d(particles.poses[:, :3, 3].cpu().numpy()),
            k=1,
            return_distance=True,
        )
        di = di[0].squeeze()
        dist = torch.tensor(di, device=particles.poses.device)

        if invalid_dist is None:
            invalid = dist > self.pen_max
        else:
            invalid = dist > invalid_dist

        m = torch.ones(len(particles), device=particles.weights.device)
        m[invalid] = 0.0
        particles.weights *= m
        drifted = torch.sum(m) == 0  # all particles have drifted away
        return particles, drifted

    def annealing(
        self, _particles: Particles, var: float, floor: int = 1000
    ) -> Particles:
        """
        Adjust number of particles based on the cluster variance (var)
        """
        particles = copy.copy(_particles)

        if torch.isinf(self.particle_var):
            """First time, skip adjustments"""
            self.particle_var = var
            self.init_particles = len(particles.weights)
            return particles
        if var == 0.0:
            """Convergence to single particle, skip"""
            return particles

        ratio = var / self.particle_var
        self.particle_var = var

        n_particles = len(particles.weights)
        N = particles.poses.shape[0]
        if ratio < 1:
            num_remove = min(
                int((1.0 - ratio) * N), abs(n_particles - floor), n_particles // 3
            )
            if not num_remove:
                return particles
            remove_idxs = torch.topk(
                particles.weights, num_remove, largest=False
            ).indices
            particles.remove(remove_idxs)
        elif ratio > 1:
            num_increase = min(int((ratio - 1.0) * N), n_particles // 3)
            if num_increase + n_particles > self.init_particles:
                return particles
            add_idxs = torch.topk(particles.weights, num_increase, largest=True).indices
            particles.add(
                particles.poses[add_idxs, :],
                particles.weights[add_idxs],
                particles.labels[add_idxs],
            )
        return particles

    def get_similarity(
        self, queries: torch.Tensor, targets: torch.Tensor, softmax=True
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


def particle_rmse(_particles: Particles, gt_pose: torch.Tensor) -> Tuple[float, float]:
    """
    root mean squared error of [trans, rot] of SE(3) particles
    """
    particles = copy.copy(_particles)

    if type(particles) is Particles:
        poses = torch.atleast_3d(particles.poses)
    else:
        poses = torch.atleast_3d(particles)

    gt_pose = gt_pose[None, :, :]

    R_diff = torch.matmul(gt_pose[:, :3, :3], poses[:, :3, :3].permute(0, 2, 1))
    T_diff = gt_pose[:, :3, 3] - poses[:, :3, 3]
    # diff = torch.inverse(gt_pose) @ poses # noisy
    e_t = torch.norm(T_diff, dim=1)

    R_diff = torch.nan_to_num(rot2euler(R_diff))
    diff_r = wrap_angles(R_diff)
    rmse_t = torch.sqrt(torch.mean((e_t) ** 2))
    rmse_r = torch.sqrt(torch.mean((diff_r) ** 2, dim=0))
    avg_rmse_r = torch.mean(rmse_r)

    return rmse_t, avg_rmse_r
