# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Simulates tactile interaction on object meshes, choose between [random, random+edges, traj, manual]
"""

import os
from os import path as osp
import numpy as np

from midastouch.viz.helpers import viz_poses_pointclouds_on_mesh
from midastouch.render.digit_renderer import digit_renderer
from midastouch.modules.misc import (
    remove_and_mkdir,
    DIRS,
    save_contactmasks,
    save_heightmaps,
    save_images,
)
from midastouch.modules.mesh import sample_poses_on_mesh
from midastouch.modules.pose import transform_pc, xyzquat_to_tf_numpy
from midastouch.data_gen.utils import random_geodesic_poses, random_manual_poses
import dill as pickle
import trimesh
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


def touch_simulator(cfg: DictConfig):
    """Tactile simulator function"""
    render_cfg = cfg.render
    obj_model = cfg.obj_model
    sampling = cfg.sampling
    num_samples = cfg.num_samples
    total_length = cfg.total_length
    save_path = cfg.save_path
    randomize = render_cfg.randomize
    headless = False

    # make paths
    if save_path is None:
        data_path = osp.join(DIRS["data"], "sim", obj_model)
        file_idx = 0
        while osp.exists(
            osp.join(data_path, str(file_idx).zfill(2), "tactile_data.pkl")
        ):
            file_idx += 1
        data_path = osp.join(data_path, str(file_idx).zfill(2))
    else:
        data_path = osp.join(save_path, obj_model)

    remove_and_mkdir(data_path)

    image_path = osp.join(data_path, "tactile_images")
    heightmap_path = osp.join(data_path, "gt_heightmaps")
    contactmasks_path = osp.join(data_path, "gt_contactmasks")
    pose_path = osp.join(data_path, "tactile_data.pkl")

    os.makedirs(image_path)
    os.makedirs(heightmap_path)
    os.makedirs(contactmasks_path)

    print(f"object: {obj_model}, data_path: {data_path} sampling: {sampling}\n")

    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")

    mesh = trimesh.load(obj_path)

    # get poses depending on the method
    if "random" in sampling:
        # random independent samples
        print(f"Generating {num_samples} random samples")
        sample_poses = sample_poses_on_mesh(
            mesh=mesh,
            num_samples=num_samples,
            edges=True if (sampling == "random+edges") else False,
        )
    elif "traj" in sampling:
        # random geodesic trajectory
        print(f"Generating random geodesic trajectory")
        sample_poses = None
        while sample_poses is None:
            sample_poses = random_geodesic_poses(
                mesh,
                shear_mag=render_cfg.shear_mag,
                total_length=total_length,
                N=num_samples,
            )
    elif "manual" in sampling:
        # manually selected waypoints trajectory
        print(f"Generating manual waypoint trajectory")
        sample_poses = random_manual_poses(
            mesh_path=obj_path, shear_mag=render_cfg.shear_mag, lc=0.001
        )
    else:
        print("Invalid sampling routine, exiting!")
        return

    # start renderer
    tac_render = digit_renderer(cfg=render_cfg, obj_path=obj_path, headless=headless)

    # remove NaNs
    batch_size = 1000
    traj_sz = sample_poses.shape[0]
    num_batches = traj_sz // batch_size
    num_batches = num_batches if (num_batches != 0) else 1

    gelposes, camposes, gelposes_meas = (
        np.empty((0, 7)),
        np.empty((0, 7)),
        np.empty((0, 7)),
    )
    gt_heightmaps, gt_masks, tactile_images = [], [], []
    for i in tqdm(range(num_batches)):
        if randomize:
            tac_render = digit_renderer(
                cfg=render_cfg, obj_path=obj_path, randomize=randomize, headless=headless
            )
        i_range = (
            np.array(range(i * batch_size, traj_sz))
            if (i == num_batches - 1)
            else np.array(range(i * batch_size, (i + 1) * batch_size))
        )
        (
            hm,
            cm,
            image,
            campose,
            gelpose,
            gelpose_meas,
        ) = tac_render.render_sensor_trajectory(
            p=sample_poses[i_range, :], mNoise=cfg.noise
        )
        gelposes = np.append(gelposes, gelpose, axis=0)
        camposes = np.append(camposes, campose, axis=0)
        gelposes_meas = np.append(gelposes_meas, gelpose_meas, axis=0)
        tactile_images = tactile_images + image
        gt_heightmaps = gt_heightmaps + hm
        gt_masks = gt_masks + cm

    # Save ground-truth pointclouds and tactile images
    print(
        f"Saving data: \nHeightmaps: {heightmap_path} \nContact masks: {contactmasks_path} \nTactile Images: {image_path}"
    )
    save_heightmaps(gt_heightmaps, heightmap_path)
    save_contactmasks(gt_masks, contactmasks_path)
    save_images(tactile_images, image_path)

    pointclouds, pointclouds_world = [None] * traj_sz, [None] * traj_sz
    for i, (h, c, p) in enumerate(zip(gt_heightmaps, gt_masks, camposes)):
        pointclouds[i] = tac_render.heightmap2Pointcloud(h, c)
    pointclouds_world = transform_pc(pointclouds, camposes)

    save_dict = {
        "gelposes": gelposes,
        "camposes": camposes,
        "gelposes_meas": gelposes_meas,
        "mNoise": cfg.noise,
    }

    print("Saving data to path: {}".format(pose_path))
    with open(pose_path, "wb") as file:
        pickle.dump(save_dict, file)

    if not headless:
        viz_gelposes = xyzquat_to_tf_numpy(gelposes)
        viz_gelposes_meas = xyzquat_to_tf_numpy(gelposes_meas)

        print("Visualizing data")
        if len(pointclouds_world) > 2500:
            pointclouds_world = pointclouds_world[::10]
            viz_gelposes, viz_gelposes_meas = (
                viz_gelposes[::10, :],
                viz_gelposes_meas[::10, :],
            )

        viz_poses_pointclouds_on_mesh(
            mesh_path=obj_path,
            poses=viz_gelposes,
            pointclouds=pointclouds_world,
            save_path=osp.join(data_path, "tactile_data"),
            decimation_factor=10,
        )
        viz_poses_pointclouds_on_mesh(
            mesh_path=obj_path,
            poses=viz_gelposes_meas,
            pointclouds=pointclouds_world,
            save_path=osp.join(data_path, "tactile_data_noisy"),
            decimation_factor=10,
        )
    return


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    touch_simulator(cfg=cfg.method)


if __name__ == "__main__":
    main()
