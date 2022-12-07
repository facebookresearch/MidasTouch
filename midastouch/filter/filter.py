# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run MidasTouch on simulated YCB-Slide data
"""

import os
from os import path as osp
import numpy as np
import torch
from midastouch.modules.particle_filter import particle_filter, particle_rmse
from midastouch.modules.misc import (
    DIRS,
    remove_and_mkdir,
    get_time,
    load_images,
    get_device,
    images_to_video,
)
from midastouch.modules.pose import extract_poses_sim
import dill as pickle

import yappi
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from midastouch.viz.visualizer import Viz
from midastouch.render.digit_renderer import digit_renderer
from midastouch.contrib.tdn.TactileDepth import TactileDepth

from midastouch.contrib.tcn_minkloc.tcn import TCN
from midastouch.modules.objects import ycb_test
import time
from tqdm import tqdm

import threading


def filter(cfg: DictConfig, viz: Viz) -> None:
    """Filtering for tactile simulation data"""
    expt_cfg, tcn_cfg, tdn_cfg = cfg.expt, cfg.tcn, cfg.tdn

    device = get_device(cpu=False)

    # print('\n----------------------------------------\n')
    # print(OmegaConf.to_yaml(cfg))
    # print('----------------------------------------\n')

    init_particles = expt_cfg.params.num_particles
    obj_model = expt_cfg.obj_model
    small_parts = False if obj_model in ycb_test else True
    log_id = str(expt_cfg.log_id).zfill(2)

    noise_ratio = expt_cfg.params.noise_ratio
    frame_rate = expt_cfg.frame_rate

    # Results saved in "output" folder
    results_path = osp.join(os.getcwd(), obj_model, log_id)
    trial_id = 0
    while osp.exists(osp.join(results_path, f"trial_{str(trial_id).zfill(2)}")):
        trial_id += 1
    results_path = osp.join(results_path, f"trial_{str(trial_id).zfill(2)}")
    if expt_cfg.ablation:
        results_path = osp.join(results_path, f"{noise_ratio}")
    remove_and_mkdir(results_path)

    # Loading data
    print("Loading dataset...")
    data_path = osp.join(DIRS["data"], "sim", obj_model, log_id)
    gt_p_cam, gt_p, meas_p = extract_poses_sim(
        osp.join(data_path, "tactile_data.pkl"), device=device
    )  # poses : (N , 4, 4)
    image_path = osp.join(data_path, "tactile_images")
    tactile_images = load_images(image_path, N=expt_cfg.max_length)  # (N, 3, H, W)
    traj_size = len(tactile_images)

    # Init pf and rendering classes
    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")
    pf = particle_filter(cfg, obj_path, noise_ratio)
    tac_render = digit_renderer(cfg=tdn_cfg.render, obj_path=obj_path)

    digit_tcn = TCN(tcn_cfg)
    digit_tdn = TactileDepth(depth_mode="vit", real=False)

    # load tactile codebook
    codebok_path = osp.join(DIRS["trees"], obj_model, "codebook.pkl")
    codebook = pickle.load(open(codebok_path, "rb"))
    codebook.to_device(device)
    heatmap_poses, _ = codebook.get_poses()
    heatmap_embeddings = codebook.get_embeddings()

    pbar = tqdm(total=traj_size, desc="processing")
    timer = dict.fromkeys(["tactile", "motion", "meas"])
    avg_timer = {"tactile": [], "motion": [], "meas": []}

    filter_stats = {
        "rmse_t": [],
        "rmse_r": [],
        "time": [],
        "traj_size": traj_size,
        "avg_time": None,
        "total_time": 0,
        "cluster_poses": [],
        "cluster_stds": [],
        "obj_name": obj_model,
        "tree_size": len(codebook),
        "noise_ratio": noise_ratio,
        "init_noise": pf.init_noise,
        "init_particles": init_particles,
        "num_particles": [],
        "log_id": log_id,
        "trial_id": trial_id,
    }
    msg = ""

    pbar.set_description(msg + "Opening visualizer...")
    if viz:
        viz.init_variables(
            obj_model=obj_model,
            mesh_path=obj_path,
            gt_pose=gt_p,
            n_particles=init_particles,
        )

    prev_idx, count = 0, 0

    # run filter
    while True:
        while viz.pause:
            time.sleep(0.01)
        current_time = filter_stats["total_time"]
        idx = int(frame_rate * current_time)
        diff = idx - prev_idx

        if idx >= traj_size:
            break
        image = tactile_images[idx]

        start_time = time.time()
        # image to heightmap
        heightmap = digit_tdn.image2heightmap(image)  # expensive
        mask = digit_tdn.heightmap2mask(heightmap, small_parts=small_parts)
        # heightmap to code
        tactile_code = digit_tcn.cloud_to_tactile_code(
            tac_render, heightmap.to(mask.device), mask
        )
        timer["tactile"] = get_time(start_time)

        # motion model
        start_time = time.time()
        if prev_idx > 0:
            # t > 0 Propagate motion model
            odom = torch.inverse(meas_p[prev_idx, :]) @ meas_p[idx, :]  # noisy
            particles = pf.motionModel(particles, odom, multiplier=1.0)
            timer["motion"] = get_time(start_time)
        else:
            # t = 0 Intialize particles
            particles = pf.init_filter(gt_p[idx, :], init_particles)
            particles.poses, _, _ = codebook.SE3_NN(particles.poses)
            timer["motion"] = get_time(start_time)

        # compute RMSE
        rmse_t, rmse_r = particle_rmse(particles, gt_p[idx, :])
        filter_stats["rmse_t"].append(rmse_t.item())
        filter_stats["rmse_r"].append(rmse_r.item())

        # get similarity from codebook
        start_time = time.time()
        _, _, nn_tactile_codes = codebook.SE3_NN(particles.poses)
        particles.weights = pf.get_similarity(
            tactile_code, nn_tactile_codes, softmax=True
        )

        # prune drifted particles
        particles, drifted = pf.remove_invalid_particles(particles)
        if drifted:
            pbar.set_description("All particles have drifted, re-projecting to surface")
            particles.poses, _, _ = codebook.SE3_NN(particles.poses)

        # cluster particles
        if count % 50 == 0:
            particles = pf.cluster_particles(particles)
        cluster_poses, cluster_stds = pf.get_cluster_centers(
            particles, method="quat_avg"
        )

        # anneal and resample
        particles = pf.annealing(particles, torch.mean(cluster_stds))
        particles = pf.resampler(particles)

        # save stats
        timer["meas"] = get_time(start_time)
        filter_stats["cluster_poses"].append(cluster_poses)
        filter_stats["cluster_stds"].append(cluster_stds)
        filter_stats["num_particles"].append(len(particles))

        iteration_time = sum(timer.values())
        filter_stats["time"].append(iteration_time)

        msg = (
            f'[RMSE: {1000 * filter_stats["rmse_t"][-1]:.1f} mm, {filter_stats["rmse_r"][-1]:.0f} deg, '
            f"{len(cluster_stds)} cluster(s) with {torch.mean(cluster_stds):.3f} sigma, P: {len(particles)}, "
            f'rate: {(1.0/filter_stats["time"][-1]):.2f} Hz] '
        )

        for key in timer:
            avg_timer[key].append(timer[key])

        if viz is not None:
            # Update visualizer
            pbar.set_description(msg + " Visualizing results")
            heatmap_weights = pf.get_similarity(
                tactile_code, heatmap_embeddings, softmax=False
            )
            viz.update(
                particles,
                cluster_poses,
                cluster_stds,
                gt_p_cam[idx, :],
                heatmap_poses,
                heatmap_weights,
                image,
                heightmap,
                mask,
                idx,
                image_savepath=osp.join(results_path, f"{idx}.png"),
            )

        prev_idx = idx
        count += 1
        filter_stats["total_time"] = sum(filter_stats["time"])
        pbar.update(diff)

    # save stats and data
    if viz is not None:
        pbar.set_description("End of sequence: saving data")
        viz.close()

    for key in avg_timer:
        avg_timer[key] = np.average(avg_timer[key])

    filter_stats["avg_time"] = sum(filter_stats["time"]) / len(filter_stats["time"])
    print(
        f"Total time: {filter_stats['total_time']:.3f}, Per iteration time: {filter_stats['avg_time']:.3f}"
    )
    print(
        f'Avg time: tactile: {avg_timer["tactile"]:.2f}, motion : {avg_timer["motion"]:.2f}, meas : {avg_timer["meas"]:.2f} '
    )

    print("---------------------------------------------------------\n\n")
    np.save(osp.join(results_path, "filter_stats.npy"), filter_stats)
    pbar.set_description("Generating video from images")
    images_to_video(results_path)  # convert saved images to .mp4
    pbar.close()
    return


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig, viz=None, profile=False):

    if profile:
        yappi.set_clock_type("wall")  # profiling
        yappi.start(builtins=True)

    if cfg.expt.render:
        viz = Viz(off_screen=cfg.expt.off_screen, zoom=1.0, window_size=0.25)

    t = threading.Thread(name="filter", target=filter, args=(cfg, viz))
    t.start()
    if viz:
        viz.plotter.app.exec_()
    t.join()

    if profile:
        stats = yappi.get_func_stats()
        stats.save(osp.join(get_original_cwd(), "filter.prof"), type="pstat")


if __name__ == "__main__":
    main()
