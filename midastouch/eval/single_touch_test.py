# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate statistic for  single-shot localization of objects"""

import os
from os import path as osp
import numpy as np

from midastouch.modules.objects import ycb_test
from midastouch.modules.misc import change_to_dir, DIRS
from midastouch.viz.helpers import viz_embedding_TSNE
import dill as pickle
from sklearn.metrics.pairwise import cosine_similarity

from midastouch.tactile_tree.tactile_tree import R3_SE3
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rc("pdf", fonttype=42)
plt.rc("ps", fonttype=42)
plt.rc("font", family="serif")

plt.rc("xtick", labelsize="small")
plt.rc("ytick", labelsize="small")

NUM_NEIGHBORS = 25


def top_n_error(embeddings, poses, n=NUM_NEIGHBORS):
    """
    Get best embedding error from top-N poses
    """
    N = poses.shape[0]

    top_n_error = np.zeros(N)

    batch_size = 5000
    num_batches = N // batch_size
    num_batches = 1 if num_batches == 0 else num_batches

    C = np.zeros((N, N))
    for i in tqdm(range(num_batches)):
        i_range = (
            np.array(range(i * batch_size, N))
            if (i == num_batches - 1)
            else np.array(range(i * batch_size, (i + 1) * batch_size))
        )
        for j in range(num_batches):
            j_range = (
                np.array(range(j * batch_size, N))
                if (j == num_batches - 1)
                else np.array(range(j * batch_size, (j + 1) * batch_size))
            )
            C[i_range[:, None], j_range] = cosine_similarity(
                np.atleast_2d(embeddings[i_range, :]),
                np.atleast_2d(embeddings[j_range, :]),
            ).squeeze()

    np.fill_diagonal(C, 0)
    for i in range(N):
        best_idxs = np.argpartition(C[i, :], -(n))[-n:]
        predicted_poses = poses[best_idxs, :]
        gt_pose = poses[i, :]
        e_t = np.linalg.norm(predicted_poses - gt_pose, axis=1)
        top_n_error[i] = np.min(e_t)

    return top_n_error


def get_random_error(poses, n=NUM_NEIGHBORS):
    """
    Get random pose error
    """
    N = poses.shape[0]

    rand_error = np.zeros(N)
    for i in range(N):
        pred_idxs = np.random.choice(N, size=n)
        predicted_poses = poses[pred_idxs, :]
        gt_pose = poses[i, :]
        e_t = np.linalg.norm(predicted_poses - gt_pose, axis=1)
        rand_error[i] = np.min(e_t)
    return np.mean(rand_error)


def plot_violin(df, method="pointcloud"):
    """
    Image or pointcloud violin plot
    """

    change_to_dir(osp.abspath(__file__))

    results_path = "single_touch"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df = pd.read_pickle(osp.join(results_path, f"error_{method}.pkl"))

    savepath = osp.join(results_path, f"violin_{method}.pdf")

    fig = plt.figure()

    df = df.sort_values(by=["median_error"], ascending=True)
    # sns.set_theme(style="whitegrid")
    palette = sns.color_palette("vlag", n_colors=len(df["key"].unique()))
    sns.violinplot(
        x="key",
        y="error",
        data=df,
        palette=palette,
        cut=0,
        gridsize=10000,
        saturation=1,
        linewidth=0.5,
    )
    # df.reset_index(level=0, inplace=True)
    # ax = sns.lineplot(x = "key", y = "median_error", data=df, color="black", legend=False, linewidth=0.5)
    # ax.lines[0].set_linestyle("--")

    plt.xlabel("YCB object models", fontsize=12)
    plt.ylabel(f"Normalized Top-{NUM_NEIGHBORS} pose error", fontsize=12)

    ax = plt.gca()

    plt.ylim([0, 1.5])
    plt.axhline(y=1.0, linestyle="--", linewidth=0.3, color=(0, 0, 0, 0.75))

    figure = plt.gcf()
    figure.set_size_inches(12, 4)
    plt.savefig(savepath, transparent=True, bbox_inches="tight", pad_inches=0)
    print("saved to ", savepath)
    plt.close()


def plot_split_violin():
    """
    Image and pointcloud violin split plot
    """

    print("Plotting split violin")
    change_to_dir(osp.abspath(__file__))

    matplotlib.use("TkAgg")

    results_path = "single_touch"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    cloud_df = pd.read_pickle(osp.join(results_path, "error_cloud.pkl"))
    image_df = pd.read_pickle(osp.join(results_path, "error_image.pkl"))

    get_overall_median(cloud_df, method="pointcloud")
    get_overall_median(image_df, method="image")

    savepath = osp.join(results_path, "violin_split.pdf")

    df = pd.DataFrame()
    df = df.append(cloud_df)
    df = df.append(image_df)

    fig = plt.figure()
    # df = df.sort_values(by=["median_error"], ascending=True)
    # sns.set_theme(style="whitegrid")
    # palette = sns.color_palette("vlag", n_colors = len(df["key"].unique()))
    muted_pal = sns.color_palette("colorblind")
    my_pal = {"cloud": muted_pal[0], "image": muted_pal[-1]}
    ax = sns.violinplot(
        x="key",
        y="error",
        data=df,
        hue="method",
        split=True,
        inner="quart",
        linewidth=0.5,
        palette=my_pal,
        cut=0,
        gridsize=1000,
        saturation=1,
    )
    ax.legend_.remove()

    sns.despine(left=True)

    plt.xlabel("YCB object models", fontsize=12)
    plt.ylabel(f"Normalized Top-{NUM_NEIGHBORS} pose error", fontsize=12)

    ax = plt.gca()

    plt.ylim([0, 1.5])
    plt.axhline(y=1.0, linestyle="--", linewidth=0.3, color=(0, 0, 0, 0.75))

    figure = plt.gcf()
    figure.set_size_inches(12, 4)
    plt.savefig(savepath, transparent=True, bbox_inches="tight", pad_inches=0)
    print("saved to ", savepath)
    plt.close()

    return


def benchmark_embeddings(obj_models, method="pointcloud"):
    """
    Benchmark embedding error
    """

    change_to_dir(osp.abspath(__file__))

    results_path = "single_touch"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df = pd.DataFrame()
    error_file = open(osp.join(results_path, f"error_{method}.txt"), "w")

    for obj_model in obj_models:

        if method == "image":
            pickle_path = osp.join(DIRS["trees"], obj_model, "image_codebook.pkl")
        else:
            pickle_path = osp.join(DIRS["trees"], obj_model, "codebook.pkl")

        print(f"Loading tree {pickle_path}")
        with open(pickle_path, "rb") as pickle_file:
            tactile_tree = pickle.load(pickle_file)

        poses, _ = tactile_tree.get_poses()
        poses = R3_SE3(poses)
        # poses = poses[:, :3]

        embeddings = tactile_tree.get_embeddings()
        obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")
        print(f"Getting top-{NUM_NEIGHBORS} {method} error")

        error_n = top_n_error(embeddings, poses, n=NUM_NEIGHBORS)
        # error_1 = top_n_error(embeddings, poses, n = 1)
        random_error = get_random_error(poses, n=NUM_NEIGHBORS)

        # normalized pose error
        # error_1 /= random_error
        error_n /= random_error

        print(error_n)

        topN = pd.DataFrame(
            {
                "error": error_n.tolist(),
                "median_error": [np.median(error_n)] * len(error_n),
                "key": [obj_model[:3]] * len(error_n),
                "method": [method] * len(error_n),
            }
        )
        df = df.append(topN)

        print(
            f"{obj_model} : Median norm. pose RMSE Top {NUM_NEIGHBORS}: {np.median(error_n):.4f}"
        )
        error_file.write(
            f"{obj_model} : Median norm. pose RMSE Top {NUM_NEIGHBORS}: {np.median(error_n):.4f}\n"
        )

        viz_embedding_TSNE(
            mesh_path=obj_path,
            samples=poses.copy(),
            clusters=error_n,
            save_path=osp.join(results_path, f"{obj_model}_cloud_error"),
            nPoints=None,
            radius_factor=50.0,
        )

    df.to_pickle(osp.join(results_path, f"error_{method}.pkl"))
    error_file.close()
    return df


def get_overall_median(df, method="pointcloud"):
    all_median = df["median_error"].median()
    print(f"Overall median error (method: {method}) = {all_median}")
    return


if __name__ == "__main__":
    config_file = "scripts/midastouch/config/config.ini"
    df = benchmark_embeddings(ycb_test, method="pointcloud")
    plot_violin(df=df)
    # plot_split_violin()
