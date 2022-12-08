# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Miscellaneous functions 
"""

import numpy as np

np.seterr(divide="ignore", invalid="ignore")
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch
import GPUtil
import time
import cv2

import os
from os import path as osp
import shutil
import ffmpeg

import matplotlib.pyplot as plt
import git
from PIL import Image
from typing import List, Tuple

plt.rc("pdf", fonttype=42)
plt.rc("ps", fonttype=42)
plt.rc("font", family="serif")
plt.rc("xtick", labelsize="small")
plt.rc("ytick", labelsize="small")

# quicklink to the root and folder directories
root = git.Repo(".", search_parent_directories=True).working_tree_dir
DIRS = {
    "root": root,
    "weights": osp.join(root, "midastouch", "model_weights"),
    "trees": osp.join(root, "midastouch", "tactile_tree", "data"),
    "data": osp.join(root, "YCB-Slide", "dataset"),
    "obj_models": osp.join(root, "YCB-Slide", "dataset", "obj_models"),
    "debug": osp.join(root, "debug"),
}


def get_device(cpu: bool = False, verbose: bool = True) -> str:
    """
    Check GPU utilization and return device for torch
    """
    if cpu:
        device = "cpu"
        if verbose:
            print("Override, using device:", device)
    else:
        try:
            deviceID = GPUtil.getFirstAvailable(
                order="first",
                maxLoad=0.8,
                maxMemory=0.8,
                attempts=5,
                interval=1,
                verbose=False,
            )
            device = torch.device(
                "cuda:" + str(deviceID[0]) if torch.cuda.is_available() else "cpu"
            )
            if verbose:
                print("Using device:", torch.cuda.get_device_name(deviceID[0]))
        except:
            device = "cpu"
            if verbose:
                print("Using device:", device)
    return device


def confusion_matrix(
    embeddings: np.ndarray, sz: int, batch_size: int = 100
) -> np.ndarray:
    """
    get pairwise cosine_similarity for embeddings and generate confusion matrix
    """
    C = np.nan * np.zeros((sz, sz))
    num_batches = sz // batch_size

    embeddings = embeddings.detach().cpu().numpy()
    if num_batches == 0:
        C = cosine_similarity(embeddings, embeddings).squeeze()
    else:
        for i in range(num_batches):
            i_range = (
                np.array(range(i * batch_size, sz))
                if (i == num_batches - 1)
                else np.array(range(i * batch_size, (i + 1) * batch_size))
            )
            for j in range(num_batches):
                j_range = (
                    np.array(range(j * batch_size, sz))
                    if (j == num_batches - 1)
                    else np.array(range(j * batch_size, (j + 1) * batch_size))
                )
                C[i_range[:, None], j_range] = cosine_similarity(
                    embeddings[i_range, :], embeddings[j_range, :]
                ).squeeze()

    # scale 0 to 1
    C = (C - np.min(C)) / np.ptp(C)  # scale [0, 1]
    return C


def color_tsne(C: np.ndarray, TSNE_init: str) -> np.ndarray:
    """
    Project high-dimensional data via TSNE and colormap
    """

    if torch.is_tensor(C):
        C = C.cpu().numpy()
    tsne = TSNE(
        n_components=1,
        verbose=1,
        perplexity=40,
        n_iter=1000,
        init=TSNE_init,
        random_state=0,
        method="exact",
    )

    tsne_encoding = tsne.fit_transform(C)
    tsne_encoding = np.squeeze(tsne_encoding)
    tsne_min, tsne_max = np.min(tsne_encoding), np.max(tsne_encoding)
    tsne_encoding = (tsne_encoding - tsne_min) / (tsne_max - tsne_min)
    colors = plt.cm.Spectral(tsne_encoding)[:, :3]
    return colors


def get_time(start_time, units="sec"):
    """
    Get difference in time since start_time
    """
    elapsedInSeconds = time.time() - start_time
    if units == "sec":
        return elapsedInSeconds
    if units == "min":
        return elapsedInSeconds / 60
    if units == "hour":
        return elapsedInSeconds / (60 * 60)


def view_subplots(image_data: List[np.ndarray], image_mosaic: List[str]) -> None:
    """
    Make subplot mosaic from image data
    """
    fig, axes = plt.subplot_mosaic(mosaic=image_mosaic, constrained_layout=True)
    for j, (label, ax) in enumerate(axes.items()):
        ax.imshow(image_data[j])
        ax.axis("off")
        ax.set_title(label)
    plt.show()


def remove_and_mkdir(results_path: str) -> None:
    """
    Remove directory (if exists) and create
    """
    if osp.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)


def change_to_dir(abspath: str, rel_dir: str) -> None:
    """
    Change path to specified directory
    """
    dname = osp.dirname(abspath)
    os.chdir(dname)
    os.chdir(rel_dir)  # root


def load_heightmap_mask(
    heightmapFile: str, maskFile: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load heightmap and contact mask from file
    """
    try:
        heightmap = cv2.imread(heightmapFile, 0).astype(np.int64)
        contactmask = cv2.imread(maskFile, 0).astype(np.int64)
    except AttributeError:
        heightmap = np.zeros(heightmap.shape).astype(np.int64)
        contactmask = np.zeros(contactmask.shape).astype(np.int64)
    return heightmap, contactmask > 255 / 2


def load_heightmaps_masks(
    heightmapFolder: str, contactmaskFolder: str
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load heightmaps and contact masks from folder
    """

    heightmapFiles = sorted(
        os.listdir(heightmapFolder), key=lambda y: int(y.split("_")[0])
    )
    contactmaskFiles = sorted(
        os.listdir(contactmaskFolder), key=lambda y: int(y.split("_")[0])
    )
    heightmaps, contactmasks = [], []

    for heightmapFile, contactmaskFile in zip(heightmapFiles, contactmaskFiles):
        heightmap, mask = load_heightmap_mask(
            os.path.join(heightmapFolder, heightmapFile),
            os.path.join(contactmaskFolder, contactmaskFile),
        )
        heightmaps.append(heightmap)
        contactmasks.append(mask)
    return heightmaps, contactmasks


def load_images(imageFolder: str, N: int = None) -> np.ndarray:
    """
    Load tactile images from folder (returns N images if specified)
    """

    try:
        imageFiles = sorted(os.listdir(imageFolder), key=lambda y: int(y.split(".")[0]))
    except:
        imageFiles = sorted(os.listdir(imageFolder))
    images = []
    for imageFile in imageFiles:
        if imageFile.endswith(".mp4"):
            continue
        im = Image.open(os.path.join(imageFolder, imageFile))
        images.append(np.array(im))
        if N is not None and len(images) == N:
            return np.stack(images)
    return np.stack(images)  # (N, 3, H, W)


def save_image(tactileImage: np.ndarray, i: int, save_path: str) -> None:
    """
    Save tactile image as .jpg file
    """
    tactileImage = Image.fromarray(tactileImage.astype("uint8"), "RGB")
    tactileImage.save("{path}/{p_i}.jpg".format(path=save_path, p_i=i))


def save_images(tactileImages: List[np.ndarray], save_path: str) -> None:
    """
    Save tactile images as .jpg files
    """
    for i, tactileImage in enumerate(tactileImages):
        save_image(tactileImage, i, save_path)


def save_heightmap(heightmap: np.ndarray, i: int, save_path: str) -> None:
    """
    Save heightmap as .jpg file
    """
    cv2.imwrite(
        "{path}/{p_i}.jpg".format(path=save_path, p_i=i), heightmap.astype("float32")
    )


def save_heightmaps(heightmaps: List[np.ndarray], save_path: str) -> None:
    """
    Save heightmaps as .jpg files
    """
    for i, heightmap in enumerate(heightmaps):
        save_heightmap(heightmap, i, save_path)


def save_contactmask(contactMask: np.ndarray, i: int, save_path: str) -> None:
    """
    Save contact mask as .jpg file
    """
    cv2.imwrite(
        "{path}/{p_i}.jpg".format(path=save_path, p_i=i),
        255 * contactMask.astype("uint8"),
    )


def save_contactmasks(contactMasks: List[np.ndarray], save_path: str) -> None:
    """
    Save contact masks as .jpg files
    """
    for i, contactMask in enumerate(contactMasks):
        save_contactmask(contactMask, i, save_path)


def index_of(val: int, in_list: list) -> int:
    """
    https://stackoverflow.com/a/49522958 : returns index of element in list
    """
    try:
        return in_list.index(val)
    except ValueError:
        return -1


def get_int(file: str) -> int:
    """
    Extract numeric value from file name
    """
    return int(file.split(".")[0])


def images_to_video(path: str) -> None:
    """
    https://stackoverflow.com/a/67152804 : list of images to .mp4
    """
    images = os.listdir(path)
    images = [im for im in images if im.endswith(".png")]
    images = sorted(images, key=get_int)

    # Execute FFmpeg sub-process, with stdin pipe as input, and jpeg_pipe input format
    process = (
        ffmpeg.input("pipe:", r="10")
        .output(osp.join(path, "video.mp4"), vcodec="libx264")
        .global_args("-loglevel", "error")
        .global_args("-y")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for image in images:
        image_path = osp.join(path, image)
        with open(image_path, "rb") as f:
            # Read the JPEG file content to jpeg_data (bytes array)
            jpeg_data = f.read()
            # Write JPEG data to stdin pipe of FFmpeg process
            process.stdin.write(jpeg_data)

    # Close stdin pipe - FFmpeg fininsh encoding the output file.
    process.stdin.close()
    process.wait()
