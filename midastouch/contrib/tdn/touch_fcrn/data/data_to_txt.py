# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Load train/test data structure and save to text file, preprocessing for depth training
"""

import numpy as np
import os
from os import path as osp

# change the data_root to the root directory of dataset with all objects
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
os.chdir(dname)

# write training/validation/testing data loader files
train_data_file = open("train_data.txt", "w")
train_label_file = open("train_label.txt", "w")

dev_data_file = open("dev_data.txt", "w")
dev_label_file = open("dev_label.txt", "w")

test_data_file = open("test_data.txt", "w")
test_label_file = open("test_label.txt", "w")

global_train_idx = 0
global_dev_idx = 0
global_test_idx = 0

# different tactile background models: add multiple folders here, below is placeholder
data_root_paths = ["/mnt/sda/fcrn/fcrn_data"]

for data_root_path in data_root_paths:
    object_folders = sorted(os.listdir(data_root_path))
    for object in object_folders:
        if object == ".DS_Store":
            continue
        _, ext = os.path.splitext(object)
        if ext == ".pickle":
            continue
        print("Object: ", object)

        # load in tactile images and ground truth height maps
        tactile_path = osp.join(data_root_path, object, "tactile_images")
        gt_heightmap_path = osp.join(data_root_path, object, "gt_heightmaps")
        gt_contactmask_path = osp.join(data_root_path, object, "gt_contactmasks")

        num_imgs = len(os.listdir(tactile_path))
        all_random_idx = np.random.permutation(num_imgs)
        num_train = int(0.8 * num_imgs)
        num_dev = int(0.1 * num_imgs)
        num_test = int(0.1 * num_imgs)

        train_idx = all_random_idx[0:num_train]
        dev_idx = all_random_idx[num_train : num_train + num_dev]
        test_idx = all_random_idx[num_train + num_dev : num_train + num_dev + num_test]

        for idx in train_idx:
            train_data_file.write(
                str(global_train_idx)
                + ","
                + tactile_path
                + "/"
                + str(idx)
                + ".jpg"
                + "\n"
            )
            train_label_file.write(
                str(global_train_idx)
                + ","
                + gt_heightmap_path
                + "/"
                + str(idx)
                + ".jpg"
                + ","
                + gt_contactmask_path
                + "/"
                + str(idx)
                + ".jpg"
                + "\n"
            )
            global_train_idx += 1

        for idx in dev_idx:
            dev_data_file.write(
                str(global_dev_idx)
                + ","
                + tactile_path
                + "/"
                + str(idx)
                + ".jpg"
                + "\n"
            )
            dev_label_file.write(
                str(global_dev_idx)
                + ","
                + gt_heightmap_path
                + "/"
                + str(idx)
                + ".jpg"
                + ","
                + gt_contactmask_path
                + "/"
                + str(idx)
                + ".jpg"
                + "\n"
            )
            global_dev_idx += 1

        for idx in test_idx:
            test_data_file.write(
                str(global_test_idx)
                + ","
                + tactile_path
                + "/"
                + str(idx)
                + ".jpg"
                + "\n"
            )
            test_label_file.write(
                str(global_test_idx)
                + ","
                + gt_heightmap_path
                + "/"
                + str(idx)
                + ".jpg"
                + ","
                + gt_contactmask_path
                + "/"
                + str(idx)
                + ".jpg"
                + "\n"
            )
            global_test_idx += 1

print(
    "Train size: {}, Val size: {}, test size: {}".format(
        global_train_idx, global_dev_idx, global_test_idx
    )
)
train_data_file.close()
train_label_file.close()
dev_data_file.close()
dev_label_file.close()
test_data_file.close()
test_label_file.close()
