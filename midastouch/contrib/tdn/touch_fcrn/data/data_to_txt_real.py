# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Load real test data structure and save to text file, preprocessing for depth test
"""

import os
from os import path as osp
import random

# change the data_root to the root directory of dataset with all objects
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
os.chdir(dname)

data_root_path = "/home/robospare/suddhu/midastouch/data/real/"
objects = sorted(os.listdir(data_root_path))

# write training/validation/testing data loader files
test_data_file = open("test_data_real.txt", "w")

global_test_idx = 0
for object in objects:
    obj_path = osp.join(data_root_path, object)
    if not osp.isdir(obj_path):
        continue
    datasets = sorted(os.listdir(obj_path))
    print("dataset: ", object)

    for dataset in datasets:
        dataset_path = osp.join(obj_path, dataset)
        if dataset == "bg" or not osp.isdir(dataset_path):
            continue
        # load in tactile images from real sensor
        tactile_path = osp.join(dataset_path, "frames")
        imgs = sorted(os.listdir(tactile_path))
        imgs = [x for x in imgs if ".jpg" in x]

        if len(imgs) > 10:
            imgs = random.sample(imgs, 10)
        for i, img in enumerate(imgs):
            test_data_file.write(str(i) + "," + tactile_path + "/" + img + "\n")
            global_test_idx += 1

print("Real test data size: {}".format(global_test_idx))
test_data_file.close()
