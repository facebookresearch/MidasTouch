#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Generate tactile codebook for all YCB objects


declare -a objModels=("004_sugar_box" "005_tomato_soup_can" "006_mustard_bottle" "021_bleach_cleanser" "025_mug" "035_power_drill" "037_scissors" "042_adjustable_wrench" "048_hammer" "055_baseball")

for obj in ${objModels[@]}; do
    python midastouch/tactile_tree/build_codebook.py expt.obj_model=$obj tdn.render.pen.max=0.001 expt.codebook_size=50000
done
