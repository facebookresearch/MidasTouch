#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Run filtering experiments for all YCB objects

declare -a objModels=("004_sugar_box" "005_tomato_soup_can" "006_mustard_bottle" "021_bleach_cleanser" "025_mug" "035_power_drill" "037_scissors" "042_adjustable_wrench" "048_hammer" "055_baseball")
# declare -a objModels=("cotter-pin" "steel-nail" "eyebolt")

for log in {0..4}; do
    for obj in ${objModels[@]}; do
        python midastouch/filter/filter.py expt.obj_model=$obj expt.log_id=$log
        # python midastouch/filter/filter_real.py expt.obj_model=$obj expt.log_id=$log
    done
done
