#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cd midastouch
echo "Downloading the TDN/TCN model weights"
gdown https://drive.google.com/drive/folders/1Zy1yFJl3-3Q3Ms0NWb2aTZDXMdEj6dW9?usp=sharing --folder
cd tactile_tree
echo "Downloading the YCB tactile codebooks"
gdown --fuzzy https://drive.google.com/file/d/165Bj9eqVJ0As5vitIoPAOU-9Jij01jpB/view?usp=sharing
unzip codebooks.zip
rm codebooks.zip
cd ../..
echo "Done!"

