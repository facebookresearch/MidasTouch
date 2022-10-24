# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to modify contents of tactile codebook and overwrite pickle file
"""

import dill as pickle
from os import path as osp
from midastouch.tactile_tree.tactile_tree import tactile_tree
from midastouch.modules.misc import DIRS
import os


def main():
    codebooks_path = osp.join(DIRS["trees"])

    objects = sorted(os.listdir(codebooks_path))

    for object in objects:
        try:
            pickle_path = osp.join(codebooks_path, object, "codebook.pkl")
            with open(pickle_path, "rb") as pickle_file:
                T = pickle.load(pickle_file)

            poses, cam_poses = T.poses, T.cam_poses

            """
            add intermediate processes here
            """

            new_T = tactile_tree(
                poses=poses, cam_poses=cam_poses, embeddings=T.embeddings
            )
            pickle_path = osp.join(codebooks_path, object, "codebook.pkl")
            print("Saving data to path: {}".format(pickle_path))
            with open(pickle_path, "wb") as file:
                pickle.dump(new_T, file)
        except:
            print(f"Error building tree: {object}")


if __name__ == "__main__":
    main()
