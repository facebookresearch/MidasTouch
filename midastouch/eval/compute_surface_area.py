# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Compute surface area of object v.s. sensor size
"""

from os import path as osp
import trimesh
from midastouch.modules.misc import DIRS
from midastouch.modules.objects import ycb_test


def compute_surface_area(obj_model):
    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")
    mesh = trimesh.load(obj_path)

    mesh_area_cm_sq = mesh.area * (10**4)
    digit_area_cm_sq = 0.02 * 0.03 * (10**4)
    ratio = mesh_area_cm_sq / digit_area_cm_sq
    print(f"{obj_model} surface area: {mesh_area_cm_sq:.3f}, ratio: {ratio:.1f}")
    return


if __name__ == "__main__":
    obj_models = ycb_test
    for obj_model in obj_models:
        compute_surface_area(obj_model)
