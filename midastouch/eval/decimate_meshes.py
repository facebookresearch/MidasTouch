# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Downsample meshes for faster rendering"""

from os import path as osp
import os
import pyvista as pv
import trimesh
from midastouch.modules.misc import DIRS

obj_paths = osp.join(DIRS["obj_models"])
objects = sorted(os.listdir(obj_paths))
for object in objects:
    stl_path = osp.join(obj_paths, object, "nontextured.stl")
    mesh_trimesh = trimesh.load(stl_path)
    mesh_pv_deci = pv.wrap(
        mesh_trimesh.simplify_quadratic_decimation(
            face_count=int(mesh_trimesh.vertices.shape[0] / 10)
        )
    )  # decimated pyvista object
    stl_path = stl_path.replace("nontextured", "nontextured_decimated")
    print(f"Saving: {stl_path}")
    mesh_pv_deci.save(stl_path)
