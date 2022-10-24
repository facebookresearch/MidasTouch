# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualize TSNE of tactile embeddings
"""

from os import path as osp
from midastouch.viz.helpers import viz_embedding_TSNE
import dill as pickle
from midastouch.modules.misc import DIRS, get_device, confusion_matrix, color_tsne
from midastouch.modules.objects import ycb_test


def viz_codebook(obj_model):
    print("model: ", obj_model)

    device = get_device(cpu=False)

    tree_path = osp.join(DIRS["trees"], obj_model, "codebook.pkl")
    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")

    codebook = pickle.load(open(tree_path, "rb"))
    codebook.to_device(device)

    poses, _ = codebook.get_poses()
    embeddings = codebook.get_embeddings()
    sz = len(codebook)
    print("Visualize tree of size: {}".format(sz))

    # euclidean TSNE is proportional to cosine distance is the features are normalized,
    # so we can skip the confusion matrix computation
    print(f"Generating feature embedding scores {embeddings.shape[1]}")
    if embeddings.shape[1] > 256:
        C = confusion_matrix(embeddings, sz)
        TSNE = color_tsne(C, "pca", osp.join(DIRS["obj_models"], obj_model))
    else:
        TSNE = color_tsne(embeddings, "pca", osp.join(DIRS["obj_models"], obj_model))

    print("Viz. TSNE")
    viz_embedding_TSNE(
        mesh_path=obj_path,
        samples=poses,
        clusters=TSNE,
        save_path=osp.join(tree_path, "tsne"),
        nPoints=500,
        radius_factor=80.0,
        off_screen=False,
    )
    return


if __name__ == "__main__":
    obj_models = ycb_test
    for obj_model in obj_models:
        viz_codebook(obj_model)
