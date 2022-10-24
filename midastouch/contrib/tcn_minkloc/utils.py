# Author: Jacek Komorowski
# Warsaw University of Technology

# Original source: https://github.com/jac99/MinkLoc3D
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import time


class ModelParams:
    def __init__(self, mode_config):
        params = mode_config

        self.model = params.model
        self.output_dim = params.output_dim  # Size of the final descriptor

        # Add gating as the last step
        if "vlad" in self.model.lower():
            self.cluster_size = params.cluster_size  # Size of NetVLAD cluster
            self.gating = params.gating  # Use gating after the NetVlad

        #######################################################################
        # Model dependent
        #######################################################################

        if "MinkFPN" in self.model:
            # Models using MinkowskiEngine
            self.mink_quantization_size = params.mink_quantization_size
            # Size of the local features from backbone network (only for MinkNet based models)
            # For PointNet-based models we always use 1024 intermediary features
            self.feature_size = params.feature_size
            if params.planes:
                self.planes = [int(e) for e in params.planes.split(",")]
            else:
                self.planes = [32, 64, 64]

            if params.layers:
                self.layers = [int(e) for e in params.layers.split(",")]
            else:
                self.layers = [1, 1, 1]

            self.num_top_down = params.num_top_down
            self.conv0_kernel_size = params.conv0_kernel_size

    def print(self):
        print("Model parameters:")
        param_dict = vars(self)
        for e in param_dict:
            print("{}: {}".format(e, param_dict[e]))

        print("")


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class MinkLocParams:
    """
    Params for training MinkLoc models
    """

    def __init__(self, config):
        """
        Configuration files
        :param path: configuration file
        """
        self.num_points = config.model.num_points
        self.dataset_folder = config.train.dataset_folder
        self.val_folder = config.train.val_folder
        self.eval_folder = config.train.eval_folder

        self.num_workers = config.train.num_workers
        self.batch_size = config.train.batch_size
        self.val_batch_size = config.train.val_batch_size
        self.max_batches = config.train.max_batches

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = config.train.batch_expansion_th
        if self.batch_expansion_th is not None:
            assert (
                0.0 < self.batch_expansion_th < 1.0
            ), "batch_expansion_th must be between 0 and 1"
            self.batch_size_limit = config.train.batch_size_limit
            # Batch size expansion rate
            self.batch_expansion_rate = config.train.batch_expansion_rate
            assert (
                self.batch_expansion_rate > 1.0
            ), "batch_expansion_rate must be greater than 1"
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = config.train.lr

        self.scheduler = config.train.scheduler
        if self.scheduler is not None:
            if self.scheduler == "CosineAnnealingLR":
                self.min_lr = config.train.min_lr
            elif self.scheduler == "MultiStepLR":
                scheduler_milestones = config.train.scheduler_milestones
                self.scheduler_milestones = [
                    int(e) for e in scheduler_milestones.split(",")
                ]
            else:
                raise NotImplementedError(
                    "Unsupported LR scheduler: {}".format(self.scheduler)
                )

        self.epochs = config.train.epochs
        self.weight_decay = config.train.weight_decay
        self.normalize_embeddings = (
            config.train.normalize_embeddings
        )  # Normalize embeddings during training and evaluation
        self.loss = config.train.loss

        if "Contrastive" in self.loss:
            self.pos_margin = config.train.pos_margin
            self.neg_margin = config.train.neg_margin
        elif "Triplet" in self.loss:
            self.margin = config.train.margin  # Margin used in loss function
        else:
            raise "Unsupported loss function: {}".format(self.loss)

        self.aug_mode = config.train.aug_mode  # Augmentation mode (1 is default)

        self.train_file = config.train.train_file
        self.val_file = config.train.val_file

        # Read model parameters
        self.model_params = ModelParams(config.model)
        # self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), "Cannot access dataset: {}".format(
            self.dataset_folder
        )

    def print(self):
        print("Parameters:")
        param_dict = vars(self)
        for e in param_dict:
            if e != "model_params":
                print("{}: {}".format(e, param_dict[e]))

        self.model_params.print()
        print("")
