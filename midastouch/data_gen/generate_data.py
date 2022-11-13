# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Collect data from YCB objects, choose between [random, traj, manual] methods
"""

from midastouch.data_gen.touch_simulator import touch_simulator
from midastouch.modules.objects import ycb_test, ycb_train
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    cfg = cfg.method
    obj_class = ycb_test if cfg.obj_class == "ycb_test" else ycb_train
    for obj_model in obj_class:
        cfg.obj_model = obj_model
        touch_simulator(cfg=cfg)


if __name__ == "__main__":
    main()
