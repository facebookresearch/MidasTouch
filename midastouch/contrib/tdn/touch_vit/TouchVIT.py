import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter

from PIL import Image

from midastouch.contrib.tdn.touch_vit.FocusOnDepth import FocusOnDepth
from midastouch.contrib.tdn.touch_vit.utils import create_dir
from midastouch.contrib.tdn.touch_vit.dataset import show
from midastouch.modules.misc import DIRS

from glob import glob
import yappi
from tqdm import tqdm
from omegaconf import DictConfig

import hydra
from hydra.utils import to_absolute_path
from hydra import compose, initialize


class TouchVIT:
    """
    Image to 3D model for DIGIT
    """

    def __init__(self, cfg: DictConfig):
        super(TouchVIT, self).__init__()

        self.config = cfg
        input_dir = to_absolute_path(self.config["General"]["path_input_images"])
        self.input_images = glob(f"{input_dir}/*.jpg") + glob(f"{input_dir}/*.png")

        self.type = self.config["General"]["type"]

        self.device = torch.device(
            self.config["General"]["device"] if torch.cuda.is_available() else "cpu"
        )
        # print("device: %s" % self.device)
        resize = self.config["Dataset"]["transforms"]["resize"]
        self.model = FocusOnDepth(
            image_size=(3, resize[0], resize[1]),
            emb_dim=self.config["General"]["emb_dim"],
            resample_dim=self.config["General"]["resample_dim"],
            read=self.config["General"]["read"],
            nclasses=0,
            hooks=self.config["General"]["hooks"],
            model_timm=self.config["General"]["model_timm"],
            type=self.type,
            patch_size=self.config["General"]["patch_size"],
        )

        path_model = os.path.join(DIRS["weights"], "FocusOnDepth.p")

        # print(f"TouchVIT path: {path_model}")
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()
        self.model.to(self.device)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((resize[0], resize[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.output_dir = self.config["General"]["path_predicted_images"]

    def image2heightmap(self, image):
        image = Image.fromarray(image)
        original_size = image.size
        image = self.transform_image(image).unsqueeze(0)
        image = image.to(self.device).float()

        output_depth, _ = self.model(image)
        output_depth = transforms.ToPILImage()(output_depth.squeeze(0).float()).resize(
            original_size, resample=Image.BICUBIC
        )
        return transforms.PILToTensor()(output_depth).squeeze().float()

    def run(self):

        yappi.set_clock_type("wall")  # profiling
        yappi.start(builtins=True)

        path_dir_depths = os.path.join(self.output_dir, "depths")
        create_dir(self.output_dir)
        create_dir(path_dir_depths)

        tensor_ims = []
        for images in self.input_images[:10]:
            pil_im = Image.open(images)
            original_size = pil_im.size
            tensor_ims.append(self.transform_image(pil_im).unsqueeze(0))

        tensor_ims = torch.vstack(tensor_ims)

        with torch.no_grad():
            print(f"Running on {len(self.input_images)} images")
            output_depth = self.image_to_3D(tensor_ims)

            # for tensor_im in tqdm(tensor_ims):
            # output_depth = 1 - output_depth
            # output_depth = transforms.ToPILImage()(
            #     output_depth.squeeze(0).float()
            # ).resize(original_size, resample=Image.BICUBIC)

            # output_depth.save(
            #     os.path.join(path_dir_depths, os.path.basename(images))
            # )

        stats = yappi.get_func_stats()
        stats.save(os.path.join(path_dir_depths, "filter.prof"), type="pstat")


if __name__ == "__main__":
    t = TouchVIT()
    t.run()
