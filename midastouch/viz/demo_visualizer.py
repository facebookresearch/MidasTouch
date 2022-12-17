# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
visualizer class for demo script 
"""

import numpy as np
import pyvista as pv
from matplotlib import cm
from pyvistaqt import BackgroundPlotter
import torch
import copy
from os import path as osp
import queue
from PIL import Image
import tkinter as tk

from midastouch.modules.misc import DIRS

pv.set_plot_theme("document")


class Viz:
    def __init__(
        self, off_screen: bool = False, zoom: float = 1.0, window_size: int = 1.0
    ):

        pv.global_theme.multi_rendering_splitting_position = 0.4
        """
            subplot(0, 0) tactile image viz
            subplot(0, 1): main viz
        """
        shape, row_weights, col_weights = (2, 2), [0.5, 0.5], [0.4, 0.6]
        groups = [(np.s_[:], 0), (0, 1), (1, 1)]

        w, h = tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight()

        self.plotter = BackgroundPlotter(
            title="MidasTouch: CoRL demo",
            lighting="three lights",
            window_size=(int(w * window_size), int(h * window_size)),
            off_screen=off_screen,
            shape=shape,
            row_weights=row_weights,
            col_weights=col_weights,
            groups=groups,
            border_color="white",
            toolbar=False,
            menu_bar=False,
            auto_update=True,
        )
        self.zoom = zoom
        self.mesh_number = 0

        self.mesh_rendered = [False] * 2
        self.viz_queue = queue.Queue(1)
        self.plotter.add_callback(self.update_viz, interval=10)

    def set_camera(self, position="yz", azimuth=45, elevation=20, zoom=None):
        (
            self.plotter.camera_position,
            self.plotter.camera.azimuth,
            self.plotter.camera.elevation,
        ) = (position, azimuth, elevation)
        if zoom is None:
            self.plotter.camera.Zoom(self.zoom)
        else:
            self.plotter.camera.Zoom(zoom)
        self.plotter.camera_set = True

    def set_mug(self, flag):
        if flag:
            self.mesh_number = 0
        else:
            self.mesh_number = 1


    def init_variables(self, mesh_path: str):
        
        self.mesh_pv_deci = [None] * 2
        for i, mesh in enumerate(mesh_path):
            if osp.exists(mesh.replace("nontextured", "nontextured_decimated")):
                mesh = mesh.replace("nontextured", "nontextured_decimated")

            self.mesh_pv_deci[i] = pv.read(mesh)  # decimated pyvista object


        # Heatmap window
        self.plotter.subplot(0, 1)

        widget_size, pos = 20, self.plotter.window_size[0] - 40

        self.set_camera()
        # txt = self.plotter.add_text(
        #     "Mug embeddings",
        #     position="top",
        #     color="black",
        #     shadow=True,
        #     font="times",
        #     font_size=15,
        #     name="Codebook text",
        # )
        self.reset_widget_0 = self.plotter.add_checkbox_button_widget(
            self.set_mug,
            value=True,
            color_off="white",
            color_on="grey",
            position=[500, 325],
            size=widget_size,
        )


        self.plotter.subplot(1, 1)
        # self.plotter.add_text(
        #     "Hammer embeddings",
        #     position="top",
        #     color="black",
        #     shadow=True,
        #     font="times",
        #     font_size=15,
        #     name="Codebook text",
        # )
        self.heatmap_mesh = [None] * 2

        # self.reset_widget_1 = self.plotter.add_checkbox_button_widget(
        #     self.set_hammer,
        #     value=True,
        #     color_off="white",
        #     color_on="grey",
        #     position=[500, 325],
        #     size=widget_size,
        # )

        # Tactile window
        self.plotter.subplot(0, 0)
        self.plotter.camera.Zoom(0.7)
        # self.plotter.add_text(
        #     "Touch to 3D",
        #     position="top",
        #     color="black",
        #     shadow=True,
        #     font="times",
        #     font_size=15,
        #     name="Tactile text",
        # )

        self.viz_count = 0
        self.image_plane, self.heightmap_plane = None, None

    def update_viz(
        self,
    ):
        if self.viz_queue.qsize():
            (
                heatmap_poses,
                heatmap_weights,
                cluster_poses,
                cluster_stds,
                image,
                heightmap,
                mask,
                frame,
                mesh_number
            ) = self.viz_queue.get()
            self.viz_tactile_image(image, heightmap, mask)

            if frame % 10 == 0:
                self.viz_heatmap(
                    heatmap_poses, heatmap_weights, cluster_poses, cluster_stds, mesh_number
                )
            # self.plotter.add_text(
            #     f"\nFrame {frame}   ",
            #     position="upper_right",
            #     color="black",
            #     shadow=True,
            #     font="times",
            #     font_size=12,
            #     name="frame text",
            #     render=True,
            # )
            self.viz_queue.task_done()

    def update(
        self,
        heatmap_poses: torch.Tensor,
        heatmap_weights: torch.Tensor,
        cluster_poses: torch.Tensor,
        cluster_stds: torch.Tensor,
        image: np.ndarray,
        heightmap: np.ndarray,
        mask: np.ndarray,
        frame: int,
    ) -> None:

        if self.viz_queue.full():
            self.viz_queue.get()
        self.viz_queue.put(
            (
                heatmap_poses,
                heatmap_weights,
                cluster_poses,
                cluster_stds,
                image,
                heightmap,
                mask,
                frame,
                self.mesh_number
            ),
            block=False,
        )

    def viz_heatmap(
        self,
        heatmap_poses: torch.Tensor,
        heatmap_weights: torch.Tensor,
        cluster_poses,
        cluster_stds,
        mesh_number
    ) -> None:

        if mesh_number == 0:
            self.plotter.subplot(0, 1)
        else:
            self.plotter.subplot(1, 1)


        heatmap_poses, heatmap_weights = (
            heatmap_poses.cpu().numpy(),
            heatmap_weights.cpu().numpy(),
        )
        heatmap_points = heatmap_poses[:, :3, 3]

        check = np.where(heatmap_weights < np.percentile(heatmap_weights, 98))[0]
        heatmap_weights = np.delete(heatmap_weights, check)
        heatmap_points = np.delete(heatmap_points, check, axis=0)
        heatmap_weights = (heatmap_weights - np.min(heatmap_weights)) / (
            np.max(heatmap_weights) - np.min(heatmap_weights)
        ) + 1.0

        # print(heatmap_weights.min(), np.percentile(heatmap_weights, 95), heatmap_weights.max())
        heatmap_weights = np.nan_to_num(heatmap_weights)
        heatmap_cloud = pv.PolyData(heatmap_points)
        heatmap_cloud["similarity"] = heatmap_weights

        selected_mesh = self.mesh_pv_deci[mesh_number]
        if self.viz_count and self.mesh_rendered[mesh_number]:
            m = selected_mesh.interpolate(
                heatmap_cloud,
                strategy="null_value",
                radius=selected_mesh.length / 50,
                sharpness=1.0,
            )
            self.plotter.update_scalars(
                mesh=self.heatmap_mesh[mesh_number], scalars=m["similarity"], render=False
            )
        else:
            self.heatmap_mesh[mesh_number] = selected_mesh.interpolate(
                heatmap_cloud,
                strategy="null_value",
                radius=selected_mesh.length / 50,
            )
            dargs = dict(
                cmap=cm.get_cmap("viridis"),
                scalars="similarity",
                interpolate_before_map=True,
                ambient=1.0,
                opacity=1.0,
                show_scalar_bar=False,
                silhouette=True,
                clim=[1.0, 1.0 + 1.0],
            )
            self.plotter.add_mesh(self.heatmap_mesh[mesh_number], **dargs)
            self.plotter.set_focus(self.heatmap_mesh[mesh_number].center)
            (
                self.plotter.camera_position,
                self.plotter.camera.azimuth,
                self.plotter.camera.elevation,
            ) = ("yz", 45, 30)
            self.plotter.camera.Zoom(1.0)
            self.plotter.camera_set = True
            self.mesh_rendered[mesh_number] = True
        self.viz_count += 1
        return

    def viz_tactile_image(
        self,
        image: np.ndarray,
        heightmap: torch.Tensor,
        mask: torch.Tensor,
        s: float = 1.8e-3,
    ) -> None:
        if self.image_plane is None:
            self.image_plane = pv.Plane(
                i_size=image.shape[1] * s,
                j_size=image.shape[0] * s,
                i_resolution=image.shape[1] - 1,
                j_resolution=image.shape[0] - 1,
            )
            self.image_plane.points[:, -1] = 0.25
            self.heightmap_plane = copy.deepcopy(self.image_plane)

        # visualize gelsight image
        self.plotter.subplot(0, 0)
        heightmap, mask = heightmap.cpu().numpy(), mask.cpu().numpy()
        image_tex = pv.numpy_to_texture(image)

        heightmap_tex = pv.numpy_to_texture(-heightmap * mask.astype(np.float32))
        self.heightmap_plane.points[:, -1] = (
            np.flip(heightmap * mask.astype(np.float32), axis=0).ravel() * (0.5 * s)
            - 0.15
        )
        self.plotter.add_mesh(
            self.image_plane,
            texture=image_tex,
            smooth_shading=False,
            show_scalar_bar=False,
            name="image",
            render=False,
        )
        self.plotter.add_mesh(
            self.heightmap_plane,
            texture=heightmap_tex,
            cmap=cm.get_cmap("plasma"),
            show_scalar_bar=False,
            name="heightmap",
            render=False,
        )

    def close(self):
        if len(self.images):
            for (im, path) in zip(self.images["im"], self.images["path"]):
                im = Image.fromarray(im.astype("uint8"), "RGB")
                im.save(path)

        self.plotter.close()
