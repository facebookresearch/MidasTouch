# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper functions for visualizer
"""

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm


def viz_poses_pointclouds_on_mesh(
    mesh_path, poses, pointclouds, save_path=None, decimation_factor=5
):
    if type(pointclouds) is not list:
        temp = pointclouds
        pointclouds = [None] * 1
        pointclouds[0] = temp

    plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)

    mesh = pv.read(mesh_path)  # pyvista object
    dargs = dict(
        color="grey",
        ambient=0.6,
        opacity=0.5,
        smooth_shading=True,
        specular=1.0,
        show_scalar_bar=False,
        render=False,
    )
    plotter.add_mesh(mesh, **dargs)
    draw_poses(plotter, mesh, poses, quiver_size=0.05)

    if poses.ndim == 2:
        spline = pv.lines_from_points(poses[:, :3])
        plotter.add_mesh(spline, line_width=3, color="k")

    final_pc = np.empty((0, 3))
    for i, pointcloud in enumerate(pointclouds):
        if pointcloud.shape[0] == 0:
            continue
        if decimation_factor is not None:
            downpcd = pointcloud[
                np.random.choice(
                    pointcloud.shape[0],
                    pointcloud.shape[0] // decimation_factor,
                    replace=False,
                ),
                :,
            ]
        else:
            downpcd = pointcloud
        final_pc = np.append(final_pc, downpcd)

    if final_pc.shape[0]:
        pc = pv.PolyData(final_pc)
        plotter.add_points(
            pc, render_points_as_spheres=True, color="#26D701", point_size=3
        )

    if save_path:
        plotter.show(screenshot=save_path)
        print(f"Save path: {save_path}.png")
    else:
        plotter.show()
    plotter.close()
    pv.close_all()


def viz_query_target_poses_on_mesh(mesh_path, query_pose, target_poses):
    plotter = pv.Plotter(window_size=[2000, 2000])

    mesh = pv.read(mesh_path)  # pyvista object
    dargs = dict(
        color="grey",
        ambient=0.6,
        opacity=0.5,
        smooth_shading=True,
        specular=1.0,
        show_scalar_bar=False,
        render=False,
    )
    plotter.add_mesh(mesh, **dargs)

    draw_poses(plotter, mesh, target_poses, opacity=0.7)
    draw_poses(plotter, mesh, query_pose)
    dargs = dict(
        color="grey",
        ambient=0.6,
        opacity=0.6,
        smooth_shading=True,
        show_edges=False,
        specular=1.0,
        show_scalar_bar=False,
    )
    plotter.add_mesh(mesh, **dargs)
    plotter.show()
    plotter.close()
    pv.close_all()


def draw_poses(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    cluster_poses: np.ndarray,
    opacity: float = 1.0,
    quiver_size=0.1,
) -> None:
    """
    Draw pose RGB coordinate axes for pose set in pyvista visualizer
    """
    quivers = pose2quiver(cluster_poses, quiver_size * mesh.length)
    quivers = [quivers["xvectors"]] + [quivers["yvectors"]] + [quivers["zvectors"]]
    names = ["xvectors", "yvectors", "zvectors"]
    colors = ["r", "g", "b"]
    cluster_centers = cluster_poses[:, :3, 3]
    for (q, c, n) in zip(quivers, colors, names):
        plotter.add_arrows(
            cluster_centers,
            q,
            color=c,
            opacity=opacity,
            show_scalar_bar=False,
            render=False,
            name=n,
        )


def draw_graph(x, y, savepath, delay, flag="t"):
    fig, ax = plt.subplots()

    plt.xlabel("Timestep", fontsize=12)
    if flag == "t":
        plt.ylabel("Avg. translation RMSE (mm)", fontsize=12)
        y = [y_ * 1000.0 for y_ in y]
    elif flag == "r":
        plt.ylabel("Avg. rotation RMSE (deg)", fontsize=12)

    # rolling avg. over 10 timesteps
    import pandas as pd

    df = pd.DataFrame()
    N = 50
    df["y"] = y
    df_smooth = df.rolling(N).mean()
    df_smooth["y"][0 : N - 1] = y[0 : N - 1]  # first 10 readings are as-is
    y = df_smooth["y"]

    N, maxy = len(x), max(y)
    (line,) = ax.plot(x, y, color="k")

    def update(num, x, y, line):
        line.set_data(x[:num], y[:num])
        line.axes.axis([0, N, 0, maxy])
        return (line,)

    ani = animation.FuncAnimation(
        fig, update, len(x), fargs=[x, y, line], interval=delay, blit=True
    )
    ani.save(savepath + ".mp4", writer="ffmpeg", codec="h264")
    fig.savefig(savepath + ".pdf", transparent=True, bbox_inches="tight", pad_inches=0)


def pose2quiver(poses, sz):
    """
    Convert pose to quiver object (RGB)
    """
    poses = np.atleast_3d(poses)
    quivers = pv.PolyData(poses[:, :3, 3])  # (N, 3) [x, y, z]
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    r = R.from_matrix(poses[:, 0:3, 0:3])
    quivers["xvectors"], quivers["yvectors"], quivers["zvectors"] = (
        r.apply(x) * sz,
        r.apply(y) * sz,
        r.apply(z) * sz,
    )
    return quivers


def viz_embedding_TSNE(
    mesh_path,
    samples,
    clusters,
    save_path,
    nPoints=500,
    radius_factor=80.0,
    off_screen=False,
):
    samples = np.atleast_2d(samples)
    samplePoints = pv.PolyData(samples[:, :3])
    samplePoints["similarity"] = clusters

    mesh_pv = pv.read(mesh_path)  # pyvista object

    mesh = mesh_pv.interpolate(
        samplePoints,
        strategy="mask_points",
        radius=mesh_pv.length / radius_factor,
    )
    p = pv.Plotter(off_screen=off_screen, window_size=[1000, 1000])

    # replace black with gray
    if clusters.ndim == 2:
        null_idx = np.all(mesh["similarity"] == np.array([0.0, 0.0, 0.0]), axis=1)
        mesh["similarity"][null_idx, :] = np.array([189 / 256, 189 / 256, 189 / 256])

    # Open a gif
    if clusters.ndim == 2:
        dargs = dict(
            scalars="similarity",
            rgb=True,
            interpolate_before_map=False,
            opacity=1,
            smooth_shading=True,
            show_scalar_bar=False,
            silhouette=True,
        )
    else:
        dargs = dict(
            scalars="similarity",
            cmap=cm.get_cmap("plasma"),
            interpolate_before_map=False,
            opacity=1,
            smooth_shading=True,
            show_scalar_bar=False,
            silhouette=True,
        )
    p.add_mesh(mesh, **dargs)

    if nPoints is not None:
        p.show(screenshot=save_path, auto_close=not off_screen)
        viewup = [0.5, 0.5, 1]
        path = p.generate_orbital_path(
            factor=8.0,
            viewup=viewup,
            n_points=nPoints,
            shift=mesh.length / (np.sqrt(3)),
        )
        p.open_movie(save_path + ".mp4")
        p.orbit_on_path(
            path, write_frames=True, viewup=[0, 0, 1], step=0.01, progress_bar=True
        )
    else:
        p.show(screenshot=save_path)
    p.close()
    pv.close_all()
