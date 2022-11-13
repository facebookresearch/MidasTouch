# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pose and mesh utilities for data generation
"""

import numpy as np
import pyvista as pv
import open3d as o3d
import potpourri3d as pp3d
import random
from scipy.spatial import KDTree
import time
import math
import trimesh
from midastouch.modules.pose import (
    pose_from_vertex_normal,
    tf_to_xyzquat,
)


def get_geodesic_path(mesh_path: str, start_point, end_point):
    """
    Get geodesic path along a mesh given start and end vertices (pyvista geodesic_distance)
    """
    mesh = pv.read(mesh_path)
    start_point_idx = np.argmin(np.linalg.norm(start_point - mesh.points, axis=1))
    end_point_idx = np.argmin(np.linalg.norm(end_point - mesh.points, axis=1))
    path_pts = mesh.geodesic(start_point_idx, end_point_idx)
    path_distance = mesh.geodesic_distance(start_point_idx, end_point_idx)
    return path_pts.points, path_distance


def random_geodesic_poses(mesh, shear_mag, total_length=0.5, N=2000):
    """Generate random points and compute geodesic trajectory"""

    cumm_length = 0.0
    num_waypoints = 1
    seg_length = total_length / float(num_waypoints)
    while seg_length > 0.25 * mesh.scale:
        num_waypoints += 1
        seg_length = total_length / float(num_waypoints)

    V, F = np.array(mesh.vertices), np.array(mesh.faces)
    print(f"num_waypoints: {num_waypoints}")
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
    path_solver = pp3d.EdgeFlipGeodesicSolver(
        V, F
    )  # shares precomputation for repeated solves

    sample_points, sample_normals = np.empty((0, 3)), np.empty((0, 3))
    seg_start = random.randint(0, V.shape[0])

    waypoints = V[seg_start, None]
    start_time = time.time()
    tree = KDTree(mesh.vertices)

    for _ in range(num_waypoints):
        geo_dist = solver.compute_distance(seg_start)
        candidates = np.argsort(np.abs(geo_dist - seg_length))

        waypoint_dist = np.linalg.norm(V[candidates, :, None] - waypoints.T, axis=1)
        waypoint_dist = np.amin(waypoint_dist, axis=1)
        mask = waypoint_dist < 0.01
        candidates = np.ma.MaskedArray(candidates, mask=mask)
        candidates = candidates.compressed()
        seg_end = candidates[0]
        seg_dist = geo_dist[seg_end]
        waypoints = np.concatenate((waypoints, V[seg_end, None]), axis=0)

        seg_points = path_solver.find_geodesic_path(v_start=seg_start, v_end=seg_end)
        # subsample path (spline) to 0.1mm per odom

        # length-based sampling
        # segmentSpline = pv.Spline(seg_points, n_segment)
        # segmentSpline = np.array(segmentSpline.points)

        _, ii = tree.query(seg_points, k=1)
        segmentNormals = mesh.vertex_normals[ii, :]

        sample_points = np.concatenate((sample_points, seg_points), axis=0)
        sample_normals = np.concatenate((sample_normals, segmentNormals), axis=0)

        cumm_length += seg_dist

        seg_start = seg_end
        if time.time() - start_time > 120:
            print("Timeout, trying again!")
            return None

    # interval-based sampling
    n_interval = math.ceil(len(sample_points) / N)
    n_interval = 1 if n_interval == 0 else n_interval
    sample_points = sample_points[::n_interval, :]
    sample_normals = sample_normals[::n_interval, :]

    delta = np.zeros(sample_points.shape[0])

    a = 1
    for i in range(1, sample_points.shape[0]):
        if i % int(sample_points.shape[0] / num_waypoints) == 0:
            a = -a
        delta[i] = delta[i - 1] + np.radians(np.random.normal(loc=a, scale=0.01))
    T = pose_from_vertex_normal(sample_points, sample_normals, shear_mag, delta)
    print(
        f"Dataset path length: {cumm_length:.4f} m, Num poses: {sample_points.shape[0]}, Time taken: {time.time() - start_time}"
    )
    return T


def random_manual_poses(mesh_path, shear_mag, lc=0.001):
    """Pick points and sample trajectory"""

    mesh = trimesh.load(mesh_path)
    tree = KDTree(mesh.vertices)

    """Get points from user input"""
    cumm_length = 0
    traj_points = pick_points(mesh_path)

    """Generate point and normals"""
    if traj_points.shape[0] == 1:
        sample_points = traj_points
    else:
        sample_points, seg_dist = get_geodesic_path(
            mesh_path, traj_points[0, :], traj_points[1, :]
        )
        n_segment = int(seg_dist / lc)
        n_interval = int(len(sample_points) / n_segment)
        n_interval = 1 if (n_interval == 0) else n_interval
        sample_points = sample_points[::n_interval, :]
        cumm_length += seg_dist
        _, ii = tree.query(sample_points, k=1)
        sample_normals = mesh.vertex_normals[ii, :]

        for i in range(1, traj_points.shape[0] - 1):
            temp, seg_dist = get_geodesic_path(
                mesh_path, traj_points[i, :], traj_points[i + 1, :]
            )
            n_segment = int(seg_dist / lc)
            n_interval = int(len(temp) / n_segment)
            temp = temp[::n_interval, :]
            sample_points = np.concatenate((sample_points, temp), axis=0)
            cumm_length += seg_dist
            _, ii = tree.query(temp, k=1)
            sample_normals = np.concatenate(
                (sample_normals, mesh.vertex_normals[ii, :]), axis=0
            )

    """visualize path"""
    # path_visual = trimesh.load_path(sample_points)
    # scene = trimesh.Scene([path_visual, mesh])
    # scene.show()

    """Convert point and normals to poses"""
    # varying delta over trajectory
    delta = np.zeros(sample_points.shape[0])
    a = 1
    for i in range(1, sample_points.shape[0]):
        if i % int(sample_points.shape[0] / 5) == 0:
            a = -a
        delta[i] = delta[i - 1] + np.radians(np.random.normal(loc=a, scale=0.01))

    T = pose_from_vertex_normal(sample_points, sample_normals, shear_mag, delta)
    print(
        f"Dataset path length: {cumm_length:.4f} m, Num poses: {sample_points.shape[0]}"
    )

    return T


def pick_points(mesh_path):
    """
    http://www.open3d.org/docs/latest/tutorial/visualization/interactive_visualization.html
    """
    print("")
    print("1) Please pick waypoints using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return np.asarray(pcd.points)[vis.get_picked_points(), :]
