# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    SE(3) pose utilities 
"""


import numpy as np
import torch
import theseus as th
from scipy.spatial.transform import Rotation as R
import dill as pickle
from typing import List, Tuple


def get_logmap_from_matrix(p: torch.Tensor) -> torch.Tensor:
    """
    Input set of rotations (4, 4, N) and return logmap
    """
    return th.SO3(tensor=p).log_map()


def tf_to_xyzquat(pose: torch.Tensor) -> torch.Tensor:
    """
    convert 4 x 4 transformation matrices to [x, y, z, qw, qx, qy, qz]
    """
    pose = torch.atleast_3d(pose)
    t = pose[:, 0:3, 3]
    q = th.SO3(tensor=pose[:, :3, :3]).to_quaternion()
    xyz_quat = torch.cat((t, q), axis=1)
    return xyz_quat  # (N, 7)


def tf_to_xyzquat_numpy(pose: torch.Tensor) -> torch.Tensor:
    """
    convert 4 x 4 transformation matrices to [x, y, z, qx, qy, qz, qw]
    """
    pose = torch.atleast_3d(pose)

    t = pose[:, 0:3, 3]
    q = th.SO3(tensor=pose[:, :3, :3]).to_quaternion()
    xyz_quat = np.concatenate((t, q), axis=1)

    # pose = np.rollaxis(pose,2) # (4, 4, N) --> (N, 4, 4)
    # r = R.from_matrix(np.array(pose[:, 0:3,0:3]))
    # q = r.as_quat() # qx, qy, qz, qw
    # xyz_quat = np.concatenate((t, q), axis=1)

    return xyz_quat  # (N, 7)


def xyzquat_to_tf(position_quat: torch.Tensor) -> torch.Tensor:
    """
    convert [x, y, z, qw, qx, qy, qz] to 4 x 4 transformation matrices
    """
    try:
        position_quat[:, 3:] = position_quat[:, 3:] / torch.norm(
            position_quat[:, 3:], dim=1
        ).reshape(-1, 1)
        th_T = th.SE3(x_y_z_quaternion=position_quat).to_matrix()
    except ValueError:
        print("Zero quat error!")
    return th_T.squeeze()


def xyzquat_to_tf_numpy(position_quat: np.ndarray) -> np.ndarray:
    """
    convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrices
    """
    try:
        position_quat = np.atleast_2d(position_quat)  # (N, 7)
        N = position_quat.shape[0]
        T = np.zeros((4, 4, N))
        T[0:3, 0:3, :] = np.moveaxis(
            R.from_quat(position_quat[:, 3:]).as_matrix(), 0, -1
        )
        T[0:3, 3, :] = position_quat[:, :3].T
        T[3, 3, :] = 1
    except ValueError:
        print("Zero quat error!")
    return T.squeeze()


def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """
    Convention change: [x, y, z, qx, qy, qz, qw] --> [x, y, z, qw, qx, qy, qz]
    """
    if quat.shape[1] == 7:
        return quat[:, [0, 1, 2, 6, 3, 4, 5]]
    else:
        return quat[:, [3, 0, 1, 2]]


def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """
    Convention change: [x, y, z, qw, qx, qy, qz] --> [x, y, z, qx, qy, qz, qw]
    """
    if quat.shape[1] == 7:
        return quat[:, [0, 1, 2, 4, 5, 6, 3]]
    else:
        return quat[:, [1, 2, 3, 0]]


def log_map_averaged(T: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Average pose computed in SE(3) lie algebra space, computed via Theseus
    """
    T_th = th.SE3(tensor=T[:, :3, :])
    log_map = T_th.log_map()
    avg_logmap = torch.sum((log_map * w[:, None]) / w.sum(), dim=0)
    avg_tf = th.SE3.exp_map(tangent_vector=avg_logmap[None, :])
    return avg_tf.to_matrix().squeeze()


def xyz_quat_averaged(T: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/mertkaraoglu/quaternion-averaging/blob/master/quaternion_averaging.py
    Weighted quaternion average based on Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, no. 4 (2007): 1193-1197.
    Args:
        a: N x 4 tensor each row representing a different data point, assumed to represent a unit-quaternion vector; i.e. [x, y, z, w]
        w: N x 1 tensor each row representing a different float for weight
    Returns:
        torch.Tensor: N x 4 tensor each row representing a different data point, represents a unit-quaternion vector; i.e. [x, y, z, w]
    """

    T = tf_to_xyzquat(T)
    T = wxyz_to_xyzw(T)
    a = T[:, 3:]  # quaternion
    # handle the antipodal configuration
    a[a[:, 3] < 0] = -1 * a[a[:, 3] < 0]

    a = a.view(-1, 4, 1)

    eigen_values, eigen_vectors = (
        torch.matmul(a.mul(w.view(-1, 1, 1)), a.transpose(1, 2))
        .sum(dim=0)
        .div(w.sum())
        .eig(True)
    )

    avg_quat = eigen_vectors[:, eigen_values.argmax(0)[0]].view(1, 4)
    # handle the antipodal configuration
    avg_quat[avg_quat[:, 3] < 0] = -1 * avg_quat[avg_quat[:, 3] < 0]

    avg_quat = xyzw_to_wxyz(avg_quat)  # quaternion portion
    avg_t = torch.sum((T[:, :3] * w[:, None]) / w.sum(), dim=0)

    avg_quat, avg_t = torch.atleast_2d(avg_quat), torch.atleast_2d(avg_t)
    out = torch.cat((avg_t, avg_quat), axis=1)
    return xyzquat_to_tf(out)


def transform_pc(pointclouds: np.ndarray, poses: np.ndarray):
    """
    Transform pointclouds by poses
    """

    if type(pointclouds) is not list:
        temp = pointclouds
        pointclouds = [None] * 1
        pointclouds[0] = temp
        poses = np.expand_dims(poses, axis=2)

    transformed_pointclouds = pointclouds
    # TODO: vectorize
    for i, (pointcloud, pose) in enumerate(
        zip(pointclouds, np.transpose(poses, (2, 0, 1)))
    ):
        pointcloud = pointcloud.T
        # 3D affine transform
        pointcloud = pose @ np.vstack([pointcloud, np.ones((1, pointcloud.shape[1]))])
        pointcloud = pointcloud / pointcloud[3, :]
        pointcloud = pointcloud[:3, :].T
        transformed_pointclouds[i] = pointcloud
    return (
        transformed_pointclouds[0] if len(pointclouds) == 1 else transformed_pointclouds
    )


def wrap_angles(angles: torch.Tensor) -> torch.Tensor:
    """
    angles : (N, 3) angles in degrees
    Wraps to [-np.pi, np.pi] or [-180, 180]
    """

    mask = angles > 180.0
    angles[mask] -= 2.0 * 180.0

    mask = angles < -180.0
    angles[mask] += 2.0 * 180.0
    return angles


def quat2euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternions to euler angles
    """
    quat = np.atleast_2d(quat)
    r = R.from_quat(quat)
    return r.as_euler("xyz", degrees=True)


def rot2euler(rot: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to euler angles
    Adapted from so3_rotation_angle() in  https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html
    """
    rot_trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    phi_cos = torch.acos((rot_trace - 1.0) * 0.5)
    return torch.rad2deg(phi_cos)

    # r = R.from_matrix(np.atleast_3d(rot.cpu().numpy()))
    # eul = r.as_euler('xyz', degrees = True)
    # return torch.tensor(eul, device= rot.device)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.euler_angles_to_matrix
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[0] @ matrices[1] @ matrices[2]


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def extract_poses_sim(
    pickle_file: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract the saved sensor poses from TACTO simulation
    """
    with open(pickle_file, "rb") as pickle_file:
        poses = pickle.load(pickle_file)
    gt_p_cam, gt_p, meas_p = (
        torch.tensor(poses["camposes"]),
        torch.tensor(poses["gelposes"]),
        torch.tensor(poses["gelposes_meas"]),
    )
    gt_p_cam, gt_p, meas_p = (
        xyzw_to_wxyz(gt_p_cam),
        xyzw_to_wxyz(gt_p),
        xyzw_to_wxyz(meas_p),
    )  # switch convention for theseus
    gt_p_cam, gt_p, meas_p = (
        xyzquat_to_tf(gt_p_cam),
        xyzquat_to_tf(gt_p),
        xyzquat_to_tf(meas_p),
    )  # convert to tf tensors
    gt_p_cam, gt_p, meas_p = (
        gt_p_cam.to(device).float(),
        gt_p.to(device).float(),
        meas_p.to(device).float(),
    )  # move to gpu
    return gt_p_cam, gt_p, meas_p


def extract_poses_real(
    pose_file: str,
    alignment_file: str,
    obj_model: str,
    device: str,
    subsample: int = 1,
    cam_dist: float = 0.022,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the saved sensor poses from real-world dataset
    """
    digit_data = np.load(pose_file, allow_pickle=True).item()
    gt_p_cam, obj_poses = (
        torch.tensor(digit_data["poses"]["DIGIT"]),
        torch.tensor(digit_data["poses"][obj_model]),
    )

    gt_p_cam, obj_poses = (
        xyzw_to_wxyz(gt_p_cam),
        xyzw_to_wxyz(obj_poses),
    )  # switch convention for theseus
    gt_p_cam, obj_poses = (
        xyzquat_to_tf(gt_p_cam),
        xyzquat_to_tf(obj_poses),
    )  # convert to tf tensors
    gt_p_cam, obj_poses = (
        gt_p_cam.to(device).float(),
        obj_poses.to(device).float(),
    )  # move to gpu

    gt_p_cam = torch.inverse(obj_poses) @ gt_p_cam  # relative to object
    gt_p_cam = clean_up_optitrack(gt_p_cam)  # clean up jumps

    traj_size = gt_p_cam.shape[0]

    gt_p = torch.zeros((traj_size, 4, 4), device=device)
    alignment = torch.tensor(
        np.load(alignment_file), device=device, dtype=torch.float32
    )
    for i in range(traj_size):
        pose = torch.eye(4, device=device)
        pose[:3, 3] = gt_p_cam[i, :3, 3]
        pose = pose @ alignment
        gt_p_cam[i, :3, 3] = pose[:3, 3]
        gt_p[i, :] = cam2gel(gt_p_cam[i, :], cam_dist=cam_dist)

    gt_p, gt_p_cam = (
        gt_p[::subsample, :],
        gt_p_cam[::subsample, :],
    )  # subsample trajectory

    return gt_p_cam, gt_p


def skew_matrix(v: np.ndarray) -> np.ndarray:
    """
    Get skew-symmetric matrix from vector
    """
    v = np.atleast_2d(v)
    # vector to its skew matrix
    mat = np.zeros((3, 3, v.shape[0]))
    mat[0, 1, :] = -1 * v[:, 2]
    mat[0, 2, :] = v[:, 1]

    mat[1, 0, :] = v[:, 2]
    mat[1, 2, :] = -1 * v[:, 0]

    mat[2, 0, :] = -1 * v[:, 1]
    mat[2, 1, :] = v[:, 0]
    return mat


def pose_from_vertex_normal(
    vertices: np.ndarray, normals: np.ndarray, shear_mag: float, delta: np.ndarray
) -> np.ndarray:
    """
    Generate SE(3) pose given
    vertices: (N, 3), normals: (N, 3), shear_mag: scalar, delta: (N, 1)
    """
    vertices = np.atleast_2d(vertices)
    normals = np.atleast_2d(normals)

    num_samples = vertices.shape[0]
    T = np.zeros((num_samples, 4, 4))  # transform from point coord to world coord
    T[:, 3, 3] = 1
    T[:, :3, 3] = vertices  # t

    # resolve ambiguous DoF
    """Find rotation of shear_vector so its orientation matches normal: np.dot(Rot, shear_vector) = normal
    https://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another """

    cos_shear_mag = np.random.uniform(
        low=np.cos(shear_mag), high=1.0, size=(num_samples,)
    )  # Base of shear cone
    shear_phi = np.random.uniform(
        low=0.0, high=2 * np.pi, size=(num_samples,)
    )  # Circle of shear cone

    # Axis v = (shear_vector \cross normal)/(||shear_vector \cross normal||)
    shear_vector = np.array(
        [
            np.sqrt(1 - cos_shear_mag**2) * np.cos(shear_phi),
            np.sqrt(1 - cos_shear_mag**2) * np.sin(shear_phi),
            cos_shear_mag,
        ]
    ).T
    shear_vector_skew = skew_matrix(shear_vector)
    v = np.einsum("ijk,jk->ik", shear_vector_skew, normals.T).T
    v = v / np.linalg.norm(v, axis=1).reshape(-1, 1)

    # find corner cases
    check = np.einsum("ij,ij->i", normals, np.array([[0, 0, 1]]))
    zero_idx_up = check > 0.9  # pointing up
    zero_idx_down = check < -0.9  # pointing down

    v_skew, sampledNormals_skew = skew_matrix(v), skew_matrix(normals)

    # Angle theta = \arccos(z_axis \dot normal)
    # elementwise: theta = np.arccos(np.dot(shear_vector,normal)/(np.linalg.norm(shear_vector)*np.linalg.norm(normal)))
    theta = np.arccos(
        np.einsum("ij,ij->i", shear_vector, normals)
        / (np.linalg.norm(shear_vector, axis=1) * np.linalg.norm(normals, axis=1))
    )

    identity_3d = np.zeros(v_skew.shape)
    np.einsum("iij->ij", identity_3d)[:] = 1
    # elementwise: Rot = np.identity(3) + v_skew*np.sin(theta) + np.linalg.matrix_power(v_skew,2) * (1-np.cos(theta)) # rodrigues
    Rot = (
        identity_3d
        + v_skew * np.sin(theta)
        + np.einsum("ijn,jkn->ikn", v_skew, v_skew) * (1 - np.cos(theta))
    )  # rodrigues

    if np.any(zero_idx_up):
        Rot[:3, :3, zero_idx_up] = np.dstack([np.identity(3)] * np.sum(zero_idx_up))
    if np.any(zero_idx_down):
        Rot[:3, :3, zero_idx_down] = np.dstack(
            [np.array([[1, 0, 0], [0, -1, -0], [0, 0, -1]])] * np.sum(zero_idx_down)
        )

    # Rotation about Z axis is still ambiguous, generating random rotation b/w [0, 2pi] about normal axis
    # elementwise: RotDelta = np.identity(3) + normal_skew*np.sin(delta[i]) + np.linalg.matrix_power(normal_skew,2) * (1-np.cos(delta[i])) # rodrigues
    RotDelta = (
        identity_3d
        + sampledNormals_skew * np.sin(delta)
        + np.einsum("ijn,jkn->ikn", sampledNormals_skew, sampledNormals_skew)
        * (1 - np.cos(delta))
    )  # rodrigues

    # elementwise:  RotDelta @ Rot
    tfs = np.einsum("ijn,jkn->ikn", RotDelta, Rot)
    T[:, :3, :3] = np.rollaxis(tfs, 2)
    return T


def clean_up_optitrack(poses):
    """
    Filter large jumps in mocap data
    """
    traj_sz = poses.shape[0]
    diff_pose_mags = []
    adjusted_count = 0
    filtered_poses = torch.empty((0, 4, 4), device=poses.device)
    for i in range(traj_sz):
        if i > 0:
            diff_pose = torch.inverse(poses[i - 1, :]) @ poses[i, :]
            diff_pose_mag = torch.norm(diff_pose[:3, 3])
            diff_pose_mags.append(diff_pose_mag)
            avg_diff_pose_mag = sum(diff_pose_mags) / len(diff_pose_mags)
            if i > 1 and diff_pose_mag > 10 * avg_diff_pose_mag:
                adjusted_count += 1
            else:
                filtered_poses = torch.cat(
                    (filtered_poses, poses[i, :][None, :]), dim=0
                )
                # print(f"Jump @ t = {i} : avg: {avg_diff_pose_mag}, curr: {diff_pose_mag}")
                # poses[i, :] = tf_to_xyzquat(xyzquat_to_tf(poses[i - 1, :]) @ xyzquat_to_tf(prev_diff_pose))
            # prev_diff_pose = diff_pose
    print(f"Adjusted {adjusted_count} / {traj_sz} object-sensor poses")
    return filtered_poses


def cam2gel(cam_pose, cam_dist):
    """
    Convert cam_pose to gel_pose
    """
    cam_tf = torch.eye(4, device=cam_pose.device)
    cam_tf[2, 3] = -cam_dist
    return cam_pose @ cam_tf[None, :]
