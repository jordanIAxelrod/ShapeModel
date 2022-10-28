"""
This file is the implementation of the shape algorithm found in
https://ieeexplore-ieee-org.proxy.lib.duke.edu/stamp/stamp.jsp?tp=&arnumber=4450597 as well as future work of theirs

This algorithm called ICMP is an iterative point cloud matching algo. It operates simultaneously on k shapes. It finds
points that are in correspondence, the Rigid pose estimations, and the root shape.


author: Jordan Axelrod
date: 10.27.22
"""

import math

import torch
from scipy.spatial import KDTree
import numpy as np


# This is the first step of the registration. We would like to make all the shapes unit length
def normalize_shape(point_cloud: torch.Tensor):
    """
    find the centroid of each point and use it to normalize the size of the data sets
    :param point_cloud: a set of K point clouds
        Shape: `(n_shapes, n_points, 3d)'
    :return: A normalized set of K shapes
    """

    centroids = torch.mean(point_cloud, dim=1)
    L2 = point_cloud - centroids.unsqueeze(2)
    distance = torch.square(L2)
    distance = torch.sum(distance, dim=2)
    centroid_size = torch.sqrt(torch.sum(distance, dim=1))
    return point_cloud / centroid_size.reshape(-1, 1, 1)


# This is the second step where we get a decent alignment of the shapes using inertia.

# Remember to plot the output of these. I would like to see if there are bones that are flipped.
def rotate_principal_moment_inertia(point_cloud: torch.Tensor):
    """
    Finds the principal moment of inertia for each shape. Then rotates each shape for alignment
    :param point_cloud: a set of K point clouds
        Shape: `(n_shapes, n_points, 3d)'
    :return: The same shapes rotated to the z axis
    """
    pca = torch.pca_lowrank(point_cloud)
    theta = torch.acos(pca.V[:, :, 0] / torch.linalg.norm(pca.V[:, :, 0], dim=1))

    rotation = torch.zeros(point_cloud.shape[0], 3, 3)

    rotation[:, 0, 0] = torch.cos(theta)
    rotation[:, 0, 1] = -torch.sin(theta)
    rotation[:, 1, 0] = torch.sin(theta)
    rotation[:, 1, 1] = torch.cos(theta)
    rotation[:, 2, 2] = 1

    aligned = torch.einsum('nkj, nlj -> nlk', rotation, point_cloud)

    return aligned


def get_closest_points(
        point_cloud: torch.Tensor,
        k: int,
        transformations: torch.Tensor,
        weights: torch.Tensor,
        trees: list[KDTree]
) -> dict:
    """
    Finds the K - 1 closest points in the other clouds and returns the points and their weights.

    :param k: the shape idx we're working on
    :param weights: the weight of each point in the cloud
        Shape: `(n_shapes, n_points)'
    :param point_cloud: a set of K point clouds transformed n times in previous repitition
        Shape: `(n_shapes, n_points, 3d)'
    :param transformations: K matrix rotations to transform from current axis to the origial(x, y, z)
        Shape: `(n_shapes, 3d, 3d)
    :param trees: The original KD-trees built initialization
    :return: a tensor of shape `(K - 1, n_points, 3d)' the K - 1 closest points one from each shape.
    """
    K, n_points, d = point_cloud.shape
    transformed_points = torch.einsum('ij, bnj', torch.linalg.inv(transformations[k]),
                                      point_cloud[torch.arange(K) != k])  # (K - 1, n_points, 3d)
    kdtree = trees[k]
    dd, ii = kdtree.query(transformed_points)  # 2 * (K - 1, n_points)
    ii = torch.Tensor(ii)  #
    # Get the weights of each point
    weights_k = torch.gather(weights[torch.arange(K) != k], -1, ii)  # (K - 1, n_points)
    points = torch.gather(point_cloud[torch.arange(K) != k], 1, ii.unsqueeze(2))  # (K - 1, n_points, 3d)
    med_dist = torch.median(torch.Tensor(dd), dim=0)[0]

    return {'points': points, 'weights': weights_k, 'median': med_dist}


def merge_operation(nearest_neighbors: torch.Tensor, weights: torch.Tensor):
    """
    Computes the centroid for the closest neighbors
    :param nearest_neighbors: A list of the index of the closest neighbor in K-1 clouds
    :param weights: the weights w_l_j_0 for the points in nearest_neigbor
    :return:
    """
    wkilj0 = (1 - ())


def weight_update():
    pass


def create_kd_trees() -> list[KDTree]:
    pass


def main(points: torch.Tensor, mu: float, global_max: int, local_max: int, lmda: float = 3):
    kdtrees = create_kd_trees()
    K, n_points, _ = points.shape
    chi_n = torch.full(K, 1e100)
    chi_o = torch.full(K, 1e100)
    t_k = torch.Tensor([np.eye(3) for _ in range(K)])

    w_hat = torch.ones((K, n_points))
    global_count = 0

    while (any(abs(chi_n - chi_o) >= mu * chi_o) or not global_count) and global_count < global_max:
        global_count += 1
        temp = chi_n
        chi_n = chi_o
        chi_o = temp
        for k in range(K):
            chi_p_n = chi_o[k]
            chi_p_o = chi_o[k]
            local_count = 0
            points_not = points[k]  # (n_points, 3d)
            q = None
            w_hat_not = w_hat[k]
            while (abs(chi_p_n - chi_p_o) >= mu * chi_p_o or not local_count) and local_count < local_max:

                # update the chi's
                temp = chi_p_n
                chi_p_n = chi_p_o
                chi_p_o = temp

                # increment the local count.
                local_count += 1

                # Get the closest points for each point in shape k
                N = get_closest_points(points, k, t_k, w_hat, kdtrees)
                N['chi_o'] = chi_o[torch.arange(K) != k].reshape(K - 1, 1).expand(-1, n_points)  # (K - 1, n_points)
                # Get the lambda double prime
                lmda_d_p = math.sqrt(
                    max(lmda, math.sqrt(2) / (math.sqrt(2) - 1))
                ) / chi_p_o * N['median']  # (K - 1, n_points)

                # whether the prime weight is within the noise envelope
                if local_count - 1:
                    boolean = torch.norm(points_not.unsqueeze(0) - q, dim=-1) <= lmda_d_p * chi_p_o
                else:
                    # since Q is not defined in the first iteration.
                    boolean = torch.ones_like(lmda_d_p)
                w_p_m = (1 - (torch.norm(points_not.unsqueeze(0) - N['points'], dim=-1)
                              / (lmda_d_p * chi_p_o)) ** 2) ** 2 * boolean  # (K - 1, n_points)
                chi_min = torch.min(N['chi_o'] + (N['weights'] <= 0) * np.inf)

                w_d_p = chi_min / N['chi_o']

                w_bar = torch.sum(N['weights'] * w_p_m * w_d_p, dim=0)

                r_k_i = (w_bar > 0) * torch.sum(N['weights'] * w_p_m * w_d_p * N['points'], dim=0) / w_bar + (
                            w_bar <= 0) * torch.median(N['points'], dim=0)
                for i in range(n_points):
                    boolean = abs(points_not - q) <= lmda_d_p * chi_p_o
