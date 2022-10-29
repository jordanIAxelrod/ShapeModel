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
    L2 = point_cloud - centroids.unsqueeze(1)
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
    _, _, V= torch.pca_lowrank(point_cloud)
    theta = torch.acos(V[:, 0, 0] / torch.linalg.norm(V[:, :, 0], dim=1))

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
    transformed_points = torch.einsum('ij, bnj -> bni', torch.linalg.inv(transformations[k][0]),
                                      point_cloud[torch.arange(K) != k])  # (K - 1, n_points, 3d)
    transformed_points -= transformations[k][1]
    kdtree = trees[k]
    dd, ii = kdtree.query(transformed_points)  # 2 * (K - 1, n_points)
    ii = torch.LongTensor(ii)  #
    # Get the weights of each point
    weights_k = torch.gather(weights[torch.arange(K) != k], -1, ii)  # (K - 1, n_points)
    points = torch.gather(point_cloud[torch.arange(K) != k], 1, ii.unsqueeze(2).expand(K - 1, n_points, 3))  # (K - 1, n_points, 3d)
    med_dist = torch.median(torch.Tensor(dd), dim=0)[0]

    return {'points': points, 'weights': weights_k, 'median': med_dist}


def create_kd_trees(points: torch.Tensor, K: int) -> list[KDTree]:
    return [KDTree(points[k]) for k in range(K)]


def weighted_mean(weights: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.sum(weights.reshape(1, -1, 1) * points, dim=0), dim=0) / torch.sum(weights)


def main(points: torch.Tensor, mu: float = .1, global_max: int = 10, local_max: int = 10, lmda: float = 3):
    """
    Runs the main loop of the algorithm
    :param points: A tensor of K point clouds
        Shape: `(K, n_points, 3d)'
    :param mu: Error goin of each local or global rounds
    :param global_max: number of outer while iterations
    :param local_max: number of inner while iterations
    :param lmda: float
    :return:
    """
    K, n_points, _ = points.shape

    # creating a decent initial guess of the shapes
    points = normalize_shape(points)
    points = rotate_principal_moment_inertia(points)
    points = points - torch.mean(points, dim=1).unsqueeze(1)

    kdtrees = create_kd_trees(points, K)

    chi_n = torch.full((K,), 1e20)
    chi_o = torch.full((K,), 1e20)
    t_k = [[torch.eye(3), torch.zeros(3)] for _ in range(K)]

    w_hat = torch.ones((K, n_points))
    global_count = 0

    while (any(abs(chi_n - chi_o) >= mu * chi_o) or not global_count) and global_count < global_max:
        global_count += 1
        temp = chi_n.clone()
        chi_n = chi_o
        chi_o = temp
        p = []
        w = []
        x = []
        for k in range(K):
            chi_p_n = chi_o[k]
            chi_p_o = chi_o[k]
            local_count = 0
            points_not = points[k]  # (n_points, 3d)

            w_hat_not = w_hat[k]
            while (abs(chi_p_n - chi_p_o) >= mu * chi_p_o or not local_count) and local_count < local_max:
                # update the chi's

                temp = chi_p_n.clone()
                chi_p_n = chi_p_o
                chi_p_o = temp

                # increment the local count.
                local_count += 1

                # reset q
                q = torch.empty(0, 3)
                qp = torch.empty(0, 3)
                # Get the closest points for each point in shape k
                N = get_closest_points(points, k, t_k, w_hat, kdtrees)
                N['chi_o'] = chi_o[torch.arange(K) != k].reshape(K - 1, 1).expand(-1, n_points)  # (K - 1, n_points)
                # Get the lambda double prime
                lmda_d_p = math.sqrt(
                    max(lmda, math.sqrt(2) / (math.sqrt(2) - 1))
                ) / chi_p_o * N['median']  # (K - 1, n_points)

                # whether the prime weight is within the noise envelope
                boolean = torch.norm(points_not.unsqueeze(0) - N['points'], dim=-1) <= lmda_d_p * chi_p_o

                # Weighted mean of the nearest points
                w_p_m = (1 - (torch.norm(points_not.unsqueeze(0) - N['points'], dim=-1)
                              / (lmda_d_p * chi_p_o)) ** 2) ** 2 * boolean  # (K - 1, n_points)
                chi_min = torch.min(N['chi_o'] + (N['weights'] <= 0) * 1e20)
                w_d_p = chi_min / N['chi_o']
                w_bar = torch.sum(N['weights'] * w_p_m * w_d_p, dim=0).reshape(1, -1, 1) + 1e-10
                r_k_i = (w_bar > 0) * torch.sum((N['weights'] * w_p_m * w_d_p).unsqueeze(2) * N['points'], dim=0) / w_bar + (
                        w_bar <= 0) * torch.median(N['points'], dim=0)[0]  # (n_points, 3d)

                # Consensus mean between the closest points, and it's self. The centroid of the points
                q = (points_not / K + (K - 1) / K * r_k_i).squeeze(0)  # (n_points * local_count, 3d)
                qp = points_not # (n_points * local_count, 3d)
                e_hat = torch.norm(q - qp, dim=-1)  # (n_points)
                sigma_hat = 1.5 * torch.nanmedian(e_hat)  # 1

                w_hat_not = (1 - (e_hat / lmda / sigma_hat) ** 2) ** 2
                w_hat_not *= (e_hat <= lmda * sigma_hat)  # (n_points)

                chi_p_n = torch.sqrt(torch.sum(w_hat_not * e_hat ** 2) / torch.sum(w_hat_not))
                p_bar = weighted_mean(w_hat_not, points_not)
                q_bar = weighted_mean(w_hat_not, q)

                p_c = (qp - p_bar)
                q_c = (q - q_bar).reshape(-1, 3)
                U, _, V = torch.pca_lowrank(torch.einsum('nd, nl -> dl', (w_hat_not.reshape(1, -1, 1) * p_c).reshape(-1, 3), q_c))
                R_hat = torch.einsum('ij, jk -> ik', V.t(), U.t())

                t_hat = q_bar - torch.einsum('ij, jk -> ik', R_hat, p_bar.unsqueeze(1)).squeeze()
                T_hat = [R_hat, t_hat]
                points_not = torch.einsum('ij, nj -> nj', T_hat[0], points_not) + T_hat[1]
                t_k[k][0] = torch.einsum('ij,jk->ik', T_hat[0], t_k[k][0])
                t_k[k][1] = torch.einsum('ij, j -> i', T_hat[0], t_k[k][1]) + T_hat[1]
            p.append(points_not.unsqueeze(0))
            w.append(w_hat_not.unsqueeze(0))
            x.append(chi_p_n.unsqueeze(0))
        points = torch.cat(p, dim=0)
        w_hat = torch.cat(w, dim=0)
        chi_n = torch.cat(x, dim=0)
    return {'Pose Estimation': t_k, 'Membership': w_hat, 'q':q}


if __name__ == '__main__':
    main(torch.randn(10, 1000, 3))