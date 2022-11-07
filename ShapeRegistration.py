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
import matplotlib.pyplot as plt

# This is the first step of the registration. We would like to make all the shapes unit length
def normalize_shape(point_cloud: torch.Tensor):
    """
    find the centroid of each point and use it to normalize the size of the data sets
    :param point_cloud: a set of K point clouds
        Shape: `(n_shapes, n_points, 3d)'
    :return: A normalized set of K shapes
    """

    centroids = torch.mean(point_cloud, dim=1)  # (n_shapes, 3d)
    L2 = point_cloud - centroids.unsqueeze(1)
    distance = torch.square(L2)
    distance = torch.sum(distance, dim=2)
    centroid_size = torch.sqrt(torch.sum(distance, dim=1)) / point_cloud.shape[0]
    return (point_cloud - centroids.unsqueeze(1)) / centroid_size.reshape(-1, 1, 1)


# This is the second step where we get a decent alignment of the shapes using inertia.

# Remember to plot the output of these. I would like to see if there are bones that are flipped.
def rotate_principal_moment_inertia(point_cloud: torch.Tensor):
    """
    Finds the principal moment of inertia for each shape. Then rotates each shape for alignment
    :param point_cloud: a set of K point clouds
        Shape: `(n_shapes, n_points, 3d)'
    :return: The same shapes rotated to the z axis
    """
    _, _, V = torch.pca_lowrank(point_cloud)
    theta = torch.acos(V[:, 0, 0] / torch.linalg.norm(V[:, :, 0], dim=1))

    rotation = torch.zeros(point_cloud.shape[0], 3, 3)

    rotation[:, 0, 0] = torch.cos(theta)
    rotation[:, 0, 1] = -torch.sin(theta)
    rotation[:, 1, 0] = torch.sin(theta)
    rotation[:, 1, 1] = torch.cos(theta)
    rotation[:, 2, 2] = 1
    rotation = torch.linalg.inv(rotation)
    aligned = torch.einsum('nkj, nlj -> nlk', rotation, point_cloud)

    return aligned, rotation


def get_closest_points(
        point_cloud: torch.Tensor,
        k: int,
        transformations: list[list[torch.Tensor]],
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
    weight_list = []
    point_list = []
    distance_list = []
    for i in range(K):
        if i == k:
            continue
        transformed_points = point_cloud[k] - transformations[i][1].reshape(1, 3)
        transformed_points = torch.einsum('ij, nj -> ni', torch.linalg.inv(transformations[i][0]),
                                          transformed_points)  # (K - 1, n_points, 3d)
        kdtree = trees[i]
        dd, ii = kdtree.query(transformed_points)  # 2 * (n_points)
        ii = torch.LongTensor(ii).squeeze()  #
        # Get the weights of each point
        weights_k = torch.gather(weights[i], -1, ii)  # (n_points)
        weight_list.append(weights_k.unsqueeze(0))
        points = torch.gather(point_cloud[i], 0,
                              ii.unsqueeze(1).expand(n_points, 3))  # (n_points, 3d)
        transform_points = torch.einsum('ij, nj -> ni', transformations[i][0], points) + transformations[i][1].reshape(1, 3)
        point_list.append(transform_points.unsqueeze(0))
        distance_list.append(dd)
    med_dist = torch.median(torch.Tensor(distance_list), dim=0)[0]
    points = torch.cat(point_list, dim=0)
    weights_k = torch.cat(weight_list, dim=0)
    return {'points': points, 'weights': weights_k, 'median': med_dist}


def create_kd_trees(points: torch.Tensor, K: int) -> list[KDTree]:
    return [KDTree(points[k]) for k in range(K)]


def weighted_mean(weights: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.sum(weights.reshape(-1, 1) * points, dim=0) / torch.sum(weights)
def print_points(thing, shapes, K, title=''):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(title)
    for i in range(K):
        transformed_shape = torch.matmul(thing['Pose Estimation'][i][0], shapes[i].T).T + thing['Pose Estimation'][i][
            1].unsqueeze(0)
        ax.scatter3D(transformed_shape[:, 0], transformed_shape[:, 1], transformed_shape[:, 2])
    plt.show()
def main(points: torch.Tensor, mu: float = .1, global_max: int = 100, local_max: int = 100, lmda: float = 3):
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

    # Create trees in the original space

    # creating a decent initial guess of the shapes
    # points = normalize_shape(points)
    centroids = - torch.mean(points, dim=1)

    points, rotation = rotate_principal_moment_inertia(points + centroids.unsqueeze(1))
    # initialize the first guess transformation
    t_k = [[torch.eye(3), torch.zeros(3)] for k in range(K)]

    kdtrees = create_kd_trees(points, K)
    chi_n = torch.full((K,), np.infty)
    chi_o = torch.full((K,), np.infty)

    print_points({'Pose Estimation':t_k}, points, K)
    w_hat = torch.ones((K, n_points))
    global_count = 0

    while (any(abs(chi_n - chi_o) >= mu * chi_o) or not global_count) and global_count < global_max:
        global_count += 1
        temp = chi_n.clone()
        chi_n = chi_o.clone()
        chi_o = temp
        p = []
        w = []
        x = []
        for k in range(K):
            chi_p_n = chi_o[k].clone()
            chi_p_o = chi_o[k].clone()
            local_count = 0
            points_not = points[k].clone()  # (n_points, 3d)

            w_hat_not = w_hat[k].clone()
            while (abs(chi_p_n - chi_p_o) >= mu * chi_p_o or not local_count) and local_count < local_max:
                # update the chi's

                temp = chi_p_n.clone()
                chi_p_n = chi_p_o.clone()
                chi_p_o = temp

                # increment the local count.
                local_count += 1

                # Get the closest points for each point in shape k
                
                # TODO Create a temp array of t_k for each k as the old points havent rotated in the perspective of this k but t_k shows them rotated..
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
                w_bar = torch.sum(N['weights'] * w_p_m * w_d_p, dim=0).reshape(-1, 1).expand(-1, 3)
                r_k_i = torch.zeros(n_points, 3)
                r_k_i[w_bar > 0] = torch.sum((N['weights'] * w_p_m * w_d_p).unsqueeze(2) * N['points'], dim=0)[
                                       w_bar > 0] / w_bar[w_bar > 0]
                r_k_i[w_bar <= 0] = torch.median(N['points'], dim=0)[0][w_bar <= 0]
                # Consensus median centroid between the closest points, and it's self. The centroid of the points
                q = (points_not / K + (K - 1) / K * r_k_i).squeeze(0)  # (n_points, 3d)
                e_hat = torch.norm(q - points_not, dim=-1)  # (n_points)
                sigma_hat = 1.5 * torch.nanmedian(e_hat)  # 1

                w_hat_not = (1 - (e_hat / lmda / sigma_hat) ** 2) ** 2
                w_hat_not *= (e_hat <= lmda * sigma_hat)  # (n_points)

                chi_p_n = torch.sqrt(torch.sum(w_hat_not * e_hat ** 2) / torch.sum(w_hat_not))
                p_bar = weighted_mean(w_hat_not, points_not)
                q_bar = weighted_mean(w_hat_not, q)
                p_c = points_not - p_bar
                q_c = q - q_bar

                H = torch.matmul((w_hat_not.unsqueeze(1) * p_c).T, q_c)
                U, _, V_t = torch.svd(H)
                R_hat = torch.matmul(V_t.t(), U.t())

                t_hat = q_bar - torch.matmul(R_hat, p_bar.unsqueeze(1)).squeeze()
                print(chi_p_o, chi_p_n)
                T_hat = [R_hat, t_hat]
                points_not = torch.matmul(T_hat[0], points_not.T).T + T_hat[1]
                t_k[k][0] = torch.matmul(T_hat[0], t_k[k][0])
                t_k[k][1] = torch.matmul(T_hat[0], t_k[k][1].unsqueeze(1)).squeeze() + T_hat[1]
            p.append(points_not.unsqueeze(0))
            w.append(w_hat_not.unsqueeze(0))
            x.append(chi_p_n.unsqueeze(0))
        points = torch.cat(p, dim=0)
        w_hat = torch.cat(w, dim=0)
        chi_n = torch.cat(x, dim=0)
    for k in range(K):
        t_k[k][0] = torch.matmul(t_k[k][0], rotation[k])
        t_k[k][1] = torch.matmul(t_k[k][0], centroids[k].unsqueeze(1)).squeeze() + t_k[k][1]
        print(t_k[k][1])
    return {'Pose Estimation': t_k, 'Membership': w_hat, 'q': q}


if __name__ == '__main__':
    main(torch.randn(10, 1000, 3))
