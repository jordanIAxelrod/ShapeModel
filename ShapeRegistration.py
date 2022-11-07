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
from scipy.spatial.transform import Rotation
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
    theta = torch.acos(V[:, 0, 2] / torch.linalg.norm(V[:, :, 0], dim=1))

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
        curr_points:torch. Tensor,
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
        # Current points in shape k transformed into the basis of the original shape i
        transformed_points = curr_points - transformations[i][1].reshape(1, 3)
        transformed_points = torch.einsum('ij, nj -> ni', torch.linalg.inv(transformations[i][0]),
                                          transformed_points)  # (K - 1, n_points, 3d)
        # Get the kdtree in question and query it for our points
        kdtree = trees[i]
        dd, ii = kdtree.query(transformed_points)  # 2 * (n_points)
        ii = torch.LongTensor(ii).squeeze()  #
        # Get the weights of each point
        weights_k = torch.gather(weights[i], -1, ii)  # (n_points)
        weight_list.append(weights_k.unsqueeze(0))
        # Get the points returned by the tree in our frame of reference
        points = torch.gather(point_cloud[i], 0,
                              ii.unsqueeze(1).expand(n_points, 3)).clone()  # (n_points, 3d)
        # Add points and distances to out list
        point_list.append(points.unsqueeze(0))
        distance_list.append(dd)
    med_dist = torch.median(torch.Tensor(distance_list), dim=0)[0]
    points = torch.cat(point_list, dim=0)
    weights_k = torch.cat(weight_list, dim=0)
    return {'points': points, 'weights': weights_k, 'median': med_dist}


def create_kd_trees(points: torch.Tensor, K: int) -> list[KDTree]:
    return [KDTree(points[k]) for k in range(K)]


def weighted_mean(weights: torch.Tensor, points: torch.Tensor) -> torch.Tensor:

    return torch.sum(weights.reshape(-1, 1) * points, dim=0) / torch.sum(weights)


def print_points(thing, shapes, K, title='', q=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(title)
    if q is not None:
        ax.scatter3D(q[:, 0], q[:, 1], q[:, 2], label='q')

    for i in range(K):
        transformed_shape = torch.matmul(thing['Pose Estimation'][i][0], shapes[i].T).T + thing['Pose Estimation'][i][
            1].unsqueeze(0)
        ax.scatter3D(transformed_shape[:, 0], transformed_shape[:, 1], transformed_shape[:, 2], label=f'{i}')
    plt.legend()
    plt.show()


def main(points: torch.Tensor, mu: float = .0001, global_max: int = 100, local_max: int = 100, lmda: float = 10):
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


    # Initialize empty transformations
    t_k = [[torch.eye(3), torch.zeros(3)] for k in range(K)]

    # Create trees in the first guess space
    kdtrees = create_kd_trees(points, K)
    chi_n = torch.full((K,), np.infty)
    chi_o = torch.full((K,), np.infty)

    print_points({'Pose Estimation': t_k}, points, K, 'inital')
    w_hat = torch.ones(K, n_points)
    global_count = 0

    while (any(-(chi_n - chi_o) >= mu * chi_o) or not global_count) and global_count < global_max:
        global_count += 1
        temp = chi_n.clone()
        chi_o = temp
        p, w, x, t = [], [], [], []
        for k in range(K):
            chi_p_n = chi_o[k].clone()
            chi_p_o = chi_o[k].clone()
            local_count = 0
            points_not = points[k].clone()  # (n_points, 3d)
            t_k_l = [t_k[k][0].clone(), t_k[k][1].clone()]

            w_hat_not = w_hat[k].clone()
            while (-(chi_p_n - chi_p_o) >= mu * chi_p_o or not local_count) and local_count < local_max:
                # update the chi's
                temp = chi_p_n.clone()
                chi_p_o = temp

                # increment the local count.
                local_count += 1

                # Get the closest points for each point in shape k
                
                N = get_closest_points(points, points_not, k, t_k, w_hat, kdtrees)
                N['chi_o'] = chi_o[torch.arange(K) != k].reshape(K - 1, 1).expand(-1, n_points)  # (K - 1, n_points)
                # Get the lambda double prime
                lmda_d_p = math.sqrt(max(lmda, math.sqrt(2) / (math.sqrt(2) - 1))) / chi_p_o * N['median']  # (K - 1, n_points)

                # whether the prime weight is within the noise envelope
                boolean = torch.linalg.norm(points_not.unsqueeze(0) - N['points'], dim=-1) <= lmda_d_p * chi_p_o
                # Weighted mean of the nearest points
                w_p_m = torch.zeros(K - 1, n_points)
                w_p_m[boolean] = (1 - (torch.linalg.norm(points_not.unsqueeze(0) - N['points'], dim=-1)[boolean]
                              / (lmda_d_p * chi_p_o).unsqueeze(0)
                                       .expand(K - 1, n_points)[boolean]) ** 2) ** 2  # (K - 1, n_points)
                chi_min = torch.min(N['chi_o'][N['weights'] > 0])
                w_d_p = chi_min / N['chi_o']
                w_bar = torch.sum(N['weights'] * w_p_m * w_d_p, dim=0)
                r_k_i = torch.zeros(n_points, 3)

                r_k_i[w_bar > 0] = torch.sum((N['weights'] * w_p_m * w_d_p).unsqueeze(2) * N['points'], dim=0)[
                                       w_bar > 0] / w_bar.unsqueeze(1)[w_bar > 0]
                r_k_i[(w_bar <= 0) | (torch.isnan(w_bar))] = torch.median(N['points'], dim=0)[0][(w_bar <= 0) | (torch.isnan(w_bar))]
                # Consensus median centroid between the closest points, and it's self. The centroid of the points
                q = points_not / K + (K - 1) / K * r_k_i  # (n_points, 3d)
                # Distance from median to its corresponding point
                e_hat = torch.linalg.norm(q - points_not, dim=-1)  # (n_points)
                sigma_hat = 1.5 * torch.nanmedian(e_hat)  # 1

                # New weights based on the distance from the median
                w_hat_not = (1 - (e_hat / lmda / sigma_hat) ** 2) ** 2
                w_hat_not *= (e_hat <= lmda * sigma_hat)  # (n_points)

                chi_p_n = torch.sqrt(torch.sum(w_hat_not * e_hat ** 2) / torch.sum(w_hat_not))
                # weighted centroid of points and median shape
                p_bar = weighted_mean(w_hat_not, points_not)
                q_bar = weighted_mean(w_hat_not, q)

                # Centered points
                p_c = points_not - p_bar
                q_c = q - q_bar

                # Covariance Matrix
                H = torch.zeros(3,3)
                for n in range(3):
                    for l in range(3):
                        H[n, l] = torch.sum(w_hat_not * p_c[:, n] * q_c[:,l])
                U, _, V = torch.svd(H)

                # Rotation Matrix
                R_hat = torch.matmul(V, U.t())
                # Translation
                t_hat = q_bar - torch.matmul(R_hat, p_bar.unsqueeze(1)).squeeze()
                # Transformation
                T_hat = [R_hat, t_hat]
                # Updated points
                points_not = torch.matmul(T_hat[0], points_not.T).T + T_hat[1]
                # Updated Translations
                t_k_l[0] = torch.matmul(T_hat[0], t_k_l[0])
                t_k_l[1] = torch.matmul(T_hat[0], t_k_l[1].unsqueeze(1)).squeeze() + T_hat[1]
                fake_points = points.clone()
                fake_points[k] = points_not

            p.append(points_not.unsqueeze(0))
            w.append(w_hat_not.unsqueeze(0))
            x.append(chi_p_n.unsqueeze(0))
            t.append(t_k_l)
        points = torch.cat(p, dim=0)
        w_hat = torch.cat(w, dim=0)
        chi_n = torch.cat(x, dim=0)
        t_k = t
        # print_points({'Pose Estimation': [[torch.eye(3), torch.zeros(3)] for _ in range(K)]}, points, K,
        #              'Point Cloud iteration', q)
    print_points({'Pose Estimation': [[torch.eye(3), torch.zeros(3)] for _ in range(K)]}, points, K, 'Point Cloud End')

    return {'Pose Estimation': t_k, 'Membership': w_hat, 'q': q}

# TODO Find how the membership maps work and output permuted shapes so points at index i are all corresponding points
if __name__ == '__main__':
    main(torch.randn(10, 1000, 2))
