import open3d
import pycpd as cpd
import torch
import numpy as np

"""
Check if CPD and hippocampus data are 
"""

def mv(points: torch.Tensor, sics: torch.Tensor, verbose=False):
    """
    Creates the mv found in number two on pages 3 and four of
    An Automated Statistical Shape Model Developmental Pipeline: Application to the Human Scapula and Humerus
    :param points: The individual point clouds
        Shape: `(k, n_points, 3)'
    :param sics: Intrinsic Consensual shape
        Shape: `(n_points, 3)'
    :return: the mv shape
        Shape: `(n_points, 3)
    """
    corr_points = []
    ty_points = []

    for k in range(points.shape[0]):
        reg = cpd.DeformableRegistration(X=sics.numpy(), Y=points[k].numpy(), beta=.000000001, alpha=200000)
        TY, _ = reg.register()
        # Get the target point that has the highest likelihood of corresponding to each point.
        corresponding_points = get_correspondences(reg.P)
        corr_points.append(corresponding_points.unsqueeze(0))
        ty_points.append(torch.Tensor(TY).unsqueeze(0))
    corr_points = torch.cat(corr_points, dim=0)
    ty_points = torch.cat(ty_points, dim=0)
    mv = torch.zeros_like(points[0])
    for i in range(corr_points.shape[1]):
        local_corr = corr_points[corr_points[:, :, 1] == i][:, 1:]
        corr_points_gather = ty_points.gather(1, local_corr.unsqueeze(2).expand(local_corr.shape[0], local_corr.shape[1], 3))
        if torch.sum(corr_points_gather) > 0:
            mv[i] = corr_points_gather.squeeze(1).mean(dim=0)
        else:
            mv[i] = sics[i]
    if verbose:
        mv_manifold = open3d.geometry.PointCloud()
        mv_manifold.points = open3d.utility.Vector3dVector(mv)
        open3d.visualization.draw_geometries([mv_manifold])
    return mv


def correspondence(points: torch.Tensor, mv:torch.Tensor, iterations: int):
    """
    Implements the correspondence between the mv and the individual samples found in section C of
    An Automated Statistical Shape Model Developmental Pipeline: Application to the Human Scapula and Humerus
    :param points: The individual point coulds
        Shape: `(k, n_points, 3)'
    :param mv: the mv shape
        Shape: `(n_points, 3)'
    :param iterations: the maximum amount of iterations
    :return: the correspondences between the mv and the individual point clouds
        Shape: `(k, n_points)'
    """
    correspondences = []
    for k in range(points.shape[0]):
        local_mv = mv.clone()
        curr_iter = 0
        reg = cpd.DeformableRegistration(X=points[k].numpy(), Y=local_mv.numpy())
        while curr_iter < iterations:

            reg.iterate()
            # Average the current transformation of mv with the corresponding vertices on the original surface
            local_mv = (torch.matmul(torch.Tensor(reg.P), points[k]) + reg.TY) / 2
            curr_iter += 1

            reg.TY = local_mv.numpy()
        # Get the max correspondence with the original surface.
        correspond = get_correspondences(reg.P)
        many_to_one = len(correspond[:, 1].unique()) != mv.shape[0]
        correspondences.append(correspond.unsqueeze(0))
        assert not many_to_one, "There is a many to one correspondence"
    return torch.cat(correspondences, dim=0)


def get_correspondences(points_matrix):
    """
    Greadily get the best next correspondence
    :param points_matrix:
    :return:
    """
    correspond = []
    print(points_matrix.shape)
    for row in range(points_matrix.shape[0]):
        entry = points_matrix[row].argmax()
        entry = [row, entry]
        correspond.append(entry)
        points_matrix[:, entry[1]] = -1e10
    tensor = torch.Tensor(correspond).long()
    return tensor
