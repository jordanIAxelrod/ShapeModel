import torch
import torch.nn as nn


class ShapeModel(nn.Module):

    def __init__(self):
        super(ShapeModel, self).__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a list of points in 3d from multiple masks, computes a PCA and returns the decomp matricies
        :param x: list of points from masks
            Shape:`(n_masks, n_points, n_points, n_points)'
        :return: The output of PCA
        """

        x = x.flatten(1)

        pca = torch.pca_lowrank(x)

        return pca
