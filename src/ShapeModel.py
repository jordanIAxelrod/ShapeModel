import torch
import torch.nn as nn
import IMCP
import CPD
import IO
import PPCA
import sklearn.decomposition as decomp
class ShapeModel(nn.Module):

    def __init__(self):
        super(ShapeModel, self).__init__()


    def forward(self, x, n_components: int) -> torch.Tensor:
        """
        Takes a list mask file_names, computes a PCA and returns the decomp matricies
        :param x: list of points from masks
            Shape:`(n_masks, n_points, 3)'
        :return: The output of PCA
        """
        x = IO.read(x)
        shapes, n_points, dim = x.shape

        # First inital Guess
        x_imcp = IMCP.normalize_shape(x)
        x_imcp = IMCP.rotate_principal_moment_inertia(x_imcp)
        # Do simultaneous alignment
        pose, weights, q = IMCP.IMCP(x_imcp)
        q = torch.cat([Q.unsqueeze(0) for Q in q], dim=0)
        weights = torch.cat([weights[k: k + 1] for k in range(shapes)], dim=0)
        # Get intrinsic consensus shape
        sics = IMCP.intrinsic_consensus_shape(q, weights, 10, n_points, 3)
        transformed_x = [(torch.matmul(pose[k][0], shapes[k]. T).T + pose[k][1].unsqueeze(0)).unsqueeze(0) for k in range(shapes)]
        transformed_x = torch.cat(transformed_x, dim=0)
        # Find the MV
        mv = CPD.mv(transformed_x, sics)
        # Get correspondence of the normalized shapes with the mv
        correspondence = CPD.correspondence(transformed_x, mv, 4)

        # Permute the orginial shapes to be in the order of the mv shape
        permuted_shapes = []
        for k in range(shapes):
            shape = x[correspondence[k]]
            permuted_shapes.append(shape.unsqueeze(0))
        permuted_shapes = torch.cat(permuted_shapes, dim=0)

        permuted_shapes = permuted_shapes.flatten(1)  # (shapes, n_points * 3)

        ppca = PPCA.PPCA()
        ppca.fit(permuted_shapes)
        pca = torch.pca_lowrank(x)

        return pca
