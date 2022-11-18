import torch
import torch.nn as nn
import IMCP
import CPD
import IO
from sklearn.decomposition import PCA

"""
To Do list. 
Register a new shape and find the parameters of the shape model that minimize the 
sum of the distance.
Create random and find the shape that minimizes the distance.
Refactor the code to seperate the IMCP and the PCA steps.
Update the attributes of the class to include the results of the pca the original data and the mv shape
"""


class ShapeModel(nn.Module):

    def __init__(self):
        super(ShapeModel, self).__init__()
        self.mv = None
        self.eig_vals = None
        self.eig_vecs = None
        self.n_points = None
        self.dim = None
        self.ppca = PCA()

    def forward(self, x, n_components: int) -> torch.Tensor:
        """
        Takes a list mask file_names, computes a PCA and returns the decomp matricies
        :param x: list of points from masks
            Shape:`(n_masks, n_points, 3)'
        :return: The output of PCA
        """
        x = IO.read(x)
        shapes, self.n_points, self.dim = x.shape

        # First inital Guess
        x_imcp = IMCP.normalize_shape(x)
        x_imcp = IMCP.rotate_principal_moment_inertia(x_imcp)
        # Do simultaneous alignment
        pose, weights, q = IMCP.IMCP(x_imcp)
        q = torch.cat([Q.unsqueeze(0) for Q in q], dim=0)
        weights = torch.cat([weights[k: k + 1] for k in range(shapes)], dim=0)
        # Get intrinsic consensus shape
        sics = IMCP.intrinsic_consensus_shape(q, weights, 10, self.n_points, 3)
        transformed_x = [(torch.matmul(pose[k][0], shapes[k].T).T + pose[k][1].unsqueeze(0)).unsqueeze(0) for k in
                         range(shapes)]
        transformed_x = torch.cat(transformed_x, dim=0)
        # Find the MV
        self.mv = CPD.mv(transformed_x, sics)
        # Get correspondence of the normalized shapes with the mv
        correspondence = CPD.correspondence(transformed_x, self.mv, 4)

        # Permute the orginial shapes to be in the order of the mv shape
        permuted_shapes = self.reorder(x, correspondence)

        permuted_shapes = permuted_shapes.flatten(1)  # (shapes, n_points * 3)

        self.ppca.fit(permuted_shapes)
        self.mean_shape = permuted_shapes.mean(dim=0)
        self.eig_vals = torch.Tensor(self.ppca.explained_variance_)
        self.eig_vecs = torch.Tensor(self.ppca.components_)

    def reorder(self, x: torch.Tensor, correspondence):
        """

        :param x:
        :param correspondence:
        :return:
        """
        permuted_shapes = []
        for k in range(x.shape[0]):
            shape = x[correspondence[k]]
            permuted_shapes.append(shape.unsqueeze(0))
        permuted_shapes = torch.cat(permuted_shapes, dim=0)
        return permuted_shapes

    def create_shape(self, alpha: torch.Tensor):
        """

        :param alpha:
        :return:
        """
        alpha = self._require_dim_len(alpha, 2)
        n_shapes, alpha_dim = alpha.shape
        assert alpha_dim <= self.eig_vals.shape[0]
        new_shape = self.mean_shape + alpha.matmul(self.eig_vecs[:alpha_dim])
        return new_shape.reshape(n_shapes, self.n_points, self.dim)

    def create_shape_approx(self, shape: torch.Tensor, n_comp: int):
        """
        Create shape approximations for many shapes
        :param shape: the shapes to transform
        :return:
        """
        shape = self._require_dim_len(shape, 3)
        dim_redux = self.ppca.transform(shape.flatten(1))[:, :n_comp]
        return self.create_shape(torch.Tensor(dim_redux))

    def register_new_shapes(self, shapes: torch.Tensor) -> torch.Tensor:
        """

        :param shapes:
        :return:
        """
        shapes = self._require_dim_len(shapes, 3)
        correspondences = CPD.correspondence(shapes, self.mv, 4)
        registered_shapes = self.reorder(shapes, correspondences)
        return registered_shapes

    def _require_dim_len(self, shape, dim):
        """

        :param shape:
        :param dim:
        :return:
        """
        if len(shape.shape) < dim:
            shape = shape.unsqueeze(0)
        return shape

    def save(self):
        IO.save(self)

    def load(self, path):
        IO.load(self, path)
