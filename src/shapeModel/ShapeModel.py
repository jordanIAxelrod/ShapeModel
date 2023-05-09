import numpy as np
import torch
import torch.nn as nn
import src.shapeModel.IMCP as IMCP
import src.shapeModel.CPD as CPD
import src.shapeModel.IO as IO
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import open3d as o3d
import icp_master.icp as icp
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
        self.final_shapes = None
        self.mv = None
        self.eig_vals = None
        self.eig_vecs = None
        self.n_points = None
        self.dim = None
        self.mean_shape = None
        self.ppca = PCA()

    def forward(self, x, verbose=False) -> torch.Tensor:
        """
        Takes a list mask file_names, computes a PCA and returns the decomp matricies
        :param x: list of points from masks
            Shape:`(n_masks, n_points, 3)'
        :return: The output of PCA
        """
        shapes, self.n_points, self.dim = x.shape

        # First inital Guess
        x_imcp, _ = self.normalize(x)
        print(x_imcp)
        # Do simultaneous alignment
        model = IMCP.IMCP(x_imcp, verbose=verbose)
        weights = model['Membership']
        pose = model['Pose Estimation']
        q = model['q']
        q = torch.cat([Q.unsqueeze(0) for Q in q], dim=0)
        weights = torch.cat([weights[k: k + 1] for k in range(shapes)], dim=0)
        # Get intrinsic consensus shape
        print(pose)
        sics, weights = IMCP.intrinsic_consensus_shape(q, weights, 10, self.n_points, 3)
        if verbose:
            manifold_sics = o3d.geometry.PointCloud()
            manifold_sics.points = o3d.utility.Vector3dVector(sics)
            manifold_sics.estimate_normals()
            radii = [0.0001, 0.0001, 0.0001, .0001]
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                manifold_sics, o3d.utility.DoubleVector(radii))
            o3d.visualization.draw_geometries([manifold_sics, rec_mesh])

            ax = plt.axes(projection='3d')
            plt.title('sics')
            ax.scatter(sics[:, 0], sics[:, 1], sics[:, 2])
            plt.show()

        transformed_x = [(torch.matmul(pose[k][0], x_imcp[k].T).T + pose[k][1].unsqueeze(0)).unsqueeze(0) for k in
                         range(shapes)]
        if verbose: 
            IMCP.print_points(model, x_imcp, shapes, "Transformed Points")
        transformed_x = torch.cat(transformed_x, dim=0)
        # Find the MV
        self.mv = CPD.mv(transformed_x, sics)
        if verbose:
            print('mv shape', self.mv, self.mv.shape)
            ax = plt.axes(projection='3d')
            plt.title('mv shape')
            ax.scatter(self.mv[:, 0], self.mv[:, 1], self.mv[:, 2])
            plt.show()
        # Get correspondence of the normalized shapes with the mv
        correspondence = CPD.correspondence(transformed_x, self.mv, 20)

        # Permute the orginial shapes to be in the order of the mv shape
        x, _ = IMCP.rotate_principal_moment_inertia(x - x.mean(dim=1).unsqueeze(1))
        for k in range(shapes):
            x[k] = torch.matmul(pose[k][0], x[k].T).T + pose[k][1]

        permuted_shapes = self.reorder(x, correspondence)
        if verbose:
            ax = plt.axes(projection='3d')
            plt.title('Permuted Shapes')
    
            for i in range(shapes):
    
                ax.scatter3D(x[i][:, 0], x[i][:, 1], x[i][:, 2], label=f'{i}')
            plt.legend()
            plt.show()
        permuted_shapes = permuted_shapes.flatten(1)  # (shapes, n_points * 3)
        self.final_shapes = permuted_shapes
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
            shape = x[k][correspondence[k, :, 1]]
            permuted_shapes.append(shape.unsqueeze(0))
        permuted_shapes = torch.cat(permuted_shapes, dim=0)
        return permuted_shapes

    def normalize(self, x):
        x_imcp = IMCP.normalize_shape(x)
        x_imcp, r = IMCP.rotate_principal_moment_inertia(x_imcp)
        return x_imcp, r

    def create_shape(self, alpha: torch.Tensor):
        """

        :param alpha:
        :return:
        """
        alpha = self._require_dim_len(alpha, 2)
        n_shapes, alpha_dim = alpha.shape
        assert alpha_dim <= self.eig_vals.shape[0]

        new_shape = self.mean_shape.unsqueeze(0) + (alpha).matmul(self.eig_vecs[:alpha_dim])
        return torch.Tensor(new_shape.reshape(n_shapes, self.n_points, self.dim))

    def create_shape_approx(self, shape: torch.Tensor, n_comp: int):
        """
        Create shape approximations for many shapes
        :param shape: the shapes to transform
        :return:
        """
        shape = self._require_dim_len(shape, 3)
        dim_redux = self.ppca.transform(shape.flatten(1))
        dim_redux[:, n_comp:] = 0
        return self.create_shape(torch.Tensor(dim_redux))

    def register_new_shapes(self, shapes: torch.Tensor) -> torch.Tensor:
        """

        :param shapes:
        :return:
        """
        shapes = self._require_dim_len(shapes, 3)
        norm_shapes, r = self.normalize(shapes)
        mean_shape = torch.concat([norm_shapes, self.mv.unsqueeze(0)])
        ax = plt.axes(projection='3d')
        plt.title("Mean shape and new shape")

        for i in range(2):
            ax.scatter(mean_shape[i, :, 0], mean_shape[i,:, 1], mean_shape[i,:, 2], label=['New', 'Mean'][i])
        plt.legend()
        plt.show()

        transformation = icp.icp(norm_shapes.squeeze(0).numpy(), self.mv.numpy())
        d4shapes = np.concatenate([norm_shapes.squeeze(0), np.ones_like(shapes[0, :, 0:1])], axis=1)
        norm_shapes = torch.Tensor(np.matmul(transformation[0], d4shapes.T).T[:, :-1])
        correspondences = CPD.correspondence(norm_shapes.unsqueeze(0), self.mv, 4)
        registered_shapes = self.reorder(shapes, correspondences)
        registered_shapes = r.matmul((registered_shapes - registered_shapes.mean(dim=1)).mT).mT
        registered_shapes = torch.cat([registered_shapes, torch.ones_like(registered_shapes[:, :, 0:1])], dim=2)
        registered_shapes = torch.Tensor(transformation[0]).matmul(registered_shapes.permute(0,2,1)).permute(0,2,1)[:, :, :-1]
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

    def get_explained_variance(self):
        explained_variance = np.cumsum(self.eig_vals) / sum(self.eig_vals)

        plt.plot([i for i in range(len(explained_variance))], explained_variance)
        plt.title("Explained Variance per mode")
        plt.show()
        return explained_variance

    def save(self):
        IO.save(self)

    def load(self, path):
        IO.load(self, path)
