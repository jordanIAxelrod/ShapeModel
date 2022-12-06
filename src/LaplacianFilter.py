import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np


def pad_to_max_len(max_len, tensor_list):
    return torch.cat(
        [F.pad(tensor, (0, 0, 0, max_len - tensor.shape[0])).unsqueeze(0) for tensor in tensor_list]
        , dim=0
    )


def get_laplace_filter(points: torch.Tensor, weights: torch.Tensor, n_neighbors: int):
    print(points[torch.isnan(points)])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=.001).fit(points.numpy())
    ii = nbrs.radius_neighbors(points, return_distance=False)
    max_neighbors = max(ii, key=lambda x: x.shape[0]).shape[0]

    neighbors = [points.gather(0, torch.LongTensor(i).unsqueeze(1).expand(-1, 3)) for i in ii]
    neighbor_weights = [weights.gather(0, torch.LongTensor(i)).unsqueeze(1) for i in ii]
    neighbors = pad_to_max_len(max_neighbors, neighbors)
    neighbor_weights = pad_to_max_len(max_neighbors, neighbor_weights).squeeze(2)

    weight_factor = (1 - (weights / .9) ** 2) ** 2 * (weights <= .9)
    non_zero_avg = torch.sum(
        neighbor_weights[:, 1:], dim=1) / torch.sum(neighbor_weights[:, 1:] != 0 + 1e-3, dim=1)
    neighbor_factor = (1 - (
            torch.max(
                torch.zeros_like(weights), weights - non_zero_avg + 0.05
            ) / 0.05
    ) ** 2) ** 2 * (weights <= 0.05)
    scale_factor = (weight_factor * neighbor_factor).unsqueeze(1)
    w_barycenter = torch.sum(neighbor_weights ** 2, dim=1) / torch.sum(neighbor_weights + 1e-5, dim=1)
    weighted_neighbors = neighbors * neighbor_weights.unsqueeze(2)
    barycenter = weighted_neighbors.mean(dim=1)
    new_points = scale_factor * barycenter + (1 - scale_factor) * points
    scale_factor = scale_factor.squeeze()
    new_weights = scale_factor * w_barycenter + (1 - scale_factor) * weights
    return new_points, new_weights


if __name__ == '__main__':
    get_laplace_filter(torch.randn(10, 3), torch.randn(10), 10)
