import torch
from sklearn.neighbors import NearestNeighbors


def get_laplace_filter(points: torch.Tensor, weights: torch.Tensor, n_neighbors: int):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points.numpy())
    ii = nbrs.kneighbors(points, return_distance=False)
    ii = torch.Tensor(ii)
    points_gather = points.unsqueeze(1).expand(-1, n_neighbors, -1)
    neighbors = points_gather.gather(0, ii.unsqueeze(1).expand(-1, -1, 3))
    neighbor_weights = weights.unsqueeze(1).expand(n_neighbors).gather(0, ii)
    weighted_neighbors = neighbors * neighbor_weights.unsqueeze(2)
    new_points = weighted_neighbors.mean(dim=1)
    return new_points


if __name__ == '__main__':
    get_laplace_filter(torch.randn(10, 3), torch.randn(10), 10)
