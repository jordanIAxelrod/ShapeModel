
import pycpd as cpd
import torch



def mv(points: torch.Tensor, sics: torch.Tensor):
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
        reg = cpd.DeformableRegistration(sics.numpy(), points[k].numpy(), alpha=1)
        TY, _ = reg.register()
        # Get the target point that has the highest likelihood of corresponding to each point.
        corresponding_points = torch.Tensor(reg.P).argmax(dim=1)
        corr_points.append(corresponding_points.unsqueeze(0))
        ty_points.append(torch.Tensor(TY).unsqueeze(0))
    corr_points = torch.cat(corr_points, dim=0)
    ty_points = torch.cat(ty_points, dim=0)
    mv = torch.zeros_like(sics)
    for i in range(corr_points.shape[1]):
        mv[i] = ty_points[corr_points == i].sum(dim=0)
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
        many_to_one = True
        curr_iter = 0
        correspond = None
        reg = cpd.DeformableRegistration(points[k].numpy(), local_mv.numpy())
        while curr_iter < iterations and many_to_one:

            reg.iterate()
            # Average the current transformation of mv with the corresponding vertices on the original surface
            local_mv = (torch.matmul(torch.Tensor(reg.P), points[k]) + reg.TY) / 2
            curr_iter += 1
            # Get the max correspondence with the original surface.
            correspond = torch.Tensor(reg.P).argmax(dim=1)
            many_to_one = len(correspond.unique()) != mv.shape[0]
            reg.TY = local_mv.numpy()
        correspondences.append(correspond.unsqueeze(0))
        assert not many_to_one, "There is a many to one correspondence"
    return torch.cat(correspondences, dim=0)

