import numpy as np
import torch
import ShapeRegistration
from scipy.spatial.transform import Rotation
from scipy.cluster.vq import kmeans


def create_random_rectangle(n_points):
    # create a unit size cube
    base_faces = [
        (0, 100, 0, 100, 0, 0),
        (0, 100, 0, 0, 0, 1),
        (0, 0, 0, 100, 0, 1),
        (0, 100, 0, 100, 1, 1),
        (1, 1, 0, 100, 0, 1),
        (0, 100, 1, 1, 0, 1)
    ]

    random_transform = Rotation.random().as_matrix()
    random_bias = np.random.randn(1, 3)

    shape = np.zeros((n_points, 3))
    random_points = np.random.uniform(low = 0, high=100, size=(n_points, 2))
    for i in range(n_points):
        side = base_faces[np.random.randint(0, 6)]
        rand_comp = random_points[i]
        for j in range(3):
            if side[2 * j] == side[2 * j + 1]:
                noise = np.random.randn(1) * 1e-2
                rand_comp = np.insert(rand_comp, j, side[2 * j] * 100 + noise)
                shape[i] = rand_comp * 4

    random_shape = np.matmul(random_transform, shape.T).T + random_bias

    return random_shape


def get_n_points(n, points):
    points = kmeans(points, n)[0]
    return points


def print_points(thing, shapes, K, q, title=''):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(title)
    if q is not None:
        ax.scatter3D(q[:, 0], q[:, 1], q[:, 2], label='q')
        plt.show()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(K):
        transformed_shape = torch.matmul(thing['Pose Estimation'][i][0], shapes[i].T).T + thing['Pose Estimation'][i][
            1].unsqueeze(0)
        ax.scatter3D(transformed_shape[:, 0], transformed_shape[:, 1], transformed_shape[:, 2], label=f'{i}')
    plt.legend()
    plt.show()


def TestMainLoop(n_points, K, n):
    shapes = []
    for i in range(K):
        print(i)
        shape = create_random_rectangle(n_points)
        shape = get_n_points(n, shape)
        shapes.append(shape)
    print(shapes)
    shapes = torch.Tensor(shapes)
    x = shapes.clone()
    print_points({'Pose Estimation': [[torch.eye(3), torch.zeros(3)] for _ in range(K)]}, shapes, K, None, 'Start')
    # creating a decent initial guess of the shapes
    # points = normalize_shape(points)
    # centroids = - torch.mean(shapes, dim=1)
    # shapes = shapes + centroids.unsqueeze(1)
    # shapes, rotation = ShapeRegistration.rotate_principal_moment_inertia(shapes)
    thing = ShapeRegistration.main(shapes)
    print(x == shapes)
    q = thing['q']

    print_points(thing, shapes, K, q, 'Finished')


if __name__ == '__main__':
    shape = torch.Tensor(create_random_rectangle(1000)).unsqueeze(0)
    shape = ShapeRegistration.rotate_principal_moment_inertia(ShapeRegistration.normalize_shape(shape))[0].squeeze()
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(shape[:, 0], shape[:, 1], shape[:, 2])
    plt.show()
    TestMainLoop(5000, 10, 500)
