import numpy as np
import torch
import ShapeRegistration
from scipy.spatial.transform import Rotation
def create_random_rectangle(n_points):
    # create a unit size cube
    base_faces = [
        (0, 2, 0, 2, 0, 0),
        (0, 2, 0, 0, 0, 1),
        (0, 0, 0, 2, 0, 1),
        (0, 2, 0, 2, 1, 1),
        (1, 1, 0, 2, 0, 1),
        (0, 2, 1, 1, 0, 1)
    ]


    random_transform = Rotation.random().as_matrix()
    random_bias = np.random.randn(1, 3)

    shape = np.zeros((n_points, 3))
    random_points = np.random.rand(n_points,2)
    for i in range(n_points):
        side = base_faces[np.random.randint(0, 6)]
        rand_comp = random_points[i]
        for j in range(3):
            if side[2*j] == side[2*j + 1]:
                noise = np.random.randn(1) * 1e-2
                rand_comp = np.insert(rand_comp, j, side[2 * j] + noise)
                shape[i] = rand_comp * 4


    random_shape = np.matmul(random_transform, shape.T).T + random_bias

    return random_shape

def print_points(thing, shapes, K, q):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if q is not None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(q[:, 0], q[:, 1], q[:, 2], label='q')
        plt.legend()
        plt.show()
    for i in range(K):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        transformed_shape = torch.matmul(thing['Pose Estimation'][i][0], shapes[i].T).T + thing['Pose Estimation'][i][
            1].unsqueeze(0)
        ax.scatter3D(transformed_shape[:, 0], transformed_shape[:, 1], transformed_shape[:, 2], label='k')
        plt.legend()
        plt.show()

def TestMainLoop(n_points, K):
    shapes = []
    for i in range(K):
        shapes.append(create_random_rectangle(n_points))

    shapes = torch.Tensor(shapes)
    x = shapes.clone()
    print_points({'Pose Estimation': [[torch.eye(3), torch.zeros(3)] for _ in range(K)]}, shapes, K, None)
    thing = ShapeRegistration.main(shapes)
    print(x == shapes)
    q = thing['q']
    print(q)
    print(thing)
    print_points(thing, shapes, K, q)

if __name__ =='__main__':
    shape = torch.Tensor(create_random_rectangle(1000)).unsqueeze(0)
    shape = ShapeRegistration.rotate_principal_moment_inertia(ShapeRegistration.normalize_shape(shape))[0].squeeze()
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(shape[:, 0], shape[:, 1], shape[:, 2])
    plt.show()
    TestMainLoop(1100, 10)
