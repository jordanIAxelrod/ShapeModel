import numpy as np
import torch
import ShapeRegistration

def create_random_rectangle(n_points):
    # create a unit size cube
    base_faces = [
        (0, 1, 0, 1, 0, 0),
        (0, 1, 0, 0, 0, 1),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 0, 1, 1, 1),
        (1, 1, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1)
    ]


    random_transform = np.random.randn(3,3)
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
                shape[i] = rand_comp


    random_shape = np.matmul(random_transform, shape.T).T + random_bias

    return random_shape


def TestMainLoop(n_points, K):
    shapes = []
    for i in range(K):
        shapes.append(create_random_rectangle(n_points))

    shapes = torch.Tensor(shapes)
    thing = ShapeRegistration.main(shapes)
    q = thing['q']
    print(q)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(q[:,0], q[:, 1], q[:, 2])
    plt.show()

if __name__ =='__main__':
    shape = create_random_rectangle(1000)
    import matplotlib.pyplot as plt

    TestMainLoop(10000, 10)
