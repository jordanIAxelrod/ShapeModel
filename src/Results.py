import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ShapeModel
import IO
from pyoints import registration

import torch
from medpy import io

def get_outline(shape):
    point_list = []
    for i in range(shape.shape[0]):
        for j in range(shape.shape[1]):
            for k in range(shape.shape[2]):
                if np.any(shape[i - 1: i + 2, j - 1: j + 2, k - 1: k + 2] == 0) and shape[i, j, k] > 0:
                    point_list.append([i, j, k])
    print(len(point_list))
    return np.array(point_list)

def read_data(folder):
    dataframe = []
    cwd = os.getcwd()
    os.chdir(folder)
    for file in os.listdir()[:6]:
        path = os.path.join(folder, file)
        shape_cloud = io.load(path)[0]
        shape_cloud = get_outline(shape_cloud)

        dataframe.append(shape_cloud)
    min_len = min(dataframe, key=lambda x: x.shape[0]).shape[0]
    for i, data in enumerate(dataframe):
        choice = np.random.choice(data.shape[0], size=(min_len,), replace=False)
        dataframe[i] = torch.Tensor(data[choice]).unsqueeze(0)

    os.chdir(cwd)
    print(os.getcwd())
    return torch.cat(dataframe, dim=0)

def create_ICMP_Model(data):
    model = ShapeModel.ShapeModel()
    model(data, verbose=True)
    model.save()
    return model


# If we want to replicate older paper
def create_ICP_Model(data):
    icp = registration.ICP(
        [3, 3, 3]
    )
    coord_dict = {str(i): data[i] for i in range(data.shape[0])}
    T, pairs, _ = icp(coord_dict)
    print(pairs)
    new_points = []
    for key in T.keys():

        loc = T[key].to_local(coord_dict[key])
        if key != '0':
            loc = loc[pairs['0'][key][0]]
        new_points.append(loc)

def get_explained_variance(model):
    explained_variance = np.cumsum(model.eigvals) / sum(model.eigvals)
    plt.plot([i for i in range(len(explained_variance))], explained_variance)
    plt.title("Explained Variance per mode")
    plt.show()

if __name__ == '__main__':
    create_ICMP_Model(read_data(r"C:\Users\jorda\Downloads\ShapeAnalysisModule_TutorialData\ShapeAnalysisModule_TutorialData\origData"))
