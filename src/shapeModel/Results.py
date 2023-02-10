import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ShapeModel
import IO

import torch
import nibabel as nib
import skimage.measure
def get_outline(shape):
    point_list = []
    shape = skimage.measure.block_reduce(shape, (4,4,4))
    for i in range(shape.shape[0]):
        for j in range(shape.shape[1]):
            for k in range(shape.shape[2]):
                if np.any(shape[i - 1: i + 2, j - 1: j + 2, k - 1: k + 2] == 0) and shape[i, j, k] > 0:
                    point_list.append([i, j, k])
    return np.array(point_list)

def read_data(folder):
    dataframe = []
    cwd = os.getcwd()
    os.chdir(folder)
    curr_dir = os.listdir()
    print(curr_dir[:18])
    print(len(curr_dir))
    for direct in curr_dir[:18]:
        os.chdir(direct)
        file = os.listdir()[0]

        shape_cloud = nib.load(file).get_fdata()
        os.chdir('..')
        shape_cloud = get_outline(shape_cloud)

        dataframe.append(shape_cloud)
    min_len = min(dataframe, key=lambda x: x.shape[0]).shape[0]
    for i, data in enumerate(dataframe):
        choice = np.random.choice(data.shape[0], size=(min_len,), replace=False)
        dataframe[i] = torch.Tensor(data[choice]).unsqueeze(0)

    os.chdir(cwd)
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
    explained_variance = np.cumsum(model.eig_vals) / sum(model.eig_vals)
    print(explained_variance)
    print(model.eig_vals)
    plt.plot([i for i in range(len(explained_variance))], explained_variance)
    plt.title("Explained Variance per mode")
    plt.show()

if __name__ == '__main__':
    ssm = create_ICMP_Model(read_data(r"C:\Users\jda_s\Box\bone_project\heart_dataset\masks"))
    # ssm = IO.load('hi', r"C:\Users\jda_s\OneDrive\Documents\Research\ShapeModel\model\20230209-121010 ICMP.pickle")
    get_explained_variance(ssm)
    print(ssm.eig_vecs)
    print(ssm.mean_shape, ssm.mean_shape.shape)
    mean_shape = ssm.mean_shape.reshape(-1, 3)
    ax = plt.axes(projection='3d')
    plt.title('mean shape original size')
    ax.scatter(mean_shape[:, 0], mean_shape[:, 1], mean_shape[:, 2])
    plt.show()
    new_shape = nib.load(r"C:\Users\jda_s\Box\bone_project\heart_dataset\masks\la_029\x.nii.gz").get_fdata()
    new_shape = get_outline(new_shape)
    choice = np.random.choice(new_shape.shape[0], size=(927,), replace=False)
    new_shape = torch.Tensor(new_shape[choice]).unsqueeze(0)
    reg_shape = ssm.register_new_shapes(torch.Tensor(new_shape))

    for i in range(ssm.eig_vals.shape[0] - 1, ssm.eig_vals.shape[0]):
        new_shape = ssm.create_shape_approx(reg_shape, i + 1)
        ax = plt.axes(projection='3d')
        plt.title('Reconstructions')
        ax.scatter(new_shape[0,:, 0], new_shape[0,:, 1], new_shape[0,:, 2])
        ax.scatter(reg_shape[0, :, 0], reg_shape[0, :, 1], reg_shape[0, :, 2])
        plt.show()
