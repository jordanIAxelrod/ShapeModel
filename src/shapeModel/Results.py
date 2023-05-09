import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.shapeModel.ShapeModel as ShapeModel

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

def read_data(folder, leave_out=1):
    dataframe = []
    cwd = os.getcwd()
    os.chdir(folder)
    curr_dir = os.listdir()
    curr_dir = curr_dir[:leave_out] + curr_dir[leave_out + 1:]

    for direct in curr_dir:
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

def create_ICMP_Model(data, verbose=True, save=False):
    model = ShapeModel.ShapeModel()
    model(data, verbose=verbose)
    if save:
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
    generality = {}
    for i in range(20):
        folder = r"C:\Users\jda_s\Box\bone_project\heart_dataset\masks"
        ssm = create_ICMP_Model(read_data(folder), i==0)
        # ssm = IO.load('hi', r"C:\Users\jda_s\OneDrive\Documents\Research\ShapeModel\model\20230209-121010 ICMP.pickle")
        get_explained_variance(ssm)
        print(ssm.eig_vecs)
        print(ssm.mean_shape, ssm.mean_shape.shape)
        cwd = os.getcwd()
        print(cwd)
        os.chdir(folder)
        curr_dir = os.listdir()[i]
        os.chdir(curr_dir)
        shape = os.listdir()[0]
        new_shape = nib.load(shape).get_fdata()
        new_shape = get_outline(new_shape)
        choice = np.random.choice(new_shape.shape[0], size=(927,), replace=False)
        new_shape = torch.Tensor(new_shape[choice]).unsqueeze(0)
        reg_shape = ssm.register_new_shapes(torch.Tensor(new_shape))
        generality[i] = []
        for j in range(1, ssm.eig_vals.shape[0]):

            new_shape1 = ssm.create_shape_approx(reg_shape, j + 1)
            dist = torch.sqrt(torch.sum(torch.square(reg_shape - new_shape1)) / (927 * 3))
            generality[i].append(dist)
        ax = plt.axes(projection='3d')
        plt.title('Reconstructions')
        ax.scatter(new_shape1[0, :, 0], new_shape1[0, :, 1], new_shape1[0, :, 2])
        ax.scatter(reg_shape[0, :, 0], reg_shape[0, :, 1], reg_shape[0, :, 2])
        os.chdir(cwd)
        print(cwd)
        plt.savefig('../img/Reconstruction.png')
        plt.show()
    averages = []
    for i in range(len(generality[0])):
        average = sum([generality[k][i] for k in generality.keys()]) / len(generality)
        averages.append(average)
    plt.title("Generality")
    plt.xlabel("PC Number")
    plt.plot(list(range(len(averages))), averages)
    plt.savefig('../img/Generality.png')
    plt.show()

