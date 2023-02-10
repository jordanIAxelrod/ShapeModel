import os

import torch
import pickle
from datetime import datetime


def read(x) -> torch.Tensor:
    pass


def save(model):
    date = datetime.today().strftime("%Y%m%d-%H%M%S")
    with open(f'../model/{date} ICMP.pickle', 'wb') as pickle_out:
        pickle.dump(model, pickle_out)


def load(model, path):
    with open(path, 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    return model
