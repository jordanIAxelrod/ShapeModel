import os

import torch
import pickle
from datetime import datetime
def read(x) -> torch.Tensor:
    pass

def save(model):
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    pickle.dump(model, f'model/{date} ICMP.pickle')

def load(model, path):
    model = pickle.load(path)
    return model