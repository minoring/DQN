import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


def from_tuple(tuple):
    return from_numpy(np.array(tuple))


def from_numpy(array):
    return torch.from_numpy(array).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
