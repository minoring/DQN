import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


def from_tuple(tuple):
    return from_numpy(np.array(tuple))


def from_numpy(array):
    return torch.from_numpy(array).float().to(device)


def from_img(img):
    """Create tensor from np.uint8 grayscale image."""
    assert img.dtype == np.uint8
    return from_numpy(img.astype(np.float32) / 255.0)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
