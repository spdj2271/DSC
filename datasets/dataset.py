import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import h5py
import scipy.io as scio
import pandas as pd


def load_dataset_xy(ds_name, dir_path=r'datasets/data/', seed=None, flatten=False):
    '''

    Args:
        ds_name: dataset name, e.g. 'FRGC', 'COIL20', 'USPS', 'MNIST_test', 'FASHION_test', 'MNIST', 'FASHION'
        dir_path: where datasets exist
        seed: shuffle seed
        flatten: whether flat raw image (N,W,H,C) into (N,-1)

    Returns:
        x:images (have been normalized between [-1,1])
        y:labels
    '''
    if not seed:
        seed = 1

    with h5py.File(f"{dir_path}/{ds_name}.h5", "r") as f:
        x = np.array(f['x'][:])
        y = np.array(f['y'][:])
    if flatten and len(x.shape) != 2:
        x = np.reshape(x, (len(x), -1))
    idx = np.random.RandomState(seed=seed).permutation(len(x))
    x = x[idx]
    y = y[idx]
    print(f"dataset ({ds_name}) is loaded, x.shape={x.shape}, y.shape={y.shape}")
    return x, y


def load_dataset(cfg, seed=None):
    x, y = load_dataset_xy(cfg.DS_NAME, seed=seed)
    ds = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(len(x))
    return ds, x, y
