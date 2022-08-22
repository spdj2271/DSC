import io
import itertools
import tensorflow as tf
import numpy as np
import csv
import os

from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def get_ACC_NMI(y, y_pred):
    y = np.array(y).squeeze()
    y_pred = np.array(y_pred).squeeze()

    y_uni = np.unique(y).astype(np.int)
    y_pred_uni = np.unique(y_pred).astype(np.int)

    if len(y_uni) != len(y_pred_uni):
        raise Exception(f"cluster number mismatched")

    for i, value in enumerate(y_pred_uni):
        y_pred[y_pred == value] = i
    for i, value in enumerate(y_uni):
        y[y == value] = i

    s = np.unique(y_pred)
    t = np.unique(y)
    N = len(np.unique(y_pred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    row, col = linear_sum_assignment(C)
    y_pred_ordered = np.array(y_pred)
    for i in range(N):
        y_pred_ordered[y_pred == row[i]] = col[i]
    acc = np.round(np.sum(y == y_pred_ordered) / len(y) * 100, 2)
    nmi = np.round(normalized_mutual_info_score(y, y_pred) * 100, 2)
    return acc, nmi


def get_n_changed_assignment(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    s = np.unique(y_pred)
    t = np.unique(y)
    N = len(np.unique(y_pred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    row, col = linear_sum_assignment(C)
    y_pred_ordered = np.array(y_pred)
    for i in range(N):
        y_pred_ordered[y_pred == row[i]] = col[i]
    return np.sum(y != y_pred_ordered)


def set_GPU():
    """GPU相关设置"""

    # 打印变量在那个设备上
    # tf.debugging.set_log_device_placement(True)
    # 获取物理GPU个数
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('物理GPU个数为：', len(gpus))
    # 设置内存自增长
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(gpu)
    print('-------------已设置完GPU内存自增长--------------')
    # 获取逻辑GPU个数
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('逻辑GPU个数为：', len(logical_gpus))


def cfg_init(cfg, x, y):
    if not os.path.exists(r'./weights'):
        os.makedirs(r'./weights')
    cfg.N_CLUSTERS = len(np.unique(y))
    if len(x.shape) > 2:
        cfg.MODEL_TYPE = 'conv'
        cfg.INPUT_SHAPE = x.shape[1:] if len(x.shape) == 4 else x.shape[1:] + (1,)  # N,W,H,C
    cfg.iter = 40
    cfg.CLUSTER.BATCH_SIZE = 32
    cfg.AUTOENCODER.WEIGTH_PATH = f'weights/weight_base_{cfg.DS_NAME}.h5'
