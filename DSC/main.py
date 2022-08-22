import time

from base_model import *
from utils.utils import *
from cluster_train import *
from datasets.dataset import *
from model_class.DSC import *
import argparse

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # dataset settings
    parser = argparse.ArgumentParser(
        description='select dataset:FRGC, COIL20, USPS, MNIST_test, FASHION_test, MNIST, FASHION')
    parser.add_argument('--dataset', default='USPS', required=False)
    args = parser.parse_args()
    ds_name_list = ['FRGC', 'COIL20', 'USPS', 'MNIST_test', 'FASHION_test', 'MNIST', 'FASHION']
    if args.dataset is None or not args.dataset in ds_name_list:
        ds_name = 'USPS'
    else:
        ds_name = args.dataset

    time_start = time.time()
    list_performance = []
    from config import cfg  # load configuration files

    # load dataset
    x, y = load_dataset_xy(ds_name)
    cfg.DS_NAME = ds_name
    # initialize parameters
    cfg_init(cfg, x, y)
    print("Running with config:\n{}".format(cfg))

    # load DSC model
    DSC = DSC(cfg)
    # pretrain autoencoder
    cfg.AUTOENCODER.WEIGTH_PATH = f'weights/weight_base_{cfg.DS_NAME}.h5'
    if not os.path.exists(cfg.AUTOENCODER.WEIGTH_PATH):
        print("*" * 20 + f"    pretraining starts, ({cfg.DS_NAME})   " + "*" * 20)
        DSC.train_reconstruction(x)
    DSC.load_reconstruction_weights(cfg)
    # clustering train via simultaneous spectral embedding and entropy minimization
    print("*" * 20 + f"    clustering training starts, ({cfg.DS_NAME})   " + "*" * 20)
    acc, nmi, _ = DSC.train(x, y, cfg)
    time_sum = time.time() - time_start
    print(f'(DSC) running time:{int(time_sum)} seconds, ACC={acc}, NMI={nmi}')
