from yacs.config import CfgNode as CN

_C = CN()

_C.AUTOENCODER = CN()
_C.AUTOENCODER.AUTOENCODER_EPOCHS = 200
_C.AUTOENCODER.AUTOENCODER_BATCH_SIZE = 256

_C.CLUSTER = CN()
_C.CLUSTER.HIDDEN_UNITS = 10

_C.LOGGER = CN()
_C.LOGGER.OUTPUT_DIR = "logs/logger"

_C.AFFINITY_MAT = CN()
# scaling parameter in Equation 3. distance of each individual to its $K=7$-th neighbor
_C.AFFINITY_MAT.LOCALSCALING = 'mid-7'
_C.AFFINITY_MAT.ADJACENT = 'knn-15'  # ['pair-wise', 'knn-20', 'knn-30', 'knn-50', 'knn-100']
_C.AFFINITY_MAT.AFFINITY = 'rw'  # ['sym', 'rw']
