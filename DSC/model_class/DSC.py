import os.path

from base_model import *
from cluster_train.common import Kmeans
from datasets.dataset import *
from cluster_train import *
from utils.utils import get_ACC_NMI


class DSC(object):
    '''
     class of DSC
    '''

    def __init__(self, cfg):
        self.model_type = cfg.MODEL_TYPE
        self.cfg = cfg
        self.model_has_decoder = True
        self.model = load_model_conv(cfg, load_weights=False)

    def load_reconstruction_weights(self, cfg, path=None):
        self.cfg = cfg if cfg else self.cfg
        if not path:
            path = cfg.AUTOENCODER.WEIGTH_PATH
        if not os.path.exists(path):
            raise Exception('no weights available')
        self.model.load_weights(path)

    def get_AE_embeddings(self, x):
        return self.model.predict(x)[:, :self.cfg.CLUSTER.HIDDEN_UNITS]

    def train_reconstruction(self, x=None, epoch=None, dataset=None, cfg=None, y=None):
        self.cfg = cfg if cfg else self.cfg
        batchsize = self.cfg.AUTOENCODER.AUTOENCODER_BATCH_SIZE
        if not epoch:
            epoch = self.cfg.AUTOENCODER.AUTOENCODER_EPOCHS
        if dataset:
            ds_xx = dataset.batch(batchsize)
        else:
            ds_xx = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(len(x)).batch(batchsize)
        train_base_model_conv(self.model, ds_xx, self.cfg, epoch, batchsize=batchsize)
        self.model.load_weights(self.cfg.AUTOENCODER.WEIGTH_PATH)

    def train(self, x, y, cfg=None):
        '''
        clustering training
        Args:
            x: image data
            y: image label
            cfg: configuration files

        Returns:

        '''
        if self.model_has_decoder:
            self.model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('embed').output)
            self.model_has_decoder = False
        # self.model.summary()
        if not cfg:
            cfg = self.cfg
        # train
        acc, nmi, assignment = train_DSC(self.model, x, y, cfg)
        return acc, nmi, assignment

    def load_final_weights(self, cfg=None, path=None):
        if self.model_has_decoder:
            self.model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('embed').output)
            self.model_has_decoder = False
        self.cfg = cfg if cfg else self.cfg
        if not path:
            path = cfg.AUTOENCODER.WEIGTH_PATH
        if not os.path.exists(path):
            raise Exception('no weights available')
        self.model.load_weights(path)

    def evaluate(self, x, y):
        n_clusters = len(np.unique(y))
        with tf.device('cpu'):
            H = self.model(x).numpy()
        assignment, _ = Kmeans(n_clusters=n_clusters, n_init=50).fit(H)
        acc, nmi = get_ACC_NMI(y, assignment)
        return acc, nmi
