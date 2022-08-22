import numpy as np
from tqdm import tqdm
from utils.affinity_matrix import get_affinity_matrix
from utils.utils import *
from tensorflow.keras import losses
from .common import *


class pbarCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(pbarCallback, self).__init__()
        self.pbar = tqdm(total=10)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        self.pbar.set_postfix(logs)

    def on_train_end(self, logs=None):
        self.pbar.close()


def train_DSC(model, x, y, cfg):
    '''
    clustering training
    Args:
        model: autoencoder object
        x: image data
        y: image label
        cfg: configuration
    Returns:

    '''

    def loss_train(y_true_batch, y_pred_batch):
        '''
        total loss function of DSC
        Args:
            y_true_batch: target matrix Y
            y_pred_batch: outputs of the encoder

        Returns:

        '''
        y_pred_batch = y_pred_batch[:, :hidden_units]
        loss = losses.mse(y_true_batch, y_pred_batch)
        return loss

    hidden_units = cfg.CLUSTER.HIDDEN_UNITS
    n_clusters = cfg.N_CLUSTERS
    batch_size = cfg.CLUSTER.BATCH_SIZE

    ite_PI = 0
    a_PI = 0
    ds_name = cfg.DS_NAME
    n_skip = 0
    n_iter = cfg.iter
    assignment = np.array([-1] * len(x))
    optimizer = tf.keras.optimizers.Adam()
    # train
    for step in range(20):
        print(f"iter {step + 1}:", end='\t|\t')
        assignment_last = np.array(assignment)
        # get autoencoder embeddings
        with tf.device('cpu'):
            H = model(x).numpy()[:, :hidden_units]
        # yield baseline
        if step == 0:
            assignment, _ = Kmeans(n_clusters=n_clusters, n_init=50).fit(H)
            acc, nmi = get_ACC_NMI(y, assignment)

        H_pi = np.array(H)
        # construct normalized affinity matrix W
        W = get_affinity_matrix(H_pi, cfg=cfg)
        # power iteration
        v = 0
        for ite_PI in range(15):
            H_pi_old = np.array(H_pi)
            v_old = np.array(v)
            H_pi = W.dot(H_pi)
            v = np.mean(H_pi - H_pi_old, -1)
            a_PI = np.linalg.norm(v - v_old, ord=np.inf)
            if a_PI <= 1e-3:
                break
        print(f'PI iteartions={ite_PI}, acceleration={np.round(a_PI, 5)}', end="\t|\t")

        # cluster on power iteration embeddings H_pi (called Z in paper)
        assignment, U_pi = Kmeans(n_clusters=n_clusters, n_init=50).fit(H_pi)

        # obtain clustering performance
        acc, nmi = get_ACC_NMI(y, assignment)
        # count the number of samples changed assignments between two optimizing steps
        n_changed_assignment = 0 if step == 0 else get_n_changed_assignment(assignment, assignment_last)
        print(f'n_changed_assignment={n_changed_assignment}, acc,nmi={acc, nmi}')

        if n_changed_assignment <= max(len(x) * 0.005, 5) and step != 0:
            model.save_weights(f'weights/weight_final_{ds_name}_NOISE.h5')
            print("end")
            break
        # construct Orthonormal transformation matrix V
        _, V = transform_matrix_V(H_pi, U_pi, n_clusters, assignment)

        # transform autoencoder embeddings into new space
        H_v = H_pi @ V
        U_v = U_pi @ V

        # construct training label (target matrix Y in paper)
        y_true = np.array(H_v)
        if step != 0:
            # the last dimension replaced with cluster centroids (Equation 16 in paper)
            y_true[:, -1] = U_v[assignment][:, -1]
        y_true = y_true @ V.T

        if step == 0:
            ds = tf.data.Dataset.from_tensor_slices((x, y_true)).batch(batch_size)
            model.compile(optimizer='adam', loss=loss_train)
            model.fit(ds, epochs=10, verbose=0)
            continue

        # construct dataset (X:image data, Y:target matrix Y)
        ds = tf.data.Dataset.from_tensor_slices((x, y_true)).skip(n_skip).batch(batch_size).repeat().take(n_iter)
        n_skip = (n_skip + n_iter) * batch_size % x.shape[0]
        # train
        for x_batch, y_batch in ds:
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                y_pred = model(x_batch)
                loss = losses.mse(y_batch, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return acc, nmi, assignment
