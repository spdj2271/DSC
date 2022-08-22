import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import warnings


def sorted_eig(X):
    e_vals, e_vecs = np.linalg.eigh(X)
    # idx = np.argsort(e_vals)
    # e_vecs = e_vecs[:, idx]
    # e_vals = e_vals[idx]
    return e_vals, e_vecs


def transform_matrix_V(H_pi, U, n_clusters, assignment):
    '''
    construct Orthonormal transformation matrix V
    Args:
        H_pi: autoencoder embeddings
        U: cluster centroids (Kmeans)
        n_clusters:
        assignment:cluster indicator

    Returns: eigenvalues, eigenvectors (called Orthonormal transformation matrix V)

    '''
    S_i = []
    for i in range(n_clusters):
        temp = H_pi[assignment == i] - U[i]
        temp = np.matmul(np.transpose(temp), temp)
        S_i.append(temp)
    S_i = np.array(S_i)
    S = np.sum(S_i, 0)
    Evals, V = sorted_eig(S)  # V[:,i] is a eigenvector
    return Evals, V.astype(np.float32)


class Kmeans(object):
    def __init__(self, n_clusters=10, n_init=10):
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=792)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn", lineno=245)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn", lineno=63)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.kmeans = None

    def fit(self, X, verbose=False):
        if verbose:
            print(f"\tKmeans n_init={self.n_init}", end='\t|\t')
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)
        self.kmeans.fit(X.astype('double'))
        self.n_init = int(self.kmeans.n_iter_ * 2)
        U = self.kmeans.cluster_centers_.astype(np.float32)
        assignment = self.kmeans.labels_
        return assignment, U

    def predict(self, x):
        return self.kmeans.predict(x.astype('double'))


def tensorboard_write_step(summary_writer, step, log_loss, acc, nmi, n_change_assignment, image=None):
    with summary_writer.as_default():
        tf.summary.scalar('train_DSC/loss', log_loss, step=step)
        tf.summary.scalar('train_DSC/acc', acc, step=step)
        tf.summary.scalar('train_DSC/nmi', nmi, step=step)
        tf.summary.scalar('train_DSC/n_change_assignment', n_change_assignment, step=step)
        if list(image):
            if len(image.shape) != 4:
                image = np.expand_dims(image, 0)
            tf.summary.image(name=f'train_DSC/embeddings_{step}', data=image, step=step)
