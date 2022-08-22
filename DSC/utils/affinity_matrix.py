from scipy.sparse import coo_matrix, csr_matrix
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
# from torch_geometric.utils import from_scipy_sparse_matrix, get_laplacian, to_scipy_sparse_matrix

from utils.utils import *
from datasets.dataset import *
from scipy import sparse
import warnings


def get_affinity_matrix(x, cfg=None, return_A=False):
    try:
        W = get_affinity_matrix_fast(x, cfg)
    except:
        W = get_affinity_matrix_slow(x, cfg)
    return W


def get_affinity_matrix_fast(x, cfg=None, return_A=False):
    ls_method = cfg.AFFINITY_MAT.LOCALSCALING
    W_method = cfg.AFFINITY_MAT.AFFINITY
    A_method = cfg.AFFINITY_MAT.ADJACENT
    assert A_method.startswith('knn')
    assert ls_method.startswith('mid') or ls_method == "None"

    warnings.filterwarnings("ignore", "RuntimeWarning")
    n_neighbors = int(A_method.split('-')[1])

    # A = kneighbors_graph(x, n_neighbors, include_self=False, n_jobs=50, mode='distance').toarray()
    A = kneighbors_graph(x, n_neighbors, include_self=False, n_jobs=50, mode='distance')
    nonzero = np.nonzero(A)
    A_nonzero = np.array(A[nonzero].A).reshape((A.shape[0], n_neighbors))
    if ls_method.startswith('mid'):
        mid_K = int(ls_method.split('-')[1])
        sigma = np.sort(A_nonzero, axis=-1)[:, -(n_neighbors - mid_K)].reshape((-1, 1))
        sigma_j = sigma[nonzero[1]].reshape((A.shape[0], -1))
        sigma_ij = sigma_j * sigma
        A = np.exp(-1 * (A_nonzero ** 2) / sigma_ij).ravel()
        A = csr_matrix((A, nonzero))
    D = A.sum(axis=-1)
    if W_method == 'rw':
        D_inv = 1 / D
        D_inv[D_inv == np.inf] = 0
        # D_inv = csr_matrix(np.diag(np.squeeze(D_inv.A)))
        D_inv = np.squeeze(D_inv.A)
        diag_idx = np.arange(A.shape[0])
        D_inv = csr_matrix((D_inv, (diag_idx, diag_idx)))
        W = D_inv @ A
    elif W_method == 'sym':
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[D_inv_sqrt == np.inf] = 0
        # D_inv_sqrt = coo_matrix(np.diag(np.squeeze(D_inv_sqrt.A)))
        D_inv_sqrt = np.squeeze(D_inv_sqrt.A)
        diag_idx = np.arange(A.shape[0])
        D_inv_sqrt = csr_matrix((D_inv_sqrt, (diag_idx, diag_idx)))
        W = D_inv_sqrt @ A @ D_inv_sqrt
    if return_A:
        return A, W
    return W


def get_affinity_matrix_slow(x, cfg=None, return_A=False):
    ls_method = cfg.AFFINITY_MAT.LOCALSCALING
    W_method = cfg.AFFINITY_MAT.AFFINITY
    A_method = cfg.AFFINITY_MAT.ADJACENT
    assert A_method.startswith('knn')
    assert ls_method.startswith('mid') or ls_method == "None"

    warnings.filterwarnings("ignore", "RuntimeWarning")
    n_neighbors = int(A_method.split('-')[1])

    A = kneighbors_graph(x, n_neighbors, include_self=False, n_jobs=50, mode='distance').toarray()
    # A = kneighbors_graph(x, n_neighbors, include_self=False, mode='distance').toarray()
    if ls_method.startswith('mid'):
        mid_K = int(ls_method.split('-')[1])
        sigma = np.sort(A, axis=-1)[:, -(n_neighbors - mid_K)].reshape((-1, 1))
        sigma_ij = sigma @ sigma.T
        A = np.where(A != 0, np.exp(-1. * (A ** 2.) / sigma_ij), 0)
    D = np.sum(A, axis=1)
    A = coo_matrix(A)
    if W_method == 'rw':
        D_inv = 1 / D
        D_inv[D_inv == np.inf] = 0
        D_inv = coo_matrix(np.diag(D_inv))
        W = D_inv @ A
    elif W_method == 'sym':
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[D_inv_sqrt == np.inf] = 0
        D_inv_sqrt = coo_matrix(np.diag(D_inv_sqrt))
        W = D_inv_sqrt @ A @ D_inv_sqrt
    if return_A:
        return A, W
    return W
