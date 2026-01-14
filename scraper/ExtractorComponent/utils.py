import numpy as np
import scipy.sparse as sp

from sklearn.cluster._k_means_lloyd import (
    lloyd_iter_chunked_dense,
    lloyd_iter_chunked_sparse,
)
from sklearn.cluster._k_means_common import (
    _inertia_dense,
    _inertia_sparse,
)
from sklearn.utils.parallel import _threadpool_controller_decorator


@_threadpool_controller_decorator(limits=1, user_api="blas")
def _spherical_kmeans_single_lloyd(X: np.ndarray,
                                   sample_weight: np.ndarray,
                                   centers_init: np.ndarray,
                                   max_iter: int = 300,
                                   tol: float = 1e-4,
                                   n_threads: int = 1):
    n_clusters = centers_init.shape[0]

    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    strict_convergence = False

    for i in range(max_iter):
        lloyd_iter(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            labels,
            center_shift,
            n_threads
        )

        centers, centers_new = (centers_new / np.linalg.norm(centers_new), 
                                centers / np.linalg.norm(centers))
        
        if np.array_equal(labels, labels_old):
            strict_convergence = True
            break
        else:
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                break

        labels_old[:] = labels

    if not strict_convergence:
        lloyd_iter(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            labels,
            center_shift,
            n_threads
        )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1
