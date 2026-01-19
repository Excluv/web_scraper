import warnings
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
from sklearn.cluster._k_means_common import (
    _inertia_dense,
    _is_same_clustering,
)
from sklearn.utils.parallel import _threadpool_controller_decorator
from sklearn.base import _fit_context
from sklearn.utils import check_array, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    validate_data,
)


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


class SphericalKMeans(KMeans):
    def __init__(self, 
                 n_clusters: int = 2,
                 init: str = "k-means++",
                 n_init: str = "auto",
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 random_state: int = 42,
                 algorithm: str = "lloyd"):
        super().__init__(
            n_clustesr=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
        self.algorithm = algorithm

    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        X = validate_data(
            self, 
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)

        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        init = self.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        x_squared_norms = row_norms(X, squared=True)
        
        kmeans_single = _spherical_kmeans_single_lloyd
        self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self._n_init):
            centers_init = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                sample_weight=sample_weight,
            ) 
            centers_init /= np.norm(centers_init, axis=1)

            labels, inertia, centers, n_iter_ = kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                tol=self._tol,
                n_threads=self._n_threads,
            )

            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(f"")

        self.cluster_centers_ = best_centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self
        