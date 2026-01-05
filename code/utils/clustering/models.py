
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from .divergences import euclidean, kullback_leibler, itakura_saito, alpha_divergence_factory



class BregmanHard(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters,
        divergence=euclidean,
        max_iter=1000,
        tol=1e-4,
        has_cov=False,
        initializer="random",
        n_init=10,
        covariance_type = "diag",
        reg_covar=1e-6,
        pretrainer=None,
        random_state=None,
        verbose=False
    ):
        """
        Bregman Hard Clustering Algorithm with strict convergence, tol,
        and inertia evolution tracking.
        """
        self.n_clusters = n_clusters
    
        self.divergence = divergence
        self.max_iter = max_iter
        self.tol = tol
        self.has_cov = has_cov
        self.initializer = initializer
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.pretrainer = pretrainer
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.verbose = verbose

    def fit(self, X):
        best_inertia = None
        best_labels = None
        best_centers = None
        best_n_iter = None
        best_history = None

        for run in range(self.n_init):
            if self.verbose:
                print(f"Run {run + 1}/{self.n_init}")
            centers = self._init_centroids(X)
            labels_old = None
            inertia_history = []

            for i in range(self.max_iter):
                labels = self._assign(X, centers)
                inertia = self._compute_inertia(X, centers, labels)
                inertia_history.append(inertia)
                if self.verbose:
                    print(f"Iteration {i}, inertia {inertia}.")
                
                # strict convergence: labels unchanged
                if labels_old is not None and np.array_equal(labels, labels_old):
                    if self.verbose:
                        print(f"Converged at iteration {i}: labels unchanged.")
                    break

                # tol-based convergence: small center shifts
                centers_new = self._update_centroids(X, labels)
                shifts = np.linalg.norm(centers_new - centers, axis=1)
                max_shift = shifts.max()
                if max_shift <= self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {i}: max shift {max_shift:.6f} <= tol {self.tol}.")
                    centers = centers_new
                    labels = self._assign(X, centers)
                    inertia = self._compute_inertia(X, centers, labels)
                    inertia_history.append(inertia)
                    break

                centers, labels_old = centers_new, labels.copy()

            final_iter = len(inertia_history) - 1
            final_inertia = inertia_history[-1]

            if best_inertia is None or final_inertia < best_inertia:
                best_inertia = final_inertia
                best_labels = labels
                best_centers = centers
                best_n_iter = final_iter
                best_history = inertia_history

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.inertia_history_ = best_history

        # After fitting, if verbose, print best run's inertia evolution
        # if self.verbose:
        #     print("Best run inertia evolution:")
        #     for i, inertia in enumerate(self.inertia_history_):
        #         print(f"Iteration {i}, inertia {inertia}.")

        return self

    def _init_centroids(self, X):
        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples, dtype=float)
        p = sample_weight / sample_weight.sum()
        seeds = self.rng.choice(
            n_samples, size=self.n_clusters, replace=False, p=p
        )
        # print("Seeds", seeds)
        return X[seeds]

    def _assign(self, X, centers):
        if self.has_cov:
            D = self.divergence(X, centers, self.cov)
        else:
  
            D = self.divergence(X, centers)
       
        return np.argmin(D, axis=1)

    def _update_centroids(self, X, labels):
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=X.dtype)
        for k in range(self.n_clusters):
            Xk = X[labels == k]
            if Xk.size == 0:
                # EMPTY CLUSTER: relocate its center to a random point
                # (or better: farthest-from-nearest strategy)
                i_farthest = np.argmax(
                    np.min(self.divergence(X, centers), axis=1)
                )
                centers[k] = X[i_farthest]
            else:
                centers[k] = Xk.mean(axis=0)
        return centers

    def _compute_inertia(self, X, centers, labels):
        D = self.divergence(X, centers)
    
        return float(D[np.arange(X.shape[0]), labels].sum())

    def predict(self, X):
        return self._assign(X, self.cluster_centers_)

    
class BregmanSoft(BregmanHard):
    """
    Bregman Soft Clustering Algorithm

    Parameters
    ----------
    n_clusters : INT
        Number of clustes.
    divergence : function
        Pairwise divergence function. The default is euclidean.
    n_iters : INT, optional
        Number of clustering iterations. The default is 1000.
    has_cov : BOOL, optional
        Specifies if the divergence requires a covariance matrix. The default is False.
    initializer : STR, optional
        Specifies if the centroids are initialized at random "rand", K-Means++ "kmeans++", or a pretrained K-Means model "pretrained". The default is "rand".
    init_iters : INT, optional
        Number of iterations for K-Means++. The default is 100.
    pretrainer : MODEL, optional
        Pretrained K-Means model to use as pretrainer.

    Returns
    -------
    None.

    """
        
    def __init__(self, *args, **kwargs):
        super(BregmanSoft, self).__init__(*args, reg_covar=1e-6, **kwargs)


     
    def fit(self, X):

        for run in range(self.n_init):
            if self.verbose:
                print(f"Run {run + 1}/{self.n_init}")
            resp, weights, means, covariances = self._intialize(X)

            for i in range(self.max_iter):
                resp = self._assign(X, weights, means, covariances)
             
                # tol-based convergence: small center shifts
                weights, means, covariances = self.m_step(X, resp)
  

    def _initialize_resp(self, X):

        n_samples = X.shape[0]

        if self.initializer == "random":
            resp = np.asarray(
                self.rng.uniform(size=(n_samples, self.n_clusters)), dtype=X.dtype
            )
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == "random_from_data":
            resp = np.zeros((n_samples, self.n_clusters), dtype=X.dtype)
            indices = self.rng.choice(
                n_samples, size=self.n_clusters, replace=False
            )
            resp[indices, np.arange(self.n_clusters)] = 1
        
        return resp
    
    def _intialize(self, X):
        """
        Initialize the model parameters.

        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.

        Returns
        -------
        None.

        """
        resp = self._initialize_resp(X)
        weights, means, covariances = self._initialize_params(X, resp, self.reg_covar, self.covariance_type)
        return resp, weights, means, covariances

    def _initialize_params(self, X, resp, reg_covar=1e-6, covariance_type="diag"):
        """
        Initialize the model parameters based on the responsibilities.

        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.
        resp : ARRAY
            Responsibilities matrix (n, k) of n samples and k clusters.
        reg_covar : FLOAT, optional
            Regularization term for covariance matrices. The default is 1e-6.
        covariance_type : STR, optional
            Type of covariance matrix. The default is "diag".

        Returns
        -------
        weights : ARRAY
            Mixing coefficients (k, ).
        means : ARRAY
            Cluster centers (k, m).
        covariances : ARRAY
            Covariance matrices (k, m, m).

        """
        n_samples = X.shape[0]
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        weights = nk / n_samples
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        # covariances = {
        #     # "full": _estimate_gaussian_covariances_full,
        #     # "tied": _estimate_gaussian_covariances_tied,
        #     "diag": self._estimate_gaussian_covariances_diag,
        #     # "spherical": _estimate_gaussian_covariances_spherical,
        # }[covariance_type](resp, X, nk, means, reg_covar)
        print("Weights shape:", weights.shape)
        print("Means shape:", means.shape)
        # print("Covariances shape:", covariances.shape)
        print("resp shape:", resp.shape)
        print("nk shape:", nk.shape)
        print("X shape:", X.shape)

        covariances = self._estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar)
        return weights, means, covariances
    
    def _estimate_gaussian_covariances_diag(self, resp, X, nk, means, reg_covar):
        """
        Estimate diagonal covariance matrices.

        Parameters
        ----------
        resp : ARRAY
            Responsibilities matrix (n, k).
        X : ARRAY
            Input data matrix (n, m).
        nk : ARRAY
            Sum of responsibilities for each cluster (k, ).
        means : ARRAY
            Cluster centers (k, m).
        reg_covar : FLOAT
            Regularization term for covariance matrices.

        Returns
        -------
        covariances : ARRAY
            Diagonal covariance matrices (k, m).

        """
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means**2
        return avg_X2 - avg_means2 + reg_covar
    

    def _assign(self, X, weights, params, covariances):
        
        D = self.divergence(X, params, covariances)

        resp = np.exp(-D) * weights
        return resp / resp.sum(axis=1).reshape(-1, 1)

    def m_step(self, X, resp):


        weights = np.mean(resp, axis=0)
        params = np.zeros((self.n_clusters, X.shape[1]), dtype=X.dtype)
        covariances = np.zeros((self.n_clusters, X.shape[1], X.shape[1]), dtype=X.dtype)
        for k in range(self.n_clusters):
            params[k] = (X * resp[:, k].reshape(-1, 1)).sum(axis=0) / resp[:, k].sum()
            if self.has_cov:
                X_mk = (X - params[k]) * resp[:, k].reshape(-1, 1)
                new_covs = np.einsum("ij,ik->jk", X_mk, X_mk) + np.eye(X.shape[1]) * self.reg_covar
                covariances[k] = new_covs / resp[:, k].sum()
           
  
        return weights, params, covariances
        


    def predict(self, X):
        """
        Prediction step.

        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.

        Returns
        -------
        y: Array
            Assigned cluster for each data point (n, )

        """
        return np.argmax(self.assignments(X), axis=1)

    def predict_proba(self, X):
        """
        Probabilities for each cluster.

        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.

        Returns
        -------
        Y: Array
            Probability of each cluster for each point (n, k)

        """
        return self.assignments(X)