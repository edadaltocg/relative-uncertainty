# import numpy as np

# def euclidean(X,Y):
#     """
#     Computes a pairwise Euclidean distance between two matrices: D_ij=||x_i-y_j||^2.

#     Parameters
#     ----------
#         X: array-like, shape=(batch_size, n_features)
#            Input batch matrix.
#         Y: array-like, shape=(n_clusters, n_features)
#            Matrix in which each row represents the mean vector of each cluster.

#     Returns
#     -------
#         D: array-like, shape=(batch_size, n_clusters)
#            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.

#     """
    
#     # same computation as the _old_euclidean function, but a new axis is added
#     # to X so that Y can be directly broadcast, speeding up computations
#     return np.sum((np.expand_dims(X, axis=1)-Y)**2, axis=-1)

import numpy as np

def mahalanobis(X, Y, cov):
    
    diff = np.expand_dims(np.expand_dims(X, axis=1)-Y, axis=-1)
    return np.sum(np.squeeze(((np.linalg.pinv(cov)@diff)*diff)), axis=-1)

def euclidean(X, Y):
    """
    Squared Euclidean (Bregman) divergence:
      D(x||y) = ||x - y||^2,
    corresponding to potential F(x) = 1/2 ||x||^2.
    """
    return np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)


def kullback_leibler(X, Y, eps=1e-15):
    """
    Kullback–Leibler divergence (I-divergence):
      D(p||q) = sum_i p_i * log(p_i / q_i) - p_i + q_i,
    for distributions p, q on the simplex.
    """
    P = np.clip(X, eps, None)
    Q = np.clip(Y, eps, None)
    # compute p log(p/q) - p + q
    return np.sum(P[:, None, :] * np.log(P[:, None, :] / Q[None, :, :])
                  - P[:, None, :] + Q[None, :, :], axis=2)


def itakura_saito(X, Y, eps=1e-15):
    """
    Itakura–Saito divergence:
      D(p||q) = sum_i (p_i / q_i) - log(p_i / q_i) - 1,
    for positive vectors p, q.
    """
    P = np.clip(X, eps, None)
    Q = np.clip(Y, eps, None)
    ratio = P[:, None, :] / Q[None, :, :]
    return np.sum(ratio - np.log(ratio) - 1, axis=2)


def mahalanobis(X, Y, A):
    """
    Mahalanobis Bregman divergence:
      D(x||y) = (x - y)^T A (x - y),
    corresponding to F(x)=1/2 x^T A x, with A positive-definite.
    """
    diff = X[:, None, :] - Y[None, :, :]
    # (n_samples, n_clusters, n_features) -> (n_samples, n_clusters)
    # using Einstein summation for efficiency
    return np.einsum('nij,ij, nij->ni', diff, A, diff)


def alpha_divergence_factory(alpha):
    """
    Returns a function to compute the Amari alpha-divergence:
      D^{(alpha)}(p||q)
    as per Wikipedia's canonical form :contentReference[oaicite:6]{index=6}.
    """
    def alpha_div(X, Y, eps=1e-15):
        P = np.clip(X, eps, None)
        Q = np.clip(Y, eps, None)
        if alpha == 0:
            # limit -> Itakura–Saito
            ratio = P[:, None, :] / Q[None, :, :]
            return np.sum(ratio - np.log(ratio) - 1, axis=2)
        elif alpha == 1:
            # limit -> KL divergence
            return kullback_leibler(X, Y, eps)
        else:
            # general alpha-divergence
            c1 = 4.0 / (1 - alpha**2)
            term = ( (1 - alpha)/2 * P[:, None, :]
                   + (1 + alpha)/2 * Q[None, :, :]
                   - (P[:, None, :]**((1 - alpha)/2)
                      * Q[None, :, :]**((1 + alpha)/2)) )
            return c1 * np.sum(term, axis=2)
    return alpha_div
