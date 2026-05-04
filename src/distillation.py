# distillation.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def distill_Y_via_regression(Y, Z, model):
    """
    Distill Y by regressing Y on Z and returning the fitted values (no residuals).

    Parameters
    ----------
    Y : array-like, shape (n_samples,) or (n_samples, n_targets)
    Z : array-like, shape (n_samples, n_features)
    model : sklearn-like regressor (must implement fit/predict)

    Returns
    -------
    Y_hat : array-like, same shape as Y
    """
    model.fit(Z, Y)
    Y_hat = model.predict(Z)
    return Y-Y_hat


def distill_Z_pca(Z, n_components):
    """
    Reduce dimensionality of Z using PCA.
    """
    pca = PCA(n_components=n_components)
    Z_reduced = pca.fit_transform(Z)
    return Z_reduced


def distill_Z_random_projection(Z, n_components, random_state=None):
    """
    Reduce dimensionality of Z using Gaussian Random Projection.
    """
    rp = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state
    )
    Z_reduced = rp.fit_transform(Z)
    return Z_reduced


def distill_Z(Z, method="pca", n_components=10, random_state=0):
    """
    General interface for Z distillation.

    Parameters
    ----------
    method : str
        "pca" or "random_projection"
    """
    if method == "pca":
        return distill_Z_pca(Z, n_components)
    elif method == "random_projection":
        return distill_Z_random_projection(Z, n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown Z distillation method: {method}")


def distill_Y(Y, Z, model, method="regression"):
    """
    General interface for Y distillation.
    """
    if method == "regression":
        return distill_Y_via_regression(Y, Z, model)
    else:
        raise ValueError(f"Unknown Y distillation method: {method}")