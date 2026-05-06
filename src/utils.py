import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import os
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression

def generate_gaussian_linear_data(
    n, beta=1.0, gamma=1.0, rho=0.6, sigma=1.0, mu=[0, 0], seed=None
):
    """
    Generate data for the model:
        Y = beta * X + gamma * Z + epsilon

    where (X, Z) are jointly Gaussian with correlation rho.

    Returns
    -------
    X : (n,)
    Z : (n,)
    Y : (n,)
    """

    rng = np.random.default_rng(seed)

    cov = [[1, rho], [rho, 1]]

    XZ = rng.multivariate_normal(mean=mu, cov=cov, size=n)

    X = XZ[:, 0]
    Z = XZ[:, 1]

    epsilon = rng.normal(0, sigma, n)

    Y = beta * X + gamma * Z + epsilon

    return X, Y,  Z


def gaussian_linear(n, seed):
    return generate_gaussian_linear_data(
        n, beta=1, gamma=1, rho=0.6, seed=seed
    )


def nonlinear_cos(n, seed):
    np.random.seed(seed)
    # Z = np.random.normal(size=(n, 1))
    # X = Z + np.random.normal(scale=0.5, size=(n, 1))
    rho = 0.6
    cov = [[1, rho], [rho, 1]]
    rng = np.random.default_rng(seed)
    XZ = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    X = XZ[:, 0]
    Z = XZ[:, 1]
    Y = np.cos(X) + Z + rng.normal(0, 1, n)

    return X, Y,  Z


def interaction_model(n, seed):
    np.random.seed(seed)
    # Z = np.random.normal(size=(n, 2))
    # X = Z[:, [0]] + np.random.normal(size=(n, 1))
    rho = 0.6
    cov = [[1, rho], [rho, 1]]
    rng = np.random.default_rng(seed)
    XZ = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)

    X = XZ[:, 0]
    Z = XZ[:, 1]
    Y = X * Z + rng.normal(0, 1, n)
    return X, Y,  Z


def heteroskedastic(n, seed):
    rng = np.random.default_rng(seed)

    rho = 0.6
    cov = [[1, rho], [rho, 1]]

    XZ = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)

    X = XZ[:, 0]
    Z = XZ[:, 1]

    noise = rng.normal(loc=0, scale=1 + np.abs(Z), size=n)
    Y = X + noise

    return X, Y,  Z

def generate_simulated_data2D(setting, n, seed=0):
    settings = {
        "gaussian_linear": gaussian_linear,
        "nonlinear_cos": nonlinear_cos,
        "interaction": interaction_model,
        "heteroskedastic": heteroskedastic,
        #"nonlinear_product": nonlinear_product,
    }

    if setting not in settings:
        raise ValueError(f"Unknown setting: {setting}")

    return settings[setting](n=n, seed=seed)

# Higher dimension
def sample_XZ(n, p, rho, rng):
    """
    Generate X scalar and Z (p-dim) jointly Gaussian with correlation rho.
    """
    Z = rng.normal(size=(n, p))
    eps = rng.normal(size=n)

    # Correlate X with first component of Z
    X = rho * Z[:, 0] + np.sqrt(1 - rho**2) * eps

    return X, Z


def gaussian_linear_multi(n, p, seed):
    rng = np.random.default_rng(seed)
    X, Z = sample_XZ(n, p, rho=0.6, rng=rng)

    beta = 1.0
    gamma = np.ones(p) / np.sqrt(p)

    Y = beta * X + Z @ gamma + rng.normal(0, 1, n)

    return X, Y,  Z

def nonlinear_cos_multi(n, p, seed):
    rng = np.random.default_rng(seed)
    X, Z = sample_XZ(n, p, rho=0.6, rng=rng)

    Y = np.cos(X) + Z[:, 0] + 0.5 * Z[:, 1 % p] + rng.normal(0, 1, n)

    return X, Y,  Z

def interaction_multi(n, p, seed):
    rng = np.random.default_rng(seed)
    X, Z = sample_XZ(n, p, rho=0.6, rng=rng)

    interaction = X * (Z[:, 0] + Z[:, 1 % p])

    Y = interaction + rng.normal(0, 1, n)

    return X, Y,  Z

def sparse_linear(n, p, seed):
    rng = np.random.default_rng(seed)
    X, Z = sample_XZ(n, p, rho=0.6, rng=rng)

    gamma = np.zeros(p)
    gamma[:3] = 1.0  # only first 3 relevant

    Y = X + Z @ gamma + rng.normal(0, 1, n)

    return X, Y,  Z

def heteroskedastic_multi(n, p, seed):
    rng = np.random.default_rng(seed)
    X, Z = sample_XZ(n, p, rho=0.6, rng=rng)

    scale = 1 + np.linalg.norm(Z, axis=1)
    noise = rng.normal(0, scale)

    Y = X + noise

    return X, Y,  Z

def nonlinear_product(n, p, seed):
    rng = np.random.default_rng(seed)
    X, Z = sample_XZ(n, p, rho=0.6, rng=rng)

    Y = np.sin(X * Z[:, 0]) + np.cos(Z[:, 1 % p]) + rng.normal(0, 1, n)

    return X, Y,  Z

def generate_simulated_dataHD(setting, n, p, seed):
    settings = {
        "gaussian_linear": gaussian_linear_multi,
        "nonlinear_cos": nonlinear_cos_multi,
        "interaction": interaction_multi,
        "sparse_linear": sparse_linear,
        "heteroskedastic": heteroskedastic_multi,
        "nonlinear_product": nonlinear_product,
    }

    if setting not in settings:
        raise ValueError(f"Unknown setting: {setting}")

    return settings[setting](n=n, p=p, seed=seed)



# Conditional sampler
def theoretical_sample_X_given_Z(
    Z, mu_x=0, mu_z=0, sigma_x=1, sigma_z=1, rho=0.5, rng=None
):
    """
    Sample X | Z from a Gaussian conditional distribution.
    """

    if rng is None:
        rng = np.random.default_rng()

    cond_mean = mu_x + rho * (sigma_x / sigma_z) * (Z - mu_z)
    cond_std = sigma_x * np.sqrt(1 - rho**2)

    return rng.normal(cond_mean, cond_std)


def sample_X_tilde_theoretical(Z, B, rng):
    n = len(Z)
    X_tilde = np.zeros((n, B))
    for b in range(B):
        X_tilde[:, b] = theoretical_sample_X_given_Z(Z, rng=rng)
    return X_tilde




# Loss functions
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred_proba):
    eps = 1e-15  # avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred_proba) +
        (1 - y_true) * np.log(1 - y_pred_proba)
    )

def zero_one_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)


def loss_chooser(setting):
    if setting=="classification_0_1":
        return zero_one_loss
    elif setting=="classif_CE":
        return cross_entropy_loss
    else:
        return mse_loss
def prediction_chooser(loss):
    if loss==cross_entropy_loss:
        return lambda m, X: m.predict_proba(X)[:,1],
    else:
        return lambda m, X: m.predict(X)





def get_base_model(model_name, random_state, n_jobs=1):
    if model_name == 'lasso':
        return LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=random_state)
    elif model_name == 'lm':
        return LinearRegression()
    elif model_name == 'RF':
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif model_name == 'NN':
        return MLPRegressor(
            hidden_layer_sizes=(32,),
            max_iter=200,
            random_state=random_state
        )
    elif model_name == 'GB':
        return GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=random_state
        )
    elif model_name == 'SL':
        estimators = [
            ('lasso', Lasso(alpha=0.01, random_state=random_state)),
            ('rf', RandomForestRegressor(
                n_estimators=30,
                max_depth=8,
                random_state=random_state,
                n_jobs=n_jobs
            )),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            n_jobs=n_jobs
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")