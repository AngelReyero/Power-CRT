from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from utils import mse_loss, cross_entropy_loss, zero_one_loss
from sklearn.base import clone

def CRT(X, X_tilde, Y, Z, T, distill_Y=None, distill_Z=None, n_jobs=5):
    # Distillation logic
    if distill_Y is not None:
        distilled_Y, distilled_Z = distill_Y(Y, Z)
    else:
        distilled_Y, distilled_Z = Y, Z

    if distill_Z is not None:
        distilled_Z = distill_Z(Z)

    # Original statistic
    T_orig = T(X, distilled_Y, distilled_Z)

    # Parallel computation
    def compute_stat(i):
        return T(X_tilde[:, i], distilled_Y, distilled_Z)

    T_b = Parallel(n_jobs=n_jobs)(
        delayed(compute_stat)(i) for i in range(X_tilde.shape[1])
    )

    return T_orig, T_b




def CRT_comparison(
    X,
    X_tilde,
    Y,
    Z,
    T_list,              # list of (name, function)
    distill_Y=None,
    distill_Z=None,
    n_jobs=5,
    model_HRT=RandomForestRegressor(),
    train_test_HRT=0.8,
):
    # Distillation
    if distill_Y is not None:
        distilled_Y = distill_Y(Y, Z)
    else:
        distilled_Y = Y

    if distill_Z is not None:
        distilled_Z = distill_Z(Z)
    else:
        distilled_Z = Z   

    results = {}

    for name, T in T_list:
        # Original statistic
        print(f"T:{name}")
        start_time = time.time()
        if name=='hrt':
            tr_i = int(X.shape[0] * train_test_HRT)
            X_train = np.column_stack([X[:tr_i], distilled_Z[:tr_i]])
            Y_train = distilled_Y[:tr_i]

            model_HRT.fit(X_train, Y_train)
            # evaluation data
            Y_eval = distilled_Y[tr_i:]

            T_orig = T(X[tr_i:], Y_eval, distilled_Z[tr_i:], model=model_HRT)

            # Parallel null statistics
            T_b = Parallel(n_jobs=n_jobs)(
                delayed(T)(X_tilde[tr_i:, i], distilled_Y[tr_i:], distilled_Z[tr_i:],
                               model=model_HRT)
                for i in range(X_tilde.shape[1])
            )

            T_b = np.array(T_b)
        else:    
            T_orig = T(X, distilled_Y, distilled_Z)

            # Parallel null statistics
            T_b = Parallel(n_jobs=n_jobs)(
                delayed(T)(X_tilde[:, i], distilled_Y, distilled_Z)
                for i in range(X_tilde.shape[1])
            )

            T_b = np.array(T_b)

        # Rank of T_orig among T_b (higher = more extreme)
        p_val = (np.sum(T_b >= T_orig) + 1)/((X_tilde.shape[1])+1)# +1 includes T_orig
        computation_time = time.time()-start_time
        results[name] = {
            "T_orig": T_orig,
            "T_b": T_b,
            "p_value": p_val,
            "time": computation_time
        }

    return results







def T_OLS(X, Y, Z):

    X_train = np.column_stack([X, Z])
    model = LinearRegression().fit(X_train, Y)

    beta_hat = model.coef_
    return abs(beta_hat[0])

def T_optCRT(X, Y, Z, i, model, predict_fn, loss_fn):
    """
    Parameters
    ----------
    X, Y, Z : arrays of length n
    i: index at which to evaluate the test
    model : callable returning a fitted model with predict()

    Returns
    -------
    T : float
    """

    # training data
    X_train = np.column_stack([X[:i], Z[:i]])
    Y_train = Y[:i]

    m_i = clone(model)
    m_i.fit(X_train, Y_train)

    # evaluation data
    X_eval = np.column_stack([X[i:], Z[i:]])
    Y_eval = Y[i:]

    preds = predict_fn(m_i, X_eval)
    return loss_fn(Y_eval, preds)



def T_jk(X, Y, Z, model, predict_fn, loss_fn):
    n = len(X)
    t_jk = 0
    for _ in range(n):
        X = np.roll(X, -1)
        Z = np.roll(Z, -1)
        Y = np.roll(Y, -1)
        t_jk += T_optCRT(X, Y, Z, i=n - 1, model=model, predict_fn=predict_fn, loss_fn=loss_fn)
    return -t_jk

def T_cv(X, Y, Z, model, predict_fn, loss_fn, derandomization=10):
    t_cv = 0
    n = len(X)
    for _ in range(derandomization):
        perm = np.random.permutation(n)
        train_test_percentage = np.random.uniform(0.5, 0.8)
        t_cv += T_optCRT(
            X[perm],
            Y[perm],
            Z[perm],
            i=int(n * train_test_percentage),
            model=model, 
            predict_fn=predict_fn,
            loss_fn=loss_fn
        )
    return -t_cv

def T_trainScore(X, Y, Z, model, predict_fn, loss_fn):
    """
    Parameters
    ----------
    X, Y, Z : arrays of length n
    model : callable returning a fitted model with predict()

    Returns
    -------
    T : float
    """
    X_train = np.column_stack([X, Z])
    m = clone(model)
    m.fit(X_train, Y)
    preds = predict_fn(m, X_train)
    return -loss_fn(Y, preds)


def T_HRT(X, Y, Z, model, predict_fn, loss_fn):
    X_eval = np.column_stack([X, Z])
    preds = predict_fn(model, X_eval)
    return -loss_fn(Y, preds)
