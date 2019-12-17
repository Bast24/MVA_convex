import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


def generate_data(n, d, noise=1.):
    """Generate random matrices X and vectors y to test the log-barrier method

    Args:
        - n : number of samples
        - d : number of features
        - noise : add gaussian noise to Y
    Returns:
        - X : (n, d) data matrix
        - Y : (n, ) objective vector
        - w : (d, ) true weights
    """
    X, Y, w = make_regression(
        n_samples=n,
        n_informative=d,
        n_features=d,
        noise=noise,
        coef=True
    )

    return X, Y, w


def build_lasso_params(X, Y, lmbda):
    """Return the QP parameters for lasso regression.

    Args:
        - X : data matrix
        - Y : Regression objective
        - lmbda : lasso parameter
    Returns:
        - Q : (n, d) data matrix
        - p : (n, ) objective vector
        - A : (d, ) true weights
        - b
    """
    n, d = X.shape
    Q = np.eye(n) / 2.
    p = Y
    A = np.vstack([X.T, -X.T])
    b = lmbda * np.ones(2 * d)

    return Q, p, A, b
