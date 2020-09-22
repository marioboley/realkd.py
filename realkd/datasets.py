"""
Access to example datasets and distributions.
"""

import numpy as np
import pandas as pd


def noisy_parity(n, d=3, variance=0.25, as_df=True, random_seed=None):
    r"""Generates observations of mixture model of Gaussian clusters centered
    at nodes of hypercube :math:`\{-1, 1\}^d` labelled according to parity of cube node.

    That is,

    .. math::
        :nowrap:

        \begin{align*}
        C &\sim \mathrm{Unif}(\{0, 1\}^d)\\
        X | C &\sim \mathrm{Norm}(C, \sigma^2 I_d)\\
        Y | C &= \prod_{i=1}^d C_i
        \end{align*}

    For example:

    >>> x, y = noisy_parity(10, random_seed=0)
    >>> x
             x1        x2        x3
    0  0.633866  0.727871  0.841850
    1 -0.794185 -0.478743 -1.064267
    2 -0.316768 -1.332597 -0.824245
    3  1.451735  1.047006  0.628250
    4  0.539137  0.771137  1.110098
    5  0.495191  0.895412  0.920387
    6  1.270423  1.107330 -0.822314
    7  0.673086  0.935193 -0.608012
    8 -0.253284  0.370467  1.756962
    9 -0.327062  1.390656  1.132228

    >>> y
    0    1
    1   -1
    2   -1
    3    1
    4    1
    5    1
    6   -1
    7   -1
    8   -1
    9   -1
    dtype: int64

    :param n: number of observations
    :param d: dimension of data
    :param variance: variance of the clusters
    :param as_df: whether to wrap return value in pandas dataframe/series
    :param random_seed: seed passed to np.random.default_rng
    :return: dataframe/matrix x and corresponding label series/arrays
    """
    rng = np.random.default_rng(seed=random_seed)
    c = rng.integers(0, 1, (n, d), endpoint=True)
    c[c == 0] = -1
    y = np.multiply.reduce(c.T)
    x = np.apply_along_axis(rng.multivariate_normal, 1, c, variance*np.eye(d))
    if as_df:
        return pd.DataFrame(x, columns=[f'x{i+1}' for i in range(d)]), pd.Series(y)
    else:
        return x, y

