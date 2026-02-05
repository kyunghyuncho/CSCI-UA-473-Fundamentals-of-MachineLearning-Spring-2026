"""Loss functions for optimization labs."""

import numpy as np


def doublewell_loss(w1, w2):
    """
    Doublewell loss function with multiple local minima.

    Parameters
    ----------
    w1 : float
        First parameter
    w2 : float
        Second parameter

    Returns
    -------
    float
        Loss value: sin(2*w1) + sin(2*w2) + 0.3*(w1^2 + w2^2)
    """
    return np.sin(2 * w1) + np.sin(2 * w2) + 0.3 * (w1**2 + w2**2)


def doublewell_gradient(w1, w2):
    """
    Compute gradient of the doublewell loss function.

    Parameters
    ----------
    w1 : float
        First parameter
    w2 : float
        Second parameter

    Returns
    -------
    np.ndarray
        Gradient vector [dL/dw1, dL/dw2]
    """
    dw1 = 2 * np.cos(2 * w1) + 0.6 * w1
    dw2 = 2 * np.cos(2 * w2) + 0.6 * w2
    return np.array([dw1, dw2])
