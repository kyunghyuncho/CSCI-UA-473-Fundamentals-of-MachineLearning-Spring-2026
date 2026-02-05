"""
Core gradient descent logic for Lab 2: Optimization
"""
import numpy as np


def line_fn(coefs):
    """
    Convert line coefficients to a callable function.

    Args:
        coefs: tuple (t0, t1) where line is y = t0*x + t1

    Returns:
        Function that evaluates the line at a given x value
    """
    t0, t1 = coefs
    return lambda x: t0 * x + t1


def update_theta(theta, batch, alpha, pts):
    """
    Perform one gradient descent update step.

    Args:
        theta: tuple (t0, t1) where t0 is slope and t1 is intercept
        batch: pandas DataFrame with columns 'x' and 'y' containing batch data
        alpha: learning rate
        pts: number of total points (for scaling)

    Returns:
        Updated theta tuple (new_t0, new_t1)
    """
    t0, t1 = theta
    batch_size = len(batch)
    val = 2 * (t0 * batch["x"] + t1 - batch["y"]) / batch_size
    return (t0 - (alpha * val * batch["x"]).sum(), t1 - (alpha * val).sum())


def make_theta_array(df, pt_idx, alpha, batch_size, init_theta, max_updates, pts):
    """
    Generate array of all theta values through gradient descent iterations.

    Args:
        df: pandas DataFrame with columns 'x' and 'y'
        pt_idx: array of point indices for batching
        alpha: learning rate
        batch_size: number of points per batch
        init_theta: initial theta values (t0, t1)
        max_updates: maximum number of update steps
        pts: total number of points

    Returns:
        List of theta tuples representing the gradient descent path
    """
    theta_arr = [init_theta]
    for i in range(max_updates):
        batch_data = df.loc[pt_idx[i * batch_size : (i + 1) * batch_size], ["x", "y"]]
        next_theta = update_theta(theta_arr[i], batch_data, alpha, pts)
        theta_arr.append(next_theta)
    return theta_arr


# ===== Polynomial Fitting Functions =====


def poly_fn(coefs):
    """
    Convert polynomial coefficients to a callable function.

    Args:
        coefs: array-like [c0, c1, c2, ...] where y = c0 + c1*x + c2*x^2 + ...

    Returns:
        Function that evaluates the polynomial at a given x value
    """

    return lambda x: np.polyval(coefs[::-1], x)


def update_poly_theta(theta, batch, alpha, degree, use_clipping=True):
    """
    Perform one gradient descent update step for polynomial regression.

    Args:
        theta: array-like of polynomial coefficients [c0, c1, c2, ...]
        batch: pandas DataFrame with columns 'x' and 'y' containing batch data
        alpha: learning rate
        degree: polynomial degree
        use_clipping: whether to apply gradient clipping for stability

    Returns:
        Updated theta array
    """

    batch_size = len(batch)
    x = batch["x"].values
    y = batch["y"].values

    # # Compute predictions
    # pred = np.zeros(batch_size)
    # for i, c in enumerate(theta):
    #     pred += c * (x**i)

    # # Compute residuals
    # residuals = pred - y

    # # Compute gradients for each coefficient
    # new_theta = []
    # alpha_eff = alpha / max(1, degree) if use_clipping else alpha
    # clip_value = 10.0
    # for i in range(degree + 1):
    #     gradient = (2 / batch_size) * np.sum(residuals * (x**i))
    #     if use_clipping:
    #         gradient = np.clip(gradient, -clip_value, clip_value)
    #     new_theta.append(theta[i] - alpha_eff * gradient)

    # return np.array(new_theta)
    # Compute predictions using vectorized operations
    powers = x[:, np.newaxis] ** np.arange(degree + 1)
    pred = np.sum(powers * theta, axis=1)

    # Compute residuals
    residuals = pred - y

    # Compute gradients for each coefficient in a vectorized way
    gradients = (2 / batch_size) * np.sum(residuals[:, np.newaxis] * powers, axis=0)

    if use_clipping:
        clip_value = 10.0
        gradients = np.clip(gradients, -clip_value, clip_value)

    alpha_eff = alpha / max(1, degree) if use_clipping else alpha
    theta - alpha_eff * gradients


def make_poly_theta_array(
    df, pt_idx, alpha, batch_size, init_theta, max_updates, degree, use_clipping=True
):
    """
    Generate array of all polynomial theta values through gradient descent iterations.

    Args:
        df: pandas DataFrame with columns 'x' and 'y'
        pt_idx: array of point indices for batching
        alpha: learning rate
        batch_size: number of points per batch
        init_theta: initial theta values (polynomial coefficients)
        max_updates: maximum number of update steps
        degree: polynomial degree
        use_clipping: whether to apply gradient clipping for stability

    Returns:
        List of theta arrays representing the gradient descent path
    """
    import numpy as np

    theta_arr = [np.array(init_theta)]
    for i in range(max_updates):
        batch_data = df.loc[pt_idx[i * batch_size : (i + 1) * batch_size], ["x", "y"]]
        next_theta = update_poly_theta(
            theta_arr[i], batch_data, alpha, degree, use_clipping
        )
        theta_arr.append(next_theta)
    return theta_arr


# ===== 2D Loss Landscape for Linear Regression =====


def linear_regression_loss_2d(w, b, X, y):
    """
    Compute MSE loss for linear regression with parameters w (slope) and b (intercept).

    Args:
        w, b: parameters (can be scalars or arrays)
        X: input features (1D array)
        y: target values (1D array)

    Returns:
        MSE loss value(s)
    """
    import numpy as np

    predictions = w * X + b
    mse = np.mean((predictions - y) ** 2)
    return mse


def linear_regression_gradient_2d(w, b, X, y):
    """
    Compute analytical gradient of MSE loss w.r.t. w and b.

    Args:
        w, b: current parameters
        X: input features (1D array)
        y: target values (1D array)

    Returns:
        Tuple (dL/dw, dL/db)
    """
    import numpy as np

    n = len(X)
    predictions = w * X + b
    errors = predictions - y

    dw = (2.0 / n) * np.sum(errors * X)
    db = (2.0 / n) * np.sum(errors)

    return dw, db
