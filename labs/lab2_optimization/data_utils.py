"""
Data generation and utilities for Lab 2: Optimization
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from labs.lab2_optimization.gradient_descent import line_fn


def generate_dataset(pts, rng):
    """
    Generate random dataset with true linear relationship plus noise.

    Args:
        pts: number of points to generate
        rng: numpy random generator

    Returns:
         Tuple of (df, fit_coefs, pt_idx) where:
            - df: DataFrame with columns 'x', 'y', 'color'
            - fit_coefs: (intercept, slope) from linear regression fit
            - pt_idx: permuted indices for batching
    """
    # Generate random coefficients for true line with negative slope
    intercept = 30 * rng.random() - 10  # Random intercept between -10 and 20
    slope = -15 * rng.random() - 5  # Random negative slope between -20 and -5
    coefs = np.array([slope, intercept])

    # Create dataframe
    df = pd.DataFrame(index=range(pts), columns=["x", "y", "color"])
    df["x"] = rng.normal(size=pts, loc=0, scale=3)

    # Generate y values with noise

    true_f = line_fn(coefs)
    df["y"] = true_f(df["x"]) + rng.normal(size=pts, loc=0, scale=7)
    df["color"] = 0

    # Fit linear regression to get optimal coefficients
    reg = LinearRegression()
    reg.fit(df[["x"]], df[["y"]])
    fit_1 = reg.coef_[0][0]
    fit_0 = reg.intercept_[0]

    fit_coefs = (fit_1, fit_0)

    return df, fit_coefs


def create_batched_indices(pts, batch_size, max_updates, rng):
    """
    Create permuted indices for batching across multiple epochs.

    Args:
        pts: total number of points
        batch_size: batch size
        max_updates: maximum number of updates
        rng: numpy random generator

    Returns:
        Array of permuted indices for batching
    """
    num_epochs = (max_updates * batch_size) // pts + 1
    pt_idx = np.concatenate([rng.permutation(range(pts)) for i in range(num_epochs)])
    return pt_idx


def format_line_equation(m, b):
    """
    Format line equation as LaTeX string.

    Args:
        m: slope
        b: intercept

    Returns:
        LaTeX formatted string for y = mx + b
    """
    return f"""$y = {round(m, 2)}x {"+" if b >= 0 else ""} {round(b, 2)}$"""


def generate_polynomial_dataset(
    pts, degree, rng, x_range=(-3, 3), noise_std=5, outlier_fraction=0.0
):
    """
    Generate random dataset with true polynomial relationship plus noise.

    Args:
        pts: number of points to generate
        degree: degree of the polynomial
        rng: numpy random generator
        x_range: tuple (min, max) for x values
        noise_std: standard deviation of noise
        outlier_fraction: fraction of training points to convert to outliers (0.0 to 1.0)

    Returns:
        Tuple of (df_train, df_val, true_coefs) where:
            - df_train: DataFrame with columns 'x', 'y', 'color' (70% of data)
            - df_val: DataFrame with columns 'x', 'y', 'color' (30% of data)
            - true_coefs: array of true polynomial coefficients
    """
    # Generate random coefficients for true polynomial
    true_coefs = rng.uniform(-10, 10, size=degree + 1)

    # Generate x values uniformly across range
    all_x = rng.uniform(x_range[0], x_range[1], size=pts)

    # Compute true y values
    from labs.lab2_optimization.gradient_descent import poly_fn

    true_f = poly_fn(true_coefs)
    all_y = true_f(all_x)

    # Add noise
    all_y += rng.normal(0, noise_std, size=pts)

    # Normalize features and targets (z-score)
    x_mean = np.mean(all_x)
    x_std = np.std(all_x) if np.std(all_x) > 0 else 1.0
    y_mean = np.mean(all_y)
    y_std = np.std(all_y) if np.std(all_y) > 0 else 1.0
    all_x = (all_x - x_mean) / x_std
    all_y = (all_y - y_mean) / y_std

    # Split into train/val (70/30)
    n_train = int(0.7 * pts)
    indices = rng.permutation(pts)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create train dataframe
    df_train = pd.DataFrame(
        {"x": all_x[train_indices], "y": all_y[train_indices], "color": 0}
    )
    df_train = df_train.reset_index(drop=True)

    # Add outliers to training set
    if outlier_fraction > 0:
        n_outliers = int(outlier_fraction * len(df_train))
        if n_outliers > 0:
            outlier_indices = rng.choice(len(df_train), size=n_outliers, replace=False)
            # Make outliers by adding large random offsets in y
            df_train.loc[outlier_indices, "y"] += rng.uniform(
                -2.5, 2.5, size=n_outliers
            )

    # Create validation dataframe
    df_val = pd.DataFrame(
        {"x": all_x[val_indices], "y": all_y[val_indices], "color": 0}
    )
    df_val = df_val.reset_index(drop=True)

    return df_train, df_val, true_coefs


def format_poly_equation(coefs):
    """
    Format polynomial equation as LaTeX string.

    Args:
        coefs: array of polynomial coefficients [c0, c1, c2, ...]

    Returns:
        LaTeX formatted string for y = c0 + c1*x + c2*x^2 + ...
    """
    terms = []
    for i, c in enumerate(coefs):
        if abs(c) < 0.01:  # Skip near-zero terms
            continue

        if i == 0:
            terms.append(f"{c:.2f}")
            continue

        sign = "+" if c >= 0 else "-"
        c_abs = f"{abs(c):.2f}"
        if i == 1:
            terms.append(f"{sign} {c_abs}x")
        else:
            terms.append(f"{sign} {c_abs}x^{i}")

    if not terms:
        return "$y = 0$"

    return f"$y = {' '.join(terms)}$"


def generate_overfitting_dataset(n_train=20, n_val=100, function_type="sine", rng=None):
    """
    Generate dataset optimized for demonstrating overfitting.

    Args:
        n_train: number of training points (small)
        n_val: number of validation points (large, dense)
        function_type: "sine" or "polynomial"
        rng: numpy random generator

    Returns:
        Tuple of (df_train, df_val, true_coefs) where:
            - df_train: DataFrame with 'x' and 'y' (sparse, high noise)
            - df_val: DataFrame with 'x' and 'y' (dense, low noise)
            - true_coefs: not used for sine but included for compatibility
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate x values
    x_train = np.sort(rng.uniform(0, 10, n_train))
    x_val = np.linspace(0, 10, n_val)

    # True function
    if function_type == "sine":
        true_train = np.sin(x_train)
        true_val = np.sin(x_val)
    else:  # polynomial
        true_train = 0.1 * x_train**3 - x_train**2 + 2 * x_train
        true_val = 0.1 * x_val**3 - x_val**2 + 2 * x_val

    # Add noise (higher for train, lower for val)
    y_train = true_train + 0.4 * rng.normal(0, 1, n_train)
    y_val = true_val + 0.1 * rng.normal(0, 1, n_val)

    # Normalize
    x_mean = np.mean(x_train)
    x_std = np.std(x_train) if np.std(x_train) > 0 else 1.0
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) if np.std(y_train) > 0 else 1.0

    x_train_norm = (x_train - x_mean) / x_std
    y_train_norm = (y_train - y_mean) / y_std
    x_val_norm = (x_val - x_mean) / x_std
    y_val_norm = (y_val - y_mean) / y_std

    # Create dataframes
    df_train = pd.DataFrame({"x": x_train_norm, "y": y_train_norm, "color": 0})

    df_val = pd.DataFrame({"x": x_val_norm, "y": y_val_norm, "color": 0})

    return df_train, df_val, None
