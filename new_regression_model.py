"""Robust timeseries regression module"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Threshold for near-zero variance in factors
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-10


def exponential_weights(
    n: int,
    half_life: float | None = None,
    lambda_: float | None = None,
) -> np.ndarray:

    if half_life is None and lambda_ is None:
        raise ValueError("Either half_life or lambda_ must be provided.")
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life must be positive.")
    if lambda_ is not None and not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1.")

    if lambda_ is None:
        lambda_ = np.exp(np.log(0.5) / half_life)

    weights: np.ndarray = lambda_ ** np.arange(n - 1, -1, -1)
    return weights / np.sum(weights)  # normalize


def robust_matrix_regression(
    df_asset_rets: pd.DataFrame,  # Shape: (n_timestamps, 1)
    df_fact_rets: pd.DataFrame,  # Shape: (n_timestamps, n_factors)
    lambda_: float,
    half_life: float = 90,
    min_periods: int = 10,
) -> Tuple[pd.Series, pd.Series]:
    """Perform robust weighted regression using matrix algebra."""
    
    # === Input Validation ===
    if df_asset_rets.empty or df_fact_rets.empty:
        raise ValueError("Input DataFrames cannot be empty")

    # Align and clean data
    df_combined = pd.concat([df_asset_rets, df_fact_rets], axis=1).dropna()
    y = df_combined.iloc[:, 0].values  # Asset returns (numpy array)
    X = df_combined.iloc[:, 1:].values  # Factor returns (numpy array)
    
    if len(y) < min_periods:
        raise ValueError(f"Need at least {min_periods} non-NaN observations")

    # === Weighting ===
    weights = exponential_weights(len(y), half_life=half_life, lambda_=lambda_)
    sqrt_weights = np.sqrt(weights)
    wX = X * sqrt_weights[:, None]  # Weighted X
    wy = y * sqrt_weights           # Weighted y

    # === Regression ===
    wX = sm.add_constant(wX)  # Add intercept
    betas = np.linalg.pinv(wX.T @ wX) @ wX.T @ wy
    residuals = y - wX @ betas  

    return betas, residuals