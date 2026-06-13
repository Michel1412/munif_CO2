"""Métricas de regressão."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    mape_epsilon: float = 1e-6,
) -> dict[str, float | None]:
    """
    Calcula MAE, MSE, RMSE, R² e MAPE.
    MAPE retorna None quando |y_true| é muito pequeno.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    mask = np.abs(y_true_arr) >= mape_epsilon
    if mask.any():
        mape = float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)
    else:
        mape = None

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def metrics_to_row(
    country: str,
    model_type: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Formata métricas como linha para CSV."""
    return {
        "country": country,
        "model": model_type,
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        "mape": metrics.get("mape"),
    }
