"""Backtesting models used to assign a forecast reliability score to each SKU."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import MSTL


MODEL_NAMES = (
    "Promedio móvil simple",
    "Suavización exponencial simple",
    "Regresión lineal",
    "Holt-Winters",
    "MSTL",
)


def _forecast_moving_average(training: np.ndarray, horizon: int, window: int = 4) -> np.ndarray:
    history = list(training.astype(float))
    predictions: list[float] = []
    for _ in range(horizon):
        prediction = float(np.mean(history[-window:]))
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions)


def _forecast_ses(training: np.ndarray, horizon: int) -> np.ndarray:
    model = ExponentialSmoothing(
        training,
        trend=None,
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.asarray(model.forecast(horizon), dtype=float)


def _forecast_linear(training: np.ndarray, horizon: int) -> np.ndarray:
    training_x = np.arange(len(training), dtype=float).reshape(-1, 1)
    future_x = np.arange(len(training), len(training) + horizon, dtype=float).reshape(-1, 1)
    return LinearRegression().fit(training_x, training).predict(future_x)


def _forecast_holt_winters(
    training: np.ndarray, horizon: int, seasonal_period: int
) -> np.ndarray:
    if len(training) < seasonal_period * 2:
        raise ValueError(f"requiere al menos {seasonal_period * 2} observaciones")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            training,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
    return np.asarray(model.forecast(horizon), dtype=float)


def _forecast_mstl(training: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    if len(training) < seasonal_period * 2:
        raise ValueError(f"requiere al menos {seasonal_period * 2} observaciones")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decomposition = MSTL(
            pd.Series(training, dtype=float), periods=(seasonal_period,)
        ).fit()
    trend = np.asarray(decomposition.trend, dtype=float)
    seasonal = decomposition.seasonal
    if isinstance(seasonal, pd.DataFrame):
        seasonal_values = seasonal.iloc[:, 0].to_numpy(dtype=float)
    else:
        seasonal_values = np.asarray(seasonal, dtype=float)

    trend_window = min(len(trend), max(seasonal_period * 2, 8))
    trend_x = np.arange(len(trend) - trend_window, len(trend), dtype=float)
    trend_slope, trend_intercept = np.polyfit(trend_x, trend[-trend_window:], 1)
    future_x = np.arange(len(training), len(training) + horizon, dtype=float)
    future_trend = trend_intercept + trend_slope * future_x
    final_cycle = seasonal_values[-seasonal_period:]
    future_seasonality = np.resize(final_cycle, horizon)
    return future_trend + future_seasonality


def backtest_sku(
    values: pd.Series | list[float] | np.ndarray,
    test_weeks: int = 13,
    seasonal_period: int = 13,
) -> tuple[dict[str, object], list[str]]:
    """Choose the successful model with the lowest RMSE on the holdout weeks."""
    series = np.asarray(values, dtype=float)
    if not np.isfinite(series).all() or (series < 0).any():
        raise ValueError("La serie debe contener demanda numérica, finita y no negativa.")
    if test_weeks < 1:
        raise ValueError("Las semanas de prueba deben ser al menos 1.")
    if seasonal_period < 2 or seasonal_period > len(series) / 2:
        raise ValueError("El periodo estacional debe ser ≥ 2 y no superar la mitad de la ventana.")
    if len(series) < 3:
        raise ValueError("Se requieren al menos 3 semanas para ejecutar backtesting.")

    horizon = min(test_weeks, len(series) - 2)
    training, actual = series[:-horizon], series[-horizon:]
    models = {
        MODEL_NAMES[0]: lambda: _forecast_moving_average(training, horizon),
        MODEL_NAMES[1]: lambda: _forecast_ses(training, horizon),
        MODEL_NAMES[2]: lambda: _forecast_linear(training, horizon),
        MODEL_NAMES[3]: lambda: _forecast_holt_winters(training, horizon, seasonal_period),
        MODEL_NAMES[4]: lambda: _forecast_mstl(training, horizon, seasonal_period),
    }
    candidates: list[dict[str, object]] = []
    model_warnings: list[str] = []
    for model_name, forecast_function in models.items():
        try:
            predicted = np.maximum(forecast_function(), 0)
            if len(predicted) != len(actual) or not np.isfinite(predicted).all():
                raise ValueError("el modelo produjo un pronóstico no válido")
            mae = float(mean_absolute_error(actual, predicted))
            rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
            candidates.append(
                {
                    "modelo_elegido": model_name,
                    "mae": mae,
                    "rmse": rmse,
                    "predicted": predicted,
                }
            )
        except Exception as error:
            model_warnings.append(f"{model_name}: {error}")

    if not candidates:
        raise ValueError("Ninguno de los modelos pudo ajustarse: " + "; ".join(model_warnings))
    chosen = min(candidates, key=lambda candidate: float(candidate["rmse"]))
    actual_mean = float(np.mean(actual))
    mae = float(chosen["mae"])
    score = max(0.0, 1 - mae / actual_mean) * 100 if actual_mean > 0 else 0.0
    mae_pct = mae / actual_mean * 100 if actual_mean > 0 else float("nan")
    r2 = float(r2_score(actual, chosen["predicted"])) if len(actual) > 1 else float("nan")
    return (
        {
            "modelo_elegido": chosen["modelo_elegido"],
            "score_pct": score,
            "mae_pct": mae_pct,
            "rmse": float(chosen["rmse"]),
            "mae": mae,
            "r2": r2,
        },
        model_warnings,
    )


def run_backtesting(
    active_window: pd.DataFrame,
    test_weeks: int = 13,
    seasonal_period: int = 13,
) -> tuple[pd.DataFrame, list[str]]:
    results: list[dict[str, object]] = []
    all_warnings: list[str] = []
    for (store, product), sku_rows in active_window.groupby(
        ["punto_venta", "producto"], sort=False
    ):
        sku_label = f"{store.replace('MotoTrack ', '')}-{product}"
        try:
            result, model_warnings = backtest_sku(
                sku_rows.sort_values("turn")["demanda"],
                test_weeks=test_weeks,
                seasonal_period=seasonal_period,
            )
            results.append({"punto_venta": store, "producto": product, **result})
            all_warnings.extend(f"{sku_label}: {warning}" for warning in model_warnings)
        except ValueError as error:
            all_warnings.append(f"{sku_label}: {error}")
            results.append(
                {
                    "punto_venta": store,
                    "producto": product,
                    "modelo_elegido": "No disponible",
                    "score_pct": np.nan,
                    "mae_pct": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                }
            )
    return pd.DataFrame(results), all_warnings