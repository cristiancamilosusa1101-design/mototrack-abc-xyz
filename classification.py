"""Active demand windows and ABC/XYZ classification rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


DETAIL_COLUMNS = [
    "Punto de venta",
    "Producto",
    "Demanda 24 sem",
    "Precio venta",
    "Costo unitario",
    "Margen unitario",
    "Utilidad",
    "% utilidad",
    "% acumulado",
    "ABC",
    "Modelo elegido",
    "Score%",
    "MAE%",
    "RMSE",
    "XYZ",
]


def active_demand_window(history: pd.DataFrame, weeks: int = 24) -> pd.DataFrame:
    """Keep each SKU's latest available turns, up to the requested window size."""
    if weeks < 1:
        raise ValueError("La ventana debe contener al menos una semana.")
    window_parts = []
    for _, sku_rows in history.groupby(["punto_venta", "producto"], sort=False):
        window_parts.append(sku_rows.sort_values("turn").tail(weeks))
    if not window_parts:
        return pd.DataFrame(columns=["punto_venta", "producto", "turn", "demanda"])
    return pd.concat(window_parts, ignore_index=True).sort_values(
        ["punto_venta", "producto", "turn"], ignore_index=True
    )


def build_classification(
    active_window: pd.DataFrame,
    costs: pd.DataFrame,
    prices: pd.DataFrame,
    forecasts: pd.DataFrame,
    abc_a_pct: float = 80,
    abc_b_pct: float = 15,
    xyz_x_threshold: float = 80,
    xyz_y_threshold: float = 60,
    window_weeks: int = 24,
) -> pd.DataFrame:
    """Combine financial contribution and forecast accuracy into ABC/XYZ classes."""
    if abc_a_pct <= 0 or abc_b_pct <= 0 or abc_a_pct + abc_b_pct >= 100:
        raise ValueError("Los cortes ABC deben ser positivos y dejar un porcentaje para C.")
    if not 0 <= xyz_y_threshold < xyz_x_threshold <= 100:
        raise ValueError("Los umbrales XYZ deben cumplir 0 ≤ Y < X ≤ 100.")
    if window_weeks < 1:
        raise ValueError("La ventana de análisis debe contener al menos una semana.")

    missing_costs = sorted(set(active_window["producto"]) - set(costs["producto"]))
    missing_prices = sorted(set(active_window["producto"]) - set(prices["producto"]))
    if missing_costs or missing_prices:
        details = []
        if missing_costs:
            details.append("sin costo: " + ", ".join(missing_costs))
        if missing_prices:
            details.append("sin precio: " + ", ".join(missing_prices))
        raise ValueError("Hay productos de demanda que no existen en " + " ni ".join(details) + ".")

    totals = active_window.groupby(["punto_venta", "producto"], as_index=False).agg(
        demanda_periodo=("demanda", "sum")
    )
    totals = totals.merge(costs[["producto", "costo_total_unitario"]], on="producto", how="left")
    totals = totals.merge(prices[["producto", "precio"]], on="producto", how="left")
    totals = totals.merge(
        forecasts[
            [
                "punto_venta",
                "producto",
                "modelo_elegido",
                "score_pct",
                "mae_pct",
                "rmse",
                "mae",
                "r2",
            ]
        ],
        on=["punto_venta", "producto"],
        how="left",
    )
    totals["margen_unitario"] = totals["precio"] - totals["costo_total_unitario"]
    totals["utilidad"] = totals["margen_unitario"] * totals["demanda_periodo"]
    totals = totals.sort_values(
        ["utilidad", "punto_venta", "producto"], ascending=[False, True, True]
    ).reset_index(drop=True)
    utility_total = float(totals["utilidad"].sum())
    totals["porcentaje_utilidad"] = (
        totals["utilidad"] / utility_total * 100 if utility_total else 0.0
    )
    totals["porcentaje_acumulado"] = totals["porcentaje_utilidad"].cumsum()

    abc_cutoff = abc_a_pct
    b_cutoff = abc_a_pct + abc_b_pct
    totals["abc"] = [
        "A"
        if index == 0 or accumulated <= abc_cutoff
        else "B"
        if accumulated <= b_cutoff
        else "C"
        for index, accumulated in enumerate(totals["porcentaje_acumulado"])
    ]
    totals["xyz"] = np.select(
        [
            totals["score_pct"] >= xyz_x_threshold,
            (totals["score_pct"] >= xyz_y_threshold)
            & (totals["score_pct"] < xyz_x_threshold),
            totals["score_pct"] < xyz_y_threshold,
        ],
        ["X", "Y", "Z"],
        default="N/D",
    )
    totals["sku"] = totals["punto_venta"].str.replace("MotoTrack ", "", regex=False) + "-" + totals["producto"]
    return totals.rename(
        columns={
            "punto_venta": "Punto de venta",
            "producto": "Producto",
            "demanda_periodo": f"Demanda {window_weeks} sem",
            "precio": "Precio venta",
            "costo_total_unitario": "Costo unitario",
            "margen_unitario": "Margen unitario",
            "utilidad": "Utilidad",
            "porcentaje_utilidad": "% utilidad",
            "porcentaje_acumulado": "% acumulado",
            "abc": "ABC",
            "modelo_elegido": "Modelo elegido",
            "score_pct": "Score%",
            "mae_pct": "MAE%",
            "rmse": "RMSE",
            "xyz": "XYZ",
        }
    )[
        [
            *(f"Demanda {window_weeks} sem" if column == "Demanda 24 sem" else column for column in DETAIL_COLUMNS),
            "sku",
            "mae",
            "r2",
        ]
    ]


def build_abc_xyz_matrix(classification: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame("", index=["A", "B", "C"], columns=["X", "Y", "Z"])
    for _, row in classification.iterrows():
        abc_class = row["ABC"]
        xyz_class = row["XYZ"]
        if abc_class in matrix.index and xyz_class in matrix.columns:
            existing = matrix.loc[abc_class, xyz_class]
            matrix.loc[abc_class, xyz_class] = (
                f"{existing}\n{row['sku']}" if existing else row["sku"]
            )
    matrix.index.name = "ABC / XYZ"
    return matrix.reset_index()