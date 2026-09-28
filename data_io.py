"""Excel parsing and local persistence for the ABC/XYZ application."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = PROJECT_ROOT / "samples"
PRICES_PATH = DATA_DIR / "precios_y_costos_actual.xlsx"
PRICES_META_PATH = DATA_DIR / "precios_y_costos_meta.json"
DEMAND_PATH = DATA_DIR / "historico_demanda.csv"
DEMAND_META_PATH = DATA_DIR / "historico_demanda_meta.json"
LAST_ANALYSIS_PATH = DATA_DIR / "ultimo_ejercicio.json"

VALID_SKUS = {
    ("MotoTrack Norte", "MOTO"),
    ("MotoTrack Norte", "CUATRIMOTO"),
    ("MotoTrack Norte", "TRACTOR"),
    ("MotoTrack Centro", "MOTO"),
    ("MotoTrack Centro", "CUATRIMOTO"),
    ("MotoTrack Centro", "TRACTOR"),
    ("MotoTrack Sur", "MOTO"),
    ("MotoTrack Sur", "CUATRIMOTO"),
}
KNOWN_PRODUCTS = {"MOTO", "CUATRIMOTO", "TRACTOR"}


class WorkbookFormatError(ValueError):
    """Raised when an uploaded workbook does not have the expected structure."""


def _normalise_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _find_sheet(workbook: pd.ExcelFile, expected: str) -> str:
    target = _normalise_text(expected)
    for sheet_name in workbook.sheet_names:
        if _normalise_text(sheet_name) == target:
            return sheet_name
    raise WorkbookFormatError(
        f"Falta la hoja '{expected}'. Hojas encontradas: "
        f"{', '.join(workbook.sheet_names) or 'ninguna'}."
    )


def _find_header_row(frame: pd.DataFrame, required: set[str], sheet_name: str) -> int:
    for row_index, row in frame.iterrows():
        values = {_normalise_text(value) for value in row.tolist()}
        if required.issubset(values):
            return int(row_index)
    expected = " | ".join(sorted(required))
    raise WorkbookFormatError(
        f"No se encontraron los encabezados '{expected}' en la hoja '{sheet_name}'."
    )


def _header_columns(frame: pd.DataFrame, row_index: int) -> dict[str, int]:
    return {
        _normalise_text(value): int(column_index)
        for column_index, value in enumerate(frame.iloc[row_index].tolist())
        if _normalise_text(value)
    }


def _number(value: object, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise WorkbookFormatError(f"Valor no numérico en {context}: {value!r}.") from error
    if not math.isfinite(number):
        raise WorkbookFormatError(f"Valor no válido en {context}: {value!r}.")
    return number


def parse_prices_workbook(source: str | Path | BinaryIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return product costs and sale prices, locating their headers by text."""
    try:
        workbook = pd.ExcelFile(source)
        costs_sheet = _find_sheet(workbook, "Costos")
        prices_sheet = _find_sheet(workbook, "Precio venta")
        costs_frame = pd.read_excel(workbook, sheet_name=costs_sheet, header=None)
        prices_frame = pd.read_excel(workbook, sheet_name=prices_sheet, header=None)
    except WorkbookFormatError:
        raise
    except Exception as error:
        raise WorkbookFormatError(f"No se pudo leer el archivo de precios y costos: {error}") from error

    cost_headers = {
        "producto",
        "costo md",
        "costo mod",
        "costo total unitario",
    }
    cost_header_row = _find_header_row(costs_frame, cost_headers, costs_sheet)
    cost_columns = _header_columns(costs_frame, cost_header_row)
    cost_rows: list[dict[str, object]] = []
    for _, row in costs_frame.iloc[cost_header_row + 1 :].iterrows():
        if all(pd.isna(value) or not str(value).strip() for value in row.tolist()):
            break
        product = str(row.iloc[cost_columns["producto"]]).strip().upper()
        if not product or product == "NAN":
            break
        cost_rows.append(
            {
                "producto": product,
                "costo_total_unitario": _number(
                    row.iloc[cost_columns["costo total unitario"]],
                    f"{costs_sheet}/{product}/Costo Total Unitario",
                ),
            }
        )

    price_headers = {"producto", "precio"}
    price_header_row = _find_header_row(prices_frame, price_headers, prices_sheet)
    price_columns = _header_columns(prices_frame, price_header_row)
    price_rows: list[dict[str, object]] = []
    for _, row in prices_frame.iloc[price_header_row + 1 :].iterrows():
        if all(pd.isna(value) or not str(value).strip() for value in row.tolist()):
            break
        product = str(row.iloc[price_columns["producto"]]).strip().upper()
        if not product or product == "NAN":
            break
        price_rows.append(
            {
                "producto": product,
                "precio": _number(
                    row.iloc[price_columns["precio"]],
                    f"{prices_sheet}/{product}/Precio",
                ),
            }
        )

    if not cost_rows:
        raise WorkbookFormatError(f"La hoja '{costs_sheet}' no contiene costos de productos.")
    if not price_rows:
        raise WorkbookFormatError(f"La hoja '{prices_sheet}' no contiene precios de productos.")
    return pd.DataFrame(cost_rows), pd.DataFrame(price_rows)


def _canonical_store(value: object) -> str | None:
    text = _normalise_text(value)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    for suffix, canonical in (
        ("mototrack norte", "MotoTrack Norte"),
        ("mototrack centro", "MotoTrack Centro"),
        ("mototrack sur", "MotoTrack Sur"),
    ):
        if text.endswith(suffix):
            return canonical
    return None


def parse_demand_workbook(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Convert dynamically positioned point-of-sale blocks into long format."""
    try:
        workbook = pd.ExcelFile(source)
        demand_sheet = _find_sheet(workbook, "Demand")
        frame = pd.read_excel(workbook, sheet_name=demand_sheet, header=None)
    except WorkbookFormatError:
        raise
    except Exception as error:
        raise WorkbookFormatError(f"No se pudo leer el archivo de demanda: {error}") from error

    records: list[dict[str, object]] = []
    turn_headers = [
        (int(row_index), int(column_index))
        for row_index, row in frame.iterrows()
        for column_index, value in enumerate(row.tolist())
        if _normalise_text(value) == "turn"
    ]
    if not turn_headers:
        raise WorkbookFormatError(
            f"No se encontró el encabezado 'Turn' en la hoja '{demand_sheet}'."
        )

    for header_row, turn_column in turn_headers:
        store_value = next(
            (
                frame.iat[row_index, turn_column]
                for row_index in range(header_row - 1, -1, -1)
                if not pd.isna(frame.iat[row_index, turn_column])
                and str(frame.iat[row_index, turn_column]).strip()
            ),
            None,
        )
        store = _canonical_store(store_value)
        if store is None:
            raise WorkbookFormatError(
                f"No se pudo identificar el punto de venta del bloque en la columna {turn_column + 1}."
            )

        product_columns: list[tuple[int, str]] = []
        for column_index in range(turn_column + 1, frame.shape[1]):
            product_value = frame.iat[header_row, column_index]
            if pd.isna(product_value) or not str(product_value).strip():
                break
            product = str(product_value).strip().upper()
            if product not in KNOWN_PRODUCTS:
                raise WorkbookFormatError(
                    f"El producto '{product}' de la hoja '{demand_sheet}' no es reconocido; "
                    "verifica las hojas Costos y Precio venta."
                )
            product_columns.append((column_index, product))
        if not product_columns:
            raise WorkbookFormatError(f"El bloque de '{store}' no contiene columnas de productos.")

        for row_index in range(header_row + 1, frame.shape[0]):
            turn_value = frame.iat[row_index, turn_column]
            if pd.isna(turn_value) or not str(turn_value).strip():
                continue
            turn_number = _number(turn_value, f"{store}/Turn")
            if not turn_number.is_integer() or turn_number < 0:
                raise WorkbookFormatError(
                    f"El turno debe ser un entero no negativo en '{store}': {turn_value!r}."
                )
            for column_index, product in product_columns:
                if (store, product) not in VALID_SKUS:
                    continue
                value = frame.iat[row_index, column_index]
                if pd.isna(value) or not str(value).strip():
                    continue
                demand = _number(value, f"{store}/{product}/turno {int(turn_number)}")
                if demand < 0:
                    raise WorkbookFormatError("La demanda no puede ser negativa.")
                records.append(
                    {
                        "punto_venta": store,
                        "producto": product,
                        "turn": int(turn_number),
                        "demanda": demand,
                    }
                )

    demand = pd.DataFrame(records, columns=["punto_venta", "producto", "turn", "demanda"])
    if demand.empty:
        raise WorkbookFormatError("La hoja 'Demand' no contiene turnos con demanda válida.")
    return demand.drop_duplicates(
        subset=["punto_venta", "producto", "turn"], keep="last"
    ).sort_values(["punto_venta", "producto", "turn"], ignore_index=True)


def merge_demand_history(history: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Upsert demand rows using (point of sale, product, turn) as the key."""
    key = ["punto_venta", "producto", "turn"]
    combined = pd.concat([history, incoming], ignore_index=True)
    return combined.drop_duplicates(subset=key, keep="last").sort_values(
        key, ignore_index=True
    )


def load_history() -> pd.DataFrame:
    if not DEMAND_PATH.exists():
        return pd.DataFrame(columns=["punto_venta", "producto", "turn", "demanda"])
    history = pd.read_csv(DEMAND_PATH)
    return history[["punto_venta", "producto", "turn", "demanda"]]


def save_history(history: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(DEMAND_PATH, index=False)


def load_metadata(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def save_metadata(path: Path, original_name: str, loaded_at: datetime | None = None) -> dict[str, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "original_name": original_name,
        "loaded_at": (loaded_at or datetime.now().astimezone()).isoformat(timespec="seconds"),
    }
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return metadata


def save_last_analysis(
    classification: pd.DataFrame,
    parameters: dict[str, object],
    warnings: list[str] | None = None,
) -> dict[str, object]:
    """Persist the last completed classification and its reproducibility metadata."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, object] = {
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parameters": parameters,
        "warnings": warnings or [],
        "classification": json.loads(
            classification.to_json(orient="records", force_ascii=False)
        ),
    }
    with LAST_ANALYSIS_PATH.open("w", encoding="utf-8") as snapshot_file:
        json.dump(snapshot, snapshot_file, ensure_ascii=False, indent=2)
    return snapshot


def load_last_analysis() -> dict[str, object] | None:
    """Load the persisted last completed classification, if one exists."""
    if not LAST_ANALYSIS_PATH.exists():
        return None
    with LAST_ANALYSIS_PATH.open("r", encoding="utf-8") as snapshot_file:
        snapshot = json.load(snapshot_file)
    if not isinstance(snapshot, dict) or not snapshot.get("classification"):
        return None
    snapshot["classification"] = pd.DataFrame(snapshot["classification"])
    return snapshot


def initialise_sample_data() -> tuple[bool, bool]:
    """Persist bundled examples on first run; return which fixtures were imported."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices_imported = False
    demand_imported = False
    sample_prices = SAMPLES_DIR / "Precios_y_costos.xlsx"
    sample_demand = SAMPLES_DIR / "Demanda.xlsx"
    if not PRICES_PATH.exists() and sample_prices.exists():
        PRICES_PATH.write_bytes(sample_prices.read_bytes())
        save_metadata(PRICES_META_PATH, sample_prices.name)
        prices_imported = True
    if not DEMAND_PATH.exists() and sample_demand.exists():
        history = parse_demand_workbook(sample_demand)
        save_history(history)
        save_metadata(DEMAND_META_PATH, sample_demand.name)
        demand_imported = True
    return prices_imported, demand_imported