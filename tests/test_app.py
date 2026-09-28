from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from classification import active_demand_window, build_classification
import data_io
from data_io import VALID_SKUS, merge_demand_history, parse_demand_workbook, parse_prices_workbook
from forecasting import backtest_sku


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


class WorkbookAndClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demand = parse_demand_workbook(SAMPLES / "Demanda.xlsx")
        cls.costs, cls.prices = parse_prices_workbook(SAMPLES / "Precios_y_costos.xlsx")
        cls.window = active_demand_window(cls.demand, weeks=24)

    def test_demand_parser_finds_eight_skus_and_omits_south_tractor(self):
        found_skus = set(
            map(
                tuple,
                self.demand[["punto_venta", "producto"]].drop_duplicates().to_numpy(),
            )
        )
        self.assertEqual(found_skus, VALID_SKUS)
        self.assertEqual(int(self.demand["turn"].max()), 213)
        self.assertEqual(
            len(
                self.demand[
                    (self.demand["punto_venta"] == "MotoTrack Sur")
                    & (self.demand["producto"] == "TRACTOR")
                ]
            ),
            0,
        )

    def test_24_week_abc_values_and_classes_match(self):
        forecasts = self.window[["punto_venta", "producto"]].drop_duplicates().copy()
        forecasts["modelo_elegido"] = "Prueba"
        forecasts["score_pct"] = 70.0
        forecasts["mae_pct"] = 30.0
        forecasts["rmse"] = 1.0
        forecasts["mae"] = 1.0
        forecasts["r2"] = 0.0
        result = build_classification(self.window, self.costs, self.prices, forecasts)

        expected = {
            "Centro-TRACTOR": (261, 1_564_695_000, "A"),
            "Centro-CUATRIMOTO": (296, 1_371_516_000, "A"),
            "Norte-CUATRIMOTO": (231, 1_070_338_500, "A"),
            "Sur-CUATRIMOTO": (204, 945_234_000, "A"),
            "Centro-MOTO": (570, 815_385_000, "A"),
            "Sur-MOTO": (433, 619_406_500, "B"),
            "Norte-MOTO": (386, 552_173_000, "B"),
            "Norte-TRACTOR": (72, 431_640_000, "C"),
        }
        actual = {
            row["sku"]: (row["Demanda 24 sem"], row["Utilidad"], row["ABC"])
            for _, row in result.iterrows()
        }
        self.assertEqual(actual, expected)
        self.assertEqual(float(result["Utilidad"].sum()), 7_370_388_000)
        self.assertTrue((result["MAE%"] == 30.0).all())
        self.assertEqual((int(self.window["turn"].min()), int(self.window["turn"].max())), (190, 213))
        self.assertNotIn("Sur-TRACTOR", set(result["sku"]))

    def test_classification_demand_label_matches_selected_window(self):
        window = active_demand_window(self.demand, weeks=12)
        forecasts = window[["punto_venta", "producto"]].drop_duplicates().copy()
        forecasts["modelo_elegido"] = "Prueba"
        forecasts["score_pct"] = 70.0
        forecasts["mae_pct"] = 30.0
        forecasts["rmse"] = 1.0
        forecasts["mae"] = 1.0
        forecasts["r2"] = 0.0

        result = build_classification(
            window,
            self.costs,
            self.prices,
            forecasts,
            window_weeks=12,
        )

        self.assertIn("Demanda 12 sem", result.columns)
        self.assertNotIn("Demanda 24 sem", result.columns)

    def test_demand_upsert_replaces_duplicate_turn(self):
        existing = pd.DataFrame(
            [{"punto_venta": "MotoTrack Norte", "producto": "MOTO", "turn": 213, "demanda": 14}]
        )
        incoming = pd.DataFrame(
            [
                {"punto_venta": "MotoTrack Norte", "producto": "MOTO", "turn": 213, "demanda": 18},
                {"punto_venta": "MotoTrack Norte", "producto": "MOTO", "turn": 214, "demanda": 12},
            ]
        )
        merged = merge_demand_history(existing, incoming)
        self.assertEqual(len(merged), 2)
        self.assertEqual(float(merged.loc[merged["turn"] == 213, "demanda"].iloc[0]), 18)

    def test_last_analysis_snapshot_persists_results_and_parameters(self):
        classification = pd.DataFrame(
            [{"sku": "Norte-MOTO", "Utilidad": 123_000, "MAE%": 12.5}]
        )
        parameters = {"analysis_weeks": 12, "turn_start": 202, "turn_end": 213}
        with TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "ultimo_ejercicio.json"
            with patch.object(data_io, "LAST_ANALYSIS_PATH", snapshot_path):
                data_io.save_last_analysis(classification, parameters, ["aviso de prueba"])
                restored = data_io.load_last_analysis()

        self.assertIsNotNone(restored)
        self.assertEqual(restored["classification"].iloc[0]["Utilidad"], 123_000)
        self.assertEqual(restored["parameters"], parameters)
        self.assertEqual(restored["warnings"], ["aviso de prueba"])

    def test_backtest_returns_model_score_and_rmse(self):
        sample_series = self.window.query(
            "punto_venta == 'MotoTrack Centro' and producto == 'TRACTOR'"
        )["demanda"]
        result, warnings = backtest_sku(sample_series, test_weeks=13, seasonal_period=5)
        self.assertIn("modelo_elegido", result)
        self.assertGreaterEqual(result["score_pct"], 0)
        self.assertGreaterEqual(result["mae_pct"], 0)
        self.assertAlmostEqual(result["score_pct"], max(0, 100 - result["mae_pct"]))
        self.assertGreaterEqual(result["rmse"], 0)
        self.assertIsInstance(warnings, list)


if __name__ == "__main__":
    unittest.main()