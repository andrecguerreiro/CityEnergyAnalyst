import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from cea.resources.radiation import workflow_metrics_report


def write_geometry_csv(path: str) -> None:
    rows = [
        {"TYPE": "roofs", "Zcoor": 12.0, "terrain_elevation": 4.0},
        {"TYPE": "undersides", "Zcoor": 6.0, "terrain_elevation": 4.0},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_sensor_csv(path: str) -> None:
    rows = [
        {
            "TYPE": "roofs",
            "AREA_m2": 10.0,
            "tilt_deg": 10.0,
            "B_deg": 12.0,
            "surface_azimuth_deg": 180.0,
            "total_rad_Whm2": 1400.0,
            "area_installed_module_m2": 7.5,
            "Xdir": 0.0,
            "Ydir": -1.0,
        }
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_totals_csv(path: str) -> None:
    rows = [
        {
            "name": "B1000",
            "E_PV_gen_kWh": 5200.0,
            "area_PV_m2": 60.0,
            "radiation_kWh": 9000.0,
        }
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class TestWorkflowMetricsReport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = self.temp_dir.name
        self.workflow1_root = os.path.join(self.base, "outputs", "data")
        self.comparison_root = self.workflow1_root
        self.out_dir = os.path.join(self.base, "metrics")

        os.makedirs(os.path.join(self.workflow1_root, "solar-radiation"), exist_ok=True)
        os.makedirs(os.path.join(self.workflow1_root, "potentials", "solar", "sensors"), exist_ok=True)

        write_geometry_csv(os.path.join(self.workflow1_root, "solar-radiation", "B1000_geometry.csv"))
        write_sensor_csv(
            os.path.join(
                self.workflow1_root,
                "potentials",
                "solar",
                "sensors",
                "B1000_PV_sensors.csv",
            )
        )
        write_totals_csv(
            os.path.join(
                self.workflow1_root,
                "potentials",
                "solar",
                "PV_PV1_total_buildings.csv",
            )
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_workflow1_only_outputs_metrics_without_deltas(self):
        argv = [
            "workflow_metrics_report",
            "--comparison-root",
            self.comparison_root,
            "--workflows",
            "WF1",
            "--workflow1-root",
            self.workflow1_root,
            "--level",
            "both",
            "--pv-panels",
            "PV1",
            "--out-dir",
            self.out_dir,
        ]
        with patch("sys.argv", argv):
            workflow_metrics_report.main()

        scenario_metrics = os.path.join(self.out_dir, "scenario_metrics.csv")
        building_metrics = os.path.join(self.out_dir, "building_metrics.csv")
        panel_summary = os.path.join(self.out_dir, "panel_placement_summary.csv")
        scenario_deltas = os.path.join(self.out_dir, "scenario_deltas.csv")
        building_deltas = os.path.join(self.out_dir, "building_deltas.csv")

        self.assertTrue(os.path.exists(scenario_metrics))
        self.assertTrue(os.path.exists(building_metrics))
        self.assertTrue(os.path.exists(panel_summary))
        self.assertFalse(os.path.exists(scenario_deltas))
        self.assertFalse(os.path.exists(building_deltas))

        scenario_df = pd.read_csv(scenario_metrics)
        building_df = pd.read_csv(building_metrics)
        self.assertEqual({"WF1"}, set(scenario_df["workflow_id"].astype(str)))
        self.assertEqual({"WF1"}, set(building_df["workflow_id"].astype(str)))
        self.assertIn("B1000", set(building_df["building"].astype(str)))


if __name__ == "__main__":
    unittest.main()
