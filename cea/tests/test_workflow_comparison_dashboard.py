import os
import tempfile
import unittest

import pandas as pd

from cea.resources.radiation.workflow_comparison_dashboard import (
    WORKFLOW_0,
    WORKFLOW_1,
    WORKFLOW_2,
    build_delta_table,
    collect_kpis,
    write_dashboard,
)


def write_geometry_csv(path: str, roof_areas: list[float]) -> None:
    rows = []
    for area in roof_areas:
        rows.append({"TYPE": "roofs", "AREA_m2": area})
    rows.append({"TYPE": "walls", "AREA_m2": 5.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_pv_total_buildings(path: str, energies: list[float], areas: list[float]) -> None:
    df = pd.DataFrame(
        {
            "name": [f"B{i + 1}" for i in range(len(energies))],
            "E_PV_gen_kWh": energies,
            "area_PV_m2": areas,
        }
    )
    df.to_csv(path, index=False)


def create_workflow_fixture(
    run_folder: str,
    workflow: str,
    roof_areas: list[float],
    energies: list[float] | None = None,
    areas: list[float] | None = None,
    panel_type: str = "TEST",
) -> None:
    workflow_folder = os.path.join(run_folder, workflow)
    solar_radiation_folder = os.path.join(workflow_folder, "solar-radiation")
    pv_folder = os.path.join(workflow_folder, "potentials", "solar")
    os.makedirs(solar_radiation_folder, exist_ok=True)
    os.makedirs(pv_folder, exist_ok=True)

    write_geometry_csv(os.path.join(solar_radiation_folder, "B1_geometry.csv"), roof_areas)

    if energies is not None and areas is not None:
        write_pv_total_buildings(
            os.path.join(pv_folder, f"PV_{panel_type}_total_buildings.csv"),
            energies,
            areas,
        )


class TestWorkflowComparisonDashboard(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_run_folder(self, scenario_name: str, run_id: str) -> str:
        run_folder = os.path.join(
            self.base,
            scenario_name,
            "outputs",
            "data",
            "roof-workflow-comparison",
            run_id,
        )
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def test_collect_and_delta_for_two_scenarios(self):
        run_a = self.create_run_folder("scenario_a", "20260309_120000")
        run_b = self.create_run_folder("scenario_b", "20260309_130000")

        create_workflow_fixture(run_a, WORKFLOW_0, [10.0, 20.0], [100.0, 20.0], [50.0, 10.0])
        create_workflow_fixture(run_a, WORKFLOW_1, [12.0, 22.0], [120.0, 24.0], [55.0, 11.0])
        create_workflow_fixture(run_a, WORKFLOW_2, [11.0, 21.0], [110.0, 22.0], [54.0, 10.0])

        create_workflow_fixture(run_b, WORKFLOW_0, [8.0, 16.0], [80.0, 16.0], [40.0, 8.0])
        create_workflow_fixture(run_b, WORKFLOW_1, [9.0, 17.0], [92.0, 18.0], [44.0, 8.0])
        create_workflow_fixture(run_b, WORKFLOW_2, [8.5, 16.5], [88.0, 17.0], [43.0, 8.0])

        kpi_df, quality_df = collect_kpis([run_a, run_b])
        delta_df = build_delta_table(kpi_df)

        self.assertEqual(6, len(kpi_df))
        self.assertEqual(0, len(quality_df))
        self.assertEqual(2, len(delta_df))

        row_a = delta_df[delta_df["scenario_name"] == "scenario_a"].iloc[0]
        self.assertAlmostEqual(24.0, row_a["dE_w1_vs_w0_kWh"], places=6)
        self.assertAlmostEqual(12.0, row_a["dE_w2_vs_w0_kWh"], places=6)
        self.assertAlmostEqual(12.0, row_a["dE_w1_vs_w2_kWh"], places=6)

    def test_quality_checks_for_missing_workflow_and_zero_area(self):
        run_folder = self.create_run_folder("scenario_c", "20260309_140000")

        create_workflow_fixture(run_folder, WORKFLOW_0, [5.0], [10.0], [0.0])
        create_workflow_fixture(run_folder, WORKFLOW_1, [6.0], [12.0], [2.0])
        # WORKFLOW_2 intentionally missing

        _, quality_df = collect_kpis([run_folder])
        issue_codes = set(quality_df["issue_code"].tolist())

        self.assertIn("missing_workflow_folder", issue_codes)
        self.assertIn("zero_pv_area", issue_codes)

    def test_write_dashboard_creates_expected_sheets(self):
        run_folder = self.create_run_folder("scenario_d", "20260309_150000")
        create_workflow_fixture(run_folder, WORKFLOW_0, [10.0], [100.0], [50.0])
        create_workflow_fixture(run_folder, WORKFLOW_1, [11.0], [110.0], [51.0])
        create_workflow_fixture(run_folder, WORKFLOW_2, [10.5], [105.0], [50.5])

        kpi_df, quality_df = collect_kpis([run_folder])
        delta_df = build_delta_table(kpi_df)

        output_path = os.path.join(self.base, "dashboard.xlsx")
        write_dashboard(output_path, kpi_df, delta_df, quality_df)

        self.assertTrue(os.path.exists(output_path))
        with pd.ExcelFile(output_path) as workbook:
            self.assertIn("scenario_workflow_kpis", workbook.sheet_names)
            self.assertIn("scenario_deltas", workbook.sheet_names)
            self.assertIn("data_quality_checks", workbook.sheet_names)


if __name__ == "__main__":
    unittest.main()
