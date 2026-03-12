import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from cea.resources.radiation import workflow_geometry_comparison_3d as workflow_3d


def write_metadata_csv(path: str, n_rows: int = 10) -> None:
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "TYPE": "roofs",
                "Xcoor": float(i),
                "Ycoor": float(i + 1),
                "Zcoor": float(i + 2),
                "Xdir": 0.0,
                "Ydir": 0.0,
                "Zdir": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class FakeBuildingGeometry:
    def __init__(self):
        self.walls = [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 2.0, 3.0],
                    [0.0, 0.0, 3.0],
                ],
                dtype=float,
            )
        ]
        self.roofs = [
            np.array(
                [
                    [0.0, 0.0, 3.0],
                    [2.0, 0.0, 3.0],
                    [2.0, 2.0, 3.0],
                    [0.0, 2.0, 3.0],
                ],
                dtype=float,
            )
        ]
        self.windows = [
            np.array(
                [
                    [0.0, 0.5, 1.0],
                    [0.0, 1.5, 1.0],
                    [0.0, 1.5, 2.0],
                    [0.0, 0.5, 2.0],
                ],
                dtype=float,
            )
        ]
        self.normals_roofs = [(0.0, 0.0, 1.0)]


class TestWorkflowGeometryComparison3D(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = self.temp_dir.name
        self.comparison_root = os.path.join(self.base, "roof-workflow-comparison")
        os.makedirs(self.comparison_root, exist_ok=True)
        self.building = "B1000"

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_workflow_folder(
        self,
        workflow: str,
        include_pickle: bool = True,
        include_metadata: bool = True,
    ) -> tuple[str, str]:
        workflow_root = os.path.join(self.comparison_root, workflow)
        pickle_path = os.path.join(workflow_root, "radiance_geometry_pickle", "zone", self.building)
        metadata_path = os.path.join(workflow_root, "solar-radiation", f"{self.building}_geometry.csv")

        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        if include_pickle:
            with open(pickle_path, "w", encoding="utf-8") as fp:
                fp.write("placeholder")

        if include_metadata:
            write_metadata_csv(metadata_path, n_rows=15)

        return pickle_path, metadata_path

    def test_missing_workflow_folder_raises(self):
        with self.assertRaises(NotADirectoryError):
            workflow_3d.load_workflow_data(
                comparison_root=self.comparison_root,
                workflow=workflow_3d.WORKFLOW_0,
                building=self.building,
                max_metadata_arrows=300,
            )

    def test_missing_geometry_pickle_raises(self):
        self.create_workflow_folder(workflow_3d.WORKFLOW_0, include_pickle=False, include_metadata=True)
        with self.assertRaises(FileNotFoundError):
            workflow_3d.load_workflow_data(
                comparison_root=self.comparison_root,
                workflow=workflow_3d.WORKFLOW_0,
                building=self.building,
                max_metadata_arrows=300,
            )

    def test_missing_metadata_csv_raises(self):
        self.create_workflow_folder(workflow_3d.WORKFLOW_0, include_pickle=True, include_metadata=False)
        with self.assertRaises(FileNotFoundError):
            workflow_3d.load_workflow_data(
                comparison_root=self.comparison_root,
                workflow=workflow_3d.WORKFLOW_0,
                building=self.building,
                max_metadata_arrows=300,
            )

    def test_downsample_dataframe_deterministic(self):
        df = pd.DataFrame({"v": np.arange(1000)})
        out_1 = workflow_3d.downsample_dataframe_deterministic(df, max_rows=300)
        out_2 = workflow_3d.downsample_dataframe_deterministic(df, max_rows=300)
        self.assertEqual(300, len(out_1))
        self.assertEqual(300, len(out_2))
        self.assertListEqual(out_1.index.tolist(), out_2.index.tolist())

    def test_create_figure_smoke(self):
        for workflow in workflow_3d.WORKFLOWS:
            self.create_workflow_folder(workflow, include_pickle=True, include_metadata=True)

        with patch.object(workflow_3d.BuildingGeometry, "load", return_value=FakeBuildingGeometry()):
            workflow_data = {
                workflow: workflow_3d.load_workflow_data(
                    comparison_root=self.comparison_root,
                    workflow=workflow,
                    building=self.building,
                    max_metadata_arrows=300,
                )
                for workflow in workflow_3d.WORKFLOWS
            }

        output_figure = os.path.join(self.comparison_root, "comparison.png")
        workflow_3d.create_figure(
            workflow_data=workflow_data,
            building=self.building,
            output_figure=output_figure,
            show=False,
        )

        self.assertTrue(os.path.exists(output_figure))


if __name__ == "__main__":
    unittest.main()
