import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from cea.resources.radiation import workflow1_building_viewer as viewer


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


def write_roof_metadata(path: str, n_rows: int = 20) -> None:
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "TYPE": "roofs",
                "Xcoor": float(i),
                "Ycoor": float(i + 1),
                "Zcoor": float(i + 2),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class TestWorkflow1BuildingViewer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = self.temp_dir.name
        self.zone_pickle_dir = os.path.join(self.base, "radiance_geometry_pickle", "zone")
        self.metadata_dir = os.path.join(self.base, "solar-radiation")
        self.export_dir = os.path.join(self.base, "images")
        os.makedirs(self.zone_pickle_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        for building in ["B1000", "B1001"]:
            with open(os.path.join(self.zone_pickle_dir, building), "w", encoding="utf-8") as fp:
                fp.write("placeholder")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_only_writes_all_building_images_with_sensor_overlay(self):
        write_roof_metadata(os.path.join(self.metadata_dir, "B1000_geometry.csv"), n_rows=30)
        write_roof_metadata(os.path.join(self.metadata_dir, "B1001_geometry.csv"), n_rows=30)

        argv = [
            "workflow1_building_viewer",
            "--zone-pickle-dir",
            self.zone_pickle_dir,
            "--metadata-dir",
            self.metadata_dir,
            "--export-images-dir",
            self.export_dir,
            "--no-gui",
            "--show-sensors",
            "--max-sensors",
            "15",
        ]
        with patch.object(viewer.BuildingGeometry, "load", return_value=FakeBuildingGeometry()):
            with patch("sys.argv", argv):
                viewer.main()

        self.assertTrue(os.path.exists(os.path.join(self.export_dir, "B1000_workflow1_3d.png")))
        self.assertTrue(os.path.exists(os.path.join(self.export_dir, "B1001_workflow1_3d.png")))

    def test_export_only_succeeds_when_some_metadata_files_are_missing(self):
        write_roof_metadata(os.path.join(self.metadata_dir, "B1000_geometry.csv"), n_rows=10)
        # B1001 metadata intentionally missing

        argv = [
            "workflow1_building_viewer",
            "--zone-pickle-dir",
            self.zone_pickle_dir,
            "--metadata-dir",
            self.metadata_dir,
            "--export-images-dir",
            self.export_dir,
            "--no-gui",
            "--show-sensors",
        ]
        with patch.object(viewer.BuildingGeometry, "load", return_value=FakeBuildingGeometry()):
            with patch("sys.argv", argv):
                viewer.main()

        self.assertTrue(os.path.exists(os.path.join(self.export_dir, "B1000_workflow1_3d.png")))
        self.assertTrue(os.path.exists(os.path.join(self.export_dir, "B1001_workflow1_3d.png")))


if __name__ == "__main__":
    unittest.main()
