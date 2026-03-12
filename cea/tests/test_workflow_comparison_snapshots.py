import os
import tempfile
import unittest

from cea.resources.radiation.workflow_comparison import snapshot_outputs


def write_file(path: str, content: str = "x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)


class TestWorkflowComparisonSnapshots(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = self.temp_dir.name
        self.scenario = os.path.join(self.base, "scenario")
        self.snapshot = os.path.join(self.base, "snapshot")
        self.prepare_scenario_tree()

    def tearDown(self):
        self.temp_dir.cleanup()

    def prepare_scenario_tree(self) -> None:
        solar_radiation = os.path.join(self.scenario, "outputs", "data", "solar-radiation")
        potentials_solar = os.path.join(self.scenario, "outputs", "data", "potentials", "solar")
        geometry_pickle = os.path.join(solar_radiation, "radiance_geometry_pickle")

        write_file(os.path.join(solar_radiation, "B1000_geometry.csv"), "TYPE,AREA_m2\nroofs,10\n")
        write_file(os.path.join(solar_radiation, "B1000_radiation.csv"), "a,b\n1,2\n")
        write_file(os.path.join(geometry_pickle, "zone", "B1000"), "pickle-data")
        write_file(os.path.join(potentials_solar, "PV_test_total_buildings.csv"), "a,b\n1,2\n")

    def test_snapshot_without_geometry_pickles(self):
        snapshot_outputs(
            scenario=self.scenario,
            workflow_snapshot_root=self.snapshot,
            include_insolation_feather=False,
            include_geometry_pickles=False,
        )

        copied_geometry = os.path.join(self.snapshot, "radiance_geometry_pickle")
        self.assertFalse(os.path.exists(copied_geometry))

    def test_snapshot_with_geometry_pickles(self):
        snapshot_outputs(
            scenario=self.scenario,
            workflow_snapshot_root=self.snapshot,
            include_insolation_feather=False,
            include_geometry_pickles=True,
        )

        copied_geometry_file = os.path.join(self.snapshot, "radiance_geometry_pickle", "zone", "B1000")
        self.assertTrue(os.path.exists(copied_geometry_file))


if __name__ == "__main__":
    unittest.main()
