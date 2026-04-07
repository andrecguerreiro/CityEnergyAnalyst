import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from cea.resources.radiation import ucea


def make_config(scenario: str) -> SimpleNamespace:
    ucea_cfg = SimpleNamespace(
        database_path="CH",
        surroundings_buffer_m=50.0,
        run_surroundings_helper=True,
        terrain_buffer_m=50.0,
        weather_source="climate.onebuilding.org",
        pv_panel="PV1",
        comparison_root="",
        workflow1_only=False,
        building_3d="",
        polygon_timeout_minutes=30.0,
        open_browser=False,
        geojson_url=ucea.DEFAULT_GEOJSON_URL,
    )
    return SimpleNamespace(scenario=scenario, ucea=ucea_cfg)


class TestUcea(unittest.TestCase):
    def setUp(self):
        local_tmp_root = os.path.join("cea", "tests", "_tmp_ucea")
        os.makedirs(local_tmp_root, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix="ucea-test-", dir=local_tmp_root)
        self.scenario = os.path.join(self.temp_dir, "scenario")
        os.makedirs(os.path.join(self.scenario, "inputs", "building-geometry"), exist_ok=True)
        with open(os.path.join(self.scenario, "inputs", "building-geometry", "roof_surfaces.geojson"), "w", encoding="utf-8") as fp:
            fp.write('{"type":"FeatureCollection","features":[]}')
        self.config = make_config(self.scenario)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_happy_path_call_order_and_args(self):
        with patch.object(ucea, "_open_geojson_io"), \
             patch.object(ucea, "_capture_polygon_coordinates", return_value=[(-9.1, 38.7), (-9.0, 38.7), (-9.0, 38.8), (-9.1, 38.7)]), \
             patch.object(ucea, "_configure_height_enrichment_paths"), \
             patch.object(ucea, "_call_api") as call_api, \
             patch.object(ucea, "_run_fixedboxtilingextension") as run_fixedbox, \
             patch.object(ucea, "_run_python_module") as run_module, \
             patch.object(ucea, "InputLocator") as locator_cls:
            locator_cls.return_value.get_zone_building_names.return_value = ["B1111"]
            ucea.main(self.config)

        expected_api = [
            call("create_polygon", self.config, filename="site", coordinates=[(-9.1, 38.7), (-9.0, 38.7), (-9.0, 38.8), (-9.1, 38.7)]),
            call("database_helper", self.config, databases_path="CH"),
            call("zone_helper", self.config),
            call("surroundings_helper", self.config, buffer=50.0),
            call("terrain_helper", self.config, buffer=50.0),
            call("weather_helper", self.config, weather="climate.onebuilding.org"),
            call("archetypes_mapper", self.config),
        ]
        self.assertEqual(expected_api, call_api.call_args_list)
        run_fixedbox.assert_called_once_with(
            self.scenario,
            [(-9.1, 38.7), (-9.0, 38.7), (-9.0, 38.8), (-9.1, 38.7)],
        )
        self.assertEqual(4, run_module.call_count)
        self.assertEqual("cea.resources.radiation.workflow_comparison", run_module.call_args_list[0].args[0])
        self.assertIn("--include-geometry-pickles", run_module.call_args_list[0].args[1])
        self.assertEqual("cea.resources.radiation.workflow_metrics_report", run_module.call_args_list[1].args[0])
        self.assertEqual("cea.resources.radiation.workflow_geometry_comparison_3d", run_module.call_args_list[2].args[0])
        self.assertEqual("cea.resources.radiation.workflow1_building_viewer", run_module.call_args_list[3].args[0])
        viewer_args = run_module.call_args_list[3].args[1]
        self.assertIn("--export-images-dir", viewer_args)
        self.assertIn(os.path.join(self.scenario, "outputs", "data", "roof-workflow-comparison", "workflow1_3d_images"), viewer_args)
        self.assertIn("--show-sensors", viewer_args)

    def test_invalid_clipboard_then_manual_coordinates(self):
        with patch.object(ucea, "_get_clipboard_text", return_value="{bad-json}"), \
             patch.object(ucea, "_timed_input", side_effect=["(1,1),(2,2),(3,3),(1,1)"]):
            coordinates = ucea._capture_polygon_coordinates(timeout_minutes=1)
        self.assertEqual([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (1.0, 1.0)], coordinates)

    def test_auto_building_selection_and_fallback(self):
        locator = MagicMock()
        locator.get_zone_building_names.return_value = ["B2000"]
        self.assertEqual("B2000", ucea._select_building_for_3d(locator, ""))
        self.assertEqual("B9999", ucea._select_building_for_3d(locator, "B9999"))
        locator.get_zone_building_names.return_value = []
        self.assertEqual("B1000", ucea._select_building_for_3d(locator, ""))

    def test_failure_propagates_stage_name_and_stops(self):
        def failing_call(script_name, config, **kwargs):
            if script_name == "database_helper":
                raise RuntimeError("helper failure")
            return None

        with patch.object(ucea, "_open_geojson_io"), \
             patch.object(ucea, "_capture_polygon_coordinates", return_value=[(-9.1, 38.7), (-9.0, 38.7), (-9.0, 38.8), (-9.1, 38.7)]), \
             patch.object(ucea, "_configure_height_enrichment_paths"), \
             patch.object(ucea, "_call_api", side_effect=failing_call), \
             patch.object(ucea, "_run_fixedboxtilingextension") as run_fixedbox, \
             patch.object(ucea, "_run_python_module") as run_module:
            with self.assertRaises(ucea.UceaStageError) as ctx:
                ucea.main(self.config)
        self.assertIn("Scenario preparation scripts failed", str(ctx.exception))
        run_fixedbox.assert_not_called()
        run_module.assert_not_called()

    def test_fixedbox_failure_propagates_stage_name_and_stops(self):
        with patch.object(ucea, "_open_geojson_io"), \
             patch.object(ucea, "_capture_polygon_coordinates", return_value=[(-9.1, 38.7), (-9.0, 38.7), (-9.0, 38.8), (-9.1, 38.7)]), \
             patch.object(ucea, "_configure_height_enrichment_paths"), \
             patch.object(ucea, "_call_api"), \
             patch.object(ucea, "_run_fixedboxtilingextension", side_effect=RuntimeError("fixedbox failed")), \
             patch.object(ucea, "_run_python_module") as run_module:
            with self.assertRaises(ucea.UceaStageError) as ctx:
                ucea.main(self.config)
        self.assertIn("Generate roof surfaces via fixedboxtilingextension failed", str(ctx.exception))
        run_module.assert_not_called()

    def test_comparison_root_defaults_to_fixed_path(self):
        expected = os.path.join(self.scenario, "outputs", "data", "roof-workflow-comparison")
        self.assertEqual(expected, ucea._resolve_comparison_root(self.config, self.scenario))
        self.config.ucea.comparison_root = os.path.join(self.scenario, "custom-root")
        self.assertEqual(os.path.abspath(self.config.ucea.comparison_root), ucea._resolve_comparison_root(self.config, self.scenario))

    def test_missing_roof_file_is_auto_created(self):
        roof_file = os.path.join(self.scenario, "inputs", "building-geometry", "roof_surfaces.geojson")
        os.remove(roof_file)
        resolved = ucea._require_roof_file(self.scenario)
        self.assertEqual(roof_file, resolved)
        self.assertTrue(os.path.exists(roof_file))

    def test_workflow1_only_runs_metrics_and_viewer_exports(self):
        self.config.ucea.workflow1_only = True
        with patch.object(ucea, "_run_cea_script") as run_cea_script, \
             patch.object(ucea, "_run_python_module") as run_module, \
             patch.object(ucea, "InputLocator") as locator_cls:
            locator_cls.return_value.get_zone_building_names.return_value = ["B1111"]
            runtime = ucea._run_experiments(self.config, self.scenario)

        self.assertEqual(
            [
                call("radiation", self.scenario),
                call("photovoltaic", self.scenario, ["--panel-on-wall", "false", "--type-pvpanel", "PV1"]),
            ],
            run_cea_script.call_args_list,
        )
        self.assertEqual(2, run_module.call_count)
        self.assertEqual("cea.resources.radiation.workflow_metrics_report", run_module.call_args_list[0].args[0])
        metrics_args = run_module.call_args_list[0].args[1]
        self.assertIn("--workflows", metrics_args)
        self.assertIn("WF1", metrics_args)
        self.assertIn("--workflow1-root", metrics_args)
        self.assertIn(os.path.join(self.scenario, "outputs", "data"), metrics_args)
        self.assertIn("--out-dir", metrics_args)
        self.assertIn(
            os.path.join(self.scenario, "outputs", "data", "solar-radiation", "workflow1_metrics"),
            metrics_args,
        )

        self.assertEqual("cea.resources.radiation.workflow1_building_viewer", run_module.call_args_list[1].args[0])
        viewer_args = run_module.call_args_list[1].args[1]
        self.assertIn("--metadata-dir", viewer_args)
        self.assertIn(os.path.join(self.scenario, "outputs", "data", "solar-radiation"), viewer_args)
        self.assertIn("--export-images-dir", viewer_args)
        self.assertIn(
            os.path.join(self.scenario, "outputs", "data", "solar-radiation", "workflow1_3d_images"),
            viewer_args,
        )
        self.assertIn("--show-sensors", viewer_args)
        self.assertFalse(runtime.comparison_ran)
        self.assertEqual(
            os.path.join(self.scenario, "outputs", "data", "solar-radiation", "workflow1_metrics"),
            runtime.metrics_output_dir,
        )
        self.assertEqual(
            os.path.join(self.scenario, "outputs", "data", "solar-radiation", "workflow1_3d_images"),
            runtime.images_output_dir,
        )


if __name__ == "__main__":
    unittest.main()
