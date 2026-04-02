"""
Run an end-to-end UX workflow for roof-comparison experiments.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Any

import cea.api
from cea.config import Configuration, parse_string_coordinate_list
from cea.inputlocator import InputLocator

DEFAULT_BUILDING_FALLBACK = "B1000"
DEFAULT_UCEA_SCENARIO = r"C:\Users\Andre\cea-scenarios\test-tilt"
DEFAULT_GEOJSON_URL = "https://geojson.io/#map=18.2/38.708267/-9.138085"
DEFAULT_INE_HEIGHT_GPKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "INE_com_NPAV_norm0.gpkg")
)
ROOF_RELATIVE_PATH = os.path.join("inputs", "building-geometry", "roof_surfaces.geojson")
DEFAULT_ROOF_SURFACES_GEOJSON ={
  "type": "FeatureCollection",
  "name": "roof",
  "crs": {
    "type": "name",
    "properties": {
      "name": "urn:ogc:def:crs:EPSG::32629"
    }
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "building": "B1003",
        "roof_id": "1"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              469412.6395426503,
              4282433.9341752175,
              9.472797393798828
            ],
            [
              469412.6864647212,
              4282433.92681034,
              9.477940559387207
            ],
            [
              469414.31193959294,
              4282439.572283552,
              11.047562599182129
            ],
            [
              469414.3269647871,
              4282439.634895555,
              11.065096855163574
            ],
            [
              469409.8246406832,
              4282440.935408365,
              10.527775764465332
            ],
            [
              469409.77771861246,
              4282440.942773244,
              10.522059440612793
            ],
            [
              469408.16019952344,
              4282435.281757862,
              8.94910717010498
            ],
            [
              469408.1372185749,
              4282435.234688073,
              8.934903144836426
            ],
            [
              469412.6395426503,
              4282433.9341752175,
              9.472797393798828
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "building": "B1010",
        "roof_id": "1"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              469413.59441232204,
              4282427.922673877,
              10.0
            ],
            [
              469413.59581565025,
              4282427.774321208,
              10.0
            ],
            [
              469413.7982432566,
              4282412.978685272,
              10.0
            ],
            [
              469413.8528995033,
              4282412.979202285,
              10.0
            ],
            [
              469419.185566033,
              4282413.053072203,
              10.0
            ],
            [
              469419.17628081434,
              4282413.209159049,
              10.0
            ],
            [
              469418.97392706113,
              4282427.996987098,
              10.0
            ],
            [
              469418.9192708139,
              4282427.996470083,
              10.0
            ],
            [
              469413.59441232204,
              4282427.922673877,
              10.0
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "building": "B1011",
        "roof_id": "1"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              469418.59153352276,
              4282423.019206328,
              11.966423988342285
            ],
            [
              469418.77892637067,
              4282423.020978953,
              11.919546127319336
            ],
            [
              469434.4574615404,
              4282423.169288452,
              8.033576011657715
            ],
            [
              469434.45687066595,
              4282423.231752736,
              8.02748966217041
            ],
            [
              469434.3834544516,
              4282430.99294008,
              7.275455951690674
            ],
            [
              469434.2741419539,
              4282430.991906048,
              7.304182052612305
            ],
            [
              469418.51752643526,
              4282430.842857726,
              11.208303451538086
            ],
            [
              469418.5183388889,
              4282430.756969338,
              11.21680736541748
            ],
            [
              469418.59153352276,
              4282423.019206328,
              11.966423988342285
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "building": "B1011",
        "roof_id": "2"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              469418.5921243973,
              4282422.956742046,
              8.0968017578125
            ],
            [
              469418.59153352276,
              4282423.019206328,
              8.103830337524414
            ],
            [
              469434.2700686871,
              4282423.167515829,
              11.851225852966309
            ],
            [
              469434.4574615404,
              4282423.169288452,
              11.896169662475586
            ],
            [
              469434.53065606416,
              4282415.431525215,
              11.037753105163574
            ],
            [
              469434.5314685154,
              4282415.345636823,
              11.02817153930664
            ],
            [
              469418.7748529911,
              4282415.19658896,
              7.263378143310547
            ],
            [
              469418.66554049647,
              4282415.19555493,
              7.235833168029785
            ],
            [
              469418.5921243973,
              4282422.956742046,
              8.0968017578125
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "building": "B1012",
        "roof_id": "1"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              469449.58283975715,
              4282419.056605149,
              11.387120246887207
            ],
            [
              469449.76242458,
              4282419.058303912,
              11.355616569519043
            ],
            [
              469465.07265349315,
              4282419.343686555,
              8.612879753112793
            ],
            [
              469465.07184104127,
              4282419.4295749515,
              8.623546600341797
            ],
            [
              469464.889882716,
              4282428.75929166,
              9.849780082702637
            ],
            [
              469464.74153003196,
              4282428.757888331,
              9.877182006835938
            ],
            [
              469449.40795088187,
              4282428.464475806,
              12.62402057647705
            ],
            [
              469449.40905877267,
              4282428.347355269,
              12.60842514038086
            ],
            [
              469449.58283975715,
              4282419.056605149,
              11.387120246887207
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "building": "B1012",
        "roof_id": "2"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              469449.7656103629,
              4282409.641000311,
              9.784828186035156
            ],
            [
              469449.9139630427,
              4282409.642403635,
              9.810946464538574
            ],
            [
              469465.2475422104,
              4282409.935815625,
              12.446929931640625
            ],
            [
              469465.24650818267,
              4282410.045128128,
              12.433055877685547
            ],
            [
              469465.07265349315,
              4282419.343686555,
              11.331050872802734
            ],
            [
              469464.9008767011,
              4282419.342061652,
              11.300966262817383
            ],
            [
              469449.58283975715,
              4282419.056605149,
              8.668949127197266
            ],
            [
              469449.5836522089,
              4282418.970716755,
              8.678633689880371
            ],
            [
              469449.7656103629,
              4282409.641000311,
              9.784828186035156
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "building": "B1013",
        "roof_id": "1"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              469433.82941004855,
              4282415.268717177,
              8.523118019104004
            ],
            [
              469433.96995468845,
              4282415.270046641,
              8.516289710998535
            ],
            [
              469450.1875402054,
              4282415.392220055,
              7.721311092376709
            ],
            [
              469450.1869493318,
              4282415.454684342,
              7.735836982727051
            ],
            [
              469450.1215417923,
              4282424.020247316,
              9.599096298217773
            ],
            [
              469449.96538107673,
              4282424.01877013,
              9.606799125671387
            ],
            [
              469433.7710719529,
              4282423.912434105,
              10.400903701782227
            ],
            [
              469433.7719582647,
              4282423.818737677,
              10.381409645080566
            ],
            [
              469433.82941004855,
              4282415.268717177,
              8.523118019104004
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "building": "B1013",
        "roof_id": "2"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              469433.7710719529,
              4282423.912434105,
              9.568986892700195
            ],
            [
              469433.92723266373,
              4282423.9139112905,
              9.577242851257324
            ],
            [
              469450.1215417923,
              4282424.020247316,
              10.431013107299805
            ],
            [
              469450.1206554805,
              4282424.113943745,
              10.411149978637695
            ],
            [
              469450.06320355786,
              4282432.663964506,
              8.523112297058105
            ],
            [
              469449.9226589137,
              4282432.662635038,
              8.51577091217041
            ],
            [
              469433.70499954215,
              4282432.548269139,
              7.66108512878418
            ],
            [
              469433.7055904179,
              4282432.485804853,
              7.675730228424072
            ],
            [
              469433.7710719529,
              4282423.912434105,
              9.568986892700195
            ]
          ]
        ]
      }
    }
  ]
}



class UceaStageError(RuntimeError):
    """Raised when a stage fails."""


@dataclass
class UceaRuntime:
    scenario: str
    comparison_root: str
    building: str
    output_figure: str
    viewer_started: bool
    comparison_ran: bool


def _log(message: str) -> None:
    print(f"[ucea] {message}")


def _run_stage(stage_name: str, fn):
    _log(f"Starting: {stage_name}")
    try:
        result = fn()
    except Exception as exc:
        raise UceaStageError(f"{stage_name} failed: {exc}") from exc
    _log(f"Completed: {stage_name}")
    return result


def _timed_input(prompt: str, timeout_seconds: float | None) -> str:
    if timeout_seconds is None:
        return input(prompt)
    if timeout_seconds <= 0:
        raise TimeoutError("Timed out while waiting for polygon input.")

    queue: Queue[str] = Queue(maxsize=1)

    def _reader() -> None:
        try:
            value = input(prompt)
        except EOFError:
            value = ""
        queue.put(value)

    thread = Thread(target=_reader, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError("Timed out while waiting for polygon input.")
    try:
        return queue.get_nowait()
    except Empty:
        return ""


def _extract_polygon_ring_from_geojson_obj(obj: dict[str, Any]) -> list[list[float]]:
    geometry = obj
    obj_type = str(obj.get("type", "")).strip()

    if obj_type == "FeatureCollection":
        features = obj.get("features", [])
        if not isinstance(features, list) or not features:
            raise ValueError("GeoJSON FeatureCollection has no features.")
        for feature in features:
            geom = feature.get("geometry") if isinstance(feature, dict) else None
            if isinstance(geom, dict) and geom.get("type") in {"Polygon", "MultiPolygon", "LineString"}:
                geometry = geom
                break
        else:
            raise ValueError("No Polygon, MultiPolygon, or LineString geometry found in FeatureCollection.")
    elif obj_type == "Feature":
        geometry = obj.get("geometry")

    if not isinstance(geometry, dict):
        raise ValueError("GeoJSON geometry is missing.")

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("GeoJSON Polygon has no coordinates.")
        ring = coordinates[0]
    elif geometry_type == "MultiPolygon":
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("GeoJSON MultiPolygon has no coordinates.")
        first_polygon = coordinates[0]
        if not isinstance(first_polygon, list) or not first_polygon:
            raise ValueError("GeoJSON MultiPolygon first polygon is invalid.")
        ring = first_polygon[0]
    elif geometry_type == "LineString":
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("GeoJSON LineString has no coordinates.")
        ring = coordinates
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

    if not isinstance(ring, list) or len(ring) < 3:
        raise ValueError("Polygon ring must contain at least three points.")
    return ring


def _format_coordinates_string(coordinates: list[tuple[float, float]]) -> str:
    return ",".join(f"({lon},{lat})" for lon, lat in coordinates)


def _parse_geojson_text_to_coordinates_string(text: str) -> str:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("GeoJSON text must decode to an object.")
    ring = _extract_polygon_ring_from_geojson_obj(payload)

    pairs: list[tuple[float, float]] = []
    for point in ring:
        if not isinstance(point, list | tuple) or len(point) < 2:
            raise ValueError("Polygon point must have at least lon/lat.")
        lon = float(point[0])
        lat = float(point[1])
        pairs.append((lon, lat))

    if pairs[0] != pairs[-1]:
        pairs.append(pairs[0])

    parse_string_coordinate_list(_format_coordinates_string(pairs))
    return _format_coordinates_string(pairs)


def _get_clipboard_text() -> str:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            text = root.clipboard_get()
        finally:
            root.destroy()
        return str(text).strip()
    except Exception:
        pass

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""
    return ""


def _parse_polygon_from_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("No input provided.")

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()

    try:
        parse_string_coordinate_list(cleaned)
        return cleaned
    except ValueError:
        pass

    return _parse_geojson_text_to_coordinates_string(cleaned)


def _read_multiline_input(first_line: str, deadline: float | None) -> str:
    lines = [first_line]
    _log("Detected JSON input. Paste remaining lines, then submit an empty line to finish.")
    while True:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Timed out while waiting for polygon input.")
        line = _timed_input("... ", remaining)
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def _capture_polygon_coordinates(timeout_minutes: float) -> list[tuple[float, float]]:
    timeout_seconds = None if timeout_minutes <= 0 else timeout_minutes * 60.0
    start = time.monotonic()

    _log("Trying to read polygon GeoJSON from clipboard.")
    clip_text = _get_clipboard_text()
    if clip_text:
        try:
            coordinates_text = _parse_polygon_from_text(clip_text)
            _log("Polygon parsed from clipboard.")
            return parse_string_coordinate_list(coordinates_text)
        except Exception as exc:
            _log(f"Clipboard did not contain a valid polygon: {exc}")
    else:
        _log("Clipboard is empty or unavailable.")

    prompt = (
        "\nPaste either GeoJSON text or coordinates in format "
        "\"(lon,lat),(lon,lat),(lon,lat)\":\n> "
    )
    while True:
        remaining_seconds = None
        if timeout_seconds is not None:
            elapsed = time.monotonic() - start
            remaining_seconds = timeout_seconds - elapsed
            if remaining_seconds <= 0:
                raise TimeoutError("Timed out while waiting for polygon input.")

        raw = _timed_input(prompt, remaining_seconds).strip()
        if not raw:
            _log("No input received. Please paste polygon data.")
            continue

        payload_text = raw
        if raw.startswith("{") or raw.startswith("["):
            deadline = None if timeout_seconds is None else start + timeout_seconds
            payload_text = _read_multiline_input(raw, deadline)

        try:
            coordinates_text = _parse_polygon_from_text(payload_text)
            return parse_string_coordinate_list(coordinates_text)
        except Exception as exc:
            _log(f"Invalid polygon input: {exc}")


def _call_api(script_name: str, config: Configuration, **kwargs) -> None:
    getattr(cea.api, script_name)(config=config, **kwargs)


def _prepare_site_polygon(config: Configuration, coordinates: list[tuple[float, float]]) -> None:
    _call_api("create_polygon", config, filename="site", coordinates=coordinates)


def _run_data_preparation(config: Configuration) -> None:
    _configure_height_enrichment_paths(config)
    _call_api("database_helper", config, databases_path=config.ucea.database_path)
    _call_api("zone_helper", config)
    if config.ucea.run_surroundings_helper:
        _call_api("surroundings_helper", config, buffer=config.ucea.surroundings_buffer_m)
    else:
        _log(
            "Skipping surroundings-helper and keeping existing surroundings geometry at "
            "inputs/geometry/surroundings.shp."
        )
    _call_api("terrain_helper", config, buffer=config.ucea.terrain_buffer_m)
    _call_api("weather_helper", config, weather=config.ucea.weather_source)
    _call_api("archetypes_mapper", config)


def _configure_height_enrichment_paths(config: Configuration) -> None:
    with config.ignore_restrictions():
        zone_height_gpkg = str(config.zone_helper.building_height_gpkg).strip()
        surroundings_height_gpkg = str(config.surroundings_helper.building_height_gpkg).strip()

        if (not zone_height_gpkg or not surroundings_height_gpkg) and os.path.exists(DEFAULT_INE_HEIGHT_GPKG):
            if not zone_height_gpkg:
                config.zone_helper.building_height_gpkg = DEFAULT_INE_HEIGHT_GPKG
                _log(f"Using default INE height GeoPackage for zone-helper: {DEFAULT_INE_HEIGHT_GPKG}")
            if not surroundings_height_gpkg:
                config.surroundings_helper.building_height_gpkg = DEFAULT_INE_HEIGHT_GPKG
                _log(f"Using default INE height GeoPackage for surroundings-helper: {DEFAULT_INE_HEIGHT_GPKG}")


def _resolve_comparison_root(config: Configuration, scenario: str) -> str:
    if config.ucea.comparison_root.strip():
        return os.path.abspath(config.ucea.comparison_root.strip())
    return os.path.join(scenario, "outputs", "data", "roof-workflow-comparison")


def _require_roof_file(scenario: str) -> str:
    roof_file = os.path.join(scenario, ROOF_RELATIVE_PATH)
    if not os.path.exists(roof_file):
        os.makedirs(os.path.dirname(roof_file), exist_ok=True)
        with open(roof_file, "w", encoding="utf-8") as fp:
            json.dump(DEFAULT_ROOF_SURFACES_GEOJSON, fp, ensure_ascii=True, indent=2)
            fp.write("\n")
        _log(f"Created temporary default roof file: {roof_file}")
    return roof_file


def _run_python_module(module: str, args: list[str]) -> None:
    command = [sys.executable, "-m", module] + args
    _log("Running command: " + " ".join(command))
    subprocess.run(command, check=True)


def _run_cea_script(script_name: str, scenario: str, extra_args: list[str] | None = None) -> None:
    command = [
        sys.executable,
        "-m",
        "cea.interfaces.cli.cli",
        script_name,
        "--scenario",
        scenario,
    ]
    if extra_args:
        command.extend(extra_args)
    _log("Running command: " + " ".join(command))
    subprocess.run(command, check=True)


def _clean_solar_outputs(scenario: str) -> None:
    targets = [
        os.path.join(scenario, "outputs", "data", "solar-radiation"),
        os.path.join(scenario, "outputs", "data", "potentials", "solar"),
    ]
    for target in targets:
        if os.path.isdir(target):
            shutil.rmtree(target)
            _log(f"Removed stale output folder: {target}")


def _select_building_for_3d(locator: InputLocator, preferred_building: str) -> str:
    preferred = preferred_building.strip()
    if preferred:
        return preferred
    try:
        building_names = locator.get_zone_building_names()
    except Exception:
        building_names = []
    if building_names:
        return building_names[0]
    _log(f"No buildings found in zone. Falling back to `{DEFAULT_BUILDING_FALLBACK}`.")
    return DEFAULT_BUILDING_FALLBACK


def _run_experiments(config: Configuration, scenario: str) -> UceaRuntime:
    if config.ucea.workflow1_only:
        roof_file = _require_roof_file(scenario)
        _log(f"Workflow1-only mode enabled. Using roof file: {roof_file}")
        _clean_solar_outputs(scenario)

        pv_panel = config.ucea.pv_panel
        _run_cea_script("radiation", scenario)
        _run_cea_script(
            "photovoltaic",
            scenario,
            ["--panel-on-wall", "false", "--type-pvpanel", pv_panel],
        )

        locator = InputLocator(scenario)
        building = _select_building_for_3d(locator, config.ucea.building_3d)
        zone_pickle_dir = os.path.join(
            scenario,
            "outputs",
            "data",
            "solar-radiation",
            "radiance_geometry_pickle",
            "zone",
        )
        viewer_started = False
        try:
            _run_python_module(
                "cea.resources.radiation.workflow1_building_viewer",
                [
                    "--zone-pickle-dir",
                    zone_pickle_dir,
                ],
            )
            viewer_started = True
        except Exception as exc:
            _log(f"Could not launch interactive workflow1 building viewer: {exc}")

        return UceaRuntime(
            scenario=scenario,
            comparison_root="",
            building=building,
            output_figure="",
            viewer_started=viewer_started,
            comparison_ran=False,
        )

    comparison_root = _resolve_comparison_root(config, scenario)
    roof_file = _require_roof_file(scenario)
    pv_panel = config.ucea.pv_panel

    _run_python_module(
        "cea.resources.radiation.workflow_comparison",
        [
            "--scenario",
            scenario,
            "--roof-file",
            roof_file,
            "--comparison-root",
            comparison_root,
            "--include-geometry-pickles",
            "--clean-first",
            "--pv-panel",
            pv_panel,
            "--harmonise-pv-azimuth-convention",
        ],
    )

    _run_python_module(
        "cea.resources.radiation.workflow_metrics_report",
        [
            "--comparison-root",
            comparison_root,
            "--level",
            "both",
            "--pv-panels",
            pv_panel,
            "--undo-pv-azimuth-harmonisation",
        ],
    )

    locator = InputLocator(scenario)
    building = _select_building_for_3d(locator, config.ucea.building_3d)
    output_figure = os.path.join(comparison_root, f"{building}_workflow_geometry_comparison_3d.png")
    _run_python_module(
        "cea.resources.radiation.workflow_geometry_comparison_3d",
        [
            "--comparison-root",
            comparison_root,
            "--building",
            building,
            "--output-figure",
            output_figure,
        ],
    )

    viewer_started = False
    try:
        # Open interactive workflow1 viewer with dropdown building selection.
        _run_python_module(
            "cea.resources.radiation.workflow1_building_viewer",
            [
                "--comparison-root",
                comparison_root,
            ],
        )
        viewer_started = True
    except Exception as exc:
        _log(f"Could not launch interactive workflow1 building viewer: {exc}")

    return UceaRuntime(
        scenario=scenario,
        comparison_root=comparison_root,
        building=building,
        output_figure=output_figure,
        viewer_started=viewer_started,
        comparison_ran=True,
    )


def _open_geojson_io(config: Configuration) -> None:
    url = config.ucea.geojson_url.strip() or DEFAULT_GEOJSON_URL
    _log(f"Go to geojson.io and draw your polygon here: {url}")
    if not config.ucea.open_browser:
        _log("Automatic browser launch is disabled. Open geojson.io manually.")
        return
    try:
        webbrowser.open(url, new=2)
        _log(f"Opened map: {url}")
    except Exception as exc:
        _log(f"Could not open browser automatically: {exc}")


def _resolve_scenario(config: Configuration) -> str:
    configured_scenario = config.scenario.strip()
    if configured_scenario:
        return os.path.abspath(configured_scenario)
    _log(f"No scenario provided. Using hard-coded default scenario: {DEFAULT_UCEA_SCENARIO}")
    return os.path.abspath(DEFAULT_UCEA_SCENARIO)


def main(config: Configuration) -> None:
    scenario = _resolve_scenario(config)
    os.makedirs(scenario, exist_ok=True)
    _log(f"Initialising experiment workflow for scenario: {scenario}")

    _run_stage("Open geojson.io", lambda: _open_geojson_io(config))
    _log(
        "Draw your polygon in geojson.io, then copy the GeoJSON text from the right panel. "
        "The runner will first try clipboard input and then ask for pasted text if needed."
    )

    coordinates = _run_stage(
        "Polygon coordinate capture",
        lambda: _capture_polygon_coordinates(config.ucea.polygon_timeout_minutes),
    )
    _run_stage("Create site polygon", lambda: _prepare_site_polygon(config, coordinates))
    _run_stage("Scenario preparation scripts", lambda: _run_data_preparation(config))
    runtime = _run_stage(
        "Workflow comparison, metrics, and 3D check",
        lambda: _run_experiments(config, scenario),
    )

    _log("Run completed successfully.")
    if runtime.comparison_ran:
        _log(f"Comparison output root: {runtime.comparison_root}")
        _log(f"Scenario metrics: {os.path.join(runtime.comparison_root, 'scenario_metrics.csv')}")
        _log(f"Building metrics: {os.path.join(runtime.comparison_root, 'building_metrics.csv')}")
        _log(f"3D building used: {runtime.building}")
        _log(f"3D figure: {runtime.output_figure}")
    else:
        _log("Workflow1-only run completed (comparison and metrics were skipped).")
        _log(f"3D building selected: {runtime.building}")
    if runtime.viewer_started:
        _log("Interactive workflow1 building viewer was launched.")


if __name__ == "__main__":
    main(Configuration())
