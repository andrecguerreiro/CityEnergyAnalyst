"""
Run an end-to-end UX workflow for roof-comparison experiments.
"""

from __future__ import annotations

import json
import os
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
DEFAULT_GEOJSON_URL = "https://geojson.io/#map=18.2/38.708267/-9.138085"
ROOF_RELATIVE_PATH = os.path.join("inputs", "building-geometry", "roof_surfaces.geojson")
DEFAULT_ROOF_SURFACES_GEOJSON: dict[str, Any] = {
    "type": "FeatureCollection",
    "name": "roof_surfaces",
    "crs": {"type": "name", "properties": {"name": "EPSG:32629"}},
    "features": [
        {
            "type": "Feature",
            "properties": {"building": "B1000", "roof_id": "r_WSW_10"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [488013.263, 4284392.486, 74.5],
                        [487990.108, 4284385.149, 70.0],
                        [487974.198, 4284437.936, 69.787491],
                        [487998.547, 4284445.259, 74.5],
                        [488013.263, 4284392.486, 74.5],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"building": "B1000", "roof_id": "r_ENE_10"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [487998.547, 4284445.259, 74.5],
                        [488021.804, 4284452.254, 70.0],
                        [488037.716, 4284400.235, 69.748958],
                        [488013.263, 4284392.486, 74.5],
                        [487998.547, 4284445.259, 74.5],
                    ]
                ],
            },
        },
    ],
}


class UceaStageError(RuntimeError):
    """Raised when a stage fails."""


@dataclass
class UceaRuntime:
    scenario: str
    comparison_root: str
    building: str
    output_figure: str


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
    _call_api("database_helper", config, databases_path=config.ucea.database_path)
    _call_api("zone_helper", config)
    _call_api("surroundings_helper", config, buffer=config.ucea.surroundings_buffer_m)
    _call_api("terrain_helper", config, buffer=config.ucea.terrain_buffer_m)
    _call_api("weather_helper", config, weather=config.ucea.weather_source)
    _call_api("archetypes_mapper", config)


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
            "--pv-panel",
            pv_panel,
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
    return UceaRuntime(
        scenario=scenario,
        comparison_root=comparison_root,
        building=building,
        output_figure=output_figure,
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


def main(config: Configuration) -> None:
    scenario = os.path.abspath(config.scenario)
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
    _log(f"Comparison output root: {runtime.comparison_root}")
    _log(f"Scenario metrics: {os.path.join(runtime.comparison_root, 'scenario_metrics.csv')}")
    _log(f"Building metrics: {os.path.join(runtime.comparison_root, 'building_metrics.csv')}")
    _log(f"3D building used: {runtime.building}")
    _log(f"3D figure: {runtime.output_figure}")


if __name__ == "__main__":
    main(Configuration())
