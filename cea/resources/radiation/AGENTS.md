# Radiation Module

## Main API
- `main(config: Configuration) -> None` in `main.py` - Runs the CEA radiation pipeline.
- `geometry_main(...) -> tuple` in `geometry_generator.py` - Builds terrain/building geometry and writes `BuildingGeometry` pickles.
- `main()` in `workflow_comparison.py` - Runs workflow0/workflow1 and snapshots outputs (optional `--include-geometry-pickles`).
- `main()` in `workflow_geometry_comparison_3d.py` - 3D side-by-side workflow geometry visualiser for one building.
- `main()` in `workflow_metrics_report.py` - Generates scenario/building metrics and explicit WF deltas from workflow snapshots.
- `main(config: Configuration)` in `ucea.py` - Runs geojson.io polygon capture, scenario helpers, workflow comparison, metrics report, and 3D check in one CLI command.

## Key Patterns
### DO: Keep `BuildingGeometry` schema stable
```python
geometry_3D_zone = {
    "name": name,
    "roofs": roof_list,
    "normals_roofs": normals_roof,
    "orientation_roofs": orientation_roofs,
}
building_geometry = BuildingGeometry(**geometry_3D_zone, terrain_elevation=elevation)
```

### DO: Align custom roof Z to OSM roof level before replacement
```python
roof_list, z_shift_m = align_custom_roofs_to_default_roof_level(custom_roofs, default_roof_list)
```

### DO: Use panel-azimuth convention in metrics for flat-roof normals
```python
# workflow_metrics_report.py
# If panel tilt > 0 but (Xdir, Ydir) == (0, 0), treat panel azimuth as south (180 deg).
panel_azimuth = panel_azimuth_deg(xdir, ydir, panel_tilt_deg)
```

### DO: Use `surface_azimuth_deg` for non-flat panel direction metrics
```python
# workflow_metrics_report.py
# Keep metrics aligned with the PV pipeline azimuth frame.
panel_azimuth = panel_azimuth_deg(xdir, ydir, panel_tilt_deg, surface_azimuth_deg=surface_azimuth_deg)
```

### DO: Use detrended shading metrics for WF comparability
```python
# workflow_metrics_report.py
# Raw shading spread mixes shading and geometry/orientation effects.
# Use detrended metrics for cross-workflow comparability.
shading_cv_aw_detrended
shading_p90_p10_ratio_detrended
```

### DO: Undo temporary PV azimuth harmonisation when binning panel directions
```python
# workflow_metrics_report.py
# Use this when workflow_comparison ran with --harmonise-pv-azimuth-convention.
python -m cea.resources.radiation.workflow_metrics_report \
  --comparison-root <...> --undo-pv-azimuth-harmonisation
```

### DO: Apply the same azimuth-undo logic to surface orientation shares
```python
# workflow_metrics_report.py
# Keep aw_surface_azimuth_deg / orientation_share_* consistent with panel orientation metrics.
surface_azimuth = surface_azimuth_deg_adjusted(surface_azimuth_deg, surface_tilt_deg)
```

### DO: Apply temporary metadata transforms only around PV runs
```python
# workflow_comparison.py
backups = apply_temp_pv_azimuth_convention_harmonisation(scenario)
run_cea_script("photovoltaic", scenario, ...)
restore_geometry_metadata_from_backups(backups)
```

### DO: Require XYZ for custom roof polygons
```python
coords = list(poly.exterior.coords)
if len(coords[0]) < 3:
    return None
```

### DO: Project custom roof vertices to one plane before OCC face creation
```python
planar_points = _project_points_to_best_fit_plane(raw_points)
points = planar_points + [planar_points[0]]
face = construct.make_polygon(points)
```

### DON'T: Change metadata columns written by `daysim.py`
```python
pd.DataFrame({"Xdir": ..., "Ydir": ..., "Zdir": ..., "TYPE": ...})
```

## Data Flow
- `zone/surroundings/terrain/envelope` -> `geometry_generator.geometry_main`.
- Geometry pickles: `outputs/data/solar-radiation/radiance_geometry_pickle`.
- `daysim.py` reads pickles and writes `<building>_geometry.csv`, `<building>_radiation.csv`, optional feather outputs.
- `workflow_comparison.py` snapshots per-workflow outputs under `roof-workflow-comparison`.
- `workflow_metrics_report.py` reads `roof-workflow-comparison` snapshots and writes:
  - `scenario_metrics.csv` / `scenario_deltas.csv`
  - `building_metrics.csv` / `building_deltas.csv`
  - `building_metrics.csv` includes `building_height_m` estimated from workflow `*_geometry.csv` (`max roof Zcoor - min underside Zcoor`; fallback: `max roof Zcoor - terrain_elevation`).
  - `panel_placement_summary.csv` (building/panel-type direction + tilt + approximate panel counts and directional energy split)
  - Includes tilt, azimuth, and orientation-share descriptors:
    - surface: `aw_surface_tilt_deg`, `aw_surface_azimuth_deg`, `orientation_share_*`
    - panel: `aw_panel_tilt_deg_area`, `aw_panel_tilt_deg_module`, `aw_panel_azimuth_deg_module`, `panel_orientation_share_*`
  - Includes both raw and detrended shading spread metrics:
    - raw: `shading_cv_aw`, `shading_p90_p10_Whm2`
    - detrended: `shading_cv_aw_detrended`, `shading_p90_p10_ratio_detrended`
- `ucea.py` orchestrates:
  - opens geojson.io and captures polygon from clipboard or pasted input
  - runs `create-polygon` (`site.shp`)
  - runs helper scripts (`database/zone/terrain/weather/archetypes`) and runs `surroundings-helper` only when `ucea:run-surroundings-helper` is `true`
  - runs `workflow_comparison.py` -> `workflow_metrics_report.py` -> `workflow_geometry_comparison_3d.py`

## Custom Roof Input Contract
- File: `inputs/building-geometry/roof_surfaces.geojson`
- Required columns:
  - `building` matching zone `name`
  - `geometry` as Polygon/MultiPolygon with XYZ coordinates
- CRS must match or be reprojectable to `zone_df.crs`.

## Related Files
- `geometry_generator.py` - Custom roof loading/injection and Z alignment.
- `daysim.py` - Sensor generation and simulation execution.
- `radiance.py` - Radiance geometry/material writing.
- `main.py` - Radiation entry point.
- `workflow_comparison.py` - Two-workflow runner and snapshot utility.
- `workflow_geometry_comparison_3d.py` - Side-by-side 3D geometry comparison.
- `workflow1_geometry_generator_documentation.txt` - Workflow1 reference notes.
- `workflow_metrics_report.py` - Scenario/building KPI and delta report utility.

## Script Defaults
- `workflow_comparison.py` and `workflow_geometry_comparison_3d.py` support editable in-script path defaults for local runs.
- `workflow_comparison.py` runs `photovoltaic` with `--panel-on-wall false` and defaults to `--pv-panel PV1` (override with `--pv-panel`).
- `workflow_comparison.py` runs with PV azimuth harmonisation enabled by default (rotate sloped roof metadata normals by 180° in XY before each PV run; metadata restored afterwards).
- Use `--no-harmonise-pv-azimuth-convention` to disable the temporary harmonisation path.
- `workflow_comparison.py` cleanup is Windows-lock tolerant: if `comparison-root` cannot be removed because files are open, it retries and then archives the locked folder with `.locked_<timestamp>`.
- `workflow_metrics_report.py` defaults to `--pv-panels PV1` (override with `--pv-panels`).
- `workflow_metrics_report.py` supports `--undo-pv-azimuth-harmonisation` to rotate non-flat panel-direction bins by +180° when PV snapshots were produced with temporary azimuth harmonisation.
- `workflow_metrics_report.py` write step is Windows-lock tolerant: if a CSV target is open, it writes to `<name>.locked_<timestamp>.csv` and continues.
- `ucea.py` defaults to opening geojson.io URL `https://geojson.io/#map=18.2/38.708267/-9.138085`.
- `ucea.py` writes outputs to `<scenario>/outputs/data/roof-workflow-comparison` unless `ucea:comparison-root` is set.
- `ucea.py` defaults to `ucea:run-surroundings-helper = true`; set it to `false` to preserve manually edited `inputs/geometry/surroundings.shp`.
- `ucea.py` auto-creates `inputs/building-geometry/roof_surfaces.geojson` with a temporary default payload if the file is missing.
- The default hardcoded roof in `ucea.py` is intentionally planar (one corrected vertex Z) to avoid OCC null-face assertions in custom roof loading.
- `ucea.py` runs `workflow_comparison.py` with `--clean-first` to avoid stale `solar-radiation` / `potentials/solar` artefacts leaking into metrics.
- `ucea.py` passes `--harmonise-pv-azimuth-convention` to `workflow_comparison.py` for PV-convention diagnostics while keeping snapshot/3D metadata restored.
- `ucea.py` passes `--undo-pv-azimuth-harmonisation` to `workflow_metrics_report.py` so WF panel-direction metrics reflect physical roof orientation after harmonised PV runs.
- Keep Matplotlib keyword tokens API-valid (example: `loc="lower center"`).
