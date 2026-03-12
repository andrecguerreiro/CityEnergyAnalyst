# Radiation Module

## Main API
- `main(config: Configuration) -> None` in `main.py` - Runs the CEA radiation pipeline.
- `geometry_main(...) -> tuple` in `geometry_generator.py` - Builds terrain/building geometry and writes `BuildingGeometry` pickles.
- `main()` in `remap_roof_metadata.py` - Remaps roof metadata for workflow2 without rerunning radiation.
- `main()` in `workflow_comparison.py` - Runs workflow0/workflow1/workflow2 and snapshots outputs (optional `--include-geometry-pickles`).
- `main()` in `workflow_geometry_comparison_3d.py` - 3D side-by-side workflow geometry visualiser for one building.
- `main()` in `workflow_metrics_report.py` - Generates scenario/building metrics and explicit WF deltas from workflow snapshots.

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

### DO: Require XYZ for custom roof polygons
```python
coords = list(poly.exterior.coords)
if len(coords[0]) < 3:
    return None
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
  - `panel_placement_summary.csv` (building/panel-type direction + tilt-bin + approximate panel counts and directional energy split)
  - Includes both surface and panel descriptors:
    - surface: `aw_surface_tilt_deg`, `orientation_share_*`
    - panel: `aw_panel_tilt_deg_area`, `aw_panel_tilt_deg_module`, `aw_panel_azimuth_deg_*`, `panel_orientation_share_*`

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
- `remap_roof_metadata.py` - Metadata-only roof remap.
- `workflow_comparison.py` - Three-workflow runner and snapshot utility.
- `workflow_geometry_comparison_3d.py` - Side-by-side 3D geometry comparison.
- `workflow1_geometry_generator_documentation.txt` - Workflow1 reference notes.
- `workflow2_remap_roof_metadata_documentation.txt` - Workflow2 reference notes.
- `workflow_metrics_report.py` - Scenario/building KPI and delta report utility.

## Script Defaults
- `workflow_comparison.py` and `workflow_geometry_comparison_3d.py` support editable in-script path defaults for local runs.
- `workflow_comparison.py` runs `photovoltaic` with `--panel-on-wall false` and defaults to `--pv-panel PV1` (override with `--pv-panel`).
- `workflow_metrics_report.py` defaults to `--pv-panels PV1` (override with `--pv-panels`).
- Keep Matplotlib keyword tokens API-valid (example: `loc="lower center"`).
