# Ultralytics Roof Export

## Main API
- `build_cea_ordered_buildings_geojson(polygon_ring_lon_lat, zone_shp_path=None) -> dict` - Load authoritative CEA footprints from scenario `zone.shp`, filter by polygon intersection, and return GeoJSON with `properties.cea_name` from `zone.shp:name`.
- `parse_polygon_text_to_ring(polygon_text) -> list[list[float]]` - Parse pasted GeoJSON text (or raw coord list) into a closed lon/lat ring.
- `run_from_geojson(polygon_geojson, building_threshold=0.15, overlap_threshold=0.25, zone_shp_path=None, output_path="roof_surfaces.geojson", write_map_output=False) -> dict` - Run inference/export directly from GeoJSON text/object (including FeatureCollection).
- `run_from_polygon_ring(polygon_ring_lon_lat, building_threshold=0.15, overlap_threshold=0.25, zone_shp_path=None, output_path="roof_surfaces.geojson", write_map_output=False) -> dict` - Run tiled inference and matching, then export `roof_surfaces.geojson` for CEA.
- `use_this_function(jsonbox, building_threshold, overlap_threshold, zone_shp_path=None, output_path="roof_surfaces.geojson", write_map_output=True, polygon_ring_lon_lat=None) -> dict` - Backward-compatible entrypoint for bbox-driven runs.
- `compute_iou_matrix(osm_buildings, predictions) -> np.ndarray` - Compute IoU matrix between footprint boxes and model predictions.
- `export_roof_surfaces_geojson(matched_data, output_path="roof_surfaces.geojson", source_crs="EPSG:3763", target_crs="EPSG:32629") -> dict` - Export one roof feature per matched roof-face tile.

## Key Patterns
### DO: Use scenario `zone.shp` as identifier source of truth
```python
zone_df = gpd.read_file(zone_shp_path).to_crs("EPSG:4326")
cea_name = row["name"]  # exact CEA name from zone.shp
```

### DO: Filter footprints by intersects(selected_polygon)
```python
selected_polygon = Polygon(ring_lon_lat)
zone_df = zone_df[zone_df.intersects(selected_polygon)]
```

### DO: Keep exactly one footprint per CEA name
```python
zone_df = zone_df.dissolve(by="name", as_index=False)
polygon = geom if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda g: g.area)
```

### DO: Keep matching IDs strict and stable
```python
cea_name = str(feature["properties"].get("cea_name", "")).strip()
if not cea_name:
    continue
matched_data.append({"building_id": cea_name, "cea_name": cea_name, ...})
```

### DO: Keep roof export IDs aligned to matched CEA names
```python
building_id = _entry_building_id(entry, fallback="unknown")
properties = {"building": building_id, "roof_id": str(roof_counter)}
```

### DO: Export roof surfaces from tiled workflow
```python
run_from_polygon_ring(
    polygon_ring_lon_lat=ring,
    zone_shp_path=r"<scenario>/inputs/building-geometry/zone.shp",
    output_path=r"<scenario>/inputs/building-geometry/roof_surfaces.geojson",
)
```

### DO: Use file-based GeoJSON for manual CLI runs (multiline-safe)
```bash
python ultralytics/fixedboxtilingextension_v2.py \
  --polygon-geojson-file area.geojson \
  --zone-shp-path "<scenario>/inputs/building-geometry/zone.shp" \
  --output-path "<scenario>/inputs/building-geometry/roof_surfaces.geojson"
```

### DON'T: Recompute or renumber building IDs in fixedbox
```python
# Avoid in this workflow:
cea_name = f"B{i + 1000}"
```

### DON'T: Fall back to bbox OSM fetch for CEA-ID-sensitive runs
```python
# Avoid:
geojson = get_osm_buildings((tl_x, tl_y), (br_x, br_y))
```

## Data Contracts
- `geojson["features"][i]["properties"]` must include:
  - `cea_name` (from `zone.shp:name`)
  - `building` (optional, derived from available zone columns)
- `matched_data` entries should include:
  - `building_id`, `cea_name`, `roof_planes`, `lines_world`, `code`, `face_data`

## Related Files
- `fixedbox_latest.py` - Active Gradio workflow using scenario `zone.shp` as ID source.
- `fixedboxtilingextension.py` - Tiled inference workflow aligned to the same `zone.shp` CEA-ID matching rules.
- `fixedboxtilingextension_v2.py` - Standalone v2 tiled inference workflow with the same CLI/API surface used by UCEA.
- `fixedboxtilingextension_v3.py` - Standalone v3 tiled inference workflow (default for UCEA) with forced roof-face orientation in `planenormal` (`ori = 0.5`) while keeping zone.shp matching and `roof_surfaces.geojson` export compatibility.
- `fixedbox_html_geojson.py` - Legacy CLI workflow (may differ from latest behaviour).
