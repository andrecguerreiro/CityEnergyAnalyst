# Ultralytics Roof Export

## Main API
- `build_cea_ordered_buildings_geojson(polygon_ring_lon_lat) -> dict` - Build CEA-ordered building footprints (`cea_name` like `B1000`) from an input polygon.
- `extract_polygon_ring_from_geojson_obj(obj) -> list[list[float]]` - Parse ring coordinates from FeatureCollection / Feature / Polygon / MultiPolygon / LineString.
- `parse_polygon_from_args() -> tuple[dict, list[list[float]]]` - Parse polygon input and return `(bbox, closed_ring_lon_lat)`.
- `compute_iou_matrix(osm_buildings, predictions) -> np.ndarray` - Compute IoU matrix between building footprints and detector boxes.
- `export_roof_surfaces_geojson(matched_data, output_path="roof.geojson", source_crs="EPSG:3763", target_crs="EPSG:32629") -> dict` - Export one roof feature per face with `properties.building` set to CEA name.

## Key Patterns
### ✅ DO: Keep CEA naming as single source of truth for roof export
```python
geojson = build_cea_ordered_buildings_geojson(polygon_ring_lon_lat)
cea_name = osm_feat["properties"]["cea_name"]
matched_data.append({"cea_name": cea_name, ...})
```

### ✅ DO: Keep IoU matching logic unchanged when only IDs are being refactored
```python
iou_mat = compute_iou_matrix(geojson["features"], buildings)
best_match_idx = np.argmax(iou_mat[i, :])
```

### ✅ DO: Export CEA names in roof features
```python
building_id = str(entry.get("cea_name", "")).strip() or "unknown"
```

### ✅ DO: Accept ucea-style GeoJSON input paths
```python
# One-argument GeoJSON text
--polygon-geojson "<FeatureCollection JSON>"

# Read JSON from stdin
Get-Content polygon.geojson | python fixedbox_html_geojson.py --polygon-stdin
```

### ✅ DO: Prompt interactively when no polygon args are provided
```python
# No args -> script asks for GeoJSON text or a file path
python fixedbox_html_geojson.py
```

### ✅ DO: Keep console output ASCII-safe and current API usage
```python
roof_prediction=np.asarray(...)
pl.show_grid(xtitle=..., ytitle=..., ztitle=...)
```

### ❌ DON'T: Write OSM IDs to `properties.building` in `roof.geojson`
```python
# Avoid
properties = {"building": osm_id}
```

### ❌ DON'T: Re-introduce bbox-only OSM fetch for matching
```python
# Avoid for ID alignment
geojson = get_osm_buildings_cached((tl_x, tl_y), (br_x, br_y))
```

## Data Contracts
- `geojson["features"][i]["properties"]` must include:
  - `cea_name` (required): sequential CEA building identifier (`B1000+`)
  - `building` (optional): building category/type
- `matched_data` entries must include:
  - `cea_name`, `roof_planes`, `lines_world`, `code`, `face_data`

## Related Files
- `fixedbox_html_geojson.py` - End-to-end roof detection, topology, matching, and GeoJSON export.
- `../cea/datamanagement/zone_helper.py` - CEA building ordering and naming pipeline (`polygon_to_zone`).
