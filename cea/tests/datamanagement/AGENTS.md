# Data Management Tests

## Main API
- `TestHeightEnrichment` in `test_height_enrichment.py` - Validates INE point-to-building height enrichment rules.
- `TestCleanGeometries` in `test_zone_helper.py` - Validates zone geometry cleaning behaviour.
- `TestAssignAttributes` in `test_zone_helper.py` - Guards CEA default `building:levels` assignment (3 floors per building, not scaled by row count).

## Key Patterns
### DO: Build deterministic geometry fixtures
```python
buildings = gpd.GeoDataFrame(...)
points = gpd.GeoDataFrame(...)
```

### DO: Test matching rules in isolation
```python
enrich_building_heights_from_points(...)
```

### DO: Cover fallback and guard behaviour
```python
# direct match, nearest-inside selection, 2-nearest fallback threshold,
# and area-median fallback from direct INE matches,
# and floor validity guards:
# - general: height_ag >= floors_ag
# - surroundings: height_ag / floors_ag > 1.0
```

### DON'T: Depend on live OSM calls in unit tests
```python
# Keep these tests independent from network and osmnx queries.
```

## Related Files
- `test_height_enrichment.py` - Shared enrichment logic tests.
- `test_zone_helper.py` - Existing geometry-cleaning regression tests.
- `../../datamanagement/height_enrichment.py` - Module under test.
