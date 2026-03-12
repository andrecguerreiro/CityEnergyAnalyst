# Tests Module

## Main API
- `python -m pytest cea/tests/...` - Run unit and integration tests.
- `test_workflow_comparison_dashboard.py` - Validates workflow dashboard KPI extraction, delta logic, and Excel export.
- `test_workflow_comparison_snapshots.py` - Validates workflow snapshot copying behaviour for optional geometry-pickle artefacts.
- `test_workflow_geometry_comparison_3d.py` - Validates 3D geometry-comparison data loading, deterministic metadata downsampling, and plot generation smoke path.

## Key Patterns
### DO: Build synthetic fixtures in temp folders
```python
with tempfile.TemporaryDirectory() as base:
    # create minimal scenario/run/workflow folder tree and CSV fixtures
```

### DO: Keep tests deterministic and file-local
```python
write_geometry_csv(...)
write_pv_total_buildings(...)
```

### DON'T: Depend on heavy runtime simulation in unit tests
```python
# avoid running radiation / photovoltaic scripts in these tests
```

## Related Files
- `test_workflow_comparison_dashboard.py` - Dashboard reporting tests.
- `test_workflow_comparison_snapshots.py` - Workflow snapshot artefact-copy tests.
- `test_workflow_geometry_comparison_3d.py` - 3D workflow geometry-comparison tests.
- `cea/resources/radiation/workflow_comparison_dashboard.py` - Reporting utility under test.
- `cea/resources/radiation/workflow_comparison.py` - Snapshot helper under test.
- `cea/resources/radiation/workflow_geometry_comparison_3d.py` - 3D comparison utility under test.
