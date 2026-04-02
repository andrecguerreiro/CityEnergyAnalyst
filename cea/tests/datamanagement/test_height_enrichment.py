import unittest

import geopandas as gpd
from shapely.geometry import Point, Polygon

from cea.datamanagement.height_enrichment import (
    apply_surroundings_floor_validity_guard,
    enrich_building_heights_from_gpkg,
    enrich_building_heights_from_points,
)


class TestHeightEnrichment(unittest.TestCase):
    def _single_building(self, height_ag=126.0, floors_ag=42, reference="CEA Assumption"):
        return gpd.GeoDataFrame(
            {
                "height_ag": [height_ag],
                "floors_ag": [floors_ag],
                "reference": [reference],
                "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            },
            crs="EPSG:3763",
        )

    def _points(self, points_with_height):
        return gpd.GeoDataFrame(
            {
                "altura(m)": [row[2] for row in points_with_height],
                "geometry": [Point(row[0], row[1]) for row in points_with_height],
            },
            crs="EPSG:3763",
        )

    def _two_buildings(self, first_height=126.0, first_floors=42, second_height=30.0, second_floors=10):
        return gpd.GeoDataFrame(
            {
                "height_ag": [first_height, second_height],
                "floors_ag": [first_floors, second_floors],
                "reference": ["CEA Assumption", "OSM - as it is"],
                "geometry": [
                    Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                    Polygon([(200, 0), (210, 0), (210, 10), (200, 10)]),
                ],
            },
            crs="EPSG:3763",
        )

    def test_inside_point_direct_match(self):
        buildings = self._single_building(height_ag=126.0, floors_ag=42)
        points = self._points([(5.0, 5.0, 9.0)])

        result = enrich_building_heights_from_points(
            buildings=buildings,
            height_points=points,
            fallback_max_distance_m=20.0,
            reference_column="reference",
        )

        self.assertEqual(result.loc[0, "height_ag"], 9.0)
        self.assertEqual(result.loc[0, "floors_ag"], 9)
        self.assertEqual(result.loc[0, "reference"], "INE")

    def test_multi_point_inside_uses_nearest_to_centroid(self):
        buildings = self._single_building(height_ag=126.0, floors_ag=42)
        points = self._points([(1.0, 1.0, 3.0), (5.1, 5.0, 12.0), (8.0, 8.0, 6.0)])

        result = enrich_building_heights_from_points(
            buildings=buildings,
            height_points=points,
            fallback_max_distance_m=20.0,
            reference_column="reference",
        )

        self.assertEqual(result.loc[0, "height_ag"], 12.0)
        self.assertEqual(result.loc[0, "floors_ag"], 12)
        self.assertEqual(result.loc[0, "reference"], "INE")

    def test_fallback_two_nearest_within_threshold(self):
        buildings = self._single_building(height_ag=30.0, floors_ag=10, reference="OSM - as it is")
        points = self._points([(12.0, 5.0, 6.0), (14.0, 5.0, 12.0)])

        result = enrich_building_heights_from_points(
            buildings=buildings,
            height_points=points,
            fallback_max_distance_m=20.0,
            reference_column="reference",
        )

        self.assertEqual(result.loc[0, "height_ag"], 9.0)
        self.assertEqual(result.loc[0, "floors_ag"], 9)
        self.assertEqual(result.loc[0, "reference"], "INE Assumption")

    def test_fallback_uses_area_median_if_second_nearest_is_too_far(self):
        buildings = self._two_buildings(first_height=126.0, first_floors=42, second_height=30.0, second_floors=10)
        points = self._points([(5.0, 5.0, 8.0), (230.0, 5.0, 6.0), (260.0, 5.0, 12.0)])

        result = enrich_building_heights_from_points(
            buildings=buildings,
            height_points=points,
            fallback_max_distance_m=20.0,
            reference_column="reference",
        )

        self.assertEqual(result.loc[0, "height_ag"], 8.0)
        self.assertEqual(result.loc[0, "reference"], "INE")
        self.assertEqual(result.loc[1, "height_ag"], 8.0)
        self.assertEqual(result.loc[1, "floors_ag"], 8)
        self.assertEqual(result.loc[1, "reference"], "INE Assumption")

    def test_fallback_rejected_if_second_nearest_is_too_far_and_no_direct_pool(self):
        buildings = self._single_building(height_ag=30.0, floors_ag=10, reference="OSM - as it is")
        points = self._points([(12.0, 5.0, 6.0), (40.0, 5.0, 12.0)])

        result = enrich_building_heights_from_points(
            buildings=buildings,
            height_points=points,
            fallback_max_distance_m=20.0,
            reference_column="reference",
        )

        self.assertEqual(result.loc[0, "height_ag"], 30.0)
        self.assertEqual(result.loc[0, "floors_ag"], 10)
        self.assertEqual(result.loc[0, "reference"], "OSM - as it is")

    def test_no_gpkg_configured_keeps_existing_values(self):
        buildings = self._single_building(height_ag=12.0, floors_ag=4, reference="OSM - as it is")

        result = enrich_building_heights_from_gpkg(
            buildings=buildings,
            building_height_gpkg=None,
            fallback_max_distance_m=20.0,
            reference_column="reference",
        )

        self.assertEqual(result.loc[0, "height_ag"], 12.0)
        self.assertEqual(result.loc[0, "floors_ag"], 4)
        self.assertEqual(result.loc[0, "reference"], "OSM - as it is")

    def test_floor_validity_guard_after_enrichment(self):
        buildings = self._single_building(height_ag=126.0, floors_ag=42)
        points = self._points([(5.0, 5.0, 6.0)])

        result = enrich_building_heights_from_points(
            buildings=buildings,
            height_points=points,
            fallback_max_distance_m=20.0,
            reference_column="reference",
        )

        self.assertEqual(result.loc[0, "height_ag"], 6.0)
        self.assertEqual(result.loc[0, "floors_ag"], 6)

    def test_surroundings_floor_validity_guard_requires_strictly_more_than_1m_per_floor(self):
        buildings = self._single_building(height_ag=3.0, floors_ag=3, reference="INE")

        result = apply_surroundings_floor_validity_guard(buildings)

        self.assertEqual(result.loc[0, "height_ag"], 3.0)
        self.assertEqual(result.loc[0, "floors_ag"], 2)


if __name__ == "__main__":
    unittest.main()
