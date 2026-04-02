import unittest

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from shapely import Point, Polygon

from cea.datamanagement import zone_helper

class TestCleanGeometries(unittest.TestCase):

    def assertGeoDataFrameEqual(self, a, b, msg, *kwargs):
        try:
            assert_geodataframe_equal(a, b, *kwargs)
        except AssertionError as e:
            raise self.failureException(msg) from e
    
    def setUp(self):
        self.addTypeEqualityFunc(gpd.GeoDataFrame, self.assertGeoDataFrameEqual)

    def test_clean_geometries(self):
        raw_geometries = gpd.GeoDataFrame(
            {
                "geometry": [Point(0,0), Polygon([(0,0), (0,1), (1,0)])]
            }
        )
        expected_output = gpd.GeoDataFrame(
            {
                "geometry": [Polygon([(0,0), (0,1), (1,0)])]
            }
        )
        output = zone_helper.clean_geometries(raw_geometries).reset_index(drop=True)
        self.assertEqual(output, expected_output)


class TestAssignAttributes(unittest.TestCase):
    def test_assign_attributes_uses_three_levels_per_building_for_cea_assumption(self):
        shapefile = gpd.GeoDataFrame(
            {
                "building": ["residential", "residential", "residential", "residential"],
                "geometry": [
                    Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                    Polygon([(2, 0), (2, 1), (3, 1), (3, 0)]),
                    Polygon([(4, 0), (4, 1), (5, 1), (5, 0)]),
                    Polygon([(6, 0), (6, 1), (7, 1), (7, 0)]),
                ],
            }
        )

        output = zone_helper.assign_attributes(
            shapefile=shapefile,
            buildings_height=None,
            buildings_floors=None,
            buildings_height_below_ground=3,
            buildings_floors_below_ground=1,
            key="B",
        )

        self.assertTrue((output["reference"] == "CEA Assumption").all())
        self.assertTrue((output["floors_ag"] == 3).all())
        self.assertTrue((output["height_ag"] == 9.0).all())
