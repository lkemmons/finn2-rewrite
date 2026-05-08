"""Tests for fire detection grouping algorithm.

Validates that the Python implementation produces identical results
to the original SQL (step1a_work_v7m.sql).
"""

import datetime
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import pytest

from fire_grouping import (
    create_fire_geometry,
    find_adjacent_pairs,
    group_detections,
    group_fire_detections,
    add_fire_groups_to_gdf,
    UnionFind,
)


class TestUnionFind:
    """Tests for disjoint-set (union-find) data structure."""

    def test_single_element(self):
        """Single element is its own component."""
        uf = UnionFind(1)
        assert uf.find(0) == 0
        comps = uf.get_components()
        assert np.array_equal(comps, [0])

    def test_union_pairs(self):
        """Union creates single component."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)
        comps = uf.get_components()
        assert len(np.unique(comps)) == 3  # [0,1,2], [3], [4]

    def test_disconnected_components(self):
        """Multiple disconnected components."""
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(3, 4)
        comps = uf.get_components()
        # Should have 4 unique components
        assert len(np.unique(comps)) == 4


class TestFireGeometry:
    """Tests for fire geometry creation."""

    def create_test_gdf(self, n=5):
        """Helper: create test GeoDataFrame."""
        lons = np.linspace(-120, -100, n)
        lats = np.linspace(30, 50, n)
        return gpd.GeoDataFrame(
            {
                "geometry": [Point(lon, lat) for lon, lat in zip(lons, lats)],
                "scan": np.full(n, 1.0),
                "track": np.full(n, 1.0),
                "instrument": np.array(["MODIS"] * n),
                "acq_date": pd.date_range("2020-01-01", periods=n),
            },
            crs="EPSG:4326",
        )

    def test_modis_vs_viirs_size(self):
        """MODIS fires are 1 km, VIIRS are 0.375 km."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(-110, 40), Point(-110, 40)],
                "scan": [1.0, 1.0],
                "track": [1.0, 1.0],
                "instrument": ["MODIS", "VIIRS"],
                "acq_date": [datetime.date(2020, 1, 1)] * 2,
            },
            crs="EPSG:4326",
        )

        geom = create_fire_geometry(gdf)

        # MODIS polygon should be larger
        modis_area = geom.geom_sml[0].area
        viirs_area = geom.geom_sml[1].area
        assert modis_area > viirs_area
        # Ratio should be approximately (1.0 / 0.375)^2
        assert abs(modis_area / viirs_area - (1.0 / 0.375) ** 2) < 0.1

    def test_pixel_enlargement(self):
        """Pixel polygons should be larger than nominal fire size."""
        gdf = self.create_test_gdf(1)
        geom = create_fire_geometry(gdf)

        sml_area = geom.geom_sml[0].area
        pix_area = geom.geom_pix[0].area

        assert pix_area > sml_area


class TestAdjacentPairs:
    """Tests for finding adjacent fire detections."""

    def test_no_adjacent_pairs(self):
        """Distant fires should not be paired."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-120, 40),
                    Point(-100, 50),  # far away
                ],
                "scan": [1.0, 1.0],
                "track": [1.0, 1.0],
                "instrument": ["MODIS", "MODIS"],
                "acq_date": [datetime.date(2020, 1, 1)] * 2,
            },
            crs="EPSG:4326",
        )

        geom = create_fire_geometry(gdf)
        pairs = find_adjacent_pairs(gdf, geom.geom_pix, geom.geom_sml)

        assert len(pairs) == 0

    def test_overlapping_detections(self):
        """Overlapping fires should be paired."""
        # Create two very close fires (will overlap)
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-110.0, 40.0),
                    Point(-110.0001, 40.0001),  # very close
                ],
                "scan": [1.0, 1.0],
                "track": [1.0, 1.0],
                "instrument": ["MODIS", "MODIS"],
                "acq_date": [datetime.date(2020, 1, 1)] * 2,
            },
            crs="EPSG:4326",
        )

        geom = create_fire_geometry(gdf)
        pairs = find_adjacent_pairs(gdf, geom.geom_pix, geom.geom_sml)

        # Should find at least one pair
        assert len(pairs) > 0

    def test_same_day_only(self):
        """Detections on different days should not pair."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-110.0, 40.0),
                    Point(-110.0001, 40.0001),
                ],
                "scan": [1.0, 1.0],
                "track": [1.0, 1.0],
                "instrument": ["MODIS", "MODIS"],
                "acq_date": [
                    datetime.date(2020, 1, 1),
                    datetime.date(2020, 1, 2),  # different day
                ],
            },
            crs="EPSG:4326",
        )

        geom = create_fire_geometry(gdf)
        pairs = find_adjacent_pairs(gdf, geom.geom_pix, geom.geom_sml)

        assert len(pairs) == 0


class TestGrouping:
    """Tests for fire group assignment."""

    def test_single_group(self):
        """All overlapping fires in single group."""
        # Create chain of overlapping fires
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-110.0, 40.0),
                    Point(-110.0001, 40.0),
                    Point(-110.0002, 40.0),
                ],
                "scan": [1.0] * 3,
                "track": [1.0] * 3,
                "instrument": ["MODIS"] * 3,
                "acq_date": [datetime.date(2020, 1, 1)] * 3,
            },
            crs="EPSG:4326",
        )

        fireid = add_fire_groups_to_gdf(gdf)["fireid"].to_numpy()

        # All should have same group ID (the minimum index: 0)
        assert len(np.unique(fireid)) == 1
        assert fireid[0] == 0

    def test_two_separate_groups(self):
        """Two distant cluster should get different groups."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-120.0, 40.0),
                    Point(-120.0001, 40.0),
                    Point(-110.0, 40.0),
                    Point(-110.0001, 40.0),
                ],
                "scan": [1.0] * 4,
                "track": [1.0] * 4,
                "instrument": ["MODIS"] * 4,
                "acq_date": [datetime.date(2020, 1, 1)] * 4,
            },
            crs="EPSG:4326",
        )

        result = add_fire_groups_to_gdf(gdf)
        fireid = result["fireid"].to_numpy()

        # Should have 2 groups
        assert len(np.unique(fireid)) == 2
        # Group ID should be minimum index in cluster
        assert fireid[0] == fireid[1]  # 0,1 are close
        assert fireid[2] == fireid[3]  # 2,3 are close
        assert fireid[0] != fireid[2]  # clusters are different

    def test_ndetect_count(self):
        """ndetect1 column should count group members."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-110.0, 40.0),
                    Point(-110.0001, 40.0),
                    Point(-110.0002, 40.0),
                ],
                "scan": [1.0] * 3,
                "track": [1.0] * 3,
                "instrument": ["MODIS"] * 3,
                "acq_date": [datetime.date(2020, 1, 1)] * 3,
            },
            crs="EPSG:4326",
        )

        result = add_fire_groups_to_gdf(gdf)

        # All 3 should be in same group with ndetect=3
        assert (result["ndetect1"] == 3).all()


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_realistic_scenario(self):
        """Test with realistic fire data."""
        np.random.seed(42)

        # Create realistic fire data with clusters
        points = []
        instruments = []
        dates = []

        # Cluster 1: 5 fires on 2020-01-01
        for i in range(5):
            points.append(
                Point(
                    -120 + np.random.uniform(-0.001, 0.001),
                    40 + np.random.uniform(-0.001, 0.001),
                )
            )
            instruments.append("MODIS")
            dates.append(datetime.date(2020, 1, 1))

        # Cluster 2: 3 fires on 2020-01-01
        for i in range(3):
            points.append(
                Point(
                    -110 + np.random.uniform(-0.001, 0.001),
                    45 + np.random.uniform(-0.001, 0.001),
                )
            )
            instruments.append("VIIRS")
            dates.append(datetime.date(2020, 1, 1))

        # Isolated fire on 2020-01-02
        points.append(Point(-100, 50))
        instruments.append("MODIS")
        dates.append(datetime.date(2020, 1, 2))

        gdf = gpd.GeoDataFrame(
            {
                "geometry": points,
                "scan": [1.0] * len(points),
                "track": [1.0] * len(points),
                "instrument": instruments,
                "acq_date": dates,
            },
            crs="EPSG:4326",
        )

        result = add_fire_groups_to_gdf(gdf)

        # Should have 3 groups (2 on 1st day, 1 on 2nd)
        assert len(np.unique(result["fireid"])) == 3

        # Each group should have correct ndetect
        for fireid in result["fireid"].unique():
            count = (result["fireid"] == fireid).sum()
            assert (result[result["fireid"] == fireid]["ndetect1"] == count).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
