"""Optimized fire detection grouping using spatial indexing and union-find.

This module replaces the SQL-based grouping (step1a_work_v7m.sql) with a pure
Python implementation using GeoPandas and spatial indexing for performance.

Key algorithms:
    1. Spatial indexing with STRtree for fast overlap detection
    2. Union-find (disjoint set) for efficient component labeling
    3. Batch processing by acquisition date
    4. Optional parallel processing by date
"""

from __future__ import annotations

import logging
from typing import NamedTuple
from multiprocessing import Pool
import functools

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box, Polygon
from shapely.strtree import STRtree

log = logging.getLogger(__name__)


class FireGeometry(NamedTuple):
    """Container for fire polygon geometries."""

    geom_sml: np.ndarray  # nominal fire size polygon
    geom_pix: np.ndarray  # pixel footprint polygon


class UnionFind:
    """Disjoint-set data structure for efficient grouping.

    Replaces the PostgreSQL pnt2grp() aggregate function.

    Uses path compression and union by rank for nearly O(1) operations.
    """

    def __init__(self, n: int):
        """Initialize with n elements (0 to n-1)."""
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.uint8)

    def find(self, x: int) -> int:
        """Find root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union two elements by rank. Returns True if merged."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def get_components(self) -> np.ndarray:
        """Get component ID (root) for each element."""
        # Compress all paths and return roots
        roots = np.array([self.find(i) for i in range(len(self.parent))])
        # Map roots to smallest ID in each component
        unique_roots = np.unique(roots)
        root_to_min = {r: r for r in unique_roots}
        component_ids = np.array(
            [root_to_min[roots[i]] for i in range(len(roots))]
        )
        return component_ids


# ---------------------------------------------------------------------------
# Fire Geometry Creation
# ---------------------------------------------------------------------------


def create_fire_geometry(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = "scan",
    track_col: str = "track",
    instrument_col: str = "instrument",
) -> FireGeometry:
    """
    Create fire polygon geometries from detection metadata.

    Replaces the SQL UPDATE statements that create geom_sml and geom_pix.

    Parameters
    ----------
    fire_gdf : GeoDataFrame with fire points (geometry column must be Points)
    scan_col : column name for scan dimension (km)
    track_col : column name for track dimension (km)
    instrument_col : column name for instrument type ('MODIS' or 'VIIRS')

    Returns
    -------
    FireGeometry with geom_sml and geom_pix arrays
    """
    lons = fire_gdf.geometry.x.to_numpy()
    lats = fire_gdf.geometry.y.to_numpy()
    scan = fire_gdf[scan_col].to_numpy()
    track = fire_gdf[track_col].to_numpy()
    instrument = fire_gdf[instrument_col].to_numpy()

    # Constants from SQL
    EARTH_CIRC = 2 * np.pi * 6370.997  # km
    PIXFAC = 1.1

    # Fire size: 1.0 km for MODIS, 0.375 km for VIIRS
    fire_size = np.where(instrument == "MODIS", 1.0, 0.375)

    # Convert km to degrees
    lat_rad = np.radians(lats)
    cos_lat = np.cos(lat_rad)

    fire_dx = 0.5 * fire_size * 360.0 / EARTH_CIRC / cos_lat
    fire_dy = 0.5 * fire_size * 360.0 / EARTH_CIRC

    pix_dx = PIXFAC * 0.5 * scan * 360.0 / EARTH_CIRC / cos_lat
    pix_dy = PIXFAC * 0.5 * track * 360.0 / EARTH_CIRC

    # Create bounding box polygons
    geom_sml = np.array(
        [
            box(lon - dx, lat - dy, lon + dx, lat + dy)
            for lon, lat, dx, dy in zip(lons, lats, fire_dx, fire_dy)
        ]
    )

    geom_pix = np.array(
        [
            box(lon - dx, lat - dy, lon + dx, lat + dy)
            for lon, lat, dx, dy in zip(lons, lats, pix_dx, pix_dy)
        ]
    )

    return FireGeometry(geom_sml=geom_sml, geom_pix=geom_pix)


# ---------------------------------------------------------------------------
# Adjacency Detection (replaces tbl_adj_det)
# ---------------------------------------------------------------------------


def find_adjacent_pairs(
    fire_gdf: gpd.GeoDataFrame,
    geom_pix: np.ndarray,
    geom_sml: np.ndarray,
    date_col: str = "acq_date",
) -> list[tuple[int, int]]:
    """
    Find pairs of detections with overlapping pixel footprints.

    Replaces the SQL `tbl_adj_det` table creation using spatial join.
    Uses STRtree spatial index for O(n log n) performance.

    Parameters
    ----------
    fire_gdf : GeoDataFrame of fire detections (must have cleanid or index)
    geom_pix : array of pixel footprint geometries
    geom_sml : array of nominal fire size geometries
    date_col : column for grouping by date

    Returns
    -------
    List of (i, j) pairs where i < j, indicating adjacent detections
    """
    pairs = []

    # Process each date separately (per SQL logic)
    for date in fire_gdf[date_col].unique():
        day_mask = fire_gdf[date_col] == date
        day_indices = np.where(day_mask)[0]

        if len(day_indices) < 2:
            continue

        # Get geometries for this day
        day_pix_geoms = geom_pix[day_indices]
        day_sml_geoms = geom_sml[day_indices]

        # Build STRtree for spatial index
        tree = STRtree(day_pix_geoms)

        # For each detection, find candidates using spatial index
        for local_i, global_i in enumerate(day_indices):
            # Query index for potential overlaps (using envelope)
            candidates = tree.query(day_pix_geoms[local_i], predicate="intersects")

            for local_j in candidates:
                if local_i >= local_j:
                    continue  # Only consider i < j

                global_j = day_indices[local_j]

                # Verify actual intersection (not just envelope)
                if day_pix_geoms[local_i].intersects(day_pix_geoms[local_j]):
                    # Optional: verify distance criteria from SQL
                    # This is implicit in the intersection check
                    pairs.append((global_i, global_j))

    return pairs


# ---------------------------------------------------------------------------
# Grouping via Union-Find (replaces pnt2grp())
# ---------------------------------------------------------------------------


def group_detections(
    fire_gdf: gpd.GeoDataFrame,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """
    Assign group IDs using union-find algorithm.

    Replaces the PostgreSQL pnt2grp() aggregate function which performs
    connected-component analysis on the adjacency pairs.

    Parameters
    ----------
    fire_gdf : GeoDataFrame with fire detections
    pairs : list of (i, j) index pairs from find_adjacent_pairs()

    Returns
    -------
    Array of group IDs (fireid), one per detection
        Each group ID is the minimum cleanid/index in that group
    """
    n = len(fire_gdf)
    uf = UnionFind(n)

    # Union all pairs
    for i, j in pairs:
        uf.union(i, j)

    # Get component roots
    components = uf.get_components()

    # Map each component root to its minimum member
    component_min = {}
    for i in range(n):
        root = components[i]
        if root not in component_min:
            component_min[root] = i
        else:
            component_min[root] = min(component_min[root], i)

    # Final group IDs: map each detection to minimum in its component
    fireid = np.array([component_min[components[i]] for i in range(n)])

    return fireid


# ---------------------------------------------------------------------------
# Complete Grouping Pipeline (Sequential)
# ---------------------------------------------------------------------------


def group_fire_detections(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = "scan",
    track_col: str = "track",
    instrument_col: str = "instrument",
    date_col: str = "acq_date",
    verbose: bool = True,
) -> tuple[np.ndarray, FireGeometry]:
    """
    Complete pipeline: create geometries, find pairs, and group.

    This is the pure Python replacement for steps in step1a_work_v7m.sql:
        1. CREATE geom_sml, geom_pix
        2. CREATE tbl_adj_det (find overlapping pixel pairs)
        3. SELECT pnt2grp() (group connected detections)
        4. UPDATE work_pnt (assign fireid1, ndetect1)

    Parameters
    ----------
    fire_gdf : GeoDataFrame of fire detections with columns:
        - geometry (Point in WGS84)
        - scan_col (float, km)
        - track_col (float, km)
        - instrument_col (str, 'MODIS' or 'VIIRS')
        - date_col (date-like)
    scan_col, track_col, instrument_col, date_col : column names
    verbose : if True, print progress messages

    Returns
    -------
    (fireid, geom) : tuple of
        - fireid : array of group IDs
        - geom : FireGeometry with geom_sml and geom_pix
    """
    if verbose:
        log.info(f"Grouping {len(fire_gdf)} detections...")

    # Step 1: Create geometries
    if verbose:
        log.info("Creating fire geometries...")
    geom = create_fire_geometry(
        fire_gdf,
        scan_col=scan_col,
        track_col=track_col,
        instrument_col=instrument_col,
    )

    # Step 2: Find adjacent pairs
    if verbose:
        log.info("Finding adjacent detection pairs...")
    pairs = find_adjacent_pairs(
        fire_gdf,
        geom.geom_pix,
        geom.geom_sml,
        date_col=date_col,
    )

    if verbose:
        log.info(f"Found {len(pairs)} adjacent pairs")

    # Step 3: Group via union-find
    if verbose:
        log.info("Assigning group IDs...")
    fireid = group_detections(fire_gdf, pairs)

    n_groups = len(np.unique(fireid))
    if verbose:
        log.info(f"Grouped into {n_groups} fire groups")

    return fireid, geom


# ---------------------------------------------------------------------------
# Parallel Processing by Date
# ---------------------------------------------------------------------------


def _group_one_day(
    date_group_tuple: tuple,
    scan_col: str = "scan",
    track_col: str = "track",
    instrument_col: str = "instrument",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Worker function for parallel date processing.

    Parameters
    ----------
    date_group_tuple : (date, day_gdf) from groupby
    scan_col, track_col, instrument_col : column names

    Returns
    -------
    (indices, fireid_array) for this day
    """
    date, day_gdf = date_group_tuple
    day_gdf = day_gdf.reset_index(drop=True)

    # Create geometries for this day
    geom = create_fire_geometry(
        day_gdf,
        scan_col=scan_col,
        track_col=track_col,
        instrument_col=instrument_col,
    )

    # Find pairs within this day
    pairs = []
    if len(day_gdf) >= 2:
        tree = STRtree(geom.geom_pix)
        for local_i in range(len(day_gdf)):
            candidates = tree.query(geom.geom_pix[local_i], predicate="intersects")
            for local_j in candidates:
                if local_i < local_j and geom.geom_pix[local_i].intersects(
                    geom.geom_pix[local_j]
                ):
                    pairs.append((local_i, local_j))

    # Group detections for this day
    fireid_day = group_detections(day_gdf, pairs)

    return day_gdf.index.to_numpy(), fireid_day


def group_fire_detections_parallel(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = "scan",
    track_col: str = "track",
    instrument_col: str = "instrument",
    date_col: str = "acq_date",
    n_workers: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, FireGeometry]:
    """
    Parallel grouping by processing each date independently.

    Parameters
    ----------
    fire_gdf : GeoDataFrame of fire detections
    scan_col, track_col, instrument_col, date_col : column names
    n_workers : number of parallel workers (default: CPU count)
    verbose : if True, print progress messages

    Returns
    -------
    (fireid, geom) : same as group_fire_detections()
    """
    if verbose:
        log.info(f"Grouping {len(fire_gdf)} detections (parallel by date)...")

    # Create geometries once for all dates
    if verbose:
        log.info("Creating fire geometries...")
    geom = create_fire_geometry(
        fire_gdf,
        scan_col=scan_col,
        track_col=track_col,
        instrument_col=instrument_col,
    )

    # Group by date for parallel processing
    date_groups = list(fire_gdf.groupby(date_col))
    n_dates = len(date_groups)

    if verbose:
        log.info(f"Processing {n_dates} dates in parallel...")

    # Process each date in parallel
    with Pool(n_workers) as pool:
        worker_fn = functools.partial(
            _group_one_day,
            scan_col=scan_col,
            track_col=track_col,
            instrument_col=instrument_col,
        )
        results = pool.map(worker_fn, date_groups)

    # Merge results
    fireid = np.zeros(len(fire_gdf), dtype=np.int64)
    for indices, fireid_day in results:
        fireid[indices] = fireid_day

    n_groups = len(np.unique(fireid))
    if verbose:
        log.info(f"Grouped into {n_groups} fire groups")

    return fireid, geom


# ---------------------------------------------------------------------------
# Integration with GeoDataFrame
# ---------------------------------------------------------------------------


def add_fire_groups_to_gdf(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = "scan",
    track_col: str = "track",
    instrument_col: str = "instrument",
    date_col: str = "acq_date",
    parallel: bool = False,
    n_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Add fireid, geom_sml, geom_pix, ndetect1 columns to fire GeoDataFrame.

    Convenience function that adds grouping results back to the input GeoDataFrame.

    Parameters
    ----------
    fire_gdf : GeoDataFrame to augment
    scan_col, track_col, instrument_col, date_col : column names
    parallel : if True, use parallel processing by date
    n_workers : number of parallel workers (default: CPU count)

    Returns
    -------
    GeoDataFrame with new columns: fireid, geom_sml, geom_pix, ndetect1
    """
    result = fire_gdf.copy()

    # Get grouping results
    if parallel:
        fireid, geom = group_fire_detections_parallel(
            result,
            scan_col=scan_col,
            track_col=track_col,
            instrument_col=instrument_col,
            date_col=date_col,
            n_workers=n_workers,
        )
    else:
        fireid, geom = group_fire_detections(
            result,
            scan_col=scan_col,
            track_col=track_col,
            instrument_col=instrument_col,
            date_col=date_col,
        )

    # Add to GeoDataFrame
    result["fireid"] = fireid
    result["geom_sml"] = geom.geom_sml
    result["geom_pix"] = geom.geom_pix

    # Calculate ndetect1 (count of detections per group)
    ndetect1 = result.groupby("fireid").size()
    result["ndetect1"] = result["fireid"].map(ndetect1)

    return result
