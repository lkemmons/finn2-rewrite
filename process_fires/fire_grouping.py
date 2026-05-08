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
from functools import partial

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
    Uses path compression and union by rank for near-linear performance.
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
        component_ids = np.array([root_to_min[roots[i]] for i in range(len(roots))])
        return component_ids


# ---------------------------------------------------------------------------
# Fire Geometry Creation
# ---------------------------------------------------------------------------

def create_fire_geometry(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = 'scan',
    track_col: str = 'track',
    instrument_col: str = 'instrument',
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
    fire_size = np.where(instrument == 'MODIS', 1.0, 0.375)

    # Convert km to degrees
    lat_rad = np.radians(lats)
    cos_lat = np.cos(lat_rad)

    fire_dx = 0.5 * fire_size * 360.0 / EARTH_CIRC / cos_lat
    fire_dy = 0.5 * fire_size * 360.0 / EARTH_CIRC

    pix_dx = PIXFAC * 0.5 * scan * 360.0 / EARTH_CIRC / cos_lat
    pix_dy = PIXFAC * 0.5 * track * 360.0 / EARTH_CIRC

    # Create bounding box polygons
    geom_sml = np.array([
        box(lon - dx, lat - dy, lon + dx, lat + dy)
        for lon, lat, dx, dy in zip(lons, lats, fire_dx, fire_dy)
    ])

    geom_pix = np.array([
        box(lon - dx, lat - dy, lon + dx, lat + dy)
        for lon, lat, dx, dy in zip(lons, lats, pix_dx, pix_dy)
    ])

    return FireGeometry(geom_sml=geom_sml, geom_pix=geom_pix)


# ---------------------------------------------------------------------------
# Adjacency Detection (replaces tbl_adj_det)
# ---------------------------------------------------------------------------

def find_adjacent_pairs(
    fire_gdf: gpd.GeoDataFrame,
    geom_pix: np.ndarray,
    geom_sml: np.ndarray,
    date_col: str = 'acq_date',
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
            candidates = tree.query(
                day_pix_geoms[local_i],
                predicate='intersects'
            )

            for local_j in candidates:
                if local_i >= local_j:
                    continue  # Only consider i < j

                global_j = day_indices[local_j]

                # Verify actual intersection (not just envelope)
                if day_pix_geoms[local_i].intersects(day_pix_geoms[local_j]):
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
# Parallel Processing by Date
# ---------------------------------------------------------------------------

def _group_single_date(
    day_data: tuple[pd.Timestamp | pd.Timestamp, gpd.GeoDataFrame, FireGeometry]
) -> tuple[pd.Timestamp, np.ndarray]:
    """
    Group detections for a single date (for parallel processing).

    Parameters
    ----------
    day_data : tuple of (date, fire_gdf, geom)

    Returns
    -------
    (date, fireid_array) for this date
    """
    date, gdf_day, geom = day_data

    # Find adjacent pairs for this date only
    pairs = find_adjacent_pairs(
        gdf_day,
        geom.geom_pix,
        geom.geom_sml,
        date_col='acq_date',
    )

    # Group detections
    fireid = group_detections(gdf_day, pairs)

    return date, fireid


def group_fire_detections_parallel(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = 'scan',
    track_col: str = 'track',
    instrument_col: str = 'instrument',
    date_col: str = 'acq_date',
    n_workers: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, FireGeometry]:
    """
    Group detections in parallel by date.

    Uses multiprocessing to process each date independently, then combines results.
    Useful for large datasets spanning many dates.

    Parameters
    ----------
    fire_gdf : GeoDataFrame of fire detections
    scan_col, track_col, instrument_col, date_col : column names
    n_workers : number of processes (default: CPU count)
    verbose : if True, print progress messages

    Returns
    -------
    (fireid, geom) : tuple of
        - fireid : array of group IDs (with global indexing)
        - geom : FireGeometry with geom_sml and geom_pix
    """
    if verbose:
        log.info(f"Grouping {len(fire_gdf)} detections in parallel...")

    # Create geometries once
    if verbose:
        log.info("Creating fire geometries...")
    geom = create_fire_geometry(
        fire_gdf,
        scan_col=scan_col,
        track_col=track_col,
        instrument_col=instrument_col,
    )

    # Prepare data for parallel processing
    dates = fire_gdf[date_col].unique()
    if verbose:
        log.info(f"Processing {len(dates)} dates...")

    worker_data = []
    date_to_indices = {}

    for date in dates:
        mask = fire_gdf[date_col] == date
        indices = np.where(mask)[0]
        date_to_indices[date] = indices

        gdf_day = fire_gdf[mask].copy()
        geom_day = FireGeometry(
            geom_sml=geom.geom_sml[indices],
            geom_pix=geom.geom_pix[indices],
        )

        worker_data.append((date, gdf_day, geom_day))

    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(_group_single_date, worker_data)

    # Combine results with original global indexing
    fireid = np.zeros(len(fire_gdf), dtype=np.int64)

    for date, day_fireid in results:
        indices = date_to_indices[date]

        # Map local fireid back to global indices
        # Group IDs need to be offset per date to avoid collisions
        date_offset = indices.min()

        for local_idx, global_idx in enumerate(indices):
            global_group_id = indices[day_fireid[local_idx]]
            fireid[global_idx] = global_group_id

    n_groups = len(np.unique(fireid))
    if verbose:
        log.info(f"Grouped into {n_groups} fire groups")

    return fireid, geom


# ---------------------------------------------------------------------------
# Complete Grouping Pipeline
# ---------------------------------------------------------------------------

def group_fire_detections(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = 'scan',
    track_col: str = 'track',
    instrument_col: str = 'instrument',
    date_col: str = 'acq_date',
    parallel: bool = False,
    n_workers: int | None = None,
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
    parallel : if True, process dates in parallel
    n_workers : number of processes for parallel mode
    verbose : if True, print progress messages

    Returns
    -------
    (fireid, geom) : tuple of
        - fireid : array of group IDs
        - geom : FireGeometry with geom_sml and geom_pix
    """
    if parallel:
        return group_fire_detections_parallel(
            fire_gdf,
            scan_col=scan_col,
            track_col=track_col,
            instrument_col=instrument_col,
            date_col=date_col,
            n_workers=n_workers,
            verbose=verbose,
        )

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
# Integration with GeoDataFrame
# ---------------------------------------------------------------------------

def add_fire_groups_to_gdf(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = 'scan',
    track_col: str = 'track',
    instrument_col: str = 'instrument',
    date_col: str = 'acq_date',
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
    parallel : if True, process dates in parallel
    n_workers : number of processes for parallel mode

    Returns
    -------
    GeoDataFrame with new columns: fireid, geom_sml, geom_pix, ndetect1
    """
    result = fire_gdf.copy()

    # Get grouping results
    fireid, geom = group_fire_detections(
        result,
        scan_col=scan_col,
        track_col=track_col,
        instrument_col=instrument_col,
        date_col=date_col,
        parallel=parallel,
        n_workers=n_workers,
    )

    # Add to GeoDataFrame
    result['fireid'] = fireid
    result['geom_sml'] = geom.geom_sml
    result['geom_pix'] = geom.geom_pix

    # Calculate ndetect1 (count of detections per group)
    ndetect1 = result.groupby('fireid').size()
    result['ndetect1'] = result['fireid'].map(ndetect1)

    return result
