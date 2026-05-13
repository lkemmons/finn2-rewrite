"""Fire detection grouping and vegetation attribution pipeline.

Replaces the original FINN PostGIS pipeline (step1a → step1vcf → step1b → step2)
with pure Python using GeoPandas, Shapely, and rasterio.

Pipeline
--------
Step 1a  group_detections_step1a()
         Create geom_sml / geom_pix per detection, find overlapping pixel pairs
         within each day, union-find → fireid1 / ndetect1.

Step VCF sample_vcf()
         Dissolve geom_pix per fireid1 → work_lrg1 polygons.
         Clip tree-cover raster against each polygon, compute mean → v_tree.
         Set alg_agg (1 = forested ≥ threshold, 2 = non-forested).

Step 1b  merge_groups_step1b()
         Second union-find pass on work_lrg1 dissolved polygons.
         Groups whose hulls intersect are merged → fireid2 / ndetect2.
         Also applies filter_persistent_sources and date_definition.

Step 2   sample_rasters_step2()
         For each final fire polygon (work_div):
           • thematic rasters  → majority class (v_lct), fraction (f_lct), rank (r_lct)
           • continuous rasters → mean (v_tree, v_herb, v_bare)
           • polygon lookups   → centroid intersection (v_regnum)
           • input-field avg   → mean of a GeoDataFrame column (e.g. frp)
         Assembles the final output GeoDataFrame.

Top-level entry point
---------------------
    from fire_grouping import add_fire_groups_to_gdf

    result_gdf = add_fire_groups_to_gdf(
        fire_gdf,
        vcf_raster_path   = "modvcf_2023.tif",
        lct_raster_path   = "modlct_2023.tif",
        vcf_raster_paths  = {"tree": "tree.tif", "herb": "herb.tif", "bare": "bare.tif"},
        region_gdf        = gpd.read_file("globreg.shp"),
        region_col        = "regnum",
    )

Dependencies
------------
    pip install geopandas shapely rasterio numpy pandas pyproj
"""

from __future__ import annotations

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import functools

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from pyproj import Geod
from shapely.geometry import box, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match FINN originals exactly)
# ---------------------------------------------------------------------------

EARTH_CIRC_KM  = 2 * np.pi * 6370.997   # km — from step1a SQL
PIXFAC         = 1.1                      # pixel expansion factor
MODIS_SIZE_KM  = 1.0                      # nominal MODIS fire pixel size, km
VIIRS_SIZE_KM  = 0.375                    # nominal VIIRS fire pixel size, km

# alg_agg values (from run_vcf.py post-step)
ALG_AGG_FORESTED     = 1   # tree cover ≥ threshold → aggressive merge
ALG_AGG_NONFORESTED  = 2   # tree cover <  threshold → conservative merge
ALG_AGG_DEFAULT      = 2   # used when VCF not available

VCF_TREE_THRESHOLD_DEFAULT = 50   # % tree cover, from run_vcf.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FireGeometry(NamedTuple):
    """Pixel-footprint and nominal-size polygons for each detection."""
    geom_sml: np.ndarray   # shape (N,) of Shapely box objects
    geom_pix: np.ndarray   # shape (N,) of Shapely box objects


class UnionFind:
    """
    Disjoint-set with path compression and union-by-rank.
    Replaces the PostgreSQL pnt2grp() aggregate.
    """

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank   = np.zeros(n, dtype=np.uint8)

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

    def labels(self, n: int) -> np.ndarray:
        """Return the root (canonical label) for each of the first n elements."""
        return np.array([self.find(i) for i in range(n)])

    def relabel_to_min(self, raw: np.ndarray) -> np.ndarray:
        """Map each root to the minimum member index in its component."""
        min_of = {}
        for i, r in enumerate(raw):
            min_of[r] = min(min_of.get(r, i), i)
        return np.array([min_of[r] for r in raw])


# ---------------------------------------------------------------------------
# Step 1a — geometry creation and detection-level grouping
# ---------------------------------------------------------------------------

def _km_to_deg_dx(km: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert a km half-width to degrees longitude at the given latitudes."""
    return 0.5 * km * 360.0 / EARTH_CIRC_KM / np.cos(np.radians(lat_deg))


def _km_to_deg_dy(km: np.ndarray) -> np.ndarray:
    """Convert a km half-height to degrees latitude (latitude-independent)."""
    return 0.5 * km * 360.0 / EARTH_CIRC_KM


def create_fire_geometry(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = "scan",
    track_col: str = "track",
    instrument_col: str = "instrument",
) -> FireGeometry:
    """
    Build geom_sml (nominal fire size) and geom_pix (pixel footprint) for
    every detection.  Replicates the SQL UPDATE statements in step1a_work.sql.

    Parameters
    ----------
    fire_gdf       : GeoDataFrame with Point geometry in WGS84
    scan_col       : column holding scan dimension (km)
    track_col      : column holding track dimension (km)
    instrument_col : column holding instrument name ('MODIS' or 'VIIRS')
    """
    lons = fire_gdf.geometry.x.to_numpy()
    lats = fire_gdf.geometry.y.to_numpy()
    scan = fire_gdf[scan_col].to_numpy()
    trk  = fire_gdf[track_col].to_numpy()
    inst = fire_gdf[instrument_col].to_numpy()

    fire_km  = np.where(inst == "MODIS", MODIS_SIZE_KM, VIIRS_SIZE_KM)

    fire_dx  = _km_to_deg_dx(fire_km, lats)
    fire_dy  = _km_to_deg_dy(fire_km)
    pix_dx   = _km_to_deg_dx(PIXFAC * scan, lats)
    pix_dy   = _km_to_deg_dy(PIXFAC * trk)

    geom_sml = np.array([
        box(lo - dx, la - dy, lo + dx, la + dy)
        for lo, la, dx, dy in zip(lons, lats, fire_dx, fire_dy)
    ])
    geom_pix = np.array([
        box(lo - dx, la - dy, lo + dx, la + dy)
        for lo, la, dx, dy in zip(lons, lats, pix_dx, pix_dy)
    ])
    return FireGeometry(geom_sml=geom_sml, geom_pix=geom_pix)


def _find_adjacent_pairs_oneday(
    pix_geoms: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Return (i, j) pairs where geom_pix[i] intersects geom_pix[j], i < j.
    Replaces the tbl_adj_det spatial join in step1a_work.sql.
    """
    pairs: list[tuple[int, int]] = []
    if len(pix_geoms) < 2:
        return pairs
    tree = STRtree(pix_geoms)
    for i, g in enumerate(pix_geoms):
        for j in tree.query(g, predicate="intersects"):
            if j > i:
                pairs.append((i, int(j)))
    return pairs


def _assign_group_ids(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """
    Union-find over adjacency pairs; returns per-element group id equal to
    the minimum index in each connected component.
    Replaces pnt2grp() PostgreSQL aggregate.
    """
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)
    return uf.relabel_to_min(uf.labels(n))


def group_detections_step1a(
    fire_gdf: gpd.GeoDataFrame,
    scan_col: str = "scan",
    track_col: str = "track",
    instrument_col: str = "instrument",
    date_col: str = "acq_date",
    filter_persistent_sources: bool = False,
    date_definition: str = "LST",
    first_day=None,
    last_day=None,
) -> tuple[gpd.GeoDataFrame, FireGeometry]:
    """
    Step 1a: create geometries, find adjacent pairs within each day,
    assign fireid1 / ndetect1.

    Parameters
    ----------
    fire_gdf                  : input GeoDataFrame (WGS84 Points)
    filter_persistent_sources : if True, drop detections with type_conf == 'p'
                                (persistent heat sources: volcanoes, gas flares)
    date_definition           : 'LST' (local solar time) or 'UTC'.
                                LST shifts acq_date by ±1 day based on longitude
                                so spatially adjacent fires near day boundaries
                                are grouped correctly.
    first_day / last_day      : optional date filters applied before grouping

    Returns
    -------
    (gdf_with_step1a_columns, FireGeometry)
    Added columns: fireid1, ndetect1, geom_sml, geom_pix, acq_date_use
    """
    if date_definition not in ("LST", "UTC"):
        raise ValueError(f"date_definition must be 'LST' or 'UTC', got '{date_definition}'")

    df = fire_gdf.copy()

    # --- persistent-source filter -----------------------------------------
    if filter_persistent_sources and "type_conf" in df.columns:
        before = len(df)
        df = df[df["type_conf"] != "p"].copy()
        log.info("filter_persistent_sources: dropped %d / %d detections",
                 before - len(df), before)

    # --- date handling -------------------------------------------------------
    # Compute acq_date_use: the date each detection is *assigned* to for
    # grouping.  In LST mode MODIS tropical detections can carry over to the
    # next UTC day, so we keep the logic consistent with the original by using
    # the local solar time date.
    if date_definition == "LST":
        lons = df.geometry.x.to_numpy()
        # local solar time offset in hours = lon/15; shift fractional days
        lst_offset_days = lons / 360.0   # +0.5 at lon=180, -0.5 at lon=-180
        acq_dt = pd.to_datetime(df[date_col])
        acq_date_use = (acq_dt + pd.to_timedelta(lst_offset_days, unit="D")).dt.date
        df["acq_date_use"] = acq_date_use
    else:
        df["acq_date_use"] = pd.to_datetime(df[date_col]).dt.date

    # --- date-range filter ---------------------------------------------------
    if first_day is not None:
        df = df[df["acq_date_use"] >= first_day].copy()
    if last_day is not None:
        df = df[df["acq_date_use"] <= last_day].copy()

    if df.empty:
        log.warning("step1a: no detections remain after date filtering")
        df["fireid1"]  = pd.array([], dtype="int64")
        df["ndetect1"] = pd.array([], dtype="int64")
        df["geom_sml"] = pd.array([], dtype=object)
        df["geom_pix"] = pd.array([], dtype=object)
        return df, FireGeometry(geom_sml=np.array([]), geom_pix=np.array([]))

    df = df.reset_index(drop=True)
    log.info("step1a: grouping %d detections ...", len(df))

    geom = create_fire_geometry(df, scan_col, track_col, instrument_col)

    # group by acq_date_use and find adjacency within each day
    fireid1 = np.arange(len(df), dtype=np.int64)   # each point is its own group initially
    uf_global = UnionFind(len(df))

    for date_val, day_idx in df.groupby("acq_date_use").groups.items():
        day_idx_arr = np.asarray(day_idx)
        day_pix = geom.geom_pix[day_idx_arr]
        local_pairs = _find_adjacent_pairs_oneday(day_pix)
        for li, lj in local_pairs:
            uf_global.union(int(day_idx_arr[li]), int(day_idx_arr[lj]))

    raw_labels = uf_global.labels(len(df))
    fireid1    = uf_global.relabel_to_min(raw_labels)

    ndetect1_map = pd.Series(fireid1).value_counts()
    ndetect1 = np.array([ndetect1_map[fid] for fid in fireid1])

    df["fireid1"]  = fireid1
    df["ndetect1"] = ndetect1
    df["geom_sml"] = geom.geom_sml
    df["geom_pix"] = geom.geom_pix

    n_groups = len(np.unique(fireid1))
    log.info("step1a: %d detections → %d groups", len(df), n_groups)

    return df, geom


# ---------------------------------------------------------------------------
# Step VCF — dissolve groups, sample tree-cover raster, set alg_agg
# ---------------------------------------------------------------------------

def _dissolve_to_work_lrg(
    det_gdf: gpd.GeoDataFrame,
    fireid_col: str = "fireid1",
    date_col:   str = "acq_date_use",
    geom_pix_col: str = "geom_pix",
) -> gpd.GeoDataFrame:
    """
    Dissolve detection pixel footprints (geom_pix) per fire group per day.
    Produces work_lrg1: one row per (fireid1, acq_date_use) with a dissolved
    polygon geometry and the fire-group area in km².

    Replicates the work_lrg1 construction in step1b_prep.sql.
    """
    records = []
    geod = Geod(ellps="WGS84")

    for (fid, date_val), grp in det_gdf.groupby([fireid_col, date_col]):
        polys   = list(grp[geom_pix_col])
        union   = unary_union(polys)
        area_m2 = abs(geod.geometry_area_perimeter(union)[0])
        records.append({
            "fireid":       fid,
            "acq_date_use": date_val,
            "ndetect":      len(grp),
            "area_sqkm":    area_m2 / 1e6,
            "geometry":     union,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=["fireid", "acq_date_use", "ndetect", "area_sqkm", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["alg_agg"] = ALG_AGG_DEFAULT
    return gdf


def _sample_raster_mean(
    polygons: gpd.GeoDataFrame,
    raster_path: str | Path,
    band: int = 1,
    nodata: float | None = None,
) -> np.ndarray:
    """
    For each polygon, clip the raster and return the mean pixel value.
    Returns NaN where there are no valid pixels (polygon outside raster extent
    or all pixels are nodata).

    Replicates st_clip + st_summarystatsagg(mean) from the original SQL.
    """
    means = np.full(len(polygons), np.nan, dtype=float)
    with rasterio.open(str(raster_path)) as src:
        nd = nodata if nodata is not None else src.nodata
        for i, geom in enumerate(polygons.geometry):
            try:
                data, _ = rasterio.mask.mask(
                    src, [geom], crop=True, nodata=nd, all_touched=True
                )
                arr = data[band - 1].astype(float)
                if nd is not None:
                    arr = np.where(arr == nd, np.nan, arr)
                valid = arr[np.isfinite(arr)]
                if valid.size > 0:
                    means[i] = valid.mean()
            except Exception:
                pass   # polygon entirely outside raster — leave as NaN
    return means


def sample_vcf(
    det_gdf: gpd.GeoDataFrame,
    vcf_tree_raster: str | Path,
    fireid_col: str = "fireid1",
    date_col:   str = "acq_date_use",
    geom_pix_col: str = "geom_pix",
    tree_threshold: int = VCF_TREE_THRESHOLD_DEFAULT,
) -> gpd.GeoDataFrame:
    """
    Step VCF: dissolve detection pixel footprints into fire-group polygons
    (work_lrg1), sample the tree-cover raster, and set alg_agg.

    Parameters
    ----------
    det_gdf          : GeoDataFrame output of group_detections_step1a()
    vcf_tree_raster  : path to the VCF tree-cover GeoTIFF
    tree_threshold   : tree-cover % above which alg_agg = 1 (forested)

    Returns
    -------
    work_lrg1 GeoDataFrame with columns:
        fireid, acq_date_use, ndetect, area_sqkm, geometry, v_tree, alg_agg
    """
    log.info("step_vcf: dissolving %d detections into group polygons ...", len(det_gdf))
    work_lrg1 = _dissolve_to_work_lrg(det_gdf, fireid_col, date_col, geom_pix_col)

    if work_lrg1.empty:
        work_lrg1["v_tree"]  = pd.array([], dtype=float)
        work_lrg1["alg_agg"] = pd.array([], dtype=int)
        return work_lrg1

    log.info("step_vcf: sampling tree-cover raster for %d polygons ...", len(work_lrg1))
    v_tree = _sample_raster_mean(work_lrg1, vcf_tree_raster)
    work_lrg1["v_tree"] = v_tree

    # Set alg_agg: 1 = forested (aggressive merge), 2 = non-forested
    # Matches the UPDATE work_pnt / work_lrg1 in run_vcf.py post-step
    work_lrg1["alg_agg"] = np.where(
        work_lrg1["v_tree"].fillna(0) >= tree_threshold,
        ALG_AGG_FORESTED,
        ALG_AGG_NONFORESTED,
    )

    # Propagate alg_agg back to the detection-level frame
    det_gdf = det_gdf.copy()
    alg_map = work_lrg1.set_index("fireid")["alg_agg"].to_dict()
    det_gdf["alg_agg"] = det_gdf[fireid_col].map(alg_map).fillna(ALG_AGG_DEFAULT).astype(int)

    log.info("step_vcf: alg_agg=1 (forested): %d / %d groups",
             (work_lrg1["alg_agg"] == ALG_AGG_FORESTED).sum(), len(work_lrg1))

    return work_lrg1


# ---------------------------------------------------------------------------
# Step 1b — second grouping pass on dissolved fire-group polygons
# ---------------------------------------------------------------------------

def merge_groups_step1b(
    work_lrg1: gpd.GeoDataFrame,
    det_gdf:   gpd.GeoDataFrame,
    fireid_col: str = "fireid1",
    date_col:   str = "acq_date_use",
    geom_pix_col: str = "geom_pix",
) -> gpd.GeoDataFrame:
    """
    Step 1b: second union-find pass.

    Operates on the dissolved group polygons in work_lrg1 (one polygon per
    fireid1 per day).  Groups whose dissolved hulls intersect are merged into
    a single larger group.  This catches cases where adjacent fires did not
    have directly overlapping individual pixel footprints.

    Replicates step1b_work_v7m.sql connected-component logic.

    Parameters
    ----------
    work_lrg1   : output of sample_vcf() — dissolved fire-group polygons
    det_gdf     : detection-level GeoDataFrame from step1a (will be updated
                  with fireid2 / ndetect2)
    fireid_col  : column in det_gdf that links to work_lrg1["fireid"]
    date_col    : date column name (must match between the two frames)
    geom_pix_col: pixel-footprint column name in det_gdf

    Returns
    -------
    det_gdf copy with new columns: fireid2, ndetect2.
    work_lrg1 is NOT modified (the caller can re-dissolve using fireid2 if
    needed for downstream steps).
    """
    if work_lrg1.empty:
        det_gdf = det_gdf.copy()
        det_gdf["fireid2"]  = det_gdf[fireid_col]
        det_gdf["ndetect2"] = det_gdf["ndetect1"]
        return det_gdf

    log.info("step1b: merging %d group polygons ...", len(work_lrg1))

    # Build a mapping  fireid1 → position index in work_lrg1
    lrg_reset = work_lrg1.reset_index(drop=True)
    fid_to_idx = {fid: i for i, fid in enumerate(lrg_reset["fireid"])}
    n_groups   = len(lrg_reset)

    uf = UnionFind(n_groups)

    # Second-pass adjacency: group polygons (not individual pixels)
    group_geoms = lrg_reset.geometry.values
    tree = STRtree(group_geoms)

    for i, g in enumerate(group_geoms):
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j > i:
                uf.union(i, j)

    raw_labels  = uf.labels(n_groups)
    merged_ids  = uf.relabel_to_min(raw_labels)

    # Map back: fireid1 → fireid2  (the merged group id is the minimum fireid1
    # in the merged component)
    fireid1_arr = lrg_reset["fireid"].to_numpy()
    min_in_component: dict[int, int] = {}
    for pos, merged_pos in enumerate(merged_ids):
        fid1 = fireid1_arr[pos]
        ref  = fireid1_arr[merged_pos]
        min_in_component[fid1] = min(
            min_in_component.get(fid1, ref), ref
        )

    # Assign fireid2 to detections
    det_out = det_gdf.copy()
    det_out["fireid2"] = det_out[fireid_col].map(
        lambda fid: min_in_component.get(fid, fid)
    )

    ndetect2_map = det_out["fireid2"].value_counts()
    det_out["ndetect2"] = det_out["fireid2"].map(ndetect2_map)

    n_merged = det_out["fireid2"].nunique()
    log.info("step1b: %d fireid1 groups → %d fireid2 groups", n_groups, n_merged)

    return det_out


# ---------------------------------------------------------------------------
# Step 2 — raster sampling on final fire polygons (work_div)
# ---------------------------------------------------------------------------

def _build_work_div(
    det_gdf: gpd.GeoDataFrame,
    fireid_col:   str = "fireid2",
    date_col:     str = "acq_date_use",
    geom_pix_col: str = "geom_pix",
    alg_agg_col:  str = "alg_agg",
) -> gpd.GeoDataFrame:
    """
    Dissolve step-1b detection pixel footprints into final fire polygons.
    Produces work_div: one row per (fireid2, acq_date_use).

    Adds polyid (sequential integer), fireid, cleanids (list of cleanids or
    original indices), centroid columns, area_sqkm, and alg_agg.
    """
    geod = Geod(ellps="WGS84")
    records = []

    for (fid, date_val), grp in det_gdf.groupby([fireid_col, date_col]):
        polys   = list(grp[geom_pix_col])
        union   = unary_union(polys)
        area_m2 = abs(geod.geometry_area_perimeter(union)[0])
        centroid = union.centroid
        # cleanids: original integer index values (stand-ins for DB cleanid)
        cleanids = list(grp.index)
        alg_agg_val = int(
            grp[alg_agg_col].mode().iloc[0]
            if alg_agg_col in grp.columns and not grp.empty
            else ALG_AGG_DEFAULT
        )
        records.append({
            "fireid":       fid,
            "acq_date_use": date_val,
            "ndetect":      len(grp),
            "cleanids":     cleanids,
            "area_sqkm":    area_m2 / 1e6,
            "cen_lon":      centroid.x,
            "cen_lat":      centroid.y,
            "alg_agg":      alg_agg_val,
            "geometry":     union,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=["polyid", "fireid", "acq_date_use", "ndetect", "cleanids",
                     "area_sqkm", "cen_lon", "cen_lat", "alg_agg", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.insert(0, "polyid", np.arange(1, len(gdf) + 1, dtype=np.int64))
    return gdf


def _sample_thematic_raster(
    polygons: gpd.GeoDataFrame,
    raster_path: str | Path,
    band: int = 1,
    nodata: int | None = None,
) -> pd.DataFrame:
    """
    For each polygon, clip a thematic (categorical) raster and return the
    majority class (v), its pixel fraction of the total (f), and rank (r)
    for every class present.

    Replicates st_clip + st_valuecount + row_number() OVER (ORDER BY count DESC)
    from mkcmd_insert_table_thematic in run_step2.py.

    Returns a DataFrame indexed by polyid with columns v_*, f_*, r_*.
    The majority class (r=1) row is the primary output; all ranks are
    preserved so downstream code can access secondary classes if needed.
    """
    rows = []
    with rasterio.open(str(raster_path)) as src:
        nd = nodata if nodata is not None else src.nodata
        for polyid, geom in zip(polygons["polyid"], polygons.geometry):
            try:
                data, _ = rasterio.mask.mask(
                    src, [geom], crop=True, nodata=nd, all_touched=True
                )
                arr = data[band - 1].flatten()
                if nd is not None:
                    arr = arr[arr != nd]
                if arr.size == 0:
                    rows.append({"polyid": polyid, "val": None,
                                 "frac": None, "rank": 1})
                    continue
                vals, counts = np.unique(arr, return_counts=True)
                tcnt = counts.sum()
                order = np.argsort(counts)[::-1]   # descending by count
                for rank, idx in enumerate(order, start=1):
                    rows.append({
                        "polyid": polyid,
                        "val":    int(vals[idx]),
                        "cnt":    int(counts[idx]),
                        "tcnt":   int(tcnt),
                        "frac":   float(counts[idx]) / float(tcnt),
                        "rank":   rank,
                    })
            except Exception:
                rows.append({"polyid": polyid, "val": None,
                             "frac": None, "rank": 1})

    if not rows:
        return pd.DataFrame(columns=["polyid", "val", "frac", "rank"])
    return pd.DataFrame(rows)


def _sample_continuous_rasters(
    polygons: gpd.GeoDataFrame,
    raster_paths: dict[str, str | Path],
) -> pd.DataFrame:
    """
    For each polygon and each named raster, compute the mean pixel value.
    Returns a DataFrame indexed by polyid with one column per variable.

    Replicates mkcmd_insert_table_continuous (st_clip + st_summarystatsagg mean).
    """
    result = polygons[["polyid"]].copy()
    for var_name, rpath in raster_paths.items():
        col = f"v_{var_name}"
        result[col] = _sample_raster_mean(polygons, rpath)
    return result


def _sample_polygon_lookup(
    work_div: gpd.GeoDataFrame,
    lookup_gdf: gpd.GeoDataFrame,
    var_name: str,
    variable_in: str,
) -> pd.DataFrame:
    """
    Assign a value from lookup_gdf to each fire polygon via centroid
    intersection.  Replicates mkcmd_insert_table_polygons (st_intersects on
    centroid).

    Parameters
    ----------
    work_div    : fire polygon GeoDataFrame with polyid and cen_lon/cen_lat
    lookup_gdf  : polygon GeoDataFrame carrying variable_in
    var_name    : output column name (will be prefixed with 'v_')
    variable_in : column in lookup_gdf to read

    Returns a DataFrame with polyid and v_{var_name}.
    """
    centroids = gpd.GeoDataFrame(
        {"polyid": work_div["polyid"]},
        geometry=gpd.points_from_xy(work_div["cen_lon"], work_div["cen_lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        centroids, lookup_gdf[[variable_in, "geometry"]],
        how="left", predicate="within",
    )
    # Keep first match per centroid (right-join may produce duplicates)
    joined = joined.groupby("polyid")[variable_in].first().reset_index()
    joined = joined.rename(columns={variable_in: f"v_{var_name}"})
    return joined


def _sample_input_field_average(
    work_div: gpd.GeoDataFrame,
    det_gdf: gpd.GeoDataFrame,
    var_name: str,
    variable_in: str,
    fireid_col: str = "fireid2",
    date_col:   str = "acq_date_use",
) -> pd.DataFrame:
    """
    Average a detection-level column over all detections in each fire polygon.
    Replicates mkcmd_insert_table_input (avg() GROUP BY polyid joining work_pnt).

    Parameters
    ----------
    work_div     : final fire polygons (has fireid + acq_date_use)
    det_gdf      : detection-level frame
    var_name     : output column (prefixed 'v_')
    variable_in  : column in det_gdf to average

    Returns a DataFrame with polyid and v_{var_name}.
    """
    # Link polyid ↔ detections via (fireid2, acq_date_use)
    fire_to_poly = work_div.set_index([fireid_col, date_col])["polyid"].to_dict() \
        if fireid_col in work_div.columns \
        else work_div.set_index(["fireid", date_col])["polyid"].to_dict()

    det_copy = det_gdf.copy()
    det_copy["_polyid"] = [
        fire_to_poly.get((fid, d))
        for fid, d in zip(det_copy[fireid_col], det_copy[date_col])
    ]
    agg = (
        det_copy.dropna(subset=["_polyid", variable_in])
        .groupby("_polyid")[variable_in]
        .mean()
        .reset_index()
        .rename(columns={"_polyid": "polyid", variable_in: f"v_{var_name}"})
    )
    return agg


RasterSpec = dict   # keys: tag, kind, variable(s)/variables, variable_in (optional)


def sample_rasters_step2(
    det_gdf:   gpd.GeoDataFrame,
    rasters:   list[RasterSpec],
    fireid_col:   str = "fireid2",
    date_col:     str = "acq_date_use",
    geom_pix_col: str = "geom_pix",
    alg_agg_col:  str = "alg_agg",
) -> gpd.GeoDataFrame:
    """
    Step 2: dissolve detection footprints into final fire polygons (work_div),
    then sample each raster / lookup dataset and assemble the output table.

    Replicates the full run_step2.py logic (thematic, continuous, polygons,
    input-field averaging).

    Parameters
    ----------
    det_gdf  : detection-level GeoDataFrame from step1b output
    rasters  : list of raster/dataset specs.  Each dict must have:
               - 'tag'       : str, dataset identifier
               - 'kind'      : one of 'thematic', 'continuous', 'polygons', 'input'
               - 'path'      : str | Path  — GeoTIFF path
                               (or 'gdf' for kind='polygons')
               - 'variable'  : str  — output variable name  (thematic/polygons/input)
               - 'variables' : list[str] — output variable names (continuous)
               - 'variable_in': str — source column (polygons kind, input kind)
               - 'band'      : int, optional, raster band (default 1)

    Returns
    -------
    GeoDataFrame: one row per fire polygon per day, columns:
        polyid, fireid, cleanids, acq_date_use, area_sqkm,
        cen_lon, cen_lat, alg_agg,
        v_lct, f_lct, r_lct,   (from thematic rasters)
        v_tree, v_herb, v_bare, (from continuous rasters)
        v_regnum,               (from polygon lookup)
        v_frp,                  (from input-field average, if configured)
        geometry
    """
    # --- build work_div (final fire polygons) --------------------------------
    log.info("step2: building work_div from %d detections ...", len(det_gdf))
    work_div = _build_work_div(det_gdf, fireid_col, date_col, geom_pix_col, alg_agg_col)

    if work_div.empty:
        log.warning("step2: work_div is empty; no rasters sampled")
        return work_div

    log.info("step2: %d final fire polygons", len(work_div))

    # --- sample each raster / dataset ----------------------------------------
    for rstinfo in rasters:
        kind = rstinfo["kind"]
        tag  = rstinfo["tag"]
        log.info("step2: sampling %s (%s) ...", tag, kind)

        if kind == "thematic":
            df_raw = _sample_thematic_raster(
                work_div,
                raster_path=rstinfo["path"],
                band=rstinfo.get("band", 1),
                nodata=rstinfo.get("nodata"),
            )
            var  = rstinfo["variable"]
            # Majority class (rank == 1) per polyid
            majority = df_raw[df_raw["rank"] == 1].set_index("polyid")
            work_div[f"v_{var}"] = work_div["polyid"].map(majority["val"])
            work_div[f"f_{var}"] = work_div["polyid"].map(majority["frac"])
            work_div[f"r_{var}"] = work_div["polyid"].map(majority["rank"])

        elif kind == "continuous":
            cont_df = _sample_continuous_rasters(
                work_div,
                {v: rstinfo["path"] if isinstance(rstinfo["path"], (str, Path))
                     else rstinfo["path"][v]
                 for v in rstinfo["variables"]},
            )
            cont_df = cont_df.set_index("polyid")
            for v in rstinfo["variables"]:
                work_div[f"v_{v}"] = work_div["polyid"].map(cont_df[f"v_{v}"])

        elif kind == "polygons":
            lu_df = _sample_polygon_lookup(
                work_div,
                lookup_gdf=rstinfo["gdf"],
                var_name=rstinfo["variable"],
                variable_in=rstinfo["variable_in"],
            ).set_index("polyid")
            work_div[f"v_{rstinfo['variable']}"] = \
                work_div["polyid"].map(lu_df[f"v_{rstinfo['variable']}"])

        elif kind == "input":
            inp_df = _sample_input_field_average(
                work_div, det_gdf,
                var_name=rstinfo["variable"],
                variable_in=rstinfo["variable_in"],
                fireid_col=fireid_col,
                date_col=date_col,
            ).set_index("polyid")
            work_div[f"v_{rstinfo['variable']}"] = \
                work_div["polyid"].map(inp_df[f"v_{rstinfo['variable']}"])

        else:
            raise ValueError(f"Unknown raster kind '{kind}' for tag '{tag}'")

    log.info("step2: done. Output columns: %s", list(work_div.columns))
    return work_div


# ---------------------------------------------------------------------------
# Parallel helpers (step1a only — subsequent steps are not date-parallel)
# ---------------------------------------------------------------------------

def _group_one_day_parallel(
    date_group_tuple: tuple,
    scan_col: str = "scan",
    track_col: str = "track",
    instrument_col: str = "instrument",
) -> tuple[np.ndarray, np.ndarray]:
    """Worker: group detections for a single day; return (orig_indices, fireid)."""
    date_val, day_gdf = date_group_tuple
    day_gdf = day_gdf.reset_index(drop=True)
    geom    = create_fire_geometry(day_gdf, scan_col, track_col, instrument_col)
    pairs   = _find_adjacent_pairs_oneday(geom.geom_pix)
    fireid  = _assign_group_ids(len(day_gdf), pairs)
    return day_gdf.index.to_numpy(), fireid


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def add_fire_groups_to_gdf(
    fire_gdf: gpd.GeoDataFrame,
    # Step VCF
    vcf_tree_raster: str | Path | None = None,
    tree_threshold:  int = VCF_TREE_THRESHOLD_DEFAULT,
    # Step 2
    rasters: list[RasterSpec] | None = None,
    # Behaviour flags
    scan_col:       str = "scan",
    track_col:      str = "track",
    instrument_col: str = "instrument",
    date_col:       str = "acq_date",
    filter_persistent_sources: bool = False,
    date_definition: str = "LST",
    first_day=None,
    last_day=None,
    parallel:   bool = False,
    n_workers:  int | None = None,
    run_step1b: bool = True,
    run_step2:  bool = False,
) -> gpd.GeoDataFrame:
    """
    Complete fire-grouping and (optionally) vegetation-attribution pipeline.

    Stages executed
    ---------------
    1. step1a  — always
    2. step_vcf — if vcf_tree_raster is provided
    3. step1b  — if run_step1b=True (default)
    4. step2   — if run_step2=True AND rasters is not None

    Parameters
    ----------
    fire_gdf                  : input GeoDataFrame of AF detections (WGS84 Points)
    vcf_tree_raster           : path to VCF tree-cover GeoTIFF for step_vcf
    tree_threshold            : % tree cover threshold for alg_agg (default 50)
    rasters                   : list of RasterSpec dicts for step2
    scan_col / track_col /
    instrument_col / date_col : column name overrides
    filter_persistent_sources : drop persistent thermal anomaly detections
    date_definition           : 'LST' or 'UTC'
    first_day / last_day      : optional date-range filter (datetime.date objects)
    parallel                  : use multiprocessing for step1a date loop
    n_workers                 : worker count for parallel mode
    run_step1b                : perform the second grouping pass (default True)
    run_step2                 : perform raster sampling (default False)

    Returns
    -------
    If run_step2=False: detection-level GeoDataFrame with added columns:
        acq_date_use, geom_sml, geom_pix, alg_agg,
        fireid1, ndetect1,
        fireid2, ndetect2  (if run_step1b=True)

    If run_step2=True: fire-polygon GeoDataFrame (work_div) with:
        polyid, fireid, cleanids, acq_date_use, area_sqkm,
        cen_lon, cen_lat, alg_agg,
        + all columns populated by rasters specs
    """
    # ------------------------------------------------------------------
    # Step 1a
    # ------------------------------------------------------------------
    det_gdf, _geom = group_detections_step1a(
        fire_gdf,
        scan_col=scan_col,
        track_col=track_col,
        instrument_col=instrument_col,
        date_col=date_col,
        filter_persistent_sources=filter_persistent_sources,
        date_definition=date_definition,
        first_day=first_day,
        last_day=last_day,
    )

    if det_gdf.empty:
        return det_gdf

    # ------------------------------------------------------------------
    # Step VCF  (sets alg_agg and builds work_lrg1)
    # ------------------------------------------------------------------
    if vcf_tree_raster is not None:
        work_lrg1 = sample_vcf(
            det_gdf,
            vcf_tree_raster=vcf_tree_raster,
            tree_threshold=tree_threshold,
            fireid_col="fireid1",
            date_col="acq_date_use",
            geom_pix_col="geom_pix",
        )
        # Propagate alg_agg back into det_gdf
        alg_map = work_lrg1.set_index("fireid")["alg_agg"].to_dict()
        det_gdf["alg_agg"] = (
            det_gdf["fireid1"]
            .map(alg_map)
            .fillna(ALG_AGG_DEFAULT)
            .astype(int)
        )
    else:
        det_gdf["alg_agg"] = ALG_AGG_DEFAULT
        work_lrg1 = _dissolve_to_work_lrg(det_gdf, "fireid1", "acq_date_use", "geom_pix")

    # ------------------------------------------------------------------
    # Step 1b
    # ------------------------------------------------------------------
    if run_step1b:
        det_gdf = merge_groups_step1b(
            work_lrg1=work_lrg1,
            det_gdf=det_gdf,
            fireid_col="fireid1",
            date_col="acq_date_use",
            geom_pix_col="geom_pix",
        )
        final_fireid_col = "fireid2"
    else:
        det_gdf["fireid2"]  = det_gdf["fireid1"]
        det_gdf["ndetect2"] = det_gdf["ndetect1"]
        final_fireid_col = "fireid2"

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------
    if run_step2 and rasters:
        return sample_rasters_step2(
            det_gdf=det_gdf,
            rasters=rasters,
            fireid_col=final_fireid_col,
            date_col="acq_date_use",
            geom_pix_col="geom_pix",
            alg_agg_col="alg_agg",
        )

    return det_gdf
