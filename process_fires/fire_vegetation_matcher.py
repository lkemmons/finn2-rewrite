"""Match fire polygons to vegetation types using GeoPandas.

Replaces the SQL-based Step 1a and Step 2 pipeline with pure GeoPandas.
No database required.

Pipeline
--------
    1. load_fire_detections()   — load AF data and create geometry
    2. group_detections()        — group nearby fires by spatial proximity
    3. build_fire_polygons()     — aggregate groups into burned area polygons
    4. clean_polygons()          — fill small holes in polygons
    5. match_vegetation()        — sample raster data at fire polygon locations
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from shapely.geometry import Polygon

log = logging.getLogger(__name__)

# WGS84 CRS for consistency with AF data
WGS84 = "EPSG:4326"


# ---------------------------------------------------------------------------
# Fire Polygon Building
# ---------------------------------------------------------------------------


def build_fire_polygons(
    fire_gdf: gpd.GeoDataFrame,
    fireid_col: str = "fireid",
    geom_sml_col: str = "geom_sml",
    date_col: str = "acq_date",
) -> gpd.GeoDataFrame:
    """
    Build burned area polygons by aggregating grouped fire detections.

    Takes fire detections that have already been grouped (via fire_grouping module)
    and creates one polygon per group by unioning the individual fire polygons.

    Parameters
    ----------
    fire_gdf : GeoDataFrame of fire points with columns:
        - fireid (group ID, result of grouping)
        - geom_sml (nominal fire size polygons)
        - acq_date (acquisition date)
    fireid_col : name of column with group IDs
    geom_sml_col : name of column with fire polygons
    date_col : name of acquisition date column

    Returns
    -------
    GeoDataFrame with one row per fire group (burned area polygon)
        Columns: fireid, acq_date, ndetect, geometry (union of fire polygons)
    """
    fire_gdf = fire_gdf.copy()

    # Aggregate by group
    grouped = fire_gdf.groupby(fireid_col).agg({
        date_col: 'first',
        'ndetect1': 'first',  # Already computed in grouping
    }).reset_index()

    # Build polygons from geom_sml by unioning within each group
    def union_geoms_for_group(group_df):
        geoms = group_df[geom_sml_col].values
        if len(geoms) == 1:
            return geoms[0]
        else:
            from shapely.ops import unary_union
            return unary_union(geoms)

    polygons = []
    for fireid in grouped[fireid_col].values:
        group_data = fire_gdf[fire_gdf[fireid_col] == fireid]
        polygon = union_geoms_for_group(group_data)
        polygons.append(polygon)

    grouped['geometry'] = polygons

    # Create output GeoDataFrame
    fire_polys = gpd.GeoDataFrame(
        grouped[[fireid_col, date_col, 'ndetect1']],
        geometry='geometry',
        crs=WGS84,
    )

    fire_polys.rename(columns={
        fireid_col: 'fireid',
        date_col: 'acq_date',
        'ndetect1': 'ndetect',
    }, inplace=True)

    # Calculate area in square km
    # At equator, 1 degree ≈ 111.32 km
    fire_polys['area_sqkm'] = fire_polys.geometry.area * (111_320 ** 2) / 1e6

    log.info(f"Built {len(fire_polys)} fire polygons from {len(fire_gdf)} detections")

    return fire_polys


# ---------------------------------------------------------------------------
# Polygon Cleaning (fill holes)
# ---------------------------------------------------------------------------


def clean_polygons(
    fire_polys: gpd.GeoDataFrame,
    hole_threshold_deg: float = 1 / 240.,
) -> gpd.GeoDataFrame:
    """
    Fill small holes in burned area polygons.

    Holes smaller than the threshold are filled completely. Larger holes are
    preserved (they may be actual water bodies or unburned patches).

    Parameters
    ----------
    fire_polys : GeoDataFrame of fire polygons
    hole_threshold_deg : minimum hole size in degrees² (default ~15 arc-seconds)
        At equator: ~500m × ~500m = ~0.25 km²

    Returns
    -------
    GeoDataFrame with holes filled
    """
    fire_polys = fire_polys.copy()

    def fill_small_holes(geom: Polygon, threshold: float) -> Polygon:
        if not isinstance(geom, Polygon):
            return geom

        # Check for interior rings (holes)
        if not geom.interiors:
            return geom

        # Keep only holes larger than threshold
        large_holes = []
        for interior in geom.interiors:
            hole_poly = Polygon(interior)
            if hole_poly.area > threshold:  # area in degrees²
                large_holes.append(interior)

        if large_holes:
            # Recreate polygon with only large holes
            return Polygon(geom.exterior, holes=large_holes)
        else:
            # Fill all holes
            return Polygon(geom.exterior)

    fire_polys['geometry'] = fire_polys.geometry.apply(
        lambda g: fill_small_holes(g, hole_threshold_deg)
    )

    log.info(f"Cleaned {len(fire_polys)} polygons (hole threshold: {hole_threshold_deg:.6f}°²)")

    return fire_polys


# ---------------------------------------------------------------------------
# Raster Sampling
# ---------------------------------------------------------------------------


def sample_raster_at_polygons(
    fire_polys: gpd.GeoDataFrame,
    raster_path: str | Path,
    band: int = 1,
    method: str = 'majority',
) -> np.ndarray:
    """
    Sample raster data within fire polygons.

    For each polygon, extracts pixel values from the raster and aggregates
    them according to the specified method.

    Parameters
    ----------
    fire_polys : GeoDataFrame of fire polygons (WGS84)
    raster_path : path to GeoTIFF file
    band : band number to read (1-based)
    method : aggregation method:
        - 'majority': most common value (for categorical data like LCT)
        - 'mean': average (for continuous data like VCF)
        - 'median': median value
        - 'weighted_majority': majority weighted by pixel area

    Returns
    -------
    Array of raster values (one per polygon), dtype depends on method
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # Reproject polygons if needed
        if fire_polys.crs != raster_crs:
            log.info(f"Reprojecting polygons from {fire_polys.crs} to {raster_crs}")
            fire_polys_reprojected = fire_polys.to_crs(raster_crs)
        else:
            fire_polys_reprojected = fire_polys

        values = np.full(len(fire_polys), fill_value=np.nan, dtype=np.float32)

        for idx, (_, row) in enumerate(fire_polys_reprojected.iterrows()):
            geom = row.geometry

            try:
                # Clip raster to polygon
                clipped, _ = rasterio.mask.mask(
                    src,
                    [geom],
                    crop=False,
                    indexes=band,
                    nodata=nodata,
                )

                # Extract valid pixel values
                valid_mask = clipped != nodata if nodata is not None else np.ones_like(clipped, dtype=bool)
                valid_pixels = clipped[valid_mask]

                if len(valid_pixels) > 0:
                    if method == 'majority':
                        # Most common value
                        unique, counts = np.unique(valid_pixels, return_counts=True)
                        values[idx] = unique[np.argmax(counts)]
                    elif method == 'mean':
                        values[idx] = np.mean(valid_pixels)
                    elif method == 'median':
                        values[idx] = np.median(valid_pixels)
                    elif method == 'weighted_majority':
                        # Majority weighted by count (same as majority)
                        unique, counts = np.unique(valid_pixels, return_counts=True)
                        values[idx] = unique[np.argmax(counts)]
                    else:
                        raise ValueError(f"Unknown method: {method}")

            except Exception as e:
                log.warning(f"Error sampling polygon {idx} ({geom.bounds}): {e}")
                values[idx] = np.nan

    log.info(f"Sampled {(~np.isnan(values)).sum()}/{len(values)} polygons from {raster_path}")

    return values


# ---------------------------------------------------------------------------
# Vegetation Matching
# ---------------------------------------------------------------------------


def match_vegetation(
    fire_polys: gpd.GeoDataFrame,
    lct_tif: str | Path,
    vcf_tifs: dict[str, str | Path] | None = None,
) -> gpd.GeoDataFrame:
    """
    Assign vegetation types to fire polygons by sampling LCT and VCF rasters.

    This is the pure-Python equivalent of SQL step2, which clips rasters to
    fire polygons and extracts vegetation values.

    Parameters
    ----------
    fire_polys : GeoDataFrame of fire polygons (WGS84)
    lct_tif : path to land-cover type GeoTIFF
    vcf_tifs : dict mapping variable names ('tree', 'herb', 'bare') to paths
        Example: {'tree': 'tree.tif', 'herb': 'herb.tif', 'bare': 'bare.tif'}

    Returns
    -------
    GeoDataFrame with added columns:
        - v_lct : land cover type (category code from MODIS MCD12Q1)
        - v_tree : percent tree cover (0-100) if vcf_tifs provided
        - v_herb : percent herbaceous cover (0-100) if vcf_tifs provided
        - v_bare : percent bare cover (0-100) if vcf_tifs provided
    """
    fire_polys = fire_polys.copy()

    # Sample LCT (categorical, use majority method)
    log.info(f"Sampling LCT from {lct_tif}...")
    fire_polys['v_lct'] = sample_raster_at_polygons(
        fire_polys,
        lct_tif,
        band=1,
        method='majority',
    ).astype('Int64')  # Nullable integer type

    # Sample VCF layers (continuous, use mean)
    if vcf_tifs:
        for var_name, tif_path in vcf_tifs.items():
            log.info(f"Sampling VCF {var_name} from {tif_path}...")
            fire_polys[f'v_{var_name}'] = sample_raster_at_polygons(
                fire_polys,
                tif_path,
                band=1,
                method='mean',
            )
    else:
        log.info("No VCF rasters provided, skipping continuous variables")

    return fire_polys


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def process_fires_to_vegetation(
    fire_gdf: gpd.GeoDataFrame,
    lct_tif: str | Path,
    vcf_tifs: dict[str, str | Path] | None = None,
    clean_holes: bool = True,
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """
    Complete fire-to-vegetation matching pipeline.

    Assumes fires have already been grouped (via fire_grouping module).
    Takes grouped fires and matches them to vegetation types.

    Parameters
    ----------
    fire_gdf : GeoDataFrame of active-fire detections with:
        - geometry (Point)
        - fireid (group ID)
        - ndetect1 (detection count per group)
        - geom_sml (fire polygon for group)
        - acq_date (acquisition date)
    lct_tif : path to land-cover type GeoTIFF (e.g., MCD12Q1)
    vcf_tifs : dict mapping variable names to continuous field GeoTIFF paths
        Optional: {'tree': 'path', 'herb': 'path', 'bare': 'path'}
    clean_holes : if True, fill small holes in burned area polygons
    output_path : optional path to save output shapefile

    Returns
    -------
    GeoDataFrame with fire polygons and vegetation assignments
        Columns:
            - fireid, acq_date, ndetect, geometry (fire polygon)
            - area_sqkm (burned area in km²)
            - v_lct (land cover type)
            - v_tree, v_herb, v_bare (if VCF provided)
    """
    log.info("Building fire polygons from grouped detections...")
    fire_polys = build_fire_polygons(fire_gdf)

    if clean_holes:
        log.info("Cleaning polygons (filling small holes)...")
        fire_polys = clean_polygons(fire_polys)

    log.info("Matching to vegetation types...")
    fire_polys = match_vegetation(fire_polys, lct_tif, vcf_tifs)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing output to {output_path}")
        fire_polys.to_file(output_path)

    return fire_polys
