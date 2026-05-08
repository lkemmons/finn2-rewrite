"""Active-fire (AF) data loader — GeoPandas edition.

Replaces the original PostGIS / ogr2ogr pipeline with pure
pandas + GeoPandas.  No database required.

Supported input formats
-----------------------
.shp  : ESRI Shapefile  (passed straight to geopandas.read_file)
.csv  : comma-separated (FIRMS standard columns expected)
.txt  : same as .csv    (FIRMS sometimes ships these as .txt)

Expected CSV/TXT columns (FIRMS active-fire format)
----------------------------------------------------
longitude, latitude, scan, track, acq_date, acq_time,
satellite, confidence, version, brightness,
bright_t31, bright_ti4, bright_ti5, frp, daynight

Usage
-----
    from af_import_geopandas import load, get_tiles_needed

    gdf = load(['MODIS_C6_Global_24h.csv', 'VIIRS_375m_24h.txt'])
    print(gdf.crs)            # EPSG:4326
    print(gdf.columns.tolist())
    print(gdf[['acq_date', 'frp']].head())

    tiles = get_tiles_needed(gdf)   # {tilename: fire_count}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# CRS used for all outputs
WGS84 = "EPSG:4326"

# Explicit dtypes for FIRMS CSV columns — prevents pandas from
# misreading confidence as int when it contains "n", "h", "l" strings
_FIRMS_DTYPES = {
    "longitude":  float,
    "latitude":   float,
    "scan":       float,
    "track":      float,
    "acq_time":   str,
    "satellite":  str,
    "confidence": str,
    "version":    str,
    "brightness": float,
    "bright_t31": float,
    "bright_ti4": float,
    "bright_ti5": float,
    "frp":        float,
    "daynight":   str,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_one(path: str | Path) -> gpd.GeoDataFrame:
    """Load a single AF file (.shp, .csv, or .txt) as a GeoDataFrame."""
    path = Path(path)

    if path.suffix == ".shp":
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84)
        return gdf.to_crs(WGS84)

    if path.suffix in (".csv", ".txt"):
        df = pd.read_csv(
            path,
            dtype=_FIRMS_DTYPES,
            parse_dates=["acq_date"],
        )

        missing = {"longitude", "latitude"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing required columns {missing}")

        geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
        return gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)

    raise ValueError(f"Unsupported file extension: {path.suffix!r}")


def load(
    fnames: str | Path | list[str | Path],
) -> gpd.GeoDataFrame:
    """
    Load one or more AF files and return a single combined GeoDataFrame.

    Parameters
    ----------
    fnames : path or list of paths (.shp / .csv / .txt)

    Returns
    -------
    GeoDataFrame in WGS84 (EPSG:4326) with a 'source_file' column
    identifying which input file each row came from.
    """
    if isinstance(fnames, (str, Path)):
        fnames = [fnames]

    parts: list[gpd.GeoDataFrame] = []
    for f in fnames:
        gdf = _load_one(f)
        gdf["source_file"] = Path(f).name
        parts.append(gdf)

    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        crs=WGS84,
    )
    return combined


# ---------------------------------------------------------------------------
# Spatial queries (previously done via PostGIS)
# ---------------------------------------------------------------------------

def check_raster_contains_fire(
    raster_gdf: gpd.GeoDataFrame,
    fire_gdf: gpd.GeoDataFrame,
) -> dict[str, int]:
    """
    Count how many fire points fall inside raster tile polygons.

    Replaces the PostGIS ST_Contains query in the original code.

    Parameters
    ----------
    raster_gdf : GeoDataFrame of raster tile polygons (the 'skeleton')
    fire_gdf   : GeoDataFrame of fire points

    Returns
    -------
    dict with keys: n_fire, n_contained, n_not_contained
    """
    n_fire = len(fire_gdf)

    if raster_gdf is None or raster_gdf.empty:
        return {"n_fire": n_fire, "n_contained": 0, "n_not_contained": n_fire}

    # sjoin keeps only points that are within at least one raster polygon
    joined = gpd.sjoin(
        fire_gdf[["geometry"]],
        raster_gdf[["geometry"]],
        how="inner",
        predicate="within",
    )
    n_contained = joined.index.nunique()   # unique fire point indices
    return {
        "n_fire":          n_fire,
        "n_contained":     n_contained,
        "n_not_contained": n_fire - n_contained,
    }


def get_tiles_needed(
    fire_gdf: gpd.GeoDataFrame,
    combined: bool = True,
) -> dict[str, int] | list[dict[str, int]]:
    """
    Determine which MODIS sinusoidal tiles contain fire points,
    and how many points each tile has.

    Replaces the PostGIS wireframe join in the original code with
    pure coordinate arithmetic (the 'Method 2' described in comments).

    MODIS tile naming
    -----------------
    The sinusoidal grid divides the globe into 36 columns (h00–h35,
    west→east) and 18 rows (v00–v17, north→south).
    Tile h18v09 is at the centre (lon=0, lat=0).

    Parameters
    ----------
    fire_gdf : GeoDataFrame of fire points in WGS84
    combined : if True (default), merge counts from all source files;
               if False, return a list with one dict per source file.

    Returns
    -------
    combined=True  → {tilename: count}
    combined=False → [{tilename: count}, ...]  one dict per source file
    """
    SINU_RADIUS = 6_371_007.181

    def _points_to_tiles(gdf: gpd.GeoDataFrame) -> dict[str, int]:
        lng = gdf.geometry.x.to_numpy()
        lat = gdf.geometry.y.to_numpy()

        # Convert degrees → sinusoidal x/y (metres)
        lat_rad = np.deg2rad(lat)
        sinu_x = np.deg2rad(lng) * SINU_RADIUS * np.cos(lat_rad)
        sinu_y = np.deg2rad(lat) * SINU_RADIUS

        # Fraction of the great circle (–0.5 … +0.5 for lon, –0.25 … +0.25 for lat)
        fx = sinu_x / (2 * np.pi * SINU_RADIUS)
        fy = sinu_y / (2 * np.pi * SINU_RADIUS)

        # h: 0–35, origin at –180°
        h = np.floor(fx * 36 + 18).astype(int)
        # v: 0–17, origin at +90° (flipped)
        v = np.floor(-fy * 36 + 9).astype(int)

        # Clamp to valid grid (guards against points exactly on ±180/±90)
        h = np.clip(h, 0, 35)
        v = np.clip(v, 0, 17)

        tilenames = [f"h{hi:02d}v{vi:02d}" for hi, vi in zip(h, v)]
        unique, counts = np.unique(tilenames, return_counts=True)
        return dict(zip(unique, counts.tolist()))

    if "source_file" not in fire_gdf.columns:
        return _points_to_tiles(fire_gdf)

    groups = [g for _, g in fire_gdf.groupby("source_file", sort=False)]
    per_file = [_points_to_tiles(g) for g in groups]

    if combined:
        merged: dict[str, int] = {}
        for d in per_file:
            for tile, cnt in d.items():
                merged[tile] = merged.get(tile, 0) + cnt
        return merged

    return per_file


# ---------------------------------------------------------------------------
# Convenience accessors (previously SQL queries)
# ---------------------------------------------------------------------------

def get_lnglat(
    fire_gdf: gpd.GeoDataFrame,
    combined: bool = True,
) -> np.ndarray | list[np.ndarray]:
    """
    Return longitude/latitude as a NumPy array.

    combined=True  → shape (N, 2)
    combined=False → list of (Ni, 2) arrays, one per source file
    """
    if "source_file" not in fire_gdf.columns or combined:
        return np.column_stack([
            fire_gdf.geometry.x.to_numpy(),
            fire_gdf.geometry.y.to_numpy(),
        ])

    return [
        np.column_stack([g.geometry.x.to_numpy(), g.geometry.y.to_numpy()])
        for _, g in fire_gdf.groupby("source_file", sort=False)
    ]


def get_dates(
    fire_gdf: gpd.GeoDataFrame,
    combined: bool = True,
) -> np.ndarray | list[np.ndarray]:
    """
    Return unique acquisition dates as a NumPy array of datetime.date objects.

    combined=True  → 1-D array of unique dates across all files
    combined=False → list of 1-D arrays, one per source file
    """
    if "acq_date" not in fire_gdf.columns:
        raise ValueError("GeoDataFrame has no 'acq_date' column.")

    def _unique_dates(gdf):
        return np.array(sorted(gdf["acq_date"].dropna().unique()))

    if "source_file" not in fire_gdf.columns or combined:
        return _unique_dates(fire_gdf)

    return [
        _unique_dates(g)
        for _, g in fire_gdf.groupby("source_file", sort=False)
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: af_import_geopandas.py file1.[csv|txt|shp] [file2 ...]")
        sys.exit(1)

    gdf = load(sys.argv[1:])
    print(f"Loaded {len(gdf):,} fire points from {len(sys.argv) - 1} file(s).")
    print(f"CRS   : {gdf.crs}")
    print(f"Dates : {get_dates(gdf)}")
    print(f"Tiles : {get_tiles_needed(gdf)}")
    print(gdf.head())
