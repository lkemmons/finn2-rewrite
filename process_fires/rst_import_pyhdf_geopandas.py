"""Import MODIS landcover (HDF4) raster data and export as shapefile for GeoPandas.

Dependencies
------------
    pip install pyhdf rasterio geopandas shapely pyproj numpy

No GDAL Python bindings or command-line tools required.

Pipeline
--------
    1. read_hdf_layer()    — read one SDS band from an HDF4 file via pyhdf
    2. build_tile_polygon()— derive the tile footprint from StructMetadata
    3. merge_to_tif()      — write selected bands to a GeoTIFF via rasterio
    4. resample_to_tiles() — warp to 10°×10° WGS84 tiles via rasterio.warp
    5. save_skeleton()     — write the skeleton GeoDataFrame to a shapefile

The output shapefile is ready for `geopandas.read_file()`.

Supported data categories
--------------------------
lct : MCD12Q1 land-cover type              (1 band)
vcf : MOD44B  vegetation continuous fields (3 bands: tree / herb / bare)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.crs
import rasterio.transform
import rasterio.warp
from pyhdf.SD import SD, SDC
from pyproj import CRS
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WGS84_CRS  = CRS.from_epsg(4326)
SINU_PROJ4 = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
)
SINU_CRS    = CRS.from_proj4(SINU_PROJ4)
SINU_RADIUS = 6_371_007.181

# Output resolution: 6 arc-seconds (~500 m at equator)
OUTPUT_RES = 0.001_666_666_666_666_667

# Default nodata value when writing GeoTIFFs
NODATA = 255

DATA_CATEGORIES: dict[str, dict] = {
    "lct": {
        "lyrnames":  ["LC_Type1"],
        "shortnames": ["lct"],
        "nodata_in": 255,
        "rsmp_alg":  rasterio.warp.Resampling.mode,
        "re_bname":  re.compile(r"^MCD12Q1\.A(\d{4})001"),
    },
    "vcf": {
        "lyrnames":  ["Percent_Tree_Cover",
                      "Percent_NonTree_Vegetation",
                      "Percent_NonVegetated"],
        "shortnames": ["tree", "herb", "bare"],
        "nodata_in": 200,
        "rsmp_alg":  rasterio.warp.Resampling.average,
        "re_bname":  re.compile(r"^MOD44B\.A(\d{4})065"),
    },
}


# ---------------------------------------------------------------------------
# pyhdf helpers
# ---------------------------------------------------------------------------

def open_hdf(hdf_path: str | Path) -> SD:
    """Open an HDF4 file for reading and return the SD object."""
    return SD(str(hdf_path), SDC.READ)


def list_datasets(hdf_path: str | Path) -> list[str]:
    """Return the names of all Scientific Datasets in an HDF4 file."""
    hdf   = open_hdf(hdf_path)
    names = list(hdf.datasets().keys())
    hdf.end()
    return names


def read_hdf_layer(hdf_path: str | Path, layer_name: str) -> np.ndarray:
    """
    Read one SDS layer from an HDF4 file and return a 2-D NumPy array.

    Raises KeyError if *layer_name* is not present in the file.
    """
    hdf       = open_hdf(hdf_path)
    available = list(hdf.datasets().keys())
    if layer_name not in available:
        hdf.end()
        raise KeyError(
            f"Layer '{layer_name}' not found in {Path(hdf_path).name}.\n"
            f"Available: {available}"
        )
    sds  = hdf.select(layer_name)
    data = sds.get()        # np.ndarray, shape (nrows, ncols)
    sds.endaccess()
    hdf.end()
    return data


# ---------------------------------------------------------------------------
# GeoTransform from StructMetadata (replaces gdal.GetGeoTransform + osr)
# ---------------------------------------------------------------------------

def _parse_struct_metadata(hdf_path: str | Path) -> dict[str, str]:
    """
    Parse the flat key=value pairs out of the HDF4 StructMetadata.0 attribute.

    MODIS HDF4 files embed a text block called StructMetadata.0 that contains
    the grid definition (corner coordinates, pixel count, projection).  We
    parse it here so we never need the GDAL HDF4 driver.
    """
    hdf  = open_hdf(hdf_path)
    meta = hdf.attributes().get("StructMetadata.0", "")
    hdf.end()

    result: dict[str, str] = {}
    for line in meta.replace("\t", "").splitlines():
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def get_geotransform(
    hdf_path: str | Path,
) -> tuple[float, float, float, float, float, float]:
    """
    Derive a GDAL-style GeoTransform from the HDF4 StructMetadata.

    Returns
    -------
    (x_ul, pixel_w, 0, y_ul, 0, -pixel_h)

    where (x_ul, y_ul) is the upper-left corner in sinusoidal metres and
    pixel_w / pixel_h are the positive pixel dimensions.
    """
    meta = _parse_struct_metadata(hdf_path)

    def _pair(key: str) -> tuple[float, float]:
        raw = meta[key].strip("()")
        a, b = raw.split(",")
        return float(a), float(b)

    ul_x, ul_y = _pair("UpperLeftPointMtrs")
    lr_x, lr_y = _pair("LowerRightMtrs")

    # Pixel dimensions from the grid's XDim / YDim counters
    xdim_key = next(k for k in meta if k.endswith("XDim"))
    ydim_key = next(k for k in meta if k.endswith("YDim"))
    ncols    = int(meta[xdim_key])
    nrows    = int(meta[ydim_key])

    pixel_w =  (lr_x - ul_x) / ncols
    pixel_h =  (ul_y - lr_y) / nrows   # positive

    return (ul_x, pixel_w, 0.0, ul_y, 0.0, -pixel_h)


# ---------------------------------------------------------------------------
# Sinusoidal boundary censor
# ---------------------------------------------------------------------------

def censor_sinu(points: np.ndarray) -> np.ndarray:
    """Clamp x-coordinates that exceed the sinusoidal valid domain at each y."""
    half_circ = np.pi * SINU_RADIUS * np.cos(points[..., 1] / SINU_RADIUS)
    points[..., 0] = np.where(
        np.abs(points[..., 0]) > half_circ,
        half_circ * np.sign(points[..., 0]),
        points[..., 0],
    )
    return points


# ---------------------------------------------------------------------------
# Skeleton (tile footprint) builder
# ---------------------------------------------------------------------------

def build_tile_polygon(hdf_path: str | Path) -> Polygon:
    """
    Build a Shapely Polygon (in sinusoidal metres) tracing the boundary of
    one MODIS HDF4 tile.

    The polygon is built by sampling 50 points along each edge of the pixel
    grid, projecting them to sinusoidal metres via the GeoTransform, then
    censoring coordinates that stray outside the valid sinusoidal domain
    (important for tiles near ±180°).

    All metadata is read from the HDF4 file itself — no GDAL required.
    """
    gt   = np.array(get_geotransform(hdf_path))
    meta = _parse_struct_metadata(hdf_path)

    xdim_key = next(k for k in meta if k.endswith("XDim"))
    ydim_key = next(k for k in meta if k.endswith("YDim"))
    nc = int(meta[xdim_key])
    nr = int(meta[ydim_key])

    # Sample 50 evenly-spaced points along each of the 4 edges → closed ring
    num = 50
    xp  = np.rint(np.linspace(0, nc, num + 1))
    yp  = np.rint(np.linspace(0, nr, num + 1))

    xy = np.zeros((num * 4 + 1, 2))
    xy[0 * num : 1 * num, 0] = 0;          xy[0 * num : 1 * num, 1] = yp[:-1]
    xy[1 * num : 2 * num, 0] = xp[:-1];    xy[1 * num : 2 * num, 1] = yp[-1]
    xy[2 * num : 3 * num, 0] = xp[-1];     xy[2 * num : 3 * num, 1] = yp[:0:-1]
    xy[3 * num : 4 * num, 0] = xp[:0:-1];  xy[3 * num : 4 * num, 1] = 0
    xy[4 * num, :] = 0
    xy = xy[::-1, :]   # clockwise works better with the censor

    # Pixel → sinusoidal metres using the affine transform
    xy = np.column_stack([
        gt[0] + gt[1] * xy[:, 0] + gt[2] * xy[:, 1],
        gt[3] + gt[4] * xy[:, 0] + gt[5] * xy[:, 1],
    ])

    xy = censor_sinu(xy)

    # Remove consecutive duplicates introduced by censoring
    keep = [0] + [i for i in range(1, len(xy))
                  if not np.array_equal(xy[i - 1], xy[i])]
    xy   = xy[keep]

    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.area <= 0:
        raise RuntimeError(f"Could not build a valid polygon for {hdf_path}")

    return poly


def build_skeleton(hdf_paths: list[str | Path]) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame with one polygon per HDF tile, reprojected to WGS84.
    """
    records = []
    for i, hdf in enumerate(hdf_paths):
        poly = build_tile_polygon(hdf)
        records.append({"id": i + 1, "name": Path(hdf).name, "geometry": poly})

    gdf = gpd.GeoDataFrame(records, crs=SINU_CRS)
    return gdf.to_crs(WGS84_CRS)


def save_skeleton(gdf: gpd.GeoDataFrame, output_path: str | Path) -> None:
    """Write the skeleton GeoDataFrame to a shapefile."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(output_path))
    log.info("Skeleton saved → %s", output_path)


# ---------------------------------------------------------------------------
# HDF → GeoTIFF  (replaces gdal_merge.py subprocess)
# ---------------------------------------------------------------------------

def merge_to_tif(
    hdf_path: str | Path,
    layer_names: list[str],
    output_tif: str | Path,
    nodata_in: int = NODATA,
) -> Path:
    """
    Extract one or more SDS layers from an HDF4 file and write them as a
    multi-band GeoTIFF using rasterio.

    Each layer becomes one band in the output file.  The sinusoidal CRS and
    affine transform are derived from the file's own StructMetadata, so no
    GDAL HDF4 driver is needed.

    Parameters
    ----------
    hdf_path    : source .hdf file
    layer_names : SDS names to extract (one band each)
    output_tif  : destination GeoTIFF path
    nodata_in   : fill / nodata value present in the source data
    """
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    gt        = get_geotransform(hdf_path)
    transform = rasterio.transform.from_origin(
        west=gt[0],
        north=gt[3],
        xsize=gt[1],
        ysize=-gt[5],   # gt[5] is negative; from_origin wants positive height
    )

    bands  = [read_hdf_layer(hdf_path, name) for name in layer_names]
    nrows, ncols = bands[0].shape

    with rasterio.open(
        output_tif,
        mode="w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=len(bands),
        dtype=bands[0].dtype,
        crs=SINU_CRS,
        transform=transform,
        nodata=nodata_in,
        compress="lzw",
        tiled=True,
    ) as dst:
        for band_idx, band_data in enumerate(bands, start=1):
            dst.write(band_data, band_idx)

    log.info("Wrote %d-band GeoTIFF → %s", len(bands), output_tif)
    return output_tif


# ---------------------------------------------------------------------------
# Resample to 10°×10° WGS84 tiles  (replaces gdalwarp subprocess)
# ---------------------------------------------------------------------------

def resample_to_tiles(
    tif_paths: list[str | Path],
    skeleton_gdf: gpd.GeoDataFrame,
    output_dir: str | Path,
    base_name: str,
    resample_alg: rasterio.warp.Resampling = rasterio.warp.Resampling.mode,
    nodata: int = NODATA,
) -> list[Path]:
    """
    Warp merged GeoTIFFs into a 36×18 grid of 10°×10° WGS84 tiles using
    rasterio.warp.reproject.  Only tiles that intersect the data skeleton
    are written.

    Replaces the gdalbuildvrt + gdalwarp subprocess loop.

    Returns a list of output GeoTIFF paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton_union = unary_union(skeleton_gdf.geometry)
    output_paths:  list[Path] = []

    # Open all source TIFs once and keep them open across the tile loop
    src_files = [rasterio.open(str(p)) for p in tif_paths]
    n_bands   = src_files[0].count
    src_dtype = src_files[0].dtypes[0]

    try:
        for i in range(36):       # longitude columns: −180 … +180
            for j in range(18):   # latitude rows:      +90 … −90
                west  = -180 + 10 * i
                south =   90 - 10 * (j + 1)
                east  = west  + 10
                north = south + 10

                if skeleton_union.disjoint(box(west, south, east, north)):
                    continue

                out_tif    = output_dir / f"{base_name}.r{i:02d}c{j:02d}.tif"
                tile_cols  = round(10 / OUTPUT_RES)
                tile_rows  = round(10 / OUTPUT_RES)
                tile_xform = rasterio.transform.from_bounds(
                    west, south, east, north, tile_cols, tile_rows
                )

                # Initialise destination with nodata; each source TIF
                # writes into it in turn (later tiles overwrite earlier ones
                # where they overlap, matching gdalwarp -overwrite behaviour).
                dest = np.full(
                    (n_bands, tile_rows, tile_cols),
                    fill_value=nodata,
                    dtype=src_dtype,
                )

                for src in src_files:
                    rasterio.warp.reproject(
                        source=rasterio.band(src, list(range(1, n_bands + 1))),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tile_xform,
                        dst_crs=WGS84_CRS,
                        resampling=resample_alg,
                        src_nodata=nodata,
                        dst_nodata=nodata,
                    )

                with rasterio.open(
                    out_tif,
                    mode="w",
                    driver="GTiff",
                    height=tile_rows,
                    width=tile_cols,
                    count=n_bands,
                    dtype=src_dtype,
                    crs=WGS84_CRS,
                    transform=tile_xform,
                    nodata=nodata,
                    compress="lzw",
                    tiled=True,
                ) as dst:
                    dst.write(dest)

                log.info("Wrote tile → %s", out_tif)
                output_paths.append(out_tif)

    finally:
        for src in src_files:
            src.close()

    return output_paths


# ---------------------------------------------------------------------------
# Importer — ties the pipeline together
# ---------------------------------------------------------------------------

class Importer:
    """Orchestrates the HDF→TIF → resample → skeleton pipeline."""

    def __init__(self, fnames: list[str | Path]):
        bname = Path(fnames[0]).name
        for cat, cfg in DATA_CATEGORIES.items():
            m = cfg["re_bname"].match(bname)
            if m:
                self._cat = cat
                self._cfg = cfg
                self.year = int(m.group(1))
                break
        else:
            raise RuntimeError(
                f"Cannot determine data category from filename: {bname}\n"
                f"Supported patterns: "
                f"{[v['re_bname'].pattern for v in DATA_CATEGORIES.values()]}"
            )

    @property
    def layer_names(self) -> list[str]:
        return self._cfg["lyrnames"]

    @property
    def short_names(self) -> list[str]:
        return self._cfg["shortnames"]

    @property
    def resample_alg(self) -> rasterio.warp.Resampling:
        return self._cfg["rsmp_alg"]

    @property
    def nodata_in(self) -> int:
        return self._cfg["nodata_in"]

    def merge(self, hdf_paths: list[str | Path], work_dir: str | Path) -> list[Path]:
        """Extract HDF layers and write one multi-band GeoTIFF per tile."""
        work_dir  = Path(work_dir)
        tif_paths = []
        for hdf in hdf_paths:
            hdf = Path(hdf)
            tif = work_dir / (hdf.stem + ".tif")
            merge_to_tif(hdf, self.layer_names, tif, self.nodata_in)
            tif_paths.append(tif)
        return tif_paths

    def resample(
        self,
        tif_paths: list[Path],
        hdf_paths: list[str | Path],
        work_dir: str | Path,
        base_name: str,
    ) -> tuple[list[Path], gpd.GeoDataFrame]:
        """Warp merged TIFs to WGS84 10°×10° tiles; also returns the skeleton."""
        skeleton = build_skeleton(hdf_paths)
        output_paths = resample_to_tiles(
            tif_paths, skeleton, work_dir, base_name, self.resample_alg, NODATA
        )
        return output_paths, skeleton


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def process(
    tag: str,
    hdf_paths: list[str | Path],
    work_dir: str | Path | None = None,
    run_merge: bool = True,
    run_resample: bool = True,
    save_skeleton_shp: bool = True,
) -> dict:
    """
    Full pipeline: read HDF tiles → write GeoTIFFs → resample → save skeleton.

    Parameters
    ----------
    tag               : label for output filenames (e.g. "lct_2020")
    hdf_paths         : list of HDF4 input files for one time step
    work_dir          : working directory (default: ./proc_<tag>)
    run_merge         : if False, skip HDF→TIF step (assumes TIFs exist)
    run_resample      : if False, skip warp step (assumes tiles exist)
    save_skeleton_shp : write skeleton to <work_dir>/skeleton.shp

    Returns
    -------
    dict with keys:
        "resampled_tifs" : list[Path]
        "skeleton_gdf"   : GeoDataFrame
        "skeleton_shp"   : Path | None
    """
    if work_dir is None:
        work_dir = Path(f"proc_{tag}")
    work_dir = Path(work_dir)

    importer  = Importer(hdf_paths)
    bname     = Path(hdf_paths[0]).name
    m         = importer._cfg["re_bname"].match(bname)
    base_name = m.group(0) if m else tag

    # Step 1 — HDF → GeoTIFF
    dir_merge = work_dir / "mrg"
    if run_merge:
        tif_paths = importer.merge(hdf_paths, dir_merge)
    else:
        tif_paths = [dir_merge / (Path(h).stem + ".tif") for h in hdf_paths]

    # Step 2 — warp to WGS84 10°×10° tiles
    dir_resamp = work_dir / "rsp"
    if run_resample:
        resampled, skeleton = importer.resample(
            tif_paths, hdf_paths, dir_resamp, base_name
        )
    else:
        resampled = sorted(dir_resamp.glob(f"{base_name}.r??c??.tif"))
        skeleton  = build_skeleton(hdf_paths)

    # Step 3 — save skeleton shapefile
    skeleton_shp = None
    if save_skeleton_shp:
        skeleton_shp = work_dir / "skeleton.shp"
        save_skeleton(skeleton, skeleton_shp)

    return {
        "resampled_tifs": resampled,
        "skeleton_gdf":   skeleton,
        "skeleton_shp":   skeleton_shp,
    }


# ---------------------------------------------------------------------------
# Quick-start usage example
# ---------------------------------------------------------------------------

def _usage_example():
    """
    >>> import glob
    >>> import geopandas as gpd
    >>> from rst_import_geopandas import list_datasets, process
    >>>
    >>> # Inspect a tile before processing
    >>> print(list_datasets("MCD12Q1.A2020001.h12v04.hdf"))
    >>>
    >>> hdf_files = sorted(glob.glob("downloads/MCD12Q1.A2020001.h??v??.hdf"))
    >>> result = process(tag="lct_2020", hdf_paths=hdf_files)
    >>>
    >>> gdf = result["skeleton_gdf"]
    >>> print(gdf.crs)     # EPSG:4326
    >>> print(gdf.head())
    >>>
    >>> # Load from the saved shapefile
    >>> gdf2 = gpd.read_file(result["skeleton_shp"])
    """
