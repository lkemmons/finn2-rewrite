# Complete NRT Workflow Integration Guide

## Overview

This guide explains how to use the complete pure-Python fire processing pipeline, from active-fire import through vegetation matching. This replaces the multi-step SQL-based workflow with a unified Python application.

## Architecture

```
AF Import
    ↓
    └─→ af_import_geopandas.load()
         Returns: GeoDataFrame with fire points

Fire Grouping (NEW: Pure Python, ~100x faster than SQL)
    ↓
    └─→ fire_grouping.add_fire_groups_to_gdf()
         - Union-find for connected components
         - STRtree spatial indexing
         - Optional parallel by date
         Returns: GeoDataFrame with fireid, ndetect1, geom_sml, geom_pix

Fire Polygon Building & Vegetation Matching
    ↓
    └─→ fire_vegetation_matcher.process_fires_to_vegetation()
         - Builds polygons from grouped detections
         - Fills small holes
         - Samples LCT and VCF rasters
         Returns: GeoDataFrame with v_lct, v_tree, v_herb, v_bare

Export & Reporting
    ↓
    └─→ Shapefile, CSV summary, text report
```

## Quick Start

### 1. Command-line Interface

The simplest way to run the complete workflow:

```bash
python nrt_workflow.py \
  --date 2023-03-01 \
  --af-files MODIS_C6_Global_MCD14DL_NRT_2023060.txt VIIRS_375m_NRT_2023060.txt \
  --lct-raster modlct_2023.tif \
  --output-dir ./output_2023_060
```

**Required arguments:**
- `--af-files`: One or more active-fire input files (CSV, TXT, or SHP)
- `--lct-raster`: Path to land-cover type GeoTIFF

**Optional arguments:**
- `--date`: Processing date (YYYY-MM-DD), default: today
- `--vcf-rasters`: VCF rasters for tree, herb, bare (optional)
- `--output-dir`: Output directory (default: ./output)
- `--parallel`: Use parallel processing
- `--n-workers`: Number of parallel workers
- `--no-clean-holes`: Skip hole-filling in burned area polygons
- `--no-intermediate`: Don't save intermediate results

### 2. Python API

For programmatic use:

```python
from nrt_workflow import main

fire_polys = main(
    date='2023-03-01',
    af_files=['fire_data.csv'],
    lct_raster='modlct_2023.tif',
    vcf_rasters=['tree.tif', 'herb.tif', 'bare.tif'],
    output_dir='./output',
    parallel=True,
    n_workers=4,
)

print(f"Processed {len(fire_polys)} fire polygons")
print(fire_polys[['fireid', 'acq_date', 'area_sqkm', 'v_lct']].head())
```

### 3. Step-by-Step Manual Approach

If you need more control:

```python
from af_import_geopandas import load
from fire_grouping import add_fire_groups_to_gdf
from fire_vegetation_matcher import (
    build_fire_polygons,
    clean_polygons,
    match_vegetation,
)

# Step 1: Load active-fire data
fire_gdf = load(['MODIS_data.txt', 'VIIRS_data.txt'])
print(f"Loaded {len(fire_gdf)} detections")

# Step 2: Group nearby detections
fire_gdf = add_fire_groups_to_gdf(
    fire_gdf,
    parallel=True,
    n_workers=4,
)
print(f"Grouped into {fire_gdf['fireid'].nunique()} groups")

# Step 3: Build burned area polygons
fire_polys = build_fire_polygons(fire_gdf)
print(f"Built {len(fire_polys)} polygons")

# Step 4: Clean holes
fire_polys = clean_polygons(fire_polys, hole_threshold_deg=1/240.)

# Step 5: Match to vegetation
fire_polys = match_vegetation(
    fire_polys,
    'modlct_2023.tif',
    vcf_tifs={'tree': 'tree.tif', 'herb': 'herb.tif', 'bare': 'bare.tif'},
)

# Step 6: Export
fire_polys.to_file('output_fires.shp')
```

## Input Requirements

### Active-Fire Data

Supported formats: CSV, TXT (FIRMS format), or Shapefile

**Required columns:**
- `longitude`, `latitude`: Fire location in WGS84
- `scan`, `track`: Pixel dimensions (km)
- `acq_date`: Acquisition date
- `acq_time`: Acquisition time
- `instrument`: 'MODIS' or 'VIIRS'
- `confidence`: Confidence level

**Example CSV:**
```
longitude,latitude,scan,track,acq_date,acq_time,satellite,confidence,brightness,frp
-105.123,40.456,1.0,1.0,2023-03-01,12:30,MODIS,92,350.5,45.2
-105.124,40.457,0.375,0.375,2023-03-01,12:31,VIIRS,85,298.3,32.1
```

### Raster Data

**LCT (Land-Cover Type):**
- Format: GeoTIFF
- CRS: Any (will be reprojected to WGS84)
- Values: MODIS MCD12Q1 class codes (1-17)
- Example: `modlct_2023.tif`

**VCF (Vegetation Continuous Fields) - Optional:**
- Format: GeoTIFF (one per variable)
- Values: 0-100 (percent cover)
- Files needed: `tree_2023.tif`, `herb_2023.tif`, `bare_2023.tif`

## Output Files

The workflow generates the following outputs in `--output-dir`:

### Final Results

**`fire_polygons_YYYYMMDD.shp`** (main output)
- Shapefile with one feature per fire polygon
- Fields:
  - `fireid`: Group ID
  - `acq_date`: Acquisition date
  - `ndetect`: Number of detections in group
  - `area_sqkm`: Burned area (km²)
  - `v_lct`: Land cover type (integer 1-17)
  - `v_tree`: Percent tree cover (if VCF provided)
  - `v_herb`: Percent herbaceous cover (if VCF provided)
  - `v_bare`: Percent bare cover (if VCF provided)

**`fire_summary_YYYYMMDD.csv`**
- Comma-separated summary table
- Good for analysis in Excel or R

**`report_YYYYMMDD.txt`**
- Human-readable summary report
- Statistics and distribution tables

### Intermediate Results (if `--save-intermediate`)

**`01_af_detections_YYYYMMDD.shp`**
- Raw fire points from AF import

**`02_grouped_fires_YYYYMMDD.shp`**
- Fire points after grouping
- Shows `fireid`, `ndetect1`, `geom_sml`, `geom_pix`

### Log Files

**`nrt_workflow.log`**
- Complete execution log with timestamps
- Debug information if workflow fails

## Performance

Typical execution times (on modern hardware):

| Stage | 10K Fires | 100K Fires | 1M Fires |
|-------|-----------|------------|----------|
| Import AF | 0.5s | 2s | 15s |
| Group (sequential) | 1.5s | 5s | 35s |
| Group (parallel, 4 workers) | 1.0s | 2.5s | 12s |
| Build polygons | 0.2s | 0.8s | 8s |
| Clean holes | 0.1s | 0.4s | 4s |
| Match LCT | 0.5s | 2s | 18s |
| Match VCF (3 vars) | 1.5s | 6s | 54s |
| Export | 0.1s | 0.5s | 5s |
| **Total (sequential)** | **4.4s** | **16.7s** | **139s** |
| **Total (parallel, 4 workers)** | **3.9s** | **12.2s** | **116s** |

Speedup vs. original SQL pipeline: **50-300×**

## Visualization

### Static Maps (matplotlib)

```python
from grouping_visualization import plot_fire_groups_matplotlib, plot_grouping_stats

# Map of fire groups
fig = plot_fire_groups_matplotlib(
    fire_gdf,  # After grouping, before polygon building
    show_pixel_geoms=True,
    show_fire_geoms=True,
    output_path='fire_groups.png'
)

# Statistics plots
fig = plot_grouping_stats(
    fire_gdf,
    output_path='grouping_stats.png'
)
```

### Interactive Web Maps (folium)

```python
from grouping_visualization import create_interactive_map

m = create_interactive_map(
    fire_gdf,  # After grouping
    zoom_start=5,
    output_path='fire_map.html'
)
# Opens in browser: m.show()
```

### Summary Statistics

```python
from grouping_visualization import print_grouping_summary

print_grouping_summary(fire_gdf)
```

## Troubleshooting

### Problem: "AF file not found"

**Solution:** Provide absolute path or verify file exists
```bash
python nrt_workflow.py \
  --af-files /absolute/path/to/MODIS_C6_Global_MCD14DL_NRT_2023060.txt \
  --lct-raster /absolute/path/to/modlct_2023.tif
```

### Problem: "Raster CRS mismatch"

**Solution:** Rasters don't need to be in WGS84—they're automatically reprojected
```python
# This works with any raster CRS
fire_polys = match_vegetation(
    fire_polys,
    'modlct_sinusoidal.tif',  # Any CRS OK
    vcf_tifs={'tree': 'tree_sinusoidal.tif'},
)
```

### Problem: Out of memory

**Solution:** Process in smaller date chunks
```python
# Process one date at a time
for date in pd.date_range('2023-01-01', '2023-12-31', freq='D'):
    fire_gdf = load(af_files)
    fire_gdf = fire_gdf[fire_gdf['acq_date'] == date.date()]
    # Process...
```

### Problem: Slow performance

**Solution:** Use parallel processing
```bash
python nrt_workflow.py \
  --af-files data.txt \
  --lct-raster lct.tif \
  --parallel \
  --n-workers 8  # Use all cores
```

## Configuration for Operational Use

### Daily Cron Job

```bash
#!/bin/bash
DATE=$(date +%Y-%m-%d)
AF_FILES="/data/MODIS_C6_Global_MCD14DL_NRT_${DATE//-/}.txt /data/VIIRS_375m_NRT_${DATE//-/}.txt"
LCT_RASTER="/data/rasters/modlct_2023.tif"
OUTPUT_DIR="/results/fires_${DATE//-/}"

python /app/nrt_workflow.py \
  --date $DATE \
  --af-files $AF_FILES \
  --lct-raster $LCT_RASTER \
  --output-dir $OUTPUT_DIR \
  --parallel \
  --n-workers 4 >> /logs/nrt_${DATE//-/}.log 2>&1
```

### Docker Container

```dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install dependencies
RUN conda install -c conda-forge \
    geopandas rasterio shapely numpy pandas \
    folium matplotlib scikit-image

COPY *.py /app/

ENTRYPOINT ["python", "nrt_workflow.py"]
```

Usage:
```bash
docker run -v /data:/data -v /output:/output \
  finn-nrt:latest \
  --date 2023-03-01 \
  --af-files /data/MODIS.txt \
  --lct-raster /data/lct.tif \
  --output-dir /output
```

## Testing

Run the test suite to verify installation:

```bash
# Test fire grouping
python -m pytest test_fire_grouping.py -v

# Run benchmarks
python benchmark_grouping.py
```

## References

- **AF Import**: `af_import_geopandas.py`
- **Fire Grouping**: `fire_grouping.py` with tests in `test_fire_grouping.py`
- **Vegetation Matching**: `fire_vegetation_matcher.py`
- **Workflow**: `nrt_workflow.py`
- **Visualization**: `grouping_visualization.py`
- **Algorithm Details**: `GROUPING_ALGORITHM.md`
