"""Complete daily NRT (near real-time) fire processing workflow.

Orchestrates the full pipeline from active-fire import through vegetation matching.

Usage
-----
    python nrt_workflow.py \
        --date 2023-03-01 \
        --af-files MODIS_C6_Global_MCD14DL_NRT_2023060.txt VIIRS_375m_NRT_2023060.txt \
        --lct-raster modlct_2023_tile_10_10.tif \
        --vcf-rasters tree_2023.tif herb_2023.tif bare_2023.tif \
        --output-dir ./output_2023_060
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from af_import_geopandas import load as load_af_data
from fire_grouping import add_fire_groups_to_gdf
from fire_vegetation_matcher import process_fires_to_vegetation
from finn2_calc_emissions_v25 import finn2_calc_emissions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('nrt_workflow.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class NRTConfig:
    """Configuration for NRT workflow."""

    def __init__(
        self,
        date: datetime.date | None = None,
        af_files: list[str | Path] | None = None,
        lct_raster: str | Path | None = None,
        vcf_rasters: dict[str, str | Path] | None = None,
        output_dir: str | Path = "./output",
        clean_holes: bool = True,
        parallel_grouping: bool = False,
        n_workers: int | None = None,
        save_intermediate: bool = True,
    ):
        """Initialize configuration."""
        self.date = date or datetime.date.today()
        self.af_files = af_files or []
        self.lct_raster = Path(lct_raster) if lct_raster else None
        self.vcf_rasters = {k: Path(v) for k, v in vcf_rasters.items()} if vcf_rasters else None
        self.output_dir = Path(output_dir)
        self.clean_holes = clean_holes
        self.parallel_grouping = parallel_grouping
        self.n_workers = n_workers
        self.save_intermediate = save_intermediate

    def validate(self) -> None:
        """Validate configuration."""
        if not self.af_files:
            raise ValueError("No AF files provided")

        for af_file in self.af_files:
            if not Path(af_file).exists():
                raise FileNotFoundError(f"AF file not found: {af_file}")

        if not self.lct_raster:
            raise ValueError("LCT raster path required")

        if not self.lct_raster.exists():
            raise FileNotFoundError(f"LCT raster not found: {self.lct_raster}")

        if self.vcf_rasters:
            for var, path in self.vcf_rasters.items():
                if not path.exists():
                    raise FileNotFoundError(f"VCF raster not found ({var}): {path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Workflow Stages
# ---------------------------------------------------------------------------

def stage_import_af(config: NRTConfig) -> gpd.GeoDataFrame:
    """
    Stage 1: Import active-fire data from CSV/TXT/SHP.

    Parameters
    ----------
    config : NRTConfig

    Returns
    -------
    GeoDataFrame of fire detections
    """
    log.info(f"\n{'='*70}")
    log.info("STAGE 1: Import Active-Fire Data")
    log.info(f"{'='*70}")

    log.info(f"Loading {len(config.af_files)} AF file(s)...")
    for af_file in config.af_files:
        log.info(f"  - {af_file}")

    fire_gdf = load_af_data(config.af_files)

    log.info(f"\nLoaded {len(fire_gdf):,} fire detections")
    log.info(f"Columns: {list(fire_gdf.columns)}")
    log.info(f"Date range: {fire_gdf['acq_date'].min()} to {fire_gdf['acq_date'].max()}")
    log.info(f"Instruments: {fire_gdf['instrument'].unique()}")
    log.info(f"Confidence: {fire_gdf['confidence'].unique()}")

    # Save intermediate results
    if config.save_intermediate:
        af_output = config.output_dir / f"01_af_detections_{config.date.strftime('%Y%m%d')}.shp"
        fire_gdf.to_file(af_output)
        log.info(f"Saved intermediate: {af_output}")

    return fire_gdf


def stage_group_fires(config: NRTConfig, fire_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Stage 2: Group nearby fire detections into fire events.

    Parameters
    ----------
    config : NRTConfig
    fire_gdf : GeoDataFrame from stage_import_af

    Returns
    -------
    GeoDataFrame with added columns: fireid, geom_sml, geom_pix, ndetect1
    """
    log.info(f"\n{'='*70}")
    log.info("STAGE 2: Group Fire Detections")
    log.info(f"{'='*70}")

    log.info(f"Grouping {len(fire_gdf):,} detections by spatial proximity...")

    fire_gdf = add_fire_groups_to_gdf(
        fire_gdf,
        parallel=config.parallel_grouping,
        n_workers=config.n_workers,
    )

    n_groups = fire_gdf['fireid'].nunique()
    log.info(f"Grouped into {n_groups:,} fire groups")
    log.info(f"Group size stats:")
    log.info(f"  Min: {fire_gdf['ndetect1'].min()}")
    log.info(f"  Max: {fire_gdf['ndetect1'].max()}")
    log.info(f"  Mean: {fire_gdf['ndetect1'].mean():.1f}")
    log.info(f"  Median: {fire_gdf['ndetect1'].median():.0f}")

    # Save intermediate results
    if config.save_intermediate:
        grouped_output = config.output_dir / f"02_grouped_fires_{config.date.strftime('%Y%m%d')}.shp"
        fire_gdf.to_file(grouped_output)
        log.info(f"Saved intermediate: {grouped_output}")

    return fire_gdf


def stage_match_vegetation(
    config: NRTConfig,
    fire_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Stage 3: Match fire polygons to vegetation types via raster sampling.

    Parameters
    ----------
    config : NRTConfig
    fire_gdf : GeoDataFrame from stage_group_fires

    Returns
    -------
    GeoDataFrame with added columns: v_lct, v_tree, v_herb, v_bare (optional)
    """
    log.info(f"\n{'='*70}")
    log.info("STAGE 3: Match Fires to Vegetation")
    log.info(f"{'='*70}")

    log.info(f"LCT Raster: {config.lct_raster}")
    if config.vcf_rasters:
        for var, path in config.vcf_rasters.items():
            log.info(f"VCF {var}: {path}")

    fire_polys = process_fires_to_vegetation(
        fire_gdf,
        config.lct_raster,
        config.vcf_rasters,
        clean_holes=config.clean_holes,
    )

    log.info(f"\nVegetation matching complete")
    log.info(f"Fire polygons: {len(fire_polys):,}")
    log.info(f"LCT values sampled: {fire_polys['v_lct'].notna().sum():,}")

    if config.vcf_rasters:
        for var in config.vcf_rasters.keys():
            col = f'v_{var}'
            if col in fire_polys.columns:
                log.info(f"{col} values sampled: {fire_polys[col].notna().sum():,}")

    return fire_polys


def stage_export_results(
    config: NRTConfig,
    fire_polys: gpd.GeoDataFrame,
) -> None:
    """
    Stage 4: Export final results.

    Parameters
    ----------
    config : NRTConfig
    fire_polys : GeoDataFrame from stage_match_vegetation
    """
    log.info(f"\n{'='*70}")
    log.info("STAGE 4: Export Results")
    log.info(f"{'='*70}")

    # Main output shapefile
    tag = config.date.strftime('%Y%m%d')
    shp_output = config.output_dir / f"fire_polygons_{tag}.shp"
    fire_polys.to_file(shp_output)
    log.info(f"Saved output shapefile: {shp_output}")

    # CSV summary
    csv_output = config.output_dir / f"fire_summary_{tag}.csv"
    summary_df = fire_polys[[
        'fireid', 'acq_date', 'ndetect', 'area_sqkm',
        'v_lct',
    ]].copy()

    # Add VCF columns if present
    for col in fire_polys.columns:
        if col.startswith('v_') and col != 'v_lct':
            summary_df[col] = fire_polys[col]

    summary_df.to_csv(csv_output, index=False)
    log.info(f"Saved CSV summary: {csv_output}")

    # Statistics report
    report_output = config.output_dir / f"report_{tag}.txt"
    with open(report_output, 'w') as f:
        f.write(f"NRT Fire Processing Report\n")
        f.write(f"Date: {config.date}\n")
        f.write(f"\n{'='*70}\n")
        f.write(f"SUMMARY STATISTICS\n")
        f.write(f"{'='*70}\n")

        f.write(f"\nFire Detections: {len(fire_polys):,}\n")
        f.write(f"Fire Groups: {fire_polys['fireid'].nunique():,}\n")
        f.write(f"Total Burned Area: {fire_polys['area_sqkm'].sum():,.1f} km²\n")

        f.write(f"\nArea Statistics:\n")
        f.write(f"  Min: {fire_polys['area_sqkm'].min():.2f} km²\n")
        f.write(f"  Max: {fire_polys['area_sqkm'].max():.2f} km²\n")
        f.write(f"  Mean: {fire_polys['area_sqkm'].mean():.2f} km²\n")
        f.write(f"  Median: {fire_polys['area_sqkm'].median():.2f} km²\n")

        f.write(f"\nLand Cover Type Distribution:\n")
        lct_counts = fire_polys['v_lct'].value_counts().sort_index()
        for lct_val, count in lct_counts.items():
            if pd.notna(lct_val):
                f.write(f"  LCT {int(lct_val)}: {count} fires\n")

        f.write(f"\nDate Distribution:\n")
        date_counts = fire_polys['acq_date'].value_counts().sort_index()
        for date_val, count in date_counts.items():
            f.write(f"  {date_val}: {count} fires\n")

    log.info(f"Saved report: {report_output}")


def print_summary(fire_polys: gpd.GeoDataFrame) -> None:
    """Print final summary to console."""
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Fire polygons: {len(fire_polys):,}")
    log.info(f"Total burned area: {fire_polys['area_sqkm'].sum():,.1f} km²")
    log.info(f"Mean polygon size: {fire_polys['area_sqkm'].mean():.2f} km²")
    log.info(f"\nVegetation matching:")
    log.info(f"  LCT sampled: {fire_polys['v_lct'].notna().sum():,}/{len(fire_polys):,}")
    if 'v_tree' in fire_polys.columns:
        log.info(f"  Tree cover: {fire_polys['v_tree'].notna().sum():,}/{len(fire_polys):,}")
    if 'v_herb' in fire_polys.columns:
        log.info(f"  Herb cover: {fire_polys['v_herb'].notna().sum():,}/{len(fire_polys):,}")
    if 'v_bare' in fire_polys.columns:
        log.info(f"  Bare cover: {fire_polys['v_bare'].notna().sum():,}/{len(fire_polys):,}")
    log.info(f"{'='*70}\n")

import sys
import os


def process_fires(fire_data, config):
    """Calculate emissions of compounds using the fire polygons and corresponding vegetation."""
    
    # Calculate emissions using the v2.5 routine
    print(f"Calculating emissions for {len(fire_polygons_with_veg)} fire events...")
    
    # This routine typically requires the fire/veg dataframe, 
    # emission factor tables, and fuel loading data.
    emissions_results = finn2_calc_emissions(
        fire_polygons_with_veg,
        ef_file=config['ef_table_path'],
        fuel_file=config['fuel_load_path'],
        output_file=emissions_output_file
    )
    
    return emissions_results


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main(
    date: str | None = None,
    af_files: list[str] | None = None,
    lct_raster: str | None = None,
    vcf_rasters: list[str] | None = None,
    output_dir: str = "./output",
    clean_holes: bool = True,
    parallel: bool = False,
    n_workers: int | None = None,
    save_intermediate: bool = True,
) -> gpd.GeoDataFrame:
    """
    Run complete NRT workflow.

    Parameters
    ----------
    date : processing date (YYYY-MM-DD format)
    af_files : list of active-fire file paths
    lct_raster : path to LCT raster file
    vcf_rasters : list of VCF raster paths (tree, herb, bare order)
    output_dir : output directory
    clean_holes : if True, fill small holes in burned area polygons
    parallel : if True, use parallel processing for grouping
    n_workers : number of parallel workers
    save_intermediate : if True, save intermediate results

    Returns
    -------
    GeoDataFrame of final fire polygons with vegetation
    """
    # Parse date
    if date:
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    else:
        date_obj = datetime.date.today()

    # Parse VCF rasters
    vcf_dict = None
    if vcf_rasters and len(vcf_rasters) > 0:
        vcf_vars = ['tree', 'herb', 'bare']
        vcf_dict = {var: path for var, path in zip(vcf_vars, vcf_rasters[:3])}

    # Create config
    config = NRTConfig(
        date=date_obj,
        af_files=af_files or [],
        lct_raster=lct_raster,
        vcf_rasters=vcf_dict,
        output_dir=output_dir,
        clean_holes=clean_holes,
        parallel_grouping=parallel,
        n_workers=n_workers,
        save_intermediate=save_intermediate,
    )

    # Validate
    try:
        config.validate()
    except (ValueError, FileNotFoundError) as e:
        log.error(f"Configuration error: {e}")
        return None

    # Run workflow
    try:
        log.info(f"Starting NRT Workflow")
        log.info(f"Date: {config.date}")
        log.info(f"Output dir: {config.output_dir}")

        # Stage 1: Import
        fire_gdf = stage_import_af(config)

        # Stage 2: Group
        fire_gdf = stage_group_fires(config, fire_gdf)

        # Stage 3: Match vegetation
        fire_polys = stage_match_vegetation(config, fire_gdf)

        # Stage 4: Export
        stage_export_results(config, fire_polys)

        # Summary
        print_summary(fire_polys)

        log.info("✓ Workflow completed successfully")
        return fire_polys

    except Exception as e:
        log.error(f"Workflow failed: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete NRT fire processing workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  # Basic usage with MODIS and VIIRS data
  python nrt_workflow.py \\
    --date 2023-03-01 \\
    --af-files MODIS_C6_Global_MCD14DL_NRT_2023060.txt VIIRS_375m_NRT_2023060.txt \\
    --lct-raster modlct_2023.tif \\
    --output-dir ./output_2023_060

  # With VCF continuous fields and parallel processing
  python nrt_workflow.py \\
    --date 2023-03-01 \\
    --af-files fire_data_2023060.csv \\
    --lct-raster modlct_2023.tif \\
    --vcf-rasters tree_2023.tif herb_2023.tif bare_2023.tif \\
    --output-dir ./output \\
    --parallel \\
    --n-workers 4
        ''',
    )

    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Processing date (YYYY-MM-DD), default: today',
    )

    parser.add_argument(
        '--af-files',
        type=str,
        nargs='+',
        required=True,
        help='Active-fire input files (CSV, TXT, or SHP)',
    )

    parser.add_argument(
        '--lct-raster',
        type=str,
        required=True,
        help='Land-cover type raster (GeoTIFF)',
    )

    parser.add_argument(
        '--vcf-rasters',
        type=str,
        nargs='*',
        help='VCF rasters (tree, herb, bare) - all optional',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory (default: ./output)',
    )

    parser.add_argument(
        '--no-clean-holes',
        action='store_false',
        dest='clean_holes',
        help='Do not fill small holes in burned area polygons',
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel processing for fire grouping',
    )

    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)',
    )

    parser.add_argument(
        '--no-intermediate',
        action='store_false',
        dest='save_intermediate',
        help='Do not save intermediate results',
    )

    args = parser.parse_args()

    # Run workflow
    fire_polys = main(
        date=args.date,
        af_files=args.af_files,
        lct_raster=args.lct_raster,
        vcf_rasters=args.vcf_rasters,
        output_dir=args.output_dir,
        clean_holes=args.clean_holes,
        parallel=args.parallel,
        n_workers=args.n_workers,
        save_intermediate=args.save_intermediate,
    )

    sys.exit(0 if fire_polys is not None else 1)
