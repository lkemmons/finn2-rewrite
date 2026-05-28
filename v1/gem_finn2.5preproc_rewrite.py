#!/usr/bin/env python3
"""
FINN3 Step 1: Fire Processing & Zonal Statistics
A pure python re-write of FINNv2.5
Loads daily active fires, clusters them into polygons, assigns global regions,
and extracts vegetation statistics from MODIS LCT and VCF HDF files.
To run on casper:
 conda activate npl-2026a
 (npl-2026a) python finn3_step1_proc_fires.py --sdate 2026100 --year 2023 --outdir {dir}
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
from pyhdf.SD import SD, SDC
from shapely.geometry import Point, box
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm

# =====================================================================
# CONSTANTS
# =====================================================================
MODIS_X_EXTENT = 20015109.354
MODIS_Y_EXTENT = 10007554.677
TILE_SIZE = 1111950.5196666666
MODIS_CRS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
EQUAL_AREA_CRS = "EPSG:6933"

# =====================================================================
# FUNCTIONS
# =====================================================================

def extract_instrument(filepath):
    filename = os.path.basename(filepath).upper()
    if 'MODIS' in filename:
        return 'MODIS'
    elif 'VIIRS' in filename:
        return 'VIIRS'
    return 'UNKNOWN'

def load_and_combine_fires(file_list):
    gdfs = []
    for filepath in file_list:
        instrument = extract_instrument(filepath)
        print(f"Loading {instrument} data from {os.path.basename(filepath)}...")
        
        df = pd.read_csv(filepath)
        initial_len = len(df)
        
        # Filter low confidence fire detections
        if instrument == 'MODIS':
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
            df = df[df['confidence'] >= 20]
        elif instrument == 'VIIRS':
            df['confidence'] = df['confidence'].astype(str).str.lower()
            valid_viirs_conf = ['nominal', 'n', 'high', 'h'] 
            df = df[df['confidence'].isin(valid_viirs_conf)]
            
        print(f"  -> Dropped {initial_len - len(df)} low confidence points.")

        # Filter out persistent sources
        len_before_type = len(df)
        type_col = 'TYPE' if 'TYPE' in df.columns else 'type' if 'type' in df.columns else None
        
        if type_col:
            df = df[df[type_col] == 0]
            type_dropped = len_before_type - len(df)
            if type_dropped > 0:
                print(f"  -> Dropped {type_dropped} persistent thermal anomalies (volcanoes/gas flares).")
        else:
            print("  -> 'type' column not found (likely NRT data). Bypassing persistent source filter.")
            
        # Convert to GeoDataFrame
        if not df.empty:
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
                crs="EPSG:4326"
            )
            gdf['instrument'] = instrument
            gdfs.append(gdf)
        else:
            print("  -> All points dropped for this file.")
        
    if gdfs:
        combined_gdf = pd.concat(gdfs, ignore_index=True)
        print(f"\nTotal combined fire points after filtering: {len(combined_gdf)}")
        return combined_gdf
    else:
        print("\nNo active fires remained after filtering.")
        return None

def load_region_shapefile(shp_path):
    if not os.path.exists(shp_path):
        print(f"Region shapefile {shp_path} not found. Please download it.")
        return None
    
    print("Loading region number into GeoPandas...")
    gdf_regions = gpd.read_file(shp_path, columns=['Region_num', 'geometry'], engine='pyogrio')
    
    if gdf_regions.crs != "EPSG:4326":
        gdf_regions = gdf_regions.to_crs("EPSG:4326")

    gdf_regions['Region_num'] = gdf_regions['Region_num'].fillna(0)
    return gdf_regions

def assign_regions_with_nearest_neighbor(grouped_centroids, regions_gdf, max_search_meters=25000):
    print(f"Assigning FINN regions (snapping up to {max_search_meters/1000} km off the coast)...")
    
    centroids_metric = grouped_centroids.to_crs("EPSG:6933")
    regions_metric = regions_gdf.to_crs("EPSG:6933")
    
    joined_metric = gpd.sjoin_nearest(
        centroids_metric, 
        regions_metric, 
        how="left", 
        max_distance=max_search_meters,
        distance_col="snap_distance_m"
    )
    
    final_centroids = joined_metric.to_crs("EPSG:4326")
    
    if 'index_right' in final_centroids.columns:
        final_centroids = final_centroids.drop(columns=['index_right'])
        
    unassigned = final_centroids['Region_num'].isna().sum()
    if unassigned > 0:
        print(f"  -> WARNING: {unassigned} fires are still further than {max_search_meters/1000} km from any region!")
        
    return final_centroids

def create_pixel_footprints(gdf):
    radius_map = {
        'MODIS': 500.0,
        'VIIRS': 187.5,
        'UNKNOWN': 500.0
    }
    distances = gdf['instrument'].map(radius_map).fillna(500.0)
    gdf["geometry"] = gdf.buffer(distances, cap_style=3)
    return gdf

def cluster_nearby_fires(gdf_fires, max_distance_meters):
    gdf_proj = gdf_fires.to_crs("EPSG:6933")
    coords = np.array((gdf_proj.geometry.x, gdf_proj.geometry.y)).T
    
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_distance_meters)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    G.add_edges_from(pairs)
    
    clusters = list(nx.connected_components(G))
    
    gdf_fires['cluster_id'] = -1
    for cluster_id, node_indices in enumerate(clusters):
        gdf_fires.iloc[list(node_indices), gdf_fires.columns.get_loc('cluster_id')] = cluster_id
        
    return gdf_fires

def group_adjacent_fires(daily_fires_gdf, max_distance_meters=1000, fill_gaps=True):
    print(f"Grouping {len(daily_fires_gdf)} fires within {max_distance_meters}m of each other...")

    if daily_fires_gdf.crs is None or daily_fires_gdf.crs.to_string() != "EPSG:4326":
        daily_fires_gdf = daily_fires_gdf.to_crs("EPSG:4326")
    
    daily_fires_gdf['orig_lon'] = daily_fires_gdf.geometry.x
    daily_fires_gdf['orig_lat'] = daily_fires_gdf.geometry.y
    
    fires_proj = daily_fires_gdf.to_crs(EQUAL_AREA_CRS)
    fires_proj = cluster_nearby_fires(fires_proj, max_distance_meters)
    fires_buffered = create_pixel_footprints(fires_proj)

    dissolved = fires_buffered.dissolve(by='cluster_id').reset_index()

    if fill_gaps:
        bridge_dist = max_distance_meters / 2.0
        dissolved['geometry'] = dissolved.geometry.buffer(bridge_dist).buffer(-bridge_dist)

    dissolved['burned_area_sqm'] = dissolved.geometry.area

    points_with_clusters = fires_proj.merge(
        dissolved[['cluster_id', 'burned_area_sqm']], 
        on='cluster_id', 
        how='left'
    )

    grouped_stats = points_with_clusters.groupby('cluster_id').agg(
        point_count=('cluster_id', 'count'),
        instruments=('instrument', lambda x: '+'.join(set(x))) if 'instrument' in points_with_clusters.columns else ('orig_lat', 'count'),
        centroid_lon=('orig_lon', 'mean'),
        centroid_lat=('orig_lat', 'mean'),
        burned_area_sqm=('burned_area_sqm', 'first')
    ).reset_index()

    if 'frp' in points_with_clusters.columns:
        grouped_stats['mean_frp'] = points_with_clusters.groupby('cluster_id')['frp'].mean().values

    grouped_centroids = gpd.GeoDataFrame(
        grouped_stats, 
        geometry=gpd.points_from_xy(grouped_stats.centroid_lon, grouped_stats.centroid_lat),
        crs="EPSG:4326"
    )
    
    cluster_polygons = dissolved[['cluster_id', 'burned_area_sqm', 'geometry']].to_crs("EPSG:4326")
    
    print(f"Reduced to {len(grouped_centroids)} contiguous fire events.")
    return grouped_centroids, cluster_polygons

def extract_multiple_pure_zonal_stats(polygons_gdf, hdf_file_list, dataset_names, pixels_per_tile, stat_type="mean"):
    if len(hdf_file_list) == 0:
        print(f"  -> CRITICAL WARNING: No HDF files provided.")
        return pd.DataFrame(index=polygons_gdf.index, columns=dataset_names)

    print(f"Projecting {len(polygons_gdf)} polygons to MODIS Sinusoidal...")
    poly_modis = polygons_gdf.to_crs(MODIS_CRS).copy()
    
    results = {
        idx: {ds: {"sum_val_area": 0.0, "sum_area": 0.0, "class_areas": {}} for ds in dataset_names}
        for idx in poly_modis.index
    }
    
    pixel_size = TILE_SIZE / pixels_per_tile
    bounds = poly_modis.bounds
    min_h = np.floor((bounds.minx + MODIS_X_EXTENT) / TILE_SIZE).astype(int)
    max_h = np.floor((bounds.maxx + MODIS_X_EXTENT) / TILE_SIZE).astype(int)
    min_v = np.floor((MODIS_Y_EXTENT - bounds.maxy) / TILE_SIZE).astype(int)
    max_v = np.floor((MODIS_Y_EXTENT - bounds.miny) / TILE_SIZE).astype(int)
    
    required_tiles = set()
    for i in range(len(poly_modis)):
        for h in range(min_h.iloc[i], max_h.iloc[i] + 1):
            for v in range(min_v.iloc[i], max_v.iloc[i] + 1):
                required_tiles.add(f"h{min(max(h, 0), 35):02d}v{min(max(v, 0), 17):02d}")

    missing_tiles = set()

    for tile in tqdm(required_tiles, desc="Multi-Variable Zonal Stats"):
        h_val, v_val = int(tile[1:3]), int(tile[4:6])
        
        tile_x_min = -MODIS_X_EXTENT + (h_val * TILE_SIZE)
        tile_y_max = MODIS_Y_EXTENT - (v_val * TILE_SIZE)
        tile_box = box(tile_x_min, tile_y_max - TILE_SIZE, tile_x_min + TILE_SIZE, tile_y_max)
        
        tile_mask = (min_h <= h_val) & (max_h >= h_val) & (min_v <= v_val) & (max_v >= v_val)
        intersecting_polys = poly_modis[tile_mask]
        
        if intersecting_polys.empty:
            continue

        file_path = next((f for f in hdf_file_list if tile.lower() in f.lower()), None)
        
        if file_path:
            try:
                hdf = SD(file_path, SDC.READ)
                arrays = {}
                for ds_name in dataset_names:
                    arr = hdf.select(ds_name).get().astype(float)
                    if stat_type == "mean":
                        arr[arr > 100] = np.nan
                    elif stat_type == "categorical":
                        arr[arr == 255] = np.nan
                    arrays[ds_name] = arr
                
                for idx, row in intersecting_polys.iterrows():
                    poly_in_tile = row.geometry.intersection(tile_box)
                    if poly_in_tile.is_empty:
                        continue
                        
                    p_minx, p_miny, p_maxx, p_maxy = poly_in_tile.bounds
                    
                    px_min = max(0, int((p_minx - tile_x_min) / pixel_size))
                    px_max = min(pixels_per_tile - 1, int((p_maxx - tile_x_min) / pixel_size))
                    py_min = max(0, int((tile_y_max - p_maxy) / pixel_size))
                    py_max = min(pixels_per_tile - 1, int((tile_y_max - p_miny) / pixel_size))
                    
                    for py in range(py_min, py_max + 1):
                        for px in range(px_min, px_max + 1):
                            pixel_xmin = tile_x_min + (px * pixel_size)
                            pixel_xmax = pixel_xmin + pixel_size
                            pixel_ymax = tile_y_max - (py * pixel_size)
                            pixel_ymin = pixel_ymax - pixel_size
                            pixel_poly = box(pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax)
                            
                            overlap_area = poly_in_tile.intersection(pixel_poly).area
                            
                            if overlap_area > 0:
                                for ds_name in dataset_names:
                                    val = arrays[ds_name][py, px]
                                    if np.isnan(val):
                                        continue
                                        
                                    if stat_type == "mean":
                                        results[idx][ds_name]["sum_val_area"] += (val * overlap_area)
                                        results[idx][ds_name]["sum_area"] += overlap_area
                                    elif stat_type == "categorical":
                                        results[idx][ds_name]["class_areas"][val] = results[idx][ds_name]["class_areas"].get(val, 0) + overlap_area
                                    
            except Exception as e:
                print(f"\n  -> ERROR reading {os.path.basename(file_path)}: {e}")
        else:
            missing_tiles.add(tile)

    if missing_tiles:
        print(f"\n  -> WARNING: Missing files for tiles: {', '.join(missing_tiles)}")

    final_outputs = {ds: [] for ds in dataset_names}
    for idx in poly_modis.index:
        for ds_name in dataset_names:
            res = results[idx][ds_name]
            if stat_type == "mean":
                if res["sum_area"] > 0:
                    final_outputs[ds_name].append(res["sum_val_area"] / res["sum_area"])
                else:
                    final_outputs[ds_name].append(None)
            elif stat_type == "categorical":
                total_area = sum(res["class_areas"].values())
                if total_area > 0:
                    fractions = {float(k): (v_area / total_area) for k, v_area in res["class_areas"].items()}
                    final_outputs[ds_name].append(fractions)
                else:
                    final_outputs[ds_name].append(None)

    return pd.DataFrame(final_outputs, index=poly_modis.index)

def export_exploded_centroids_to_csv(grouped_centroids, output_csv_path):
    print("Preparing data for CSV export...")
    export_df = grouped_centroids.copy()
    
    export_df['lct_items'] = export_df['lct_fractions'].apply(
        lambda x: list(x.items()) if isinstance(x, dict) else []
    )
    
    export_df = export_df.explode('lct_items')
    export_df['lct_type'] = export_df['lct_items'].apply(lambda x: x[0] if isinstance(x, tuple) else None)
    export_df['lct_fraction'] = export_df['lct_items'].apply(lambda x: x[1] if isinstance(x, tuple) else None)
    export_df['lct_type'] = export_df['lct_type'].astype('Int64')
    
    columns_to_drop = ['lct_fractions', 'lct_items']
    if 'geometry' in export_df.columns:
        columns_to_drop.append('geometry')
    if 'instruments' in export_df.columns:
        columns_to_drop.append('instruments')
    if 'burned_area_sqm' in export_df.columns:
        columns_to_drop.append('burned_area_sqm')
    
    export_df = export_df.drop(columns=columns_to_drop)
    export_df.to_csv(output_csv_path, index=False)
    print(f"  -> Successfully saved exploded records to {output_csv_path}")
    return export_df

def save_intermediate_geometries(centroids_gdf, polygons_gdf, output_prefix="outputs/finn_intermediate"):
    out_file = f"{output_prefix}_geometries.gpkg"
    print(f"Saving spatial data to {out_file}...")
    
    cent_clean = centroids_gdf.copy()
    if 'lct_fractions' in cent_clean.columns:
        cent_clean['lct_fractions'] = cent_clean['lct_fractions'].astype(str)
        
    poly_clean = polygons_gdf.copy()
    if 'lct_fractions' in poly_clean.columns:
        poly_clean['lct_fractions'] = poly_clean['lct_fractions'].astype(str)

    cent_clean.to_file(out_file, layer='centroids', driver="GPKG")
    poly_clean.to_file(out_file, layer='polygons', driver="GPKG")
    print("  -> Save complete.")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="FINN3 Step 1: Process Fires and Extract Vegetation")
    parser.add_argument("--sdate", type=str, required=True, help="Date for fire count files (e.g., 2026100)")
    parser.add_argument("--year", type=str, required=True, help="Year to use for LCT and VCF files (e.g., 2023)")
    parser.add_argument("--indir", type=str, default='/glade/derecho/scratch/emmons/finn3_inputs/', help="Base input directory")
    parser.add_argument("--outdir", type=str, default='/glade/derecho/scratch/emmons/finn3_output/', help="Output directory")
    parser.add_argument("--regions", type=str, default='/glade/u/home/emmons/EMISSIONS/finn2-rewrite/process_fires/finn_run_data/All_Countries.shp', help="Path to global regions shapefile")
    
    args = parser.parse_args()

    # 1. Setup paths
    os.makedirs(args.outdir, exist_ok=True)
    
    file_mod = os.path.join(args.indir, f'MODIS_C6_1_Global_MCD14DL_NRT_{args.sdate}.txt')
    file_vrs = os.path.join(args.indir, f'SUOMI_VIIRS_C2_Global_VNP14IMGTDL_NRT_{args.sdate}.txt')
    
    path_lct = os.path.join(args.indir, f'modis_lct_{args.year}')
    path_vcf = os.path.join(args.indir, f'modis_vcf_{args.year}')
    
    # Restrict HDF files by the requested Year
    lct_files = glob.glob(f"{path_lct}/*MCD12Q1.A{args.year}*.hdf") + glob.glob(f"{path_lct}/*MCD12Q1.A{args.year}*.HDF")
    vcf_files = glob.glob(f"{path_vcf}/*MOD44B.A{args.year}*.hdf") + glob.glob(f"{path_vcf}/*MOD44B.A{args.year}*.HDF")

    print(f"--- Running FINN3 Step 1 ---")
    print(f"Date: {args.sdate} | Base Vegetation Year: {args.year}")
    print(f"Found {len(lct_files)} LCT files and {len(vcf_files)} VCF files.\n")

    # 2. Load and combine fire pixels
    file_list = [file_mod, file_vrs]
    gdf_fires = load_and_combine_fires(file_list)
    if gdf_fires is None:
        print("No fires to process. Exiting.")
        return

    # 3. Group adjacent fires
    grouped_centroids, grouped_polygons = group_adjacent_fires(gdf_fires, max_distance_meters=1000, fill_gaps=True)
    grouped_centroids['area_sqkm'] = grouped_centroids['burned_area_sqm'] / 1e6
    grouped_polygons['area_sqkm'] = grouped_polygons['burned_area_sqm'] / 1e6

    # 4. Assign region numbers
    print("Assigning Region Numbers...")
    regions_gdf = load_region_shapefile(args.regions)
    if regions_gdf is not None:
        grouped_centroids = assign_regions_with_nearest_neighbor(grouped_centroids, regions_gdf, max_search_meters=100000)
        grouped_centroids['Region_num'] = grouped_centroids['Region_num'].fillna(0).astype(int)

    # 5. Extract Zonal Statistics
    grouped_polygons['lct_fractions'] = extract_multiple_pure_zonal_stats(
        grouped_polygons, 
        lct_files, 
        ["LC_Type1"], 
        pixels_per_tile=2400, 
        stat_type="categorical"
    )

    vcf_variables_orig = ["Percent_Tree_Cover", "Percent_NonTree_Vegetation", "Percent_NonVegetated"]
    vcf_variables_new = ["vcf_tree", "vcf_herb", "vcf_bare"]

    grouped_polygons[vcf_variables_new] = extract_multiple_pure_zonal_stats(
        grouped_polygons, 
        vcf_files, 
        vcf_variables_orig, 
        pixels_per_tile=4800, 
        stat_type="mean"
    ).values

    # Merge polygon stats back to centroids
    columns_to_merge = ['cluster_id', 'lct_fractions', 'vcf_tree', 'vcf_herb', 'vcf_bare']
    grouped_centroids = grouped_centroids.merge(
        grouped_polygons[columns_to_merge], 
        on='cluster_id', 
        how='left'
    )

    # 6. Save Outputs
    output_csv = os.path.join(args.outdir, f"processed_fires_{args.sdate}.csv")
    export_exploded_centroids_to_csv(grouped_centroids, output_csv)

    output_prefix = os.path.join(args.outdir, f"fire_polygons_lct{args.year}_{args.sdate}")
    save_intermediate_geometries(grouped_centroids, grouped_polygons, output_prefix=output_prefix)
    
    print("\n--- Pipeline Run Successfully Completed ---")

if __name__ == "__main__":
    main()
