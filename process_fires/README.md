Satellite fire detections from various datasets (MODIS, VIIRS, etc.) are matched to vegetation maps.
1. Read MODIS land-cover type (LCT) and vegetation continuous fields (VCF); annual files (rst_import_pyhdf_geopandas.py)
2. Read fire detections; daily files (af_import_geopandas.py)
3. Create fire location map - merging MODIS and VIIRS, eliminating overlaps.
4. Find vegetation type for each fire location
5. For forest vegetation types, find polygons encompassing neighboring fire pixels to fill in missing detections
6. Estimate biomass burned and burned area (tree vs herb)
