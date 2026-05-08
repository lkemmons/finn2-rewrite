"""Visualization and debugging tools for fire grouping.

Provides interactive maps and diagnostic plots for fire detection grouping.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import folium
from folium import plugins

log = logging.getLogger(__name__)


def plot_fire_groups_matplotlib(
    fire_gdf: gpd.GeoDataFrame,
    figsize: tuple = (14, 10),
    show_pixel_geoms: bool = True,
    show_fire_geoms: bool = True,
    cmap: str = 'tab20',
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot fire detections and groups using matplotlib.

    Parameters
    ----------
    fire_gdf : GeoDataFrame with fireid, geometry (Point), geom_sml, geom_pix
    figsize : figure size
    show_pixel_geoms : if True, show pixel footprint polygons
    show_fire_geoms : if True, show nominal fire size polygons
    cmap : matplotlib colormap name
    output_path : if provided, save figure to this path

    Returns
    -------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color by group
    unique_groups = np.unique(fire_gdf['fireid'])
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_groups)))
    group_colors = {g: colors[i] for i, g in enumerate(unique_groups)}

    # Plot pixel geometries (background)
    if show_pixel_geoms and 'geom_pix' in fire_gdf.columns:
        for idx, row in fire_gdf.iterrows():
            color = group_colors[row['fireid']]
            x, y = row['geom_pix'].exterior.xy
            ax.fill(x, y, color=color, alpha=0.2, edgecolor='none')
            ax.plot(x, y, color=color, linewidth=0.5, alpha=0.5)

    # Plot nominal fire geometries
    if show_fire_geoms and 'geom_sml' in fire_gdf.columns:
        for idx, row in fire_gdf.iterrows():
            color = group_colors[row['fireid']]
            x, y = row['geom_sml'].exterior.xy
            ax.fill(x, y, color=color, alpha=0.5, edgecolor=color, linewidth=0.8)

    # Plot fire points
    for group_id, color in group_colors.items():
        mask = fire_gdf['fireid'] == group_id
        ax.scatter(
            fire_gdf[mask].geometry.x,
            fire_gdf[mask].geometry.y,
            c=[color],
            s=50,
            marker='*',
            edgecolors='black',
            linewidths=0.5,
            label=f'Group {group_id}',
            zorder=10,
        )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Fire Detection Groups ({len(unique_groups)} groups, {len(fire_gdf)} detections)')
    ax.grid(True, alpha=0.3)

    if len(unique_groups) <= 20:
        ax.legend(loc='upper left', fontsize=8, ncol=2)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved plot to {output_path}")

    return fig


def plot_grouping_stats(
    fire_gdf: gpd.GeoDataFrame,
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot statistics about fire grouping.

    Parameters
    ----------
    fire_gdf : GeoDataFrame with fireid, ndetect1, acq_date
    figsize : figure size
    output_path : if provided, save figure to this path

    Returns
    -------
    matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Group size distribution
    group_sizes = fire_gdf.groupby('fireid').size()
    axes[0, 0].hist(group_sizes, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Detections per Group')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Group Size Distribution (n={len(group_sizes)} groups)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Detections per day
    if 'acq_date' in fire_gdf.columns:
        daily_counts = fire_gdf.groupby('acq_date').size()
        axes[0, 1].plot(daily_counts.index, daily_counts.values, marker='o')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Detections')
        axes[0, 1].set_title('Detections per Day')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Instrument distribution
    if 'instrument' in fire_gdf.columns:
        inst_counts = fire_gdf['instrument'].value_counts()
        axes[1, 0].bar(inst_counts.index, inst_counts.values, edgecolor='black', alpha=0.7)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Detections by Instrument')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Confidence distribution
    if 'confidence' in fire_gdf.columns:
        fire_gdf['confidence_num'] = pd.to_numeric(fire_gdf['confidence'], errors='coerce')
        confidence_data = fire_gdf['confidence_num'].dropna()
        if len(confidence_data) > 0:
            axes[1, 1].hist(confidence_data, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Confidence Distribution')
            axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved plot to {output_path}")

    return fig


def create_interactive_map(
    fire_gdf: gpd.GeoDataFrame,
    center: Optional[tuple] = None,
    zoom_start: int = 4,
    output_path: Optional[str] = None,
) -> folium.Map:
    """
    Create interactive map with fire groups using folium.

    Parameters
    ----------
    fire_gdf : GeoDataFrame with geometry, fireid, ndetect1
    center : (lat, lon) for map center (default: centroid of all fires)
    zoom_start : initial zoom level
    output_path : if provided, save map to this HTML file

    Returns
    -------
    folium Map object
    """
    # Calculate center
    if center is None:
        bounds = fire_gdf.total_bounds  # [minx, miny, maxx, maxy]
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # Create base map
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap',
    )

    # Add fire markers grouped by color
    unique_groups = np.unique(fire_gdf['fireid'])
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_groups)))

    for group_idx, group_id in enumerate(unique_groups):
        group_data = fire_gdf[fire_gdf['fireid'] == group_id]
        color_rgb = colors[group_idx]
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(color_rgb[0] * 255),
            int(color_rgb[1] * 255),
            int(color_rgb[2] * 255),
        )

        for idx, row in group_data.iterrows():
            lat, lon = row.geometry.y, row.geometry.x

            # Create popup with info
            popup_text = f"""<b>Fire Group {group_id}</b><br>
            Lon: {lon:.4f}, Lat: {lat:.4f}<br>
            Detections in group: {row['ndetect1']}"""

            if 'acq_date' in row:
                popup_text += f"<br>Date: {row['acq_date']}"
            if 'instrument' in row:
                popup_text += f"<br>Instrument: {row['instrument']}"
            if 'confidence' in row:
                popup_text += f"<br>Confidence: {row['confidence']}"

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=popup_text,
                color=color_hex,
                fill=True,
                fillColor=color_hex,
                fillOpacity=0.7,
                weight=1,
            ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    if output_path:
        m.save(output_path)
        log.info(f"Saved interactive map to {output_path}")

    return m


def print_grouping_summary(fire_gdf: gpd.GeoDataFrame) -> None:
    """
    Print summary statistics about fire grouping.

    Parameters
    ----------
    fire_gdf : GeoDataFrame with grouping results
    """
    print("\n" + "=" * 60)
    print("FIRE GROUPING SUMMARY")
    print("=" * 60)

    print(f"\nTotal detections: {len(fire_gdf):,}")
    print(f"Total groups: {fire_gdf['fireid'].nunique():,}")

    # Group statistics
    group_sizes = fire_gdf.groupby('fireid').size()
    print(f"\nGroup size statistics:")
    print(f"  Min: {group_sizes.min()}")
    print(f"  Max: {group_sizes.max()}")
    print(f"  Mean: {group_sizes.mean():.2f}")
    print(f"  Median: {group_sizes.median():.0f}")

    # Single detection groups
    single_groups = (group_sizes == 1).sum()
    print(f"\nSingle-detection groups: {single_groups} ({100*single_groups/len(group_sizes):.1f}%)")

    # Multi-detection groups
    multi_groups = (group_sizes > 1).sum()
    print(f"Multi-detection groups: {multi_groups} ({100*multi_groups/len(group_sizes):.1f}%)")

    # By date
    if 'acq_date' in fire_gdf.columns:
        print(f"\nBy acquisition date:")
        daily = fire_gdf.groupby('acq_date').agg({
            'fireid': 'count',
            'ndetect1': 'sum',  # this double-counts, but shows total
        })
        daily.columns = ['detections', 'grouped_detections']
        for date, row in daily.iterrows():
            print(f"  {date}: {row['detections']} detections")

    # By instrument
    if 'instrument' in fire_gdf.columns:
        print(f"\nBy instrument:")
        for inst in fire_gdf['instrument'].unique():
            count = (fire_gdf['instrument'] == inst).sum()
            print(f"  {inst}: {count}")

    print("\n" + "=" * 60)
