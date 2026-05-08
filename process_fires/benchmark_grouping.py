"""Benchmark fire grouping performance."""

import datetime
import time

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from fire_grouping import group_fire_detections, group_fire_detections_parallel


def create_realistic_fire_data(n_fires: int, n_clusters: int = 10) -> gpd.GeoDataFrame:
    """Create synthetic fire data with spatial clustering."""
    np.random.seed(42)

    points = []
    instruments = []
    dates = []

    # Create clusters
    cluster_centers = np.random.uniform(
        low=[-180, -90],
        high=[180, 90],
        size=(n_clusters, 2)
    )

    fires_per_cluster = n_fires // n_clusters
    date = datetime.date(2020, 1, 1)

    for center_lon, center_lat in cluster_centers:
        for _ in range(fires_per_cluster):
            points.append(Point(
                center_lon + np.random.normal(0, 0.001),
                center_lat + np.random.normal(0, 0.001)
            ))
            instruments.append(np.random.choice(["MODIS", "VIIRS"]))
            dates.append(date)

    gdf = gpd.GeoDataFrame(
        {
            "geometry": points,
            "scan": np.random.uniform(0.8, 1.2, len(points)),
            "track": np.random.uniform(0.8, 1.2, len(points)),
            "instrument": instruments,
            "acq_date": dates,
        },
        crs="EPSG:4326",
    )

    return gdf


def benchmark_grouping(sizes: list[int]) -> dict:
    """Benchmark grouping at different data sizes."""
    results = {}

    for n in sizes:
        print(f"\nBenchmarking with {n:,} fires...")

        gdf = create_realistic_fire_data(n, n_clusters=max(5, n // 100))

        start = time.time()
        fireid, geom = group_fire_detections(gdf, verbose=False)
        elapsed = time.time() - start

        n_groups = len(np.unique(fireid))

        results[n] = {
            "time_sec": elapsed,
            "fires_per_sec": n / elapsed,
            "n_groups": n_groups,
        }

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {n / elapsed:,.0f} fires/sec")
        print(f"  Groups: {n_groups:,}")

    return results


def benchmark_parallel_vs_sequential(size: int = 10000) -> None:
    """Compare parallel vs sequential performance."""
    print(f"\nComparing parallel vs sequential on {size:,} fires...\n")

    gdf = create_realistic_fire_data(size, n_clusters=max(5, size // 100))

    # Sequential
    print("Sequential processing:")
    start = time.time()
    fireid_seq, _ = group_fire_detections(gdf, verbose=False)
    time_seq = time.time() - start
    print(f"  Time: {time_seq:.2f}s")

    # Parallel
    print("\nParallel processing:")
    start = time.time()
    fireid_par, _ = group_fire_detections_parallel(gdf, verbose=False)
    time_par = time.time() - start
    print(f"  Time: {time_par:.2f}s")

    # Verify same results
    assert np.array_equal(fireid_seq, fireid_par), "Results differ!"
    print(f"\n✓ Results match")
    print(f"Speedup: {time_seq / time_par:.1f}x")


if __name__ == "__main__":
    print("Fire Detection Grouping Performance Benchmark")
    print("=" * 50)

    # Benchmark different sizes
    sizes = [1_000, 5_000, 10_000, 50_000]
    results = benchmark_grouping(sizes)

    print("\n" + "=" * 50)
    print("Summary:")
    print("-" * 50)
    print(f"{'Fires':<12} {'Time (s)':<12} {'Rate (fires/s)':<15} {'Groups':<10}")
    print("-" * 50)
    for n, r in results.items():
        print(
            f"{n:<12,} {r['time_sec']:<12.2f} {r['fires_per_sec']:<15,.0f} "
            f"{r['n_groups']:<10,}"
        )

    # Compare parallel vs sequential
    benchmark_parallel_vs_sequential(size=10_000)
