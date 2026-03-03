"""
Benchmark script comparing JAX DBSCAN with scikit-learn DBSCAN.

This script benchmarks both implementations on various dataset sizes
and reports performance metrics, memory usage, and accuracy.
"""

import gc
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_blobs, make_circles

from loguru import logger

from dbscan import JaxDBScan


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def generate_dataset(
    n_samples: int, dataset_type: str = "blobs", random_state: int = 42
):
    """Generate a synthetic dataset for benchmarking.

    Args:
        n_samples: Number of samples to generate
        dataset_type: Type of dataset ('blobs', 'circles', 'moons')
        random_state: Random seed for reproducibility

    Returns:
        X: numpy array of shape (n_samples, n_features)
    """
    if dataset_type == "blobs":
        X, _ = make_blobs(  # type: ignore
            n_samples=n_samples,
            n_features=2,
            centers=5,
            cluster_std=0.5,
            random_state=random_state,
        )
    elif dataset_type == "circles":
        X, _ = make_circles(
            n_samples=n_samples,
            noise=0.05,
            factor=0.5,
            random_state=random_state,
        )
    elif dataset_type == "moons":
        from sklearn.datasets import make_moons

        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return X


def benchmark_sklearn(
    X: np.ndarray, eps: float, min_samples: int, n_runs: int = 3
) -> dict[str, Any]:
    """Benchmark scikit-learn DBSCAN.

    Args:
        X: Input data
        eps: Maximum distance between two samples
        min_samples: Minimum samples in neighborhood
        n_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    gc.collect()
    jax.clear_caches()
    mem_before = get_memory_usage()

    times = []
    for _ in range(n_runs):
        start = time.time()
        dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(X)
        end = time.time()
        times.append(end - start)

    mem_after = get_memory_usage()

    return {
        "implementation": "scikit-learn",
        "time_mean": np.mean(times),
        "time_std": np.std(times),
        "time_min": np.min(times),
        "time_max": np.max(times),
        "memory_mb": mem_after - mem_before,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),  # type: ignore
        "n_noise": np.sum(labels == -1),  # type: ignore
    }


def benchmark_jax(
    X: np.ndarray, eps: float, min_pts: int, memory_mode: str = "auto", n_runs: int = 3
) -> dict[str, Any]:
    """Benchmark JAX DBSCAN.

    Args:
        X: Input data
        eps: Maximum distance between two samples
        min_pts: Minimum samples in neighborhood
        memory_mode: Memory mode to use
        n_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    X_jax = jnp.array(X)

    gc.collect()
    jax.clear_caches()
    mem_before = get_memory_usage()

    # Warm-up run for JIT compilation
    model = JaxDBScan(eps=eps, min_pts=min_pts, memory_mode=memory_mode)  # type: ignore
    _ = model.fit_predict(X_jax)

    # Benchmark runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        labels = model.fit_predict(X_jax)
        end = time.time()
        times.append(end - start)

    mem_after = get_memory_usage()
    labels_np = np.array(labels)  # type: ignore

    return {
        "implementation": f"JAX ({memory_mode})",
        "time_mean": np.mean(times),
        "time_std": np.std(times),
        "time_min": np.min(times),
        "time_max": np.max(times),
        "memory_mb": mem_after - mem_before,
        "n_clusters": len(set(labels_np)) - (1 if -1 in labels_np else 0),
        "n_noise": np.sum(labels_np == -1),
    }


def print_benchmark_header(title: str):
    """Print a formatted benchmark section header."""
    logger.info("\n" + "=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def print_benchmark_results(results: list[dict[str, Any]]):
    """Print benchmark results in a formatted table.

    Args:
        results: List of benchmark result dictionaries
    """
    logger.info(
        f"\n{'Implementation':<25} {'Time (s)':<12} {'Speedup':<10} {'Memory (MB)':<12} {'Clusters':<10} {'Noise':<10}"
    )
    logger.info("-" * 90)

    baseline_time = results[0]["time_mean"]

    for r in results:
        speedup = baseline_time / r["time_mean"]
        logger.info(
            f"{r['implementation']:<25} "
            f"{r['time_mean']:<12.4f} "
            f"{speedup:<10.2f}x "
            f"{r['memory_mb']:<12.1f} "
            f"{r['n_clusters']:<10} "
            f"{r['n_noise']:<10}"
        )


def run_scaling_benchmark():
    """Run benchmark scaling with different dataset sizes."""
    print_benchmark_header("Scaling Benchmark: Dataset Size Impact")

    dataset_sizes = [5000, 10000, 20000, 50000]
    eps = 0.15
    min_samples = 5

    all_results = []

    for n_samples in dataset_sizes:
        logger.info(f"\n--- Dataset Size: {n_samples:,} samples ---")

        X = generate_dataset(n_samples, dataset_type="blobs", random_state=42)

        # Benchmark scikit-learn
        logger.info("Running scikit-learn DBSCAN...")
        sklearn_result = benchmark_sklearn(X, eps, min_samples, n_runs=2)
        all_results.append(sklearn_result)

        # Benchmark JAX in different modes
        for mode in ["standard", "chunked", "ann"]:
            if mode == "standard" and n_samples > 20000:
                continue  # Skip standard mode for large datasets

            logger.info(f"Running JAX DBSCAN ({mode} mode)...")
            jax_result = benchmark_jax(X, eps, min_samples, memory_mode=mode, n_runs=2)
            all_results.append(jax_result)

    print_benchmark_results(all_results)


def run_dataset_types_benchmark():
    """Run benchmark on different dataset types."""
    print_benchmark_header("Dataset Types Benchmark")

    n_samples = 15000
    dataset_types = ["blobs", "circles", "moons"]
    eps = 0.15
    min_samples = 5

    all_results = []

    for dataset_type in dataset_types:
        logger.info(f"\n--- Dataset Type: {dataset_type.capitalize()} ---")

        X = generate_dataset(n_samples, dataset_type=dataset_type, random_state=42)

        # Benchmark scikit-learn
        logger.info("Running scikit-learn DBSCAN...")
        sklearn_result = benchmark_sklearn(X, eps, min_samples, n_runs=3)
        all_results.append(sklearn_result)

        # Benchmark JAX
        logger.info("Running JAX DBSCAN...")
        jax_result = benchmark_jax(X, eps, min_samples, memory_mode="auto", n_runs=3)
        all_results.append(jax_result)

    print_benchmark_results(all_results)


def run_memory_modes_benchmark():
    """Run benchmark comparing different JAX memory modes."""
    print_benchmark_header("JAX Memory Modes Benchmark")

    n_samples = 20000
    X = generate_dataset(n_samples, dataset_type="blobs", random_state=42)
    eps = 0.15
    min_samples = 5

    all_results = []

    memory_modes = ["standard", "chunked", "sparse", "ann"]

    for mode in memory_modes:
        logger.info(f"\n--- Memory Mode: {mode.upper()} ---")
        logger.info(f"Running JAX DBSCAN ({mode} mode)...")
        jax_result = benchmark_jax(X, eps, min_samples, memory_mode=mode, n_runs=3)
        all_results.append(jax_result)

    print_benchmark_results(all_results)


def run_main_benchmark():
    """Run the main benchmark with default settings."""
    print_benchmark_header("JAX vs Scikit-learn DBSCAN Benchmark")

    n_samples = 20000
    eps = 0.15
    min_samples = 5

    logger.info("\nDataset Configuration:")
    logger.info("  - Samples: {n_samples:,}")
    logger.info("  - Features: 2")
    logger.info("  - Dataset type: blobs")
    logger.info("  - eps: {eps}")
    logger.info("  - min_samples: {min_samples}")
    logger.info("  - JAX devices: {len(jax.devices())}")
    logger.info("  - Platform: {jax.devices()[0].platform}")

    X = generate_dataset(n_samples, dataset_type="blobs", random_state=42)

    # Benchmark scikit-learn
    logger.info("\n--- scikit-learn DBSCAN ---")
    sklearn_result = benchmark_sklearn(X, eps, min_samples, n_runs=5)
    all_results = [sklearn_result]

    # Benchmark JAX in different modes
    logger.info("\n--- JAX DBSCAN ---")
    for mode in ["standard", "chunked"]:
        logger.info(f"Running {mode} mode...")
        jax_result = benchmark_jax(X, eps, min_samples, memory_mode=mode, n_runs=5)
        all_results.append(jax_result)

    print_benchmark_results(all_results)

    # Print detailed statistics
    logger.info("\n" + "=" * 70)
    logger.info("  Detailed Statistics")
    logger.info("=" * 70)
    for r in all_results:
        logger.info(f"\n{r['implementation']}:")
        logger.info(f"  Mean time:   {r['time_mean']:.4f} ± {r['time_std']:.4f} s")
        logger.info(f"  Min time:    {r['time_min']:.4f} s")
        logger.info(f"  Max time:    {r['time_max']:.4f} s")
        logger.info(f"  Memory:      {r['memory_mb']:.1f} MB")
        logger.info(f"  Clusters:    {r['n_clusters']}")
        logger.info(
            f"  Noise:       {r['n_noise']} ({r['n_noise']/n_samples*100:.1f}%)"
        )


def main():
    """Run all benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark JAX vs scikit-learn DBSCAN")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="main",
        choices=["main", "scaling", "types", "modes"],
        help="Which benchmark to run",
    )
    args = parser.parse_args()

    logger.info("\n" + "=" * 70)
    logger.info("  DBSCAN Benchmark Suite")
    logger.info("  Comparing JAX vs Scikit-learn Implementations")
    logger.info("=" * 70)

    if args.benchmark == "main":
        run_main_benchmark()
    elif args.benchmark == "scaling":
        run_scaling_benchmark()
    elif args.benchmark == "types":
        run_dataset_types_benchmark()
    elif args.benchmark == "modes":
        run_memory_modes_benchmark()

    logger.info("\n" + "=" * 70)
    logger.info("  Benchmark Complete")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
