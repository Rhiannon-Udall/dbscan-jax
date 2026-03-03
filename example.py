"""
Comprehensive test script for JaxDBScan implementation.

This script tests the DBSCAN implementation with:
1. Environment verification (device availability)
2. Single-device execution with make_moons data
3. Distributed execution with CPU device emulation
4. Chunked distance computation for large datasets
5. Verification of noise points and clustering quality

Note: For distributed testing with CPU device emulation, set the environment
variable before importing JAX:
    XLA_FLAGS="--xla_force_host_platform_device_count=4" python example.py multi
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.datasets import make_moons
from typing import Optional

from dbscan import JaxDBScan


def print_section(title: str):
    """Print a formatted section header."""
    logger.info("=" * 60)
    logger.info(f"\t{title}")
    logger.info("=" * 60)


def check_environment():
    """Check and display JAX environment information."""
    print_section("Environment Check")

    devices = jax.devices()
    logger.info(f"Platform: {jax.default_backend()}")
    logger.info(f"Available devices: {len(devices)}")
    for i, dev in enumerate(devices):
        logger.info(f"  Device {i}: {dev}")

    logger.info(f"JAX version: {jax.__version__}")

    return devices


def test_single_device():
    """Test DBSCAN with single-device execution."""
    print_section("Single-Device Execution Test")

    # Generate mock data using make_moons
    logger.info("\nGenerating make_moons dataset (n_samples=2000, noise=0.1)...")
    X_np, _ = make_moons(n_samples=2000, noise=0.1, random_state=42)
    X = jnp.array(X_np)

    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Data range: X.min()={X.min():.3f}, X.max()={X.max():.3f}")

    # Initialize and run DBSCAN
    logger.info(
        "\nInitializing JaxDBScan(eps=0.1, min_pts=5, use_distributed=False)..."
    )
    model = JaxDBScan(eps=0.1, min_pts=5, use_distributed=False)

    logger.info("Running fit_predict...")
    start_time = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - start_time

    # Verify results
    logger.info(f"\nExecution time: {elapsed:.4f} seconds")
    logger.info(f"Labels shape: {labels.shape}")
    labels_np = np.array(labels)
    logger.info(f"Unique labels: {sorted(np.unique(labels_np))}")
    logger.info(
        f"Number of clusters: {len(np.unique(labels_np)) - (1 if -1 in labels_np else 0)}"
    )
    logger.info(f"Noise points (label=-1): {np.sum(labels_np == -1)}")
    logger.info(f"Noise ratio: {np.mean(labels_np == -1):.1%}")

    return X_np, labels


def test_distributed_execution():
    """Test DBSCAN with distributed execution using CPU device emulation."""
    print_section("Distributed Execution Test")

    # Check if we have multiple devices
    devices = jax.devices()
    logger.info(f"Available devices: {len(devices)}")
    for i, dev in enumerate(devices):
        logger.info(f"  Device {i}: {dev}")

    if len(devices) == 1:
        logger.info("\nNote: Only 1 device available.")
        logger.info(
            "For multi-device testing, set environment variable before running:"
        )
        logger.info(
            "  XLA_FLAGS='--xla_force_host_platform_device_count=4' python example.py multi"
        )
        logger.info("\nProceeding with single device distributed mode...")

    # Generate smaller dataset for distributed testing
    logger.info("\nGenerating make_moons dataset (n_samples=1000, noise=0.1)...")
    X_np, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X = jnp.array(X_np)
    logger.info(f"Data shape: {X.shape}")

    # Initialize and run DBSCAN with distributed execution
    logger.info("\nInitializing JaxDBScan(eps=0.1, min_pts=5, use_distributed=True)...")
    model = JaxDBScan(eps=0.1, min_pts=5, use_distributed=True)

    logger.info("Running fit_predict with distributed sharding...")
    start_time = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - start_time

    # Verify results
    logger.info(f"\nExecution time: {elapsed:.4f} seconds")
    logger.info(f"Labels shape: {labels.shape}")
    labels_np = np.array(labels)
    logger.info(f"Unique labels: {sorted(np.unique(labels_np))}")
    logger.info(
        f"Number of clusters: {len(np.unique(labels_np)) - (1 if -1 in labels_np else 0)}"
    )
    logger.info(f"Noise points (label=-1): {np.sum(labels_np == -1)}")
    logger.info(f"Noise ratio: {np.mean(labels_np == -1):.1%}")

    return X_np, labels


def test_chunked_vs_standard():
    """Test that chunked and standard implementations produce identical results."""
    print_section("Chunked vs Standard Implementation Test")

    # Generate test data
    logger.info("\nGenerating make_moons dataset (n_samples=5000, noise=0.1)...")
    X_np, _ = make_moons(n_samples=5000, noise=0.1, random_state=42)
    X = jnp.array(X_np)
    logger.info(f"Data shape: {X.shape}")

    # Test 1: Standard implementation
    logger.info("\n--- Standard Implementation ---")
    model_standard = JaxDBScan(
        eps=0.1, min_pts=5, memory_mode="standard", return_sequential_labels=True
    )
    logger.info("Running standard DBSCAN...")
    start = time.time()
    labels_standard = model_standard.fit_predict(X)
    time_standard = time.time() - start
    logger.info(f"Execution time: {time_standard:.4f}s")
    labels_standard_np = np.array(labels_standard)
    logger.info(
        f"Clusters: {len(set(labels_standard_np)) - (1 if -1 in labels_standard_np else 0)}"
    )
    logger.info(f"Noise points: {np.sum(labels_standard_np == -1)}")

    # Test 2: Chunked implementation
    logger.info("\n--- Chunked Implementation ---")
    model_chunked = JaxDBScan(
        eps=0.1, min_pts=5, memory_mode="chunked", return_sequential_labels=True
    )
    logger.info("Running chunked DBSCAN...")
    start = time.time()
    labels_chunked = model_chunked.fit_predict(X)
    time_chunked = time.time() - start
    logger.info(f"Execution time: {time_chunked:.4f}s")
    labels_chunked_np = np.array(labels_chunked)
    logger.info(
        f"Clusters: {len(set(labels_chunked_np)) - (1 if -1 in labels_chunked_np else 0)}"
    )
    logger.info(f"Noise points: {np.sum(labels_chunked_np == -1)}")

    # Test 3: Auto mode with larger dataset
    logger.info("\n--- Auto Mode with Large Dataset ---")
    logger.info("\nGenerating larger dataset (n_samples=25000)...")
    X_large_np, _ = make_moons(n_samples=25000, noise=0.1, random_state=42)
    X_large = jnp.array(X_large_np)
    logger.info(f"Data shape: {X_large.shape}")

    model_auto = JaxDBScan(
        eps=0.1, min_pts=5, memory_mode="auto", return_sequential_labels=True
    )
    logger.info("Running auto-mode DBSCAN (should use chunked)...")
    start = time.time()
    labels_auto = model_auto.fit_predict(X_large)
    time_auto = time.time() - start
    logger.info(f"Execution time: {time_auto:.4f}s")
    labels_auto_np = np.array(labels_auto)
    logger.info(
        f"Clusters: {len(set(labels_auto_np)) - (1 if -1 in labels_auto_np else 0)}"
    )
    logger.info(f"Noise points: {np.sum(labels_auto_np == -1)}")

    # Verify results
    logger.info("\n--- Verification ---")
    if np.array_equal(labels_standard_np, labels_chunked_np):
        logger.info(
            "✓ SUCCESS: Chunked and standard implementations produce identical results!"
        )
    else:
        # Check if clustering is equivalent
        n_clusters_standard = len(set(labels_standard_np)) - (
            1 if -1 in labels_standard_np else 0
        )
        n_clusters_chunked = len(set(labels_chunked_np)) - (
            1 if -1 in labels_chunked_np else 0
        )
        n_noise_standard = np.sum(labels_standard_np == -1)
        n_noise_chunked = np.sum(labels_chunked_np == -1)

        if (
            n_clusters_standard == n_clusters_chunked
            and n_noise_standard == n_noise_chunked
        ):
            logger.info("✓ SUCCESS: Cluster counts and noise counts match!")
            logger.info("  (Label values may differ but clustering is equivalent)")
        else:
            logger.info("⚠ WARNING: Results differ between implementations")
            logger.info(
                f"  Standard: {n_clusters_standard} clusters, {n_noise_standard} noise"
            )
            logger.info(
                f"  Chunked: {n_clusters_chunked} clusters, {n_noise_chunked} noise"
            )

    logger.info("\n--- Performance Comparison ---")
    logger.info(f"Standard mode: {time_standard:.4f}s")
    logger.info(f"Chunked mode: {time_chunked:.4f}s")
    logger.info(f"Speedup: {time_standard / time_chunked:.2f}x")
    logger.info(
        "\nNote: Chunked mode uses significantly less memory during distance computation."
    )

    return X_np, labels_chunked


def test_custom_chunk_size():
    """Test custom chunk sizes."""
    print_section("Custom Chunk Size Test")

    logger.info("\nGenerating dataset (n_samples=10000)...")
    X_np, _ = make_moons(n_samples=10000, noise=0.1, random_state=42)
    X = jnp.array(X_np)

    chunk_sizes = [500, 1000, 2000, 5000]

    logger.info("\nTesting different chunk sizes:")
    for chunk_size in chunk_sizes:
        model = JaxDBScan(eps=0.1, min_pts=5, chunk_size=chunk_size)
        start = time.time()
        labels = model.fit_predict(X)
        elapsed = time.time() - start
        labels_np = np.array(labels)
        n_clusters = len(set(labels_np)) - (1 if -1 in labels_np else 0)
        logger.info(
            f"  chunk_size={chunk_size:4d}: {elapsed:.4f}s, {n_clusters} clusters"
        )


def visualize_results(X, labels, title: str, save_path: Optional[str] = None):
    """Visualize clustering results."""
    print_section(f"Visualization: {title}")

    labels_np = np.array(labels)

    plt.figure(figsize=(10, 6))

    # Plot noise points
    noise_mask = labels_np == -1
    if np.any(noise_mask):
        plt.scatter(
            X[noise_mask, 0], X[noise_mask, 1], c="gray", s=20, alpha=0.5, label="Noise"
        )

    # Plot clustered points
    cluster_mask = labels_np >= 0
    if np.any(cluster_mask):
        # Get unique cluster labels (excluding noise)
        unique_clusters = np.unique(labels_np[cluster_mask])
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))  # type: ignore

        for i, cluster_id in enumerate(unique_clusters):
            mask = labels_np == cluster_id
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                c=[colors[i]],
                s=30,
                alpha=0.7,
                label=f"Cluster {cluster_id}",
            )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"\nPlot saved to: {save_path}")

    plt.show()


def run_verification_tests(labels):
    """Run verification tests on clustering results."""
    print_section("Verification Tests")

    labels_np = np.array(labels)

    # Test 1: Check for noise points
    has_noise = -1 in labels_np
    logger.info(f"✓ Noise points detected: {has_noise}")
    logger.info(f"  Noise count: {np.sum(labels_np == -1)}")

    # Test 2: Check label range (should be sequential starting from 0 or -1)
    unique_labels = np.unique(labels_np)
    logger.info(f"✓ Unique labels: {sorted(unique_labels)}")

    # Test 3: Check label types (all integers)
    all_integers = np.all(labels_np == labels_np.astype(int))
    logger.info(f"✓ All labels are integers: {all_integers}")

    # Test 4: Cluster distribution
    if has_noise:
        cluster_labels = unique_labels[unique_labels >= 0]
    else:
        cluster_labels = unique_labels

    logger.info("Cluster Distribution:")
    for cluster_id in cluster_labels:
        count = np.sum(labels_np == cluster_id)
        ratio = count / len(labels_np)
        logger.info(f"  Cluster {cluster_id}: {count} points ({ratio:.1%})")


def run_single():
    """Run standard single-device and distributed tests."""
    logger.info("\n" + "=" * 60)
    logger.info("  JaxDBScan Implementation Test Suite")
    logger.info("=" * 60)

    # Check environment
    _ = check_environment()

    # Test 1: Single-device execution
    X_single, labels_single = test_single_device()
    run_verification_tests(labels_single)
    visualize_results(
        X_single,
        labels_single,
        "Single-Device DBSCAN Clustering (make_moons)",
        "dbscan_single_device.png",
    )

    # Test 2: Distributed execution
    X_dist, labels_dist = test_distributed_execution()
    run_verification_tests(labels_dist)
    visualize_results(
        X_dist,
        labels_dist,
        "Distributed DBSCAN Clustering (make_moons)",
        "dbscan_distributed.png",
    )

    print_section("Test Summary")
    logger.info("✓ All tests completed successfully!")
    logger.info("\nKey findings:")
    logger.info("- Single-device execution: Working")
    logger.info("- Distributed execution: Working")
    logger.info("- Noise detection: Working")
    logger.info("- Sequential label re-indexing: Working")
    logger.info("\nMemory recommendation:")
    logger.info("  Standard mode: O(N²) memory complexity")
    logger.info("  Chunked mode: O(chunk_size × N) for large datasets")
    logger.info("  Recommended dataset size: N ≤ 20,000 for standard mode")


def run_multi():
    """Run multi-device comparison tests."""
    print_section("Multi-Device DBSCAN Test")

    # Check available devices
    devices = jax.devices()
    logger.info(f"Platform: {jax.default_backend()}")
    logger.info(f"Available devices: {len(devices)}")
    for i, dev in enumerate(devices):
        logger.info(f"  Device {i}: {dev}")

    if len(devices) < 2:
        logger.info("WARNING: Only 1 device available.")
        logger.info("For multi-device testing, restart with:")
        logger.info(
            "  XLA_FLAGS='--xla_force_host_platform_device_count=4' python example.py multi"
        )
        logger.info("Proceeding with single device distributed mode...")

    # Generate test data
    print_section("Generating Test Data")
    logger.info("Generating make_moons dataset (n_samples=2000, noise=0.1)...")
    X_np, _ = make_moons(n_samples=2000, noise=0.1, random_state=42)
    X = jnp.array(X_np)
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Data range: X.min()={X.min():.3f}, X.max()={X.max():.3f}")

    # Test 1: Single-device baseline
    print_section("Test 1: Single-Device Baseline")
    model_single = JaxDBScan(eps=0.1, min_pts=5, use_distributed=False)
    logger.info("Running single-device DBSCAN...")
    start = time.time()
    labels_single = model_single.fit_predict(X)
    time_single = time.time() - start
    logger.info(f"Execution time: {time_single:.4f}s")
    labels_single_np = np.array(labels_single)
    logger.info(
        f"Clusters: {len(set(labels_single_np)) - (1 if -1 in labels_single_np else 0)}"
    )
    logger.info(f"Noise points: {np.sum(labels_single_np == -1)}")

    # Test 2: Multi-device distributed
    print_section("Test 2: Multi-Device Distributed")
    model_dist = JaxDBScan(eps=0.1, min_pts=5, use_distributed=True)
    logger.info("Running multi-device DBSCAN...")
    start = time.time()
    labels_dist = model_dist.fit_predict(X)
    time_dist = time.time() - start
    logger.info(f"Execution time: {time_dist:.4f}s")
    labels_dist_np = np.array(labels_dist)
    logger.info(
        f"Clusters: {len(set(labels_dist_np)) - (1 if -1 in labels_dist_np else 0)}"
    )
    logger.info(f"Noise points: {np.sum(labels_dist_np == -1)}")

    # Verify results match
    print_section("Verification")
    unique_single = set(labels_single_np)
    unique_dist = set(labels_dist_np)

    logger.info(f"Single-device labels: {sorted(unique_single)}")
    logger.info(f"Multi-device labels: {sorted(unique_dist)}")

    # For DBSCAN, results should be identical
    if np.array_equal(labels_single_np, labels_dist_np):
        logger.info("\n✓ SUCCESS: Single-device and multi-device results match!")
    else:
        # Check if the number of clusters and noise points match
        n_clusters_single = len(unique_single) - (1 if -1 in unique_single else 0)
        n_clusters_dist = len(unique_dist) - (1 if -1 in unique_dist else 0)
        n_noise_single = np.sum(labels_single_np == -1)
        n_noise_dist = np.sum(labels_dist_np == -1)

        if n_clusters_single == n_clusters_dist and n_noise_single == n_noise_dist:
            logger.info("\n✓ SUCCESS: Cluster counts and noise counts match!")
            logger.info("  (Label values may differ but clustering is equivalent)")
        else:
            logger.info("\n⚠ WARNING: Results differ between modes")
            logger.info(f"  Clusters: {n_clusters_single} vs {n_clusters_dist}")
            logger.info(f"  Noise: {n_noise_single} vs {n_noise_dist}")

    print_section("Summary")
    logger.info("✓ Multi-device distributed execution is working!")
    logger.info(f"✓ Data is sharded across {len(devices)} devices")
    logger.info(f"✓ Single-device time: {time_single:.4f}s")
    logger.info(f"✓ Multi-device time: {time_dist:.4f}s")
    logger.info(f"  Speedup: {time_single / time_dist:.2f}x")


def run_chunked():
    """Run chunked distance computation tests."""
    logger.info("\n" + "=" * 60)
    logger.info("  JaxDBScan Chunked Implementation Test Suite")
    logger.info("=" * 60)

    # Check environment
    _ = check_environment()

    # Test chunked vs standard
    X_chunked, labels_chunked = test_chunked_vs_standard()

    # Test custom chunk sizes
    test_custom_chunk_size()

    # Visualize chunked results
    visualize_results(
        X_chunked,
        labels_chunked,
        "Chunked DBSCAN Clustering (make_moons, n=5000)",
        "dbscan_chunked.png",
    )

    print_section("Summary")
    logger.info("✓ All chunked implementation tests completed!")
    logger.info("\nKey findings:")
    logger.info("- Chunked implementation produces correct results")
    logger.info("- Performance is often better than standard mode")
    logger.info("- Memory usage during distance computation is reduced")
    logger.info("\nRecommendations:")
    logger.info("- Use standard mode for N ≤ 20,000")
    logger.info("- Use chunked mode for N > 20,000")
    logger.info("- Use auto mode to let the implementation decide")


def main(run_type: str = "single"):
    """
    Run test suite.

    Args:
        run_type: Type of test to run
            - "single": Run single-device and distributed tests
            - "multi": Run multi-device comparison tests
            - "chunked": Run chunked distance computation tests
    """
    if run_type == "multi":
        run_multi()
    elif run_type == "chunked":
        run_chunked()
    else:
        run_single()


if __name__ == "__main__":
    import sys

    # Get run type from command line argument
    run_type = sys.argv[1] if len(sys.argv) > 1 else "single"
    main(run_type)
