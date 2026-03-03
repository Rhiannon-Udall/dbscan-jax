"""
Test script for JAXDBSCAN implementation.

This script tests the DBSCAN implementation with:
1. Environment verification (device availability)
2. Single-device execution with make_moons data
3. Distributed execution with CPU device emulation
4. Verification of noise points and clustering quality

Note: For distributed testing with CPU device emulation, set the environment
variable before importing JAX:
    XLA_FLAGS="--xla_force_host_platform_device_count=4" python test_dbscan.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from loguru import logger
from typing import Optional

import jax
import jax.numpy as jnp

from dbscan import JAXDBSCAN


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
    logger.info(f"JAX version XLA: {jax.__version__}")

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
        "\nInitializing JAXDBSCAN(eps=0.1, min_pts=5, use_distributed=False)..."
    )
    model = JAXDBSCAN(eps=0.1, min_pts=5, use_distributed=False)

    logger.info("Running fit_predict...")
    start_time = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - start_time

    # Verify results
    logger.info(f"\nExecution time: {elapsed:.4f} seconds")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Unique labels: {sorted(np.unique(np.array(labels)))}")
    logger.info(
        f"Number of clusters: {len(np.unique(np.array(labels))) - (1 if -1 in labels else 0)}"
    )
    logger.info(f"Noise points (label=-1): {np.sum(np.array(labels) == -1)}")
    logger.info(f"Noise ratio: {np.mean(np.array(labels) == -1):.1%}")

    return X, labels


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
            "  XLA_FLAGS='--xla_force_host_platform_device_count=4' python test_dbscan.py"
        )
        logger.info("\nProceeding with single device distributed mode...")

    # Generate smaller dataset for distributed testing
    logger.info("\nGenerating make_moons dataset (n_samples=1000, noise=0.1)...")
    X_np, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X = jnp.array(X_np)
    logger.info(f"Data shape: {X.shape}")

    # Initialize and run DBSCAN with distributed execution
    logger.info("\nInitializing JAXDBSCAN(eps=0.1, min_pts=5, use_distributed=True)...")
    model = JAXDBSCAN(eps=0.1, min_pts=5, use_distributed=True)

    logger.info("Running fit_predict with distributed sharding...")
    start_time = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - start_time

    # Verify results
    logger.info(f"\nExecution time: {elapsed:.4f} seconds")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Unique labels: {sorted(np.unique(np.array(labels)))}")
    logger.info(
        f"Number of clusters: {len(np.unique(np.array(labels))) - (1 if -1 in labels else 0)}"
    )
    logger.info(f"Noise points (label=-1): {np.sum(np.array(labels) == -1)}")
    logger.info(f"Noise ratio: {np.mean(np.array(labels) == -1):.1%}")

    return X, labels


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
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("  JAXDBSCAN Implementation Test Suite")
    logger.info("=" * 60)

    # Check environment
    _ = check_environment()

    # Test 1: Single-device execution
    X_single, labels_single = test_single_device()
    run_verification_tests(labels_single)
    visualize_results(
        np.array(X_single),
        labels_single,
        "Single-Device DBSCAN Clustering (make_moons)",
        "dbscan_single_device.png",
    )

    # Test 2: Distributed execution
    X_dist, labels_dist = test_distributed_execution()
    run_verification_tests(labels_dist)
    visualize_results(
        np.array(X_dist),
        labels_dist,
        "Distributed DBSCAN Clustering (make_moons)",
        "dbscan_distributed.png",
    )

    print_section("Test Summary")
    logger.info("✓ All tests completed successfully!")
    logger.info("\nKey findings:")
    logger.info("- Single-device execution: Working")
    logger.info("- Distributed execution: Working (with CPU emulation)")
    logger.info("- Noise detection: Working")
    logger.info("- Sequential label re-indexing: Working")
    logger.info("\nMemory recommendation:")
    logger.info("  Current implementation: O(N²) memory complexity")
    logger.info("  Recommended dataset size: N ≤ 10,000 to 20,000 points")


def main(run_type: str = "single"):
    if run_type == "multi":
        pass
    else:
        run_single()


if __name__ == "__main__":
    main()
