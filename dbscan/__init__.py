"""
Distributed JAX Implementation of DBSCAN Clustering Algorithm.

This module provides a JAX-based implementation of the DBSCAN (Density-Based
Spatial Clustering of Applications with Noise) algorithm using matrix-based
label propagation. It supports both single-device and multi-device distributed
execution through a unified interface.

Memory Constraints:
    Due to the O(N²) distance matrix computation, this implementation is
    recommended for datasets with N ≤ 10,000 to 20,000 points on typical
    hardware. For larger datasets, consider implementing chunked/block
    distance computation.

Example:
    >>> import jax.numpy as jnp
    >>> from dbscan import JAXDBSCAN
    >>> X = jnp.array([[0, 0], [1, 1], [5, 5]])
    >>> model = JAXDBSCAN(eps=0.5, min_pts=2)
    >>> labels = model.fit_predict(X)
"""

from dbscan.core import JAXDBSCAN

__all__ = ["JAXDBSCAN"]
__version__ = "0.1.0"
