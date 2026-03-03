"""
Core DBSCAN implementation using JAX with matrix-based label propagation.

This module implements the DBSCAN clustering algorithm adapted for JAX's
functional programming paradigm. Instead of traditional graph traversal,
it uses label propagation via matrix operations and jax.lax.while_loop.

The algorithm consists of these main steps:
1. Compute pairwise distances and adjacency matrix
2. Identify core points (degree ≥ min_pts)
3. Propagate labels among core points using connected components
4. Assign boundary points and noise labels
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils


class JaxDBScan:
    """
    JAX-based implementation of DBSCAN clustering using label propagation.

    Supports both standard JIT compilation and distributed execution across
    multiple devices. The implementation uses a matrix-based approach compatible
    with JAX/XLA's static shape requirements.

    Parameters:
    -----------
    eps : float
        Maximum distance between two samples for one to be considered as
        in the neighborhood of the other.
    min_pts : int
        Number of samples in a neighborhood for a point to be considered
        as a core point.
    use_distributed : bool, default=False
        Whether to use distributed execution across multiple devices.
        If True, data is sharded across available devices.
    return_sequential_labels : bool, default=True
        If True, re-index cluster labels to be sequential (0, 1, 2, ...).
        If False, labels are derived from original point indices.

    Attributes:
    -----------
    eps : float
        The epsilon neighborhood radius.
    min_pts : int
        The minimum number of points to form a core point.
    use_distributed : bool
        Whether distributed mode is enabled.
    labels_ : jax.Array
        Cluster labels for each point (-1 for noise).

    Notes:
    ------
    Memory Complexity: O(N²) due to the dense pairwise distance matrix.
    Recommended for datasets with N ≤ 10,000 to 20,000 points.

    Examples:
    ---------
    >>> import jax.numpy as jnp
    >>> from dbscan import JaxDBScan
    >>> X = jnp.array([[0, 0], [0.1, 0.1], [5, 5]])
    >>> model = JaxDBScan(eps=0.5, min_pts=2)
    >>> labels = model.fit_predict(X)
    >>> print(labels)  # [0, 0, -1] or similar
    """

    def __init__(
        self,
        eps: float,
        min_pts: int,
        use_distributed: bool = False,
        return_sequential_labels: bool = True,
    ):
        self.eps = eps
        self.min_pts = min_pts
        self.use_distributed = use_distributed
        self.return_sequential_labels = return_sequential_labels
        self._jit_fit_predict = None
        self.labels_ = None

    def _core_algorithm(self, X: jax.Array) -> jax.Array:
        """
        Core DBSCAN algorithm using matrix-based label propagation.

        This method implements the main algorithm logic:
        1. Compute pairwise Euclidean distances
        2. Build adjacency matrix (points within eps)
        3. Identify core points (degree ≥ min_pts)
        4. Propagate labels among core points using while_loop
        5. Assign boundary points and noise labels

        Parameters:
        -----------
        X : jax.Array
            Input data of shape (N, D) where N is the number of samples
            and D is the number of dimensions.

        Returns:
        --------
        labels : jax.Array
            Cluster labels of shape (N,). Non-sequential unique integers
            for each cluster, with -1 indicating noise points.
        """
        N = X.shape[0]

        # 1. Pairwise Distances (O(N²) memory)
        # For N > 20,000, consider implementing chunked/block distance computation
        diff = X[:, None, :] - X[None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1)

        # 2. Adjacency Matrix
        A = dists <= self.eps

        # 3. Core Points Identification
        degrees = jnp.sum(A, axis=1)
        core_mask = degrees >= self.min_pts

        # 4. Core-to-Core Adjacency Mask
        A_core = A & core_mask[:, None] & core_mask[None, :]

        # 5. Connected Components (Label Propagation)
        # Initialize each point with its own index as a distinct label
        init_labels = jnp.arange(N)

        def cond_fn(state):
            _, changed = state
            return changed

        def body_fn(state):
            labels, _ = state
            # Propagate maximum label from neighboring core points
            neighbor_labels = jnp.where(A_core, labels[None, :], -1)
            new_labels = jnp.max(neighbor_labels, axis=1)  # type: ignore

            # Only update labels for core points
            new_labels = jnp.where(core_mask, new_labels, labels)

            # Check for convergence
            changed = jnp.any(new_labels != labels)
            return new_labels, changed

        # Run while loop until labels stop changing
        final_labels, _ = jax.lax.while_loop(cond_fn, body_fn, (init_labels, True))

        # 6. Boundary and Noise Assignment
        # Identify non-core points adjacent to core points
        A_border = A & (~core_mask[:, None]) & core_mask[None, :]
        border_neighbor_labels = jnp.where(A_border, final_labels[None, :], -1)  # type: ignore
        border_labels = jnp.max(border_neighbor_labels, axis=1)

        is_border = jnp.any(A_border, axis=1)

        # Combine labels:
        # Core -> propagated label
        # Border -> max adjacent core label
        # Noise -> -1
        out_labels = jnp.where(
            core_mask,
            final_labels,  # type: ignore
            jnp.where(is_border, border_labels, -1),
        )  # type: ignore

        return out_labels

    def _reindex_labels(self, labels: jax.Array) -> jax.Array:
        """
        Re-index cluster labels to be sequential (0, 1, 2, ...).

        Converts non-sequential labels (derived from point indices) to
        sequential cluster labels starting from 0, preserving noise as -1.

        This uses pure JAX operations for compatibility with JIT compilation.

        Parameters:
        -----------
        labels : jax.Array
            Original labels with non-sequential cluster IDs.

        Returns:
        --------
        jax.Array
            Re-indexed labels with sequential cluster IDs (0, 1, 2, ...).
            Noise points remain labeled as -1.
        """
        # Get unique labels and sort them
        unique_labels = jnp.unique(labels)

        # Create a mapping from original labels to sequential indices
        # For each unique label, find its index in the sorted array
        # This gives us sequential IDs, but we need to adjust for -1

        # First, map each label to its position in unique_labels
        # Using searchsorted to find indices
        indices = jnp.searchsorted(unique_labels, labels)

        # Check if -1 is the first unique label (it would be, since sorted)
        has_noise = (unique_labels.size > 0) & (unique_labels[0] == -1)

        # Adjust indices: if -1 exists, subtract 1 from all indices after -1's position
        # and set the -1 position to -1
        offset = jnp.where(has_noise, 1, 0)
        adjusted_indices = jnp.where(labels == -1, -1, indices - offset)

        return adjusted_indices

    def fit_predict(self, X: jax.Array) -> jax.Array:
        """
        Execute DBSCAN clustering on the input data.

        JIT compiles the core algorithm on the first call for efficiency.
        Applies sharding specifications if use_distributed is True.

        Parameters:
        -----------
        X : jax.Array
            Input data of shape (N, D) where N is the number of samples
            and D is the number of dimensions.

        Returns:
        --------
        labels : jax.Array
            Cluster labels for each point. Noise points are labeled as -1.
            If return_sequential_labels=True, cluster IDs are sequential (0, 1, 2, ...).
            Otherwise, labels are non-sequential unique integers.

        Raises:
        -------
        RuntimeError
            If no devices are available when use_distributed=True.

        Examples:
        ---------
        >>> import jax.numpy as jnp
        >>> from dbscan import JaxDBScan
        >>> X = jnp.array([[0, 0], [0.1, 0], [0, 0.1], [5, 5]])
        >>> model = JaxDBScan(eps=0.5, min_pts=2)
        >>> labels = model.fit_predict(X)
        """
        if self._jit_fit_predict is None:
            if self.use_distributed:
                devices = jax.devices()
                if len(devices) == 0:
                    raise RuntimeError("No devices available for distributed execution")

                # Create a 1D device mesh across all available devices
                device_mesh = mesh_utils.create_device_mesh((len(devices),))
                mesh = Mesh(device_mesh, axis_names=("batch",))

                # Shard the data along the first axis ('batch')
                in_sharding = NamedSharding(mesh, PartitionSpec("batch", None))
                out_sharding = NamedSharding(mesh, PartitionSpec("batch"))

                self._jit_fit_predict = jax.jit(
                    self._core_algorithm,
                    in_shardings=(in_sharding,),
                    out_shardings=out_sharding,
                )
            else:
                self._jit_fit_predict = jax.jit(self._core_algorithm)

        # Execute the compiled function
        labels = self._jit_fit_predict(X)

        # Re-index labels if requested
        if self.return_sequential_labels:
            labels = self._reindex_labels(labels)

        self.labels_ = labels
        return labels

    def fit(self, X: jax.Array) -> "JaxDBScan":
        """
        Fit DBSCAN clustering model to the input data.

        Performs clustering and stores labels in the labels_ attribute.

        Parameters:
        -----------
        X : jax.Array
            Input data of shape (N, D).

        Returns:
        --------
        self : JaxDBScan
            Returns the fitted instance.

        Examples:
        ---------
        >>> model = JaxDBScan(eps=0.5, min_pts=5)
        >>> model.fit(X)
        >>> print(model.labels_)
        """
        self.fit_predict(X)
        return self
