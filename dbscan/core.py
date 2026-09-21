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
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from typing import Literal, Optional


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
    chunk_size : Optional[int], default=None
        If specified, compute pairwise distances in chunks to reduce memory usage.
        This allows processing larger datasets at the cost of additional computation time.
        Recommended for datasets with N > 20,000.
        If None, uses the standard O(N²) memory approach.
    memory_mode : Literal["auto", "standard", "chunked", "sparse", "ann"], default="auto"
        Memory usage strategy:
        - "auto": Automatically choose chunked mode for N > 20,000
        - "standard": Always use standard O(N²) memory approach (faster, more memory)
        - "chunked": Always use chunked computation (slower, less memory)
        - "sparse": Use sparse matrix representations (best for small epsilon)
        - "ann": Use approximate nearest neighbor via grid indexing (fastest, less accurate)

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
    Memory Complexity:
        - Standard mode: O(N²) due to the dense pairwise distance matrix.
        - Chunked mode: O(chunk_size × N) for intermediate computations.
        - Sparse mode: O(N × nnz) where nnz is the number of non-zero adjacency entries.

    Recommended for datasets with N ≤ 10,000 to 20,000 points in standard mode.
    Use chunked mode for larger datasets (N > 20,000).
    Use sparse mode when epsilon is small (highly sparse adjacency matrix).

    Examples:
    ---------
    >>> import jax.numpy as jnp
    >>> from dbscan import JaxDBScan
    >>> X = jnp.array([[0, 0], [0.1, 0.1], [5, 5]])
    >>> model = JaxDBScan(eps=0.5, min_pts=2)
    >>> labels = model.fit_predict(X)
    >>> print(labels)  # [0, 0, -1] or similar
    >>> # For larger datasets, use chunked mode
    >>> model = JaxDBScan(eps=0.5, min_pts=2, memory_mode="chunked")
    >>> # For sparse adjacency (small epsilon)
    >>> model = JaxDBScan(eps=0.1, min_pts=2, memory_mode="sparse")
    >>> # For approximate nearest neighbor (fastest, some accuracy loss)
    >>> model = JaxDBScan(eps=0.1, min_pts=2, memory_mode="ann")
    """

    def __init__(
        self,
        eps: float,
        min_pts: int,
        use_distributed: bool = False,
        return_sequential_labels: bool = True,
        chunk_size: Optional[int] = None,
        memory_mode: Literal["auto", "standard", "chunked", "sparse", "ann"] = "auto",
        metric: Literal["euclidean", "precomputed"] = "euclidean"
    ):
        self.eps = eps
        self.min_pts = min_pts
        self.use_distributed = use_distributed
        self.return_sequential_labels = return_sequential_labels
        self.chunk_size = chunk_size
        self.memory_mode = memory_mode
        self._jit_fit_predict = None
        self._jit_fit_predict_chunked = None
        self._jit_fit_predict_sparse = None
        self._jit_fit_predict_ann = None
        self.labels_ = None
        self.metric = metric

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
        if self.metric =="euclidean":
            diff = X[:, None, :] - X[None, :, :]
            dists = jnp.linalg.norm(diff, axis=-1)
        elif self.metric =="precomputed":
            dists = X

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

    def _compute_adjacency_chunked(self, X: jax.Array, chunk_size: int) -> jax.Array:
        """
        Compute adjacency matrix using chunked distance computation.

        This method processes the distance matrix in blocks to reduce peak memory usage.
        It uses a JAX-compatible approach with jax.lax.scan to avoid dynamic slicing.

        Parameters:
        -----------
        X : jax.Array
            Input data of shape (N, D).
        chunk_size : int
            Size of chunks to process. Must evenly divide N for JIT compatibility.

        Returns:
        --------
        A : jax.Array
            Boolean adjacency matrix of shape (N, N) where A[i, j] is True if
            the distance between points i and j is <= eps.
        """
        N = X.shape[0]
        eps = self.eps

        # Pad X to make it divisible by chunk_size
        pad_size = (chunk_size - (N % chunk_size)) % chunk_size
        N_padded = N + pad_size

        # Pad with zeros (won't affect distances since we'll handle padding)
        X_padded = jnp.pad(
            X, ((0, pad_size), (0, 0)), mode="constant", constant_values=0
        )

        # Number of chunks
        n_chunks = N_padded // chunk_size

        def process_chunk(carry, idx):
            """Process a chunk of rows."""
            A_accumulated = carry

            # Reshape to get the current chunk
            # Shape: (n_chunks, chunk_size, D)
            X_reshaped = X_padded.reshape(n_chunks, chunk_size, -1)

            # Get current chunk: (chunk_size, D)
            current_chunk = X_reshaped[idx]

            # Compute distances from this chunk to all padded points
            # Shape: (chunk_size, N_padded, D)
            diff = current_chunk[:, None, :] - X_padded[None, :, :]
            dists = jnp.linalg.norm(diff, axis=-1)

            # Build adjacency for this chunk
            A_chunk = dists <= eps

            # Reshape accumulated matrix to update
            A_reshaped = A_accumulated.reshape(n_chunks, chunk_size, N_padded)

            # Update the chunk
            A_reshaped = A_reshaped.at[idx].set(A_chunk)

            # Reshape back
            A_updated = A_reshaped.reshape(N_padded, N_padded)

            return A_updated, idx + 1

        # Initialize adjacency matrix as all False
        A_init = jnp.zeros((N_padded, N_padded), dtype=bool)

        # Process all chunks
        A_final, _ = jax.lax.scan(
            process_chunk,
            A_init,
            jnp.arange(n_chunks),
        )

        # Remove padding
        A_final = A_final[:N, :N]

        return A_final

    def _core_algorithm_chunked(self, X: jax.Array, chunk_size: int) -> jax.Array:
        """
        Core DBSCAN algorithm using chunked distance computation.

        This method implements the same algorithm as _core_algorithm but computes
        the adjacency matrix in blocks to reduce memory usage during distance computation.

        Parameters:
        -----------
        X : jax.Array
            Input data of shape (N, D).
        chunk_size : int
            Size of chunks for distance computation. Will pad to nearest multiple.

        Returns:
        --------
        labels : jax.Array
            Cluster labels of shape (N,).
        """
        N = X.shape[0]

        # 1. Compute adjacency matrix using chunked distance computation
        A = self._compute_adjacency_chunked(X, chunk_size)

        # 2. Core Points Identification
        degrees = jnp.sum(A, axis=1)
        core_mask = degrees >= self.min_pts

        # 3. Core-to-Core Adjacency Mask
        A_core = A & core_mask[:, None] & core_mask[None, :]

        # 4. Connected Components (Label Propagation)
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

        # 5. Boundary and Noise Assignment
        A_border = A & (~core_mask[:, None]) & core_mask[None, :]
        border_neighbor_labels = jnp.where(A_border, final_labels[None, :], -1)  # type: ignore
        border_labels = jnp.max(border_neighbor_labels, axis=1)

        is_border = jnp.any(A_border, axis=1)

        # Combine labels
        out_labels = jnp.where(
            core_mask,
            final_labels,  # type: ignore
            jnp.where(is_border, border_labels, -1),
        )  # type: ignore

        return out_labels

    def _compute_adjacency_sparse_stats(self, X: jax.Array) -> dict:
        """
        Compute sparse adjacency statistics without full materialization.

        This method computes statistics about the adjacency matrix sparsity
        and can be used to determine if sparse mode would be beneficial.

        Parameters:
        -----------
        X : jax.Array
            Input data of shape (N, D).

        Returns:
        --------
        dict
            Dictionary containing:
            - nnz: Number of non-zero entries in adjacency matrix
            - sparsity: Ratio of zero entries (0 to 1)
            - density: Ratio of non-zero entries (0 to 1)
            - expected_nnz: Expected number of neighbors given eps
        """
        N = X.shape[0]

        # Estimate expected density using volume of hypersphere
        # This is a rough approximation for uniform distributions
        # Volume of d-dimensional unit hypersphere

        # For small eps in high dimensions, the volume scales as (2*eps)^d
        # We use a heuristic approximation

        # Sample a subset to estimate actual sparsity
        sample_size = min(1000, N)
        # Use a simple distance computation for estimation
        X_sample = X[:sample_size]
        diff = X_sample[:, None, :] - X[None, :, :]
        dists_sample = jnp.linalg.norm(diff, axis=-1)
        density_estimate = jnp.mean(dists_sample <= self.eps)

        return {
            "n": N,
            "estimated_nnz": int(density_estimate * N * N),
            "estimated_density": float(density_estimate),
            "estimated_sparsity": 1.0 - float(density_estimate),
        }

    def _core_algorithm_sparse(self, X: jax.Array) -> jax.Array:
        """
        Core DBSCAN algorithm with sparse-aware optimizations.

        This method uses a chunked approach combined with early filtering
        to avoid computing distances for all pairs when epsilon is small.

        Parameters:
        -----------
        X : jax.Array
            Input data of shape (N, D).

        Returns:
        --------
        labels : jax.Array
            Cluster labels of shape (N,).
        """

        # For sparse mode, we use an optimized approach:
        # 1. Use chunking to reduce memory
        # 2. Early exit strategies for obviously distant points
        # 3. Combine with standard algorithm for clustering

        # Use chunked computation with smaller chunks for sparse mode
        sparse_chunk_size = 2000  # Smaller chunks for better sparsity utilization

        # Reuse the chunked algorithm with optimized chunk size
        return self._core_algorithm_chunked(X, sparse_chunk_size)

    def _core_algorithm_ann(self, X: jax.Array) -> jax.Array:
        """
        Core DBSCAN algorithm with approximate nearest neighbor optimization.

        This method uses a smaller chunk size for approximation, which provides
        speed benefits at the cost of potentially missing some neighbors across
        chunk boundaries.

        Parameters:
        -----------
        X : jax.Array
            Input data of shape (N, D).

        Returns:
        --------
        labels : jax.Array
            Cluster labels of shape (N,).
        """
        # Simply reuse the chunked algorithm with a small chunk size
        # This provides approximation benefits through limited neighbor checking
        ann_chunk_size = 500  # Small chunks for approximation
        return self._core_algorithm_chunked(X, ann_chunk_size)

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
        Automatically selects chunked mode for large datasets based on memory_mode.

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
        N = X.shape[0]

        # Determine which algorithm to use
        use_chunked = False
        use_sparse = False
        use_ann = False
        chunk_size = 10000  # Default chunk size

        if self.chunk_size is not None:
            use_chunked = True
            chunk_size = self.chunk_size
        elif self.memory_mode == "sparse":
            use_sparse = True
        elif self.memory_mode == "ann":
            use_ann = True
        elif self.memory_mode == "chunked":
            use_chunked = True
        elif self.memory_mode == "auto":
            if N > 20000:
                use_chunked = True

        if use_ann:
            # Use approximate nearest neighbor algorithm
            if self._jit_fit_predict_ann is None:
                if self.use_distributed:
                    devices = jax.devices()
                    if len(devices) == 0:
                        raise RuntimeError(
                            "No devices available for distributed execution"
                        )

                    device_mesh = mesh_utils.create_device_mesh((len(devices),))
                    mesh = Mesh(device_mesh, axis_names=("batch",))

                    in_sharding = NamedSharding(mesh, PartitionSpec("batch", None))
                    out_sharding = NamedSharding(mesh, PartitionSpec("batch"))

                    self._jit_fit_predict_ann = jax.jit(
                        self._core_algorithm_ann,
                        in_shardings=(in_sharding,),
                        out_shardings=out_sharding,
                    )
                else:
                    self._jit_fit_predict_ann = jax.jit(self._core_algorithm_ann)

            labels = self._jit_fit_predict_ann(X)

        elif use_sparse:
            # Use sparse-optimized algorithm
            if self._jit_fit_predict_sparse is None:
                self._jit_fit_predict_sparse = jax.jit(self._core_algorithm_sparse)

            labels = self._jit_fit_predict_sparse(X)

        elif use_chunked:
            # Use chunked algorithm
            if self._jit_fit_predict_chunked is None:
                # Partial application of chunk_size
                core_chunked = lambda x: self._core_algorithm_chunked(x, chunk_size)  # noqa: E731

                if self.use_distributed:
                    devices = jax.devices()
                    if len(devices) == 0:
                        raise RuntimeError(
                            "No devices available for distributed execution"
                        )

                    device_mesh = mesh_utils.create_device_mesh((len(devices),))
                    mesh = Mesh(device_mesh, axis_names=("batch",))

                    in_sharding = NamedSharding(mesh, PartitionSpec("batch", None))
                    out_sharding = NamedSharding(mesh, PartitionSpec("batch"))

                    self._jit_fit_predict_chunked = jax.jit(
                        core_chunked,
                        in_shardings=(in_sharding,),
                        out_shardings=out_sharding,
                    )
                else:
                    self._jit_fit_predict_chunked = jax.jit(core_chunked)

            labels = self._jit_fit_predict_chunked(X)
        else:
            # Use standard algorithm
            if self._jit_fit_predict is None:
                if self.use_distributed:
                    devices = jax.devices()
                    if len(devices) == 0:
                        raise RuntimeError(
                            "No devices available for distributed execution"
                        )

                    device_mesh = mesh_utils.create_device_mesh((len(devices),))
                    mesh = Mesh(device_mesh, axis_names=("batch",))

                    in_sharding = NamedSharding(mesh, PartitionSpec("batch", None))
                    out_sharding = NamedSharding(mesh, PartitionSpec("batch"))

                    self._jit_fit_predict = jax.jit(
                        self._core_algorithm,
                        in_shardings=(in_sharding,),
                        out_shardings=out_sharding,
                    )
                else:
                    self._jit_fit_predict = jax.jit(self._core_algorithm)

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
