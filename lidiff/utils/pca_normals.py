from __future__ import annotations

"""PCA-based normal estimation with k-NN and optional orientation.

Prefers SciPy's cKDTree for nearest neighbors, falls back to scikit-learn.
"""

from typing import Optional, Tuple
from collections import deque
import warnings

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None  # type: ignore
    _HAS_SCIPY = False

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    NearestNeighbors = None  # type: ignore
    _HAS_SKLEARN = False


def _build_knn(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) for k-NN of each point.

    distances: (N, k), indices: (N, k)
    """
    n = points.shape[0]
    k_eff = min(max(1, k), n)
    if _HAS_SCIPY and cKDTree is not None:
        tree = cKDTree(points)
        dists, inds = tree.query(points, k=k_eff)
        if k_eff == 1:
            # cKDTree returns shape (N,) when k=1; normalize shapes
            dists = dists[:, None]
            inds = inds[:, None]
        return dists, inds
    if _HAS_SKLEARN and NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
        nn.fit(points)
        dists, inds = nn.kneighbors(points, return_distance=True)
        return dists, inds
    raise ImportError("Neither SciPy nor scikit-learn kNN available. Install scipy or scikit-learn.")


def estimate_normals_pca(points: np.ndarray, k: int = 30, orient_to_cam: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate surface normals via PCA on k-nearest neighbors.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3)
    k : int, optional
        Number of nearest neighbors for PCA (default 30). Minimum used is 3.
    orient_to_cam : Optional[np.ndarray], optional
        If provided as (3,), flip normals so that dot(n, cam - p) >= 0.

    Returns
    -------
    np.ndarray
        Normals of shape (N, 3), dtype float32, unit-length where possible.
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {points.shape}")

    n_points = points.shape[0]
    if n_points < 3:
        return np.zeros((n_points, 3), dtype=np.float32)

    k_pca = max(3, int(k))
    k_pca = min(k_pca, n_points)

    _, knn_indices = _build_knn(points, k_pca)

    normals = np.zeros((n_points, 3), dtype=np.float32)

    # Compute PCA normal per point
    for i in range(n_points):
        idx = knn_indices[i]
        # Remove duplicates; optionally drop self if present
        unique_idx = np.unique(idx)
        if unique_idx.size < 3:
            normals[i] = 0.0
            continue
        neigh = points[unique_idx].astype(np.float64, copy=False)
        center = neigh.mean(axis=0)
        centered = neigh - center
        # Covariance eigen-decomposition: eigenvector with smallest eigenvalue is the normal
        try:
            cov = centered.T @ centered / max(1, centered.shape[0])
            # Ensure symmetric
            cov = (cov + cov.T) * 0.5
            evals, evecs = np.linalg.eigh(cov)
            normal64 = evecs[:, 0]
            normal = normal64.astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Normalize
        norm = float(np.linalg.norm(normal))
        if norm > 0.0 and np.isfinite(norm):
            normals[i] = normal / norm
        else:
            normals[i] = 0.0

    # Consistency propagation on small kNN graph
    k_cons = min(10, n_points)
    _, graph_indices = _build_knn(points, k_cons)

    magnitudes = np.linalg.norm(normals, axis=1)
    visited = np.zeros(n_points, dtype=bool)

    def bfs(seed: int) -> None:
        if visited[seed]:
            return
        queue = deque([seed])
        visited[seed] = True
        while queue:
            cur = queue.popleft()
            n_cur = normals[cur]
            for j in graph_indices[cur]:
                if j == cur:
                    continue
                if not visited[j]:
                    # Flip to agree with current if necessary
                    if np.dot(n_cur, normals[j]) < 0.0:
                        normals[j] = -normals[j]
                    visited[j] = True
                    queue.append(int(j))

    # Start BFS from strongest normals first to reduce flips from zeros
    order = np.argsort(-magnitudes)
    for seed in order:
        if magnitudes[seed] == 0.0:
            continue
        bfs(int(seed))

    # Optional: orient to camera
    if orient_to_cam is not None:
        cam = np.asarray(orient_to_cam, dtype=np.float32).reshape(3)
        view = cam[None, :] - points.astype(np.float32, copy=False)
        vnorm = np.linalg.norm(view, axis=1, keepdims=True)
        valid = vnorm[:, 0] > 0.0
        view_unit = np.zeros_like(view)
        view_unit[valid] = view[valid] / vnorm[valid]
        dots = np.einsum("ij,ij->i", normals, view_unit)
        flip_mask = dots < 0.0
        normals[flip_mask] = -normals[flip_mask]

    # Final normalization and NaN safety
    with np.errstate(invalid="ignore", divide="ignore"):
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0.0
        normals[nonzero] = normals[nonzero] / norms[nonzero]
        normals[~np.isfinite(normals)] = 0.0

    return normals.astype(np.float32, copy=False)

