"""LiDAR degeneracy detection via ICP translation Hessian.

Implements a Zhang 2016 ICRA-inspired degeneracy probe that runs
independently of KISS-ICP's internal state. For each frame pair (prev, cur)
already registered by KISS-ICP, we compute a translation-block Hessian
``H_t = sum_i n_i n_i^T`` from point-to-plane correspondences between the
current frame's source points and the previous frame's points. The three
eigenvalues of H_t measure how well each translation axis is observed; a
large condition number (or near-zero smallest eigenvalue) marks directional
degeneracy such as long straight corridors or tunnels.

The 3x3 translation block is used rather than the full 6x6 Jacobian because
the mixed translation/rotation units of the 6x6 form make its condition
number depend on an arbitrary length scale. The 3x3 block is dimensionally
consistent and directly corresponds to the unobserved translation direction
in KITTI Seq 01 (highway) degeneracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree


@dataclass
class DegeneracyScore:
    """Per-frame degeneracy descriptor.

    Attributes:
        lambda_min: Smallest eigenvalue of the 3x3 translation Hessian.
        lambda_max: Largest eigenvalue of the 3x3 translation Hessian.
        cond_number: ``lambda_max / max(lambda_min, eps)``. Unbounded from
            above; grows when the smallest eigenvalue collapses.
        eig_direction: Unit vector aligned with the least-observed
            translation direction (eigenvector of ``lambda_min``).
        n_corr: Number of correspondences used to build the Hessian.
    """

    lambda_min: float
    lambda_max: float
    cond_number: float
    eig_direction: np.ndarray
    n_corr: int

    @classmethod
    def null(cls) -> DegeneracyScore:
        """Placeholder for frames where the probe cannot run (e.g. frame 0)."""
        return cls(
            lambda_min=float("nan"),
            lambda_max=float("nan"),
            cond_number=float("nan"),
            eig_direction=np.array([np.nan, np.nan, np.nan], dtype=np.float64),
            n_corr=0,
        )

    @property
    def is_null(self) -> bool:
        return self.n_corr == 0 or not np.isfinite(self.cond_number)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deterministic voxel downsample via integer coordinate hashing.

    Keeps the first point encountered per voxel (not centroid) so that the
    returned points are actual observations, which matters when normals are
    later computed on the same set.
    """
    if points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)

    # Shift to non-negative so we can bit-pack into one int64.
    cmin = coords.min(axis=0)
    shifted = coords - cmin  # all >= 0

    # 21 bits per axis -> range [0, 2^21) voxels per axis (~2.1M * voxel_size meters).
    # Comfortably covers any LiDAR scene; falls back to slow path if exceeded.
    BITS = 21
    if int(shifted.max(initial=0)) >= (1 << BITS):
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        return points[np.sort(unique_idx)]

    key = (shifted[:, 0] << (2 * BITS)) | (shifted[:, 1] << BITS) | shifted[:, 2]
    _, unique_idx = np.unique(key, return_index=True)
    return points[np.sort(unique_idx)]


def _pca_normals(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-point PCA normals and linearity/planarity-style quality scores.

    Args:
        points: ``(N, 3)`` sample points in world frame.
        k: Neighborhood size for PCA (including the query point).

    Returns:
        normals: ``(N, 3)`` unit normal vectors (smallest-eigenvalue
            direction of the local covariance).
        quality: ``(N,)`` planarity score ``(l2 - l1) / l3`` where
            ``l1 <= l2 <= l3`` are local eigenvalues. Larger values indicate
            the local surface is well approximated by a plane.
    """
    n = points.shape[0]
    if n == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    tree = KDTree(points)
    k_use = min(k, n)
    _, idx = tree.query(points, k=k_use, workers=-1)
    if k_use == 1:
        idx = idx.reshape(-1, 1)

    neighborhoods = points[idx]  # (N, k, 3)
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k_use - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    normals = eigvecs[:, :, 0]  # smallest-eigenvalue direction = normal

    l1 = eigvals[:, 0]
    l2 = eigvals[:, 1]
    l3 = eigvals[:, 2]
    denom = np.where(l3 > 1e-12, l3, 1.0)
    quality = (l2 - l1) / denom  # high when planar
    return normals.astype(np.float64), quality.astype(np.float64)


class DegeneracyAnalyzer:
    """ICP translation-Hessian degeneracy probe.

    Runs alongside KISS-ICP per frame on the already-registered source
    points (in world frame) versus the previous frame's points. Returns a
    per-frame :class:`DegeneracyScore`.

    Cost per frame on KITTI (~5000 downsampled points, k=10 PCA): ~30-60 ms
    on a warm numpy/scipy stack. The probe is independent of KISS-ICP's
    internal ICP so it does not affect registration results.
    """

    def __init__(
        self,
        max_correspondences: int = 5000,
        normal_k: int = 10,
        max_nn_dist: float = 1.0,
        voxel_size: float = 0.5,
        min_quality: float = 0.0,
        mode: str = "3x3",
        length_scale: float = 10.0,
    ) -> None:
        """Initialize analyzer.

        Args:
            max_correspondences: Upper bound on correspondences kept after
                NN filtering. If exceeded, a uniform random subsample is
                drawn for speed. Results are insensitive above ~2000.
            normal_k: Neighborhood size for PCA normal estimation on the
                target point set.
            max_nn_dist: Reject correspondences whose nearest-neighbor
                distance exceeds this (meters). Guards against
                ``ghost`` correspondences when consecutive frames have
                minimal overlap.
            voxel_size: Pre-downsample voxel size applied to both source
                and target before NN search. Controls compute cost.
            min_quality: Optional PCA quality gate on normals. ``0.0``
                accepts every normal; ``>0`` rejects noisy neighborhoods.
            mode: ``"3x3"`` (default) builds the translation-block Hessian
                ``H_t = sum n_i n_i^T``. ``"6x6"`` builds the full
                point-to-plane SE(3) Hessian
                ``H = sum J_i^T J_i`` where
                ``J_i = [(p_i x n_i) / L, n_i]`` with ``p_i`` relative to
                the frame centroid and ``L = length_scale``. The 6x6 form
                includes rotation DOF but requires a dimensional length
                scale; it is provided as a sanity comparison mode.
            length_scale: Characteristic length (meters) used to normalize
                the rotation block in ``6x6`` mode so that the Hessian
                eigenvalues stay scale-free. Only used when
                ``mode == "6x6"``.
        """
        self.max_correspondences = max_correspondences
        self.normal_k = normal_k
        self.max_nn_dist = max_nn_dist
        self.voxel_size = voxel_size
        self.min_quality = min_quality
        self.mode = mode
        self.length_scale = length_scale
        if mode not in ("3x3", "6x6"):
            raise ValueError(f"mode must be '3x3' or '6x6', got {mode!r}")
        self._rng = np.random.default_rng(0)

    def analyze(
        self,
        source_world: np.ndarray,
        target_world: np.ndarray,
    ) -> DegeneracyScore:
        """Compute degeneracy score for one frame pair.

        Args:
            source_world: ``(N, 3)`` current-frame points in world frame
                (i.e. transformed by KISS-ICP's ``last_pose``).
            target_world: ``(M, 3)`` previous-frame points in world frame.

        Returns:
            :class:`DegeneracyScore` for the pair. Returns
            :meth:`DegeneracyScore.null` when the probe cannot produce a
            usable Hessian (insufficient correspondences, degenerate
            eigendecomposition).
        """
        if source_world.size == 0 or target_world.size == 0:
            return DegeneracyScore.null()

        src = _voxel_downsample(
            np.ascontiguousarray(source_world, dtype=np.float64), self.voxel_size
        )
        tgt = _voxel_downsample(
            np.ascontiguousarray(target_world, dtype=np.float64), self.voxel_size
        )
        if src.shape[0] < 10 or tgt.shape[0] < self.normal_k:
            return DegeneracyScore.null()

        # Cap source size for speed. Target stays full to keep normal PCA
        # neighborhoods dense; the NN query dominates runtime on src size.
        if src.shape[0] > self.max_correspondences:
            sel = self._rng.choice(src.shape[0], self.max_correspondences, replace=False)
            src = src[sel]

        # Normals and quality on the target set (computed once per call).
        target_normals, target_quality = _pca_normals(tgt, k=self.normal_k)

        # NN from source into target.
        tree = KDTree(tgt)
        dists, nn_idx = tree.query(src, k=1, workers=-1)
        dists = np.asarray(dists)
        nn_idx = np.asarray(nn_idx)
        mask = dists <= self.max_nn_dist
        if self.min_quality > 0.0:
            mask &= target_quality[nn_idx] >= self.min_quality

        if mask.sum() < 10:
            return DegeneracyScore.null()

        matched_src = src[mask]  # (K, 3) source points surviving NN gate
        nn_idx = nn_idx[mask]
        normals = target_normals[nn_idx]  # (K, 3)
        # Guard against degenerate zero normals (very sparse PCA failure).
        norm_mag = np.linalg.norm(normals, axis=1)
        good = norm_mag > 1e-8
        normals = normals[good]
        matched_src = matched_src[good]
        if normals.shape[0] < 10:
            return DegeneracyScore.null()

        if self.mode == "3x3":
            # Translation-block Hessian: H_t = sum_i n_i n_i^T.
            hessian = normals.T @ normals  # (3, 3)
        else:
            # Full 6x6 point-to-plane SE(3) Hessian with length-scale
            # normalization. p is taken relative to the matched-source
            # centroid (translation-invariant).
            p_rel = matched_src - matched_src.mean(axis=0)
            cross = np.cross(p_rel, normals) / self.length_scale  # (K, 3)
            jac = np.hstack([cross, normals])  # (K, 6): [rx,ry,rz, tx,ty,tz]
            hessian = jac.T @ jac  # (6, 6)

        eigvals, eigvecs = np.linalg.eigh(hessian)
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[-1])
        if lam_max <= 0.0:
            return DegeneracyScore.null()
        # Guard against exact zero in lam_min (theoretically possible).
        safe_lam_min = max(lam_min, 1e-12)
        cond = lam_max / safe_lam_min
        # For 6x6, eig_direction is in R^6. Truncate to the last 3 components
        # (translation axis) so the returned vector stays compatible with the
        # CSV/plotting schema.
        eig_full = np.ascontiguousarray(eigvecs[:, 0], dtype=np.float64)
        eig_direction = eig_full[-3:] if eig_full.shape[0] == 6 else eig_full
        return DegeneracyScore(
            lambda_min=lam_min,
            lambda_max=lam_max,
            cond_number=float(cond),
            eig_direction=eig_direction,
            n_corr=int(normals.shape[0]),
        )
