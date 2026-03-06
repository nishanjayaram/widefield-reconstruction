"""
Basis-function inversion for the NV-diamond stress reconstruction.

Algorithm (SPEC.md Section 6)
------------------------------
1. Enumerate traction basis: each culet DOF k gets a unit traction e_k.
2. For each k: solve K u_k = e_k (reusing the MUMPS factorisation in ForwardSolver).
3. Evaluate sigma(u_k) at the NV measurement points → column k of influence matrix A.
4. Stack data d = [D_g(x_m)] for all 4 orientations and N_meas points.
5. Regularised least squares: c = argmin ||d - A c||^2 + lam * ||c||^2.
6. Reconstruct stress: sigma(x) = sum_k c_k * sigma(u_k)(x).

Dimensions
----------
  N_meas   : number of NV measurement points (default 11593 per orientation)
  N_B      : number of traction basis functions = len(culet_dofs)
  A        : (4 * N_meas, N_B)  float64
  d        : (4 * N_meas,)       float64
  c        : (N_B,)              float64

Memory note
-----------
A is computed and stored entirely in RAM. Each column requires one forward
solve (back-substitution) + one stress interpolation + N_meas evaluations.
For N_B ~ 3000 and N_meas ~ 11593, A is about 140 MB — easily fits in memory.
The N_B back-solves are the computational bottleneck but share the same
MUMPS factorisation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.linalg

from .forward import ForwardSolver
from .nv_coupling import build_coupling_matrices, dg_all


class BasisSolver:
    """
    Constructs the influence matrix A and solves the regularised inverse problem.

    Parameters
    ----------
    solver : ForwardSolver
        Pre-built and factorised forward solver (mesh already loaded, K factored).
    nv_coords : (N_meas, 2) or (N_meas, 3) ndarray
        (x, y) or (x, y, z) coordinates of NV measurement points in µm.
        If (N_meas, 2), z=0 (culet surface) is assumed.
    nv_data : (4, N_meas) ndarray
        Measured Delta_D for each NV orientation (GHz).
    culet_dof_subset : array-like or None
        Subset of culet DOF indices to use as basis.
        None (default) uses all culet DOFs from the solver.
    theta_misc_deg : float
        Miscut polar angle (degrees).
    phi_misc_deg : float
        Effective miscut azimuthal angle (degrees).
    verbose : bool
    """

    def __init__(
        self,
        solver: ForwardSolver,
        nv_coords: np.ndarray,
        nv_data: np.ndarray,
        culet_dof_subset: np.ndarray | None = None,
        theta_misc_deg: float = -3.5,
        phi_misc_deg: float = 174.4,
        verbose: bool = True,
    ) -> None:
        self.solver  = solver
        self.verbose = verbose

        # NV measurement coordinates
        nv_coords = np.asarray(nv_coords, dtype=np.float64)
        if nv_coords.ndim == 1:
            nv_coords = nv_coords.reshape(-1, 1)
        if nv_coords.shape[1] == 2:
            # Append z=0 column
            nv_coords = np.column_stack([nv_coords, np.zeros(len(nv_coords))])
        self.nv_coords = nv_coords          # (N_meas, 3)

        self.nv_data = np.asarray(nv_data, dtype=np.float64)  # (4, N_meas)
        assert self.nv_data.shape == (4, len(self.nv_coords)), \
            f"nv_data shape {self.nv_data.shape} inconsistent with " \
            f"{len(self.nv_coords)} measurement points."

        # Coupling matrices
        self.nvs, self.M_list = build_coupling_matrices(theta_misc_deg, phi_misc_deg)

        # Traction basis DOFs
        if culet_dof_subset is not None:
            self.basis_dofs = np.asarray(culet_dof_subset, dtype=np.intp)
        else:
            self.basis_dofs = solver.culet_dofs

        self.N_meas = len(self.nv_coords)
        self.N_B    = len(self.basis_dofs)

        if verbose:
            print(f"BasisSolver: N_meas={self.N_meas}, N_B={self.N_B}")
            print(f"  Influence matrix A will be ({4 * self.N_meas}, {self.N_B})")
            mem_mb = 4 * self.N_meas * self.N_B * 8 / 1e6
            print(f"  Estimated memory for A: {mem_mb:.1f} MB")

        self.A: np.ndarray | None = None   # built lazily by build_influence_matrix()

    # ---------------------------------------------------------------------- #
    # Step 2: Build influence matrix A                                        #
    # ---------------------------------------------------------------------- #

    def build_influence_matrix(
        self,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 100,
    ) -> np.ndarray:
        """
        Build the influence matrix A of shape (4 * N_meas, N_B).

        Each column k corresponds to one traction basis DOF:
          1. Solve K u_k = e_k (unit load at DOF k).
          2. Evaluate sigma(u_k) at NV measurement points.
          3. A[(g*N_meas):(g+1)*N_meas, k] = M_g : sigma_k  for g=0,1,2,3.

        Parameters
        ----------
        checkpoint_path : str or Path or None
            If given, save A to this .npy file every `checkpoint_every` columns.
        checkpoint_every : int
            Checkpoint interval (number of columns).

        Returns
        -------
        A : (4 * N_meas, N_B) ndarray [GHz/GPa_unit_load]
        """
        N = self.N_meas
        A = np.zeros((4 * N, self.N_B), dtype=np.float64)

        t0 = time.perf_counter()
        for k, dof_idx in enumerate(self.basis_dofs):
            # Solve unit-load forward problem
            u_k = self.solver.solve_unit_load(int(dof_idx))

            # Stress at NV measurement points: (N_meas, 6)
            sigma_k = self.solver.stress_at_coords_batch(u_k, self.nv_coords)

            # Replace NaN (points outside mesh) with 0
            sigma_k = np.where(np.isnan(sigma_k), 0.0, sigma_k)

            # D_g contributions: (4, N_meas)
            D_k = dg_all(sigma_k, self.M_list)

            # Pack into column k of A
            for g in range(4):
                A[g * N:(g + 1) * N, k] = D_k[g]

            # Progress and checkpointing
            if self.verbose and (k + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate    = (k + 1) / elapsed
                eta     = (self.N_B - k - 1) / rate
                print(f"  [{k+1}/{self.N_B}]  {elapsed:.0f}s elapsed, "
                      f"ETA {eta:.0f}s  ({rate:.1f} solves/s)")

            if checkpoint_path is not None and (k + 1) % checkpoint_every == 0:
                np.save(checkpoint_path, A)
                if self.verbose:
                    print(f"  Checkpoint saved to {checkpoint_path}")

        elapsed = time.perf_counter() - t0
        if self.verbose:
            print(f"Influence matrix built in {elapsed:.1f}s.")

        if checkpoint_path is not None:
            np.save(checkpoint_path, A)

        self.A = A
        return A

    # ---------------------------------------------------------------------- #
    # Step 3: Regularised least squares                                       #
    # ---------------------------------------------------------------------- #

    def data_vector(self) -> np.ndarray:
        """
        Flatten the 4 NV data arrays into the data vector d of length 4*N_meas.

        d[g*N_meas : (g+1)*N_meas] = nv_data[g]
        """
        return self.nv_data.ravel(order="C")

    def solve_regularized(
        self,
        lambda_reg: float = 1e-3,
        A: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Tikhonov regularised least squares: c = (A^T A + lam I)^{-1} A^T d.

        Parameters
        ----------
        lambda_reg : float
            Regularisation parameter (scales identity penalty).
        A : ndarray or None
            Influence matrix to use. If None, uses self.A (must be pre-built).

        Returns
        -------
        c : (N_B,) ndarray — traction coefficients
        """
        if A is None:
            A = self.A
        if A is None:
            raise RuntimeError("Call build_influence_matrix() first.")

        d = self.data_vector()
        if self.verbose:
            print(f"Solving regularised LSQ: A={A.shape}, lam={lambda_reg:.2e}")

        # Normal equations: (A^T A + lam I) c = A^T d
        AtA = A.T @ A
        Atd = A.T @ d
        AtA[np.diag_indices_from(AtA)] += lambda_reg

        c, *_ = scipy.linalg.lstsq(AtA, Atd, assume_a="pos")
        if self.verbose:
            residual = np.linalg.norm(d - A @ c)
            print(f"  ||d - A c|| = {residual:.4e} GHz")
        return c

    def solve_svd(
        self,
        n_components: int | None = None,
        lambda_reg: float = 0.0,
        A: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve via truncated SVD: c = V S^{-1} U^T d  (with optional Tikhonov).

        Parameters
        ----------
        n_components : int or None
            Number of singular values to keep. None keeps all.
        lambda_reg : float
            Tikhonov regularisation added to singular values
            (filter: s / (s^2 + lam) instead of 1/s).
        A : ndarray or None
            Influence matrix to use. If None, uses self.A.

        Returns
        -------
        c : (N_B,) ndarray — traction coefficients
        svals : (min(4*N_meas, N_B),) ndarray — singular values
        """
        if A is None:
            A = self.A
        if A is None:
            raise RuntimeError("Call build_influence_matrix() first.")

        d = self.data_vector()
        if self.verbose:
            print(f"SVD solve: A={A.shape}, n_components={n_components}, lam={lambda_reg:.2e}")

        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        if n_components is not None:
            U, s, Vt = U[:, :n_components], s[:n_components], Vt[:n_components]

        # Tikhonov filter factors
        filters = s / (s ** 2 + lambda_reg) if lambda_reg > 0 else 1.0 / s

        c = Vt.T @ (filters * (U.T @ d))
        if self.verbose:
            residual = np.linalg.norm(d - A @ c)
            print(f"  ||d - A c|| = {residual:.4e} GHz")
        return c, s

    # ---------------------------------------------------------------------- #
    # Step 4: Reconstruct stress from traction coefficients                   #
    # ---------------------------------------------------------------------- #

    def reconstruct_stress(
        self,
        c: np.ndarray,
        eval_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct the full stress field from traction coefficients.

        sigma(x) = sum_k c_k * sigma(u_k)(x)

        This requires re-solving all N_B forward problems.  For large N_B it
        can be expensive; use the checkpoint path in build_influence_matrix to
        cache the solutions if needed.

        Parameters
        ----------
        c : (N_B,) ndarray — traction coefficients from solve_regularized / solve_svd
        eval_coords : (M, 3) ndarray — evaluation coordinates (µm)

        Returns
        -------
        sigma_total : (M, 6) ndarray [GPa] — reconstructed Voigt stress
        """
        eval_coords = np.asarray(eval_coords, dtype=np.float64)
        M_pts = len(eval_coords)
        sigma_total = np.zeros((M_pts, 6), dtype=np.float64)

        if self.verbose:
            print(f"Reconstructing stress at {M_pts} points "
                  f"(N_B={self.N_B} solves) ...")

        for k, (dof_idx, ck) in enumerate(zip(self.basis_dofs, c)):
            if abs(ck) < 1e-15:
                continue
            u_k = self.solver.solve_unit_load(int(dof_idx))
            sigma_k = self.solver.stress_at_coords_batch(u_k, eval_coords)
            sigma_k = np.where(np.isnan(sigma_k), 0.0, sigma_k)
            sigma_total += ck * sigma_k

            if self.verbose and (k + 1) % 200 == 0:
                print(f"  [{k+1}/{self.N_B}]")

        return sigma_total

    # ---------------------------------------------------------------------- #
    # Convenience: L-curve for choosing lambda                               #
    # ---------------------------------------------------------------------- #

    def lcurve(
        self,
        lambdas: np.ndarray | None = None,
        A: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the L-curve (residual norm vs solution norm) for a range of
        regularisation parameters.

        Parameters
        ----------
        lambdas : 1-D array or None
            Regularisation parameters to scan. Defaults to 50 log-spaced
            values from 1e-6 to 1e2.
        A : ndarray or None

        Returns
        -------
        lambdas : (K,) ndarray
        residuals : (K,) ndarray — ||d - A c||
        norms     : (K,) ndarray — ||c||
        """
        if A is None:
            A = self.A
        if A is None:
            raise RuntimeError("Call build_influence_matrix() first.")
        if lambdas is None:
            lambdas = np.logspace(-6, 2, 50)

        d = self.data_vector()
        residuals = np.zeros(len(lambdas))
        norms     = np.zeros(len(lambdas))

        AtA = A.T @ A
        Atd = A.T @ d

        for i, lam in enumerate(lambdas):
            M = AtA.copy()
            M[np.diag_indices_from(M)] += lam
            c, *_ = scipy.linalg.lstsq(M, Atd, assume_a="pos")
            residuals[i] = np.linalg.norm(d - A @ c)
            norms[i]     = np.linalg.norm(c)

        return lambdas, residuals, norms


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_nv_data(data_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """
    Load the four NV data files and return coordinates + data matrix.

    Files expected: nv{0..3}_roi_centered_minusZFS.txt
    Format: CSV with header x_um_centered_to_ellipse,y_um_centered_to_ellipse,delta_f_GHz

    Returns
    -------
    coords : (N_meas, 2) ndarray  — (x, y) in µm (same grid for all 4 orientations)
    data   : (4, N_meas) ndarray  — Delta_D in GHz
    """
    data_dir = Path(data_dir)
    coords = None
    data_list = []
    for g in range(4):
        fname = data_dir / f"nv{g}_roi_centered_minusZFS.txt"
        arr = np.loadtxt(fname, delimiter=",", skiprows=1)
        if coords is None:
            coords = arr[:, :2]
        data_list.append(arr[:, 2])
    return coords, np.array(data_list)


if __name__ == "__main__":
    import sys

    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "mesh/anvil_preview.msh"
    data_dir  = sys.argv[2] if len(sys.argv) > 2 else "data"

    from .forward import ForwardSolver

    solver = ForwardSolver(mesh_path, verbose=True)
    coords, data = load_nv_data(data_dir)

    basis = BasisSolver(solver, coords, data, verbose=True)
    A = basis.build_influence_matrix(checkpoint_path="solver/A_matrix.npy")

    # Quick solve with Tikhonov
    c = basis.solve_regularized(lambda_reg=1e-2)
    np.save("solver/traction_coeffs.npy", c)
    print("Done. Traction coefficients saved to solver/traction_coeffs.npy")
