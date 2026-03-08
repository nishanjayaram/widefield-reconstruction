"""
LSQR inversion for NV-diamond stress reconstruction.

Uses the matrix-free NVOperator (operators.py) wrapped in scipy's LSQR.
Tikhonov regularization is applied by augmenting the system:

    [  A  ]       [ d ]
    [ sqrt(lam)*I ] c = [ 0 ]

which LSQR solves as a single least-squares problem.

Usage
-----
    # With real data:
    from solver.forward import ForwardSolver
    from solver.operators import NVOperator
    from solver.invert import run_lsqr, load_nv_data

    solver  = ForwardSolver("mesh/anvil_fine_tet.msh")
    coords, data = load_nv_data("data")
    op      = NVOperator(solver, coords)
    result  = run_lsqr(op, data, lambda_reg=1e-3, max_iter=200)

    # Reconstruct final stress:
    from solver.invert import reconstruct_final_stress
    sigma = reconstruct_final_stress(solver, op, result["c"])
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from .operators import NVOperator
from .forward import ForwardSolver, TAG_CULET_SAMPLE
from .nv_coupling import build_coupling_matrices, dg_all


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nv_data(data_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """
    Load the four NV data files.

    Returns
    -------
    coords : (N_meas, 2) ndarray  — (x, y) in µm
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


# ---------------------------------------------------------------------------
# Tikhonov-augmented LSQR
# ---------------------------------------------------------------------------

class AugmentedOperator:
    """
    Augments the operator A with Tikhonov regularization:

        A_aug = [A; sqrt(lam) * I]    (shape: (N_data + N_B, N_B))
        d_aug = [d; 0]

    This lets LSQR minimize ||A c - d||^2 + lam * ||c||^2 natively.
    """

    def __init__(self, op: NVOperator, lambda_reg: float) -> None:
        self.op     = op
        self.lam_sq = float(np.sqrt(lambda_reg))
        self.shape  = (op.N_data + op.N_B, op.N_B)
        self.dtype  = np.float64

    def matvec(self, c: np.ndarray) -> np.ndarray:
        Ac  = self.op.matvec(c)
        reg = self.lam_sq * c
        return np.concatenate([Ac, reg])

    def rmatvec(self, r: np.ndarray) -> np.ndarray:
        r_data = r[:self.op.N_data]
        r_reg  = r[self.op.N_data:]
        return self.op.rmatvec(r_data) + self.lam_sq * r_reg

    def as_linear_operator(self) -> spla.LinearOperator:
        return spla.LinearOperator(
            shape=self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=self.dtype,
        )


def run_lsqr(
    op: NVOperator,
    nv_data: np.ndarray,
    lambda_reg: float = 1e-3,
    max_iter: int = 200,
    atol: float = 1e-6,
    btol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Run Tikhonov-regularised LSQR.

    Parameters
    ----------
    op         : NVOperator
    nv_data    : (4, N_meas) ndarray — measured Delta_D [GHz]
    lambda_reg : Tikhonov regularisation parameter
    max_iter   : maximum LSQR iterations
    atol, btol : LSQR convergence tolerances
    verbose    : print per-iteration progress

    Returns
    -------
    dict with keys:
      c          : (N_B,) traction coefficients
      t_pixels   : (N_meas, 3) traction field [GPa]
      residual   : ||d - A c|| [GHz]
      n_iter     : number of LSQR iterations
      istop      : LSQR stop condition code
    """
    # Flatten data: [D_0(x_1),..,D_0(x_N), D_1(x_1),.., D_3(x_N)]
    d = nv_data.ravel(order="C")   # (4*N_meas,)

    # Augmented system
    aug = AugmentedOperator(op, lambda_reg)
    d_aug = np.concatenate([d, np.zeros(op.N_B)])

    A_lo = aug.as_linear_operator()

    if verbose:
        print(f"LSQR: shape=({aug.shape[0]}, {aug.shape[1]}), "
              f"lambda={lambda_reg:.2e}, max_iter={max_iter}")

    t0 = time.perf_counter()

    # Callback for per-iteration printing
    iter_count = [0]
    def _callback(xk):
        iter_count[0] += 1
        if verbose and iter_count[0] % 10 == 0:
            Axk = op.matvec(xk)
            res = np.linalg.norm(d - Axk)
            print(f"  iter {iter_count[0]:4d}  ||r|| = {res:.4e} GHz")

    result = spla.lsqr(
        A_lo,
        d_aug,
        atol=atol,
        btol=btol,
        iter_lim=max_iter,
        show=False,
        calc_var=False,
    )

    c, istop, itn, r1norm = result[0], result[1], result[2], result[3]
    elapsed = time.perf_counter() - t0

    # Final residual on original (non-augmented) system
    Ac = op.matvec(c)
    residual = float(np.linalg.norm(d - Ac))

    if verbose:
        print(f"LSQR done: {itn} iterations in {elapsed:.1f}s")
        print(f"  istop={istop}, ||d - Ac|| = {residual:.4e} GHz")
        print(f"  ||c|| = {np.linalg.norm(c):.4e} GPa")

    return {
        "c":       c,
        "t_nodes": c.reshape(op.N_nodes, 3),
        "residual": residual,
        "n_iter":  itn,
        "istop":   istop,
    }


# ---------------------------------------------------------------------------
# Final stress reconstruction
# ---------------------------------------------------------------------------

def reconstruct_final_stress(
    solver: ForwardSolver,
    op: NVOperator,
    c: np.ndarray,
    eval_coords: np.ndarray | None = None,
) -> dict:
    """
    Reconstruct the full 3D stress field from traction coefficients c.

    Uses superposition: form one combined load vector, one forward solve.

    Parameters
    ----------
    solver      : ForwardSolver
    op          : NVOperator (carries pixel coords + surface projection)
    c           : (N_B,) traction coefficients [GPa]
    eval_coords : (M, 3) ndarray or None (defaults to pixel locations)

    Returns
    -------
    dict with keys:
      sigma_voigt : (M, 6) stress [GPa] in Voigt order
      eval_coords : (M, 3) evaluation coordinates
      D_predicted : (4, M) predicted D_g values [GHz]
    """
    t_nodes = c.reshape(op.N_nodes, 3)
    b = op._traction_to_load_vector(t_nodes)
    u_final = solver.solve_load_vector(b)

    if eval_coords is None:
        eval_coords = op.pixel_coords

    sigma_voigt = solver.stress_at_coords_batch(u_final, eval_coords)
    sigma_voigt = np.where(np.isnan(sigma_voigt), 0.0, sigma_voigt)

    _, M_list = build_coupling_matrices()
    D_pred = dg_all(sigma_voigt, M_list)   # (4, M)

    return {
        "sigma_voigt": sigma_voigt,
        "eval_coords": eval_coords,
        "D_predicted": D_pred,
        "u":           u_final,
    }


# ---------------------------------------------------------------------------
# Synthetic data test
# ---------------------------------------------------------------------------

def run_synthetic_test(
    mesh_path: str = "mesh/anvil_preview.msh",
    lambda_reg: float = 1e-3,
    max_iter: int = 50,
) -> None:
    """
    End-to-end test on synthetic data.

    1. Load mesh, build operator.
    2. Define a known non-uniform traction (Gaussian bump in z).
    3. Forward solve to generate synthetic D_g maps.
    4. Add noise, invert with LSQR.
    5. Compare recovered traction to truth.
    """
    print("=" * 60)
    print("Synthetic inversion test")
    print("=" * 60)

    # Build solver
    solver = ForwardSolver(mesh_path, verbose=True)

    # Build a small pixel grid inside the sample chamber for the test
    # (use a coarser grid to keep cost down)
    x_grid = np.arange(-60, 61, 5, dtype=float)
    y_grid = np.arange(-40, 41, 5, dtype=float)
    xx, yy = np.meshgrid(x_grid, y_grid)
    # Keep only points inside the ellipse
    ELL_A, ELL_B = 72.0, 50.0
    mask = (xx / ELL_A)**2 + (yy / ELL_B)**2 < 0.9
    px = xx[mask].ravel()
    py = yy[mask].ravel()
    pz = np.zeros_like(px)
    pixel_coords = np.column_stack([px, py, pz])
    N_meas = len(pixel_coords)
    print(f"Synthetic test grid: {N_meas} pixels")

    op = NVOperator(solver, pixel_coords)

    # True traction: Gaussian bump in z-direction (normal pressure)
    # c is now indexed by culet node (not pixel), so evaluate Gaussian at node coords.
    sigma_x0, sigma_y0 = 20.0, 15.0   # Gaussian widths (µm)
    node_coords = solver.V.tabulate_dof_coordinates()[op._culet_node_list]  # (N_nodes, 3)
    node_x, node_y = node_coords[:, 0], node_coords[:, 1]
    t_true_z_nodes = 20.0 * np.exp(-(node_x**2 / sigma_x0**2 + node_y**2 / sigma_y0**2))
    c_true = np.zeros(op.N_B)
    c_true[2::3] = t_true_z_nodes   # z-component at each node

    # For comparison with pixels, evaluate at pixel coords too
    t_true_z = 20.0 * np.exp(-(px**2 / sigma_x0**2 + py**2 / sigma_y0**2))
    print(f"True traction: Gaussian in z, max={t_true_z_nodes.max():.1f} GPa")

    # Forward: generate synthetic D_g
    d_synthetic = op.matvec(c_true)
    D_synth = d_synthetic.reshape(4, N_meas)
    print(f"Synthetic D_g range: [{d_synthetic.min():.4f}, {d_synthetic.max():.4f}] GHz")

    # Add 1% noise
    noise_level = 0.01 * np.abs(d_synthetic).mean()
    rng = np.random.default_rng(0)
    d_noisy = d_synthetic + rng.standard_normal(len(d_synthetic)) * noise_level
    print(f"Noise level: {noise_level:.4e} GHz")

    # Invert
    result = run_lsqr(
        op, d_noisy.reshape(4, N_meas),
        lambda_reg=lambda_reg,
        max_iter=max_iter,
        verbose=True,
    )

    c_rec = result["c"]
    t_rec_z = c_rec[2::3]   # z-component at each culet node

    # Recovery quality: compare at node positions
    corr = float(np.corrcoef(t_true_z_nodes, t_rec_z)[0, 1])
    rms  = float(np.sqrt(np.mean((t_true_z_nodes - t_rec_z)**2)))
    print(f"\nRecovery (at {op.N_nodes} culet nodes):")
    print(f"  Pearson r (z-traction): {corr:.4f}")
    print(f"  RMS error:              {rms:.4f} GPa")
    print(f"  True max t_z:           {t_true_z_nodes.max():.4f} GPa")
    print(f"  Recovered max t_z:      {t_rec_z.max():.4f} GPa")

    if corr > 0.8:
        print("\nSYNTHETIC TEST PASSED")
    else:
        print("\nSYNTHETIC TEST FAILED — check adjoint and regularization")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"

    if mode == "synthetic":
        mesh = sys.argv[2] if len(sys.argv) > 2 else "mesh/anvil_preview.msh"
        run_synthetic_test(mesh_path=mesh)

    elif mode == "real":
        mesh        = sys.argv[2] if len(sys.argv) > 2 else "mesh/anvil_fine_tet.msh"
        data        = sys.argv[3] if len(sys.argv) > 3 else "data"
        lam         = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
        s_gasket    = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
        s_sample    = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0

        solver  = ForwardSolver(mesh, verbose=True)
        coords, nv_data = load_nv_data(data)
        op      = NVOperator(solver, coords,
                             traction_grid_spacing=s_gasket,
                             sample_grid_spacing=s_sample)

        # Adjoint check before inverting
        op.check_adjoint(n_trials=2)

        result = run_lsqr(op, nv_data, lambda_reg=lam, max_iter=200)
        np.save("solver/traction_coeffs.npy", result["c"])
        print("Traction coefficients saved to solver/traction_coeffs.npy")

        # Reconstruct stress and save
        recon = reconstruct_final_stress(solver, op, result["c"])
        np.save("solver/sigma_culet.npy", recon["sigma_voigt"])
        np.save("solver/D_predicted.npy", recon["D_predicted"])
        print("Stress and predicted D_g saved.")
