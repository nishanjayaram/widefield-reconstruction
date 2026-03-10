#!/usr/bin/env python3
"""
run_analysis.py — Complete NV-diamond stress inversion pipeline.

Run from the project root (widefield-reconstruction/):

    python run_analysis.py

Or with custom options:

    python run_analysis.py \\
        --mesh  mesh/medium_mesh_COMSOL_tet_mesh.xdmf \\
        --data  data \\
        --out   results \\
        --lam   5e-8 \\
        --s-sample 2.0 \\
        --s-gasket 10.0 \\
        --max-iter 400

Mesh auto-conversion
--------------------
If --mesh points to a .nas file, the pipeline converts:
    .nas → .msh  (via mesh/convert_nas_to_tet.py)
    .msh → .xdmf (via mesh/convert_msh_to_xdmf.py)
and then loads the .xdmf. Converted files are written alongside the input.

If --mesh points to a .msh file, the .msh → .xdmf conversion is run.

If --mesh points to a .xdmf file (or the _mesh.xdmf variant), it is loaded directly.

Outputs (written to --out, default: results/)
---------------------------------------------
    traction_coeffs.npy     — (N_B,) traction coefficients [GPa]
    traction_grid_xy.npy    — (N_B/3, 2) traction grid x,y positions [µm]
    sigma_culet.npy         — (N_meas, 6) Voigt stress at NV pixel coords [GPa]
    D_predicted.npy         — (4, N_meas) predicted ΔD [GHz]
    stress_culet.csv        — columns: x,y,sigma_xx,sigma_yy,sigma_zz,sigma_yz,sigma_xz,sigma_xy [µm / GPa]
    plots/sigma_zz.pdf
    plots/tau.pdf
    plots/stress_maps.pdf
    plots/Dg_comparison.pdf
    plots/Dg_scatter.pdf
    plots/traction_field.pdf

Environment
-----------
    conda activate dac-recon   # or your equivalent env with DOLFINx + gmsh + meshio
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Defaults (best parameters from local L-curve analysis)
# ---------------------------------------------------------------------------
DEFAULT_MESH      = "mesh/medium_mesh_COMSOL_tet_mesh.xdmf"
DEFAULT_DATA      = "data"
DEFAULT_OUT       = "results"
DEFAULT_LAM       = 5e-8
DEFAULT_S_SAMPLE  = 2.0    # µm — fine traction grid in sample ellipse
DEFAULT_S_GASKET  = 10.0   # µm — coarse traction grid in gasket
DEFAULT_MAX_ITER  = 400


# ---------------------------------------------------------------------------
# Mesh conversion helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=True)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")


def resolve_mesh(mesh_arg: str) -> str:
    """
    Given any of .nas / .msh / .xdmf, ensure XDMF files exist and return
    the path to the _mesh.xdmf file that ForwardSolver expects.
    """
    p = Path(mesh_arg).resolve()
    stem = p.with_suffix("")

    if p.suffix == ".nas":
        msh_path  = stem.with_suffix(".msh")
        xdmf_path = stem.parent / (stem.name + "_mesh.xdmf")
        if not msh_path.exists():
            _run([sys.executable, "mesh/convert_nas_to_tet.py",
                  "--input", str(p), "--output", str(msh_path)],
                 f"Converting NAS → MSH: {p.name}")
        if not xdmf_path.exists():
            _run([sys.executable, "mesh/convert_msh_to_xdmf.py",
                  "--input", str(msh_path), "--output", str(msh_path)],
                 f"Converting MSH → XDMF: {msh_path.name}")
        return str(xdmf_path)

    elif p.suffix == ".msh":
        xdmf_path = p.parent / (stem.name + "_mesh.xdmf")
        if not xdmf_path.exists():
            _run([sys.executable, "mesh/convert_msh_to_xdmf.py",
                  "--input", str(p), "--output", str(p)],
                 f"Converting MSH → XDMF: {p.name}")
        return str(xdmf_path)

    elif p.suffix == ".xdmf":
        # Accept either bare .xdmf or the _mesh.xdmf variant
        if p.name.endswith("_mesh.xdmf"):
            return str(p)
        mesh_xdmf = p.parent / (stem.name + "_mesh.xdmf")
        if mesh_xdmf.exists():
            return str(mesh_xdmf)
        return str(p)

    else:
        raise ValueError(f"Unrecognised mesh extension: {p.suffix}  "
                         f"(expected .nas, .msh, or .xdmf)")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_stress_csv(sigma_voigt: np.ndarray, coords: np.ndarray, path: Path) -> None:
    """
    Write (x, y, sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy)
    to a CSV file.  Voigt order: [xx, yy, zz, yz, xz, xy] (FEniCSx convention).
    """
    x = coords[:, 0]
    y = coords[:, 1]
    header = "x_um,y_um,sigma_xx_GPa,sigma_yy_GPa,sigma_zz_GPa,sigma_yz_GPa,sigma_xz_GPa,sigma_xy_GPa"
    data = np.column_stack([x, y, sigma_voigt])
    np.savetxt(str(path), data, delimiter=",", header=header, comments="")
    print(f"  Stress CSV: {path}  ({len(data):,} rows)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NV-diamond stress inversion — full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mesh",      default=DEFAULT_MESH,
                        help="Mesh file (.nas, .msh, or .xdmf)")
    parser.add_argument("--data",      default=DEFAULT_DATA,
                        help="Directory containing nv{0-3}_roi_*_minusZFS.txt")
    parser.add_argument("--out",       default=DEFAULT_OUT,
                        help="Output directory for all results and plots")
    parser.add_argument("--lam",       type=float, default=DEFAULT_LAM,
                        help="Tikhonov regularisation parameter")
    parser.add_argument("--s-sample",  type=float, default=DEFAULT_S_SAMPLE,
                        help="Traction grid spacing inside sample ellipse (µm)")
    parser.add_argument("--s-gasket",  type=float, default=DEFAULT_S_GASKET,
                        help="Traction grid spacing in gasket annulus (µm)")
    parser.add_argument("--max-iter",  type=int,   default=DEFAULT_MAX_ITER,
                        help="Maximum LSQR iterations")
    parser.add_argument("--skip-adjoint-check", action="store_true",
                        help="Skip the adjoint consistency check (saves ~30s)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Resolve mesh → XDMF
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Step 1: Resolve mesh")
    print("="*60)
    mesh_xdmf = resolve_mesh(args.mesh)
    print(f"  Mesh XDMF: {mesh_xdmf}")

    # ------------------------------------------------------------------
    # 2. Load ForwardSolver
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Step 2: Build forward solver (assemble K, MUMPS factorisation)")
    print("="*60)
    from solver.forward   import ForwardSolver
    from solver.operators import NVOperator
    from solver.invert    import run_lsqr, reconstruct_final_stress
    from solver.invert    import load_nv_data

    t0 = time.perf_counter()
    solver = ForwardSolver(mesh_xdmf, verbose=True)
    print(f"  ForwardSolver ready in {time.perf_counter()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Load NV data + build operator
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Step 3: Load NV data and build NVOperator")
    print("="*60)
    coords, nv_data = load_nv_data(args.data)
    print(f"  NV pixels: {len(coords):,}")

    op = NVOperator(
        solver, coords,
        traction_grid_spacing=args.s_gasket,
        sample_grid_spacing=args.s_sample,
    )

    # ------------------------------------------------------------------
    # 4. Adjoint check
    # ------------------------------------------------------------------
    if not args.skip_adjoint_check:
        print("\n" + "="*60)
        print("  Step 4: Adjoint consistency check")
        print("="*60)
        op.check_adjoint(n_trials=2)

    # ------------------------------------------------------------------
    # 5. LSQR inversion
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  Step 5: LSQR inversion  (λ={args.lam:.1e}, max_iter={args.max_iter})")
    print("="*60)
    result = run_lsqr(
        op, nv_data,
        lambda_reg=args.lam,
        max_iter=args.max_iter,
        verbose=True,
    )
    c = result["c"]

    # ------------------------------------------------------------------
    # 6. Reconstruct stress
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Step 6: Reconstruct stress field")
    print("="*60)
    recon = reconstruct_final_stress(solver, op, c)
    sigma_voigt = recon["sigma_voigt"]   # (N_meas, 6)
    D_predicted = recon["D_predicted"]  # (4, N_meas)

    # ------------------------------------------------------------------
    # 7. Save NPY outputs
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Step 7: Save results")
    print("="*60)
    np.save(out_dir / "traction_coeffs.npy", c)
    np.save(out_dir / "sigma_culet.npy",     sigma_voigt)
    np.save(out_dir / "D_predicted.npy",     D_predicted)
    if hasattr(op, "_traction_grid_xy"):
        np.save(out_dir / "traction_grid_xy.npy", op._traction_grid_xy)
    print(f"  NPY files written to {out_dir}/")

    # Save CSV
    save_stress_csv(sigma_voigt, coords, out_dir / "stress_culet.csv")

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n=== Inversion Summary ===")
    voigt_labels = ["σ_xx", "σ_yy", "σ_zz", "σ_yz", "σ_xz", "σ_xy"]
    for i, lab in enumerate(voigt_labels):
        v = sigma_voigt[:, i]
        print(f"  {lab}: mean={v.mean():.2f}, std={v.std():.2f}, "
              f"min={v.min():.2f}, max={v.max():.2f} GPa")

    from scipy.stats import pearsonr
    nv_labels = ["NV0", "NV1", "NV2", "NV3"]
    _, D_measured = load_nv_data(args.data)
    d_flat = D_measured.reshape(-1)
    d_pred_flat = D_predicted.reshape(-1)
    total_rms = float(np.sqrt(np.mean((d_flat - d_pred_flat)**2)))
    print(f"\nD_g residuals:")
    for g in range(4):
        dm = D_measured[g]
        dp = D_predicted[g]
        r, _ = pearsonr(dm, dp)
        rms  = float(np.sqrt(np.mean((dm - dp)**2)))
        print(f"  {nv_labels[g]}: RMS={rms:.4f} GHz, r={r:.4f}")
    total_l2 = float(np.sqrt(np.sum((D_measured - D_predicted)**2)))
    print(f"  Total L2: {total_l2:.4f} GHz,  per-pixel RMS: {total_rms:.4f} GHz")

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Step 8: Generate plots")
    print("="*60)
    # Temporarily point solver outputs where visualize.py expects them
    # (it loads from solver/ by default; we redirect via symlinks or direct calls)
    from solver.visualize import (
        plot_sigma_zz, plot_tau, plot_stress_maps,
        plot_Dg_comparison, plot_Dg_scatter, plot_traction_field,
    )
    _, D_measured = load_nv_data(args.data)
    plot_sigma_zz(sigma_voigt, coords, plot_dir)
    plot_tau(sigma_voigt, coords, plot_dir)
    plot_stress_maps(sigma_voigt, coords, plot_dir)
    plot_Dg_comparison(D_measured, D_predicted, coords, plot_dir)
    plot_Dg_scatter(D_measured, D_predicted, plot_dir)
    if hasattr(op, "_traction_grid_xy"):
        grid_xy = op._traction_grid_xy
        if len(c) == 3 * len(grid_xy):
            plot_traction_field(c, grid_xy, plot_dir)

    print(f"\nAll plots saved to: {plot_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
