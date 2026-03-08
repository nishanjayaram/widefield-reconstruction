"""
Visualization of NV-diamond stress inversion results.

Loads:
  solver/traction_coeffs.npy  — (N_B,) traction coefficients [GPa]
  solver/sigma_culet.npy      — (N_meas, 6) stress at pixel locs [GPa], Voigt order
  solver/D_predicted.npy      — (4, N_meas) predicted D_g [GHz]
  data/nv{0-3}_roi_...txt     — measured D_g [GHz]

Saves plots to solver/plots/.

Usage
-----
    conda run -n dac-recon python -m solver.visualize [--data data] [--out solver/plots]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nv_data(data_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    coords = None
    data_list = []
    for g in range(4):
        arr = np.loadtxt(data_dir / f"nv{g}_roi_centered_minusZFS.txt",
                         delimiter=",", skiprows=1)
        if coords is None:
            coords = arr[:, :2]
        data_list.append(arr[:, 2])
    return coords, np.array(data_list)   # coords: (N,2), data: (4,N)


def scatter_map(ax, x, y, v, title, cmap="RdBu_r", symmetric=False, unit=""):
    """Scatter plot colored by v on irregular pixel grid."""
    if symmetric:
        vmax = np.nanpercentile(np.abs(v), 99)
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        sc = ax.scatter(x, y, c=v, cmap=cmap, norm=norm, s=1.5, rasterized=True)
    else:
        sc = ax.scatter(x, y, c=v, cmap=cmap, s=1.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label=unit, fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)


# ---------------------------------------------------------------------------
# Plot 1: Stress maps on culet
# ---------------------------------------------------------------------------

VOIGT_LABELS = ["σ_xx", "σ_yy", "σ_zz", "σ_yz", "σ_xz", "σ_xy"]


def plot_stress_maps(sigma_voigt: np.ndarray, coords: np.ndarray, out_dir: Path) -> None:
    """6-panel figure: all Voigt stress components."""
    x, y = coords[:, 0], coords[:, 1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for i, (ax, label) in enumerate(zip(axes, VOIGT_LABELS)):
        v = sigma_voigt[:, i]
        scatter_map(ax, x, y, v, label, cmap="RdBu_r", symmetric=True, unit="GPa")
    fig.suptitle("Stress on culet (GPa)", fontsize=14)
    fig.tight_layout()
    out = out_dir / "stress_maps.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def _to_grid(x, y, v, spacing=2.0, smooth_sigma=2.0):
    """
    Interpolate scattered (x,y,v) onto a regular grid and apply Gaussian smoothing.

    The FEM stress is piecewise-linear (P1 within each ~5 µm element) and
    discontinuous across element boundaries.  Griddata faithfully captures
    those jumps, making the map look blocky.  A Gaussian filter with
    smooth_sigma (in grid cells, so σ_physical = sigma * spacing µm) removes
    element-boundary artifacts while preserving features at the traction-grid
    scale (~10 µm).
    """
    gx = np.arange(x.min(), x.max() + spacing, spacing)
    gy = np.arange(y.min(), y.max() + spacing, spacing)
    GX, GY = np.meshgrid(gx, gy)
    VG = griddata(np.c_[x, y], v, (GX, GY), method="linear")

    if smooth_sigma > 0:
        mask = np.isnan(VG)
        # Fill NaN with mean before filtering so boundaries don't bleed
        fill = np.where(mask, np.nanmean(VG), VG)
        VG = gaussian_filter(fill, sigma=smooth_sigma)
        VG[mask] = np.nan   # restore boundary NaNs

    return GX, GY, VG


def plot_sigma_zz(sigma_voigt: np.ndarray, coords: np.ndarray, out_dir: Path) -> None:
    """σ_zz map — gridded, viridis, fixed colorbar 16–28 GPa."""
    x, y = coords[:, 0], coords[:, 1]
    GX, GY, VG = _to_grid(x, y, sigma_voigt[:, 2], spacing=2.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(GX, GY, VG, cmap="viridis", vmin=16, vmax=28,
                       shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax, label="σ_zz (GPa)", fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("σ_zz (normal stress on culet)")
    fig.tight_layout()
    out = out_dir / "sigma_zz.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_tau(sigma_voigt: np.ndarray, coords: np.ndarray, out_dir: Path) -> None:
    """
    In-plane shear traction magnitude |τ| = sqrt(σ_xz² + σ_yz²) with quiver arrows.

    Voigt convention:
      sigma_voigt[:, 4] = σ_yz  (index 4: (1,2))
      sigma_voigt[:, 5] = σ_xz  (index 5: (0,2))
    """
    x, y = coords[:, 0], coords[:, 1]
    tau_x = sigma_voigt[:, 5]   # σ_xz
    tau_y = sigma_voigt[:, 4]   # σ_yz
    tau_mag = np.sqrt(tau_x**2 + tau_y**2)

    # Background: gridded magnitude
    GX, GY, MG = _to_grid(x, y, tau_mag, spacing=2.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(GX, GY, MG, cmap="viridis", vmin=0, vmax=8,
                       shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax, label="|τ| (GPa)", fraction=0.046, pad=0.04)

    # Quiver: coarser grid (8 µm spacing), unit-length arrows
    arrow_spacing = 8.0
    _, _, TX = _to_grid(x, y, tau_x, spacing=arrow_spacing)
    _, _, TY = _to_grid(x, y, tau_y, spacing=arrow_spacing)
    agx = np.arange(x.min(), x.max() + arrow_spacing, arrow_spacing)
    agy = np.arange(y.min(), y.max() + arrow_spacing, arrow_spacing)
    AGX, AGY = np.meshgrid(agx, agy)

    valid = np.isfinite(TX) & np.isfinite(TY)
    mag_a = np.sqrt(TX**2 + TY**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(valid & (mag_a > 0), TX / mag_a, 0.0)
        uy = np.where(valid & (mag_a > 0), TY / mag_a, 0.0)

    ax.quiver(AGX[valid], AGY[valid], ux[valid], uy[valid],
              color="black", scale=30, width=0.003, headwidth=3, headlength=4,
              pivot="middle", alpha=0.7)

    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(r"|τ| = $\sqrt{σ_{xz}^2 + σ_{yz}^2}$ (in-plane shear traction)")
    fig.tight_layout()
    out = out_dir / "tau.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: D_predicted vs D_measured per NV orientation
# ---------------------------------------------------------------------------

NV_LABELS = ["NV0", "NV1", "NV2", "NV3"]


def plot_Dg_comparison(
    D_measured: np.ndarray,   # (4, N)
    D_predicted: np.ndarray,  # (4, N)
    coords: np.ndarray,       # (N, 2)
    out_dir: Path,
) -> None:
    """2×4 figure: measured | predicted for each NV orientation."""
    x, y = coords[:, 0], coords[:, 1]
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    for g in range(4):
        dm = D_measured[g]
        dp = D_predicted[g]
        res = dm - dp

        vmin = min(dm.min(), dp.min())
        vmax = max(dm.max(), dp.max())

        # Measured
        sc = axes[g, 0].scatter(x, y, c=dm, cmap="viridis",
                                vmin=vmin, vmax=vmax, s=1.5, rasterized=True)
        plt.colorbar(sc, ax=axes[g, 0], label="GHz", fraction=0.046, pad=0.04)
        axes[g, 0].set_title(f"{NV_LABELS[g]} measured")
        axes[g, 0].set_aspect("equal")

        # Predicted
        sc = axes[g, 1].scatter(x, y, c=dp, cmap="viridis",
                                vmin=vmin, vmax=vmax, s=1.5, rasterized=True)
        plt.colorbar(sc, ax=axes[g, 1], label="GHz", fraction=0.046, pad=0.04)
        axes[g, 1].set_title(f"{NV_LABELS[g]} predicted")
        axes[g, 1].set_aspect("equal")

        # Residual
        rvmax = np.percentile(np.abs(res), 99)
        norm = TwoSlopeNorm(vcenter=0, vmin=-rvmax, vmax=rvmax)
        sc = axes[g, 2].scatter(x, y, c=res, cmap="RdBu_r",
                                norm=norm, s=1.5, rasterized=True)
        plt.colorbar(sc, ax=axes[g, 2], label="GHz", fraction=0.046, pad=0.04)
        rms = float(np.sqrt(np.mean(res**2)))
        axes[g, 2].set_title(f"{NV_LABELS[g]} residual (RMS={rms:.4f} GHz)")
        axes[g, 2].set_aspect("equal")

        for ax in axes[g]:
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")

    fig.suptitle("D_g: measured vs predicted (GHz)", fontsize=14)
    fig.tight_layout()
    out = out_dir / "Dg_comparison.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Scatter plot — predicted vs measured (per orientation)
# ---------------------------------------------------------------------------

def plot_Dg_scatter(
    D_measured: np.ndarray,
    D_predicted: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    for g, ax in enumerate(axes):
        dm = D_measured[g]
        dp = D_predicted[g]
        corr = float(np.corrcoef(dm, dp)[0, 1])
        rms = float(np.sqrt(np.mean((dm - dp)**2)))
        ax.scatter(dm, dp, s=1, alpha=0.3, rasterized=True)
        lo = min(dm.min(), dp.min())
        hi = max(dm.max(), dp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("Measured D_g (GHz)")
        ax.set_ylabel("Predicted D_g (GHz)")
        ax.set_title(f"{NV_LABELS[g]}: r={corr:.4f}, RMS={rms:.4f} GHz")
    fig.suptitle("Predicted vs measured D_g", fontsize=13)
    fig.tight_layout()
    out = out_dir / "Dg_scatter.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4: Traction field
# ---------------------------------------------------------------------------

def plot_traction_field(
    traction_coeffs: np.ndarray,   # (N_B,) = (3*N_nodes,)
    node_coords: np.ndarray,       # (N_nodes, 3) — culet node positions
    out_dir: Path,
) -> None:
    N_nodes = len(node_coords)
    t = traction_coeffs[:3 * N_nodes].reshape(N_nodes, 3)
    x, y = node_coords[:, 0], node_coords[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["t_x (GPa)", "t_y (GPa)", "t_z (GPa)"]
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        scatter_map(ax, x, y, t[:, i], lab, cmap="RdBu_r", symmetric=True, unit="GPa")
    fig.suptitle("Recovered surface traction on culet", fontsize=13)
    fig.tight_layout()
    out = out_dir / "traction_field.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_summary(sigma_voigt, D_measured, D_predicted, traction_coeffs, coords):
    print("\n=== Summary ===")
    print(f"N_meas = {len(coords)}")
    print(f"Traction ||c|| = {np.linalg.norm(traction_coeffs):.4e} GPa")
    print(f"\nStress stats on culet (GPa):")
    for i, lab in enumerate(VOIGT_LABELS):
        v = sigma_voigt[:, i]
        print(f"  {lab}: mean={v.mean():.4f}, std={v.std():.4f}, "
              f"min={v.min():.4f}, max={v.max():.4f}")

    print(f"\nD_g residuals (GHz):")
    for g in range(4):
        dm, dp = D_measured[g], D_predicted[g]
        rms = float(np.sqrt(np.mean((dm - dp)**2)))
        corr = float(np.corrcoef(dm, dp)[0, 1])
        print(f"  NV{g}: RMS={rms:.4f} GHz, r={corr:.4f}")

    total_rms = float(np.sqrt(np.mean((D_measured - D_predicted)**2)))
    print(f"  Total RMS: {total_rms:.4f} GHz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize inversion results")
    parser.add_argument("--data",    default="data",         help="data directory")
    parser.add_argument("--solver",  default="solver",       help="solver output directory")
    parser.add_argument("--out",     default="solver/plots", help="output directory for plots")
    args = parser.parse_args()

    solver_dir = Path(args.solver)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    traction_coeffs = np.load(solver_dir / "traction_coeffs.npy")
    sigma_voigt     = np.load(solver_dir / "sigma_culet.npy")
    D_predicted     = np.load(solver_dir / "D_predicted.npy")

    # Load measurements
    coords, D_measured = load_nv_data(args.data)

    print(f"  traction_coeffs: {traction_coeffs.shape}")
    print(f"  sigma_culet:     {sigma_voigt.shape}")
    print(f"  D_predicted:     {D_predicted.shape}")
    print(f"  D_measured:      {D_measured.shape}")
    print(f"  coords:          {coords.shape}")

    print_summary(sigma_voigt, D_measured, D_predicted, traction_coeffs, coords)

    # Load node coordinates for traction plot (requires NVOperator)
    print("\nGenerating plots...")
    plot_sigma_zz(sigma_voigt, coords, out_dir)
    plot_tau(sigma_voigt, coords, out_dir)
    plot_stress_maps(sigma_voigt, coords, out_dir)
    plot_Dg_comparison(D_measured, D_predicted, coords, out_dir)
    plot_Dg_scatter(D_measured, D_predicted, out_dir)

    # Traction field: need culet node coords from mesh
    try:
        from solver.forward import ForwardSolver
        from solver.operators import NVOperator
        mesh_path = Path(args.solver).parent / "mesh" / "anvil_preview.msh"
        if not mesh_path.exists():
            mesh_path = Path("mesh/anvil_preview.msh")
        _solver = ForwardSolver(str(mesh_path), verbose=False)
        _op = NVOperator(_solver, coords)
        node_coords = _solver.V.tabulate_dof_coordinates()[_op._culet_node_list]
        if len(traction_coeffs) == 3 * len(node_coords):
            plot_traction_field(traction_coeffs, node_coords, out_dir)
        else:
            print(f"  Skipping traction plot: size mismatch "
                  f"(coeffs {len(traction_coeffs)} vs 3*nodes {3*len(node_coords)})")
    except Exception as e:
        print(f"  Skipping traction plot: {e}")

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
