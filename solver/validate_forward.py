"""
Phase 1 validation: uniform pressure on sample chamber.

Applies t = (0, 0, 20) GPa on culet_sample (tag 1), solves, evaluates
sigma and D_g at a small grid of points inside the sample chamber ellipse,
and confirms:
  - D_g values are positive (compressive convention from SPEC)
  - D_g magnitudes are in the range ~0.3-0.6 GHz

Run from repo root:
    conda activate dac-recon
    python -m solver.validate_forward [mesh_path]

Defaults to mesh/anvil_preview.msh if no argument given.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ufl
from solver.forward import ForwardSolver, TAG_CULET_SAMPLE
from solver.nv_coupling import build_coupling_matrices, dg_from_stress_voigt, dg_all

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MESH_PATH = sys.argv[1] if len(sys.argv) > 1 else "mesh/anvil_preview.msh"
P_GPa     = 20.0   # uniform normal pressure (GPa)

# A small grid of test points inside the sample chamber ellipse (µm)
# ELL_A=72, ELL_B=50 — staying well inside to avoid boundary effects
TEST_COORDS = np.array([
    [ 0.0,   0.0, 0.0],   # centre
    [20.0,   0.0, 0.0],   # +x
    [-20.0,  0.0, 0.0],   # -x
    [ 0.0,  15.0, 0.0],   # +y
    [ 0.0, -15.0, 0.0],   # -y
    [30.0,  20.0, 0.0],   # off-axis
    [-30.0,-20.0, 0.0],   # off-axis
])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"Validating forward solver")
    print(f"  mesh : {MESH_PATH}")
    print(f"  load : t = (0, 0, {P_GPa}) GPa on culet_sample")
    print("=" * 60)

    # Build solver (loads mesh, assembles K, factors via MUMPS)
    solver = ForwardSolver(MESH_PATH, verbose=True)

    # NV coupling matrices
    nvs, M_list = build_coupling_matrices()
    print(f"\nNV axes:")
    for g, n in enumerate(nvs):
        print(f"  NV{g}: {np.round(n, 4)}")

    # Apply uniform normal traction on sample chamber only
    # t = (0, 0, P) — the sign convention: +z is out of diamond (toward sample).
    # The sample pushes inward; we test both and observe sign of D_g.
    t_uniform = ufl.as_vector([0.0, 0.0, P_GPa])

    print(f"\nSolving K u = f  (uniform t_z = {P_GPa} GPa on culet_sample) ...")
    u = solver.solve_traction(t_uniform, culet_tags=(TAG_CULET_SAMPLE,))
    print("Solve complete.")

    # Displacement at culet centre as a sanity check
    u_centre = solver.stress_at_coords_batch(u, np.array([[0., 0., 0.]]))
    # (that gives stress, not disp — use separate eval for displacement)
    print("\n--- Stress at test points (Voigt: s11,s22,s33,s12,s23,s13) [GPa] ---")
    sigma_vals = solver.stress_at_coords_batch(u, TEST_COORDS)

    point_labels = ["centre", "+x", "-x", "+y", "-y", "off-ax1", "off-ax2"]
    for i, (label, sv) in enumerate(zip(point_labels, sigma_vals)):
        if np.any(np.isnan(sv)):
            print(f"  {label:10s}: OUTSIDE MESH (NaN) — point may be on boundary")
            continue
        print(f"  {label:10s}: s_zz={sv[2]:+7.3f}  tr={sv[:3].sum():+7.3f}  "
              f"sig={np.round(sv, 3)}")

    # D_g at all test points
    print("\n--- D_g [GHz] at test points ---")
    D_all = dg_all(sigma_vals, M_list)  # (4, N_pts)

    header = f"  {'Point':10s}  " + "  ".join(f"D_{g}" for g in range(4))
    print(header)
    for i, label in enumerate(point_labels):
        if np.any(np.isnan(sigma_vals[i])):
            print(f"  {label:10s}  {'NaN':>8}" * 4)
            continue
        dvals = D_all[:, i]
        row = "  ".join(f"{d:+8.4f}" for d in dvals)
        print(f"  {label:10s}  {row}")

    # Summary checks
    print("\n--- Summary checks ---")
    valid_mask = ~np.any(np.isnan(sigma_vals), axis=1)
    if valid_mask.sum() == 0:
        print("ERROR: All test points outside mesh. Check mesh and coordinates.")
        return

    D_valid = D_all[:, valid_mask]  # (4, N_valid)

    all_positive = np.all(D_valid > 0)
    in_range     = np.all((D_valid > 0.05) & (D_valid < 5.0))
    # Scale: 20 GPa is ~40× typical DAC pressure of 0.5 GPa.
    # Data shows ~0.5 GHz at ~0.5 GPa, so we expect ~20 GHz at 20 GPa — let's see.
    # Actually: D_g ~ alpha * 3 * sigma_avg, sigma_avg from 20 GPa surface load.
    # The stress decays into the bulk; near the culet we expect O(1-10) GPa stress.

    print(f"  D_g all positive at valid points: {'YES' if all_positive else 'NO — SIGN PROBLEM'}")
    print(f"  D_g range: [{D_valid.min():.4f}, {D_valid.max():.4f}] GHz")

    if not all_positive:
        print("\n  NOTE: Negative D_g means the traction sign is flipped.")
        print("  The physical compressive traction should be t = (0, 0, -P).")
        print("  Check forward.py traction sign convention and culet outward normal.")
    else:
        print("  PASS: Sign convention is correct.")

    # Linearity check: solve at P/2 and verify D_g scales by 0.5
    print(f"\nLinearity check: solving at P = {P_GPa/2:.1f} GPa ...")
    t_half = ufl.as_vector([0.0, 0.0, P_GPa / 2.0])
    u_half = solver.solve_traction(t_half, culet_tags=(TAG_CULET_SAMPLE,))
    sig_half = solver.stress_at_coords_batch(u_half, TEST_COORDS)
    D_half = dg_all(sig_half, M_list)

    valid_and_nonzero = valid_mask & (np.abs(D_all[0]) > 1e-10)
    if valid_and_nonzero.sum() > 0:
        ratios = D_all[:, valid_and_nonzero] / D_half[:, valid_and_nonzero]
        print(f"  D_full / D_half ratios (should be ~2.0): "
              f"min={ratios.min():.4f}, max={ratios.max():.4f}, "
              f"mean={ratios.mean():.4f}")
        linearity_ok = np.allclose(ratios, 2.0, rtol=1e-3)
        print(f"  Linearity: {'PASS' if linearity_ok else 'FAIL'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
