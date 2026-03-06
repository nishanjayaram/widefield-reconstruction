"""
Standalone physics validation tests (no FEM required).

Tests:
  1. C_lab is symmetric and positive definite.
  2. NV axes are unit vectors.
  3. D_g under isotropic compression matches the expected analytic value.
  4. Voigt round-trip: voigt_to_tensor -> tensor_to_voigt reproduces C_cubic.
  5. Rotation orthogonality: R is a rotation (R^T R = I, det R = 1).

Run with:
    python -m solver.test_physics
or:
    cd widefield-reconstruction && python solver/test_physics.py
"""

import sys
import numpy as np

# Allow running as script without installing as package
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from solver.crystal import (
    build_C_lab, C_cubic_voigt, voigt_to_tensor, tensor_to_voigt,
    miscut_rotation, C11, C12, C44
)
from solver.nv_coupling import (
    build_nv_axes, build_coupling_matrices,
    dg_from_stress_voigt, ALPHA, DELTA, BETA
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name: str, condition: bool, info: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  ({info})" if info else ""))
    return condition


def test_cubic_voigt_roundtrip():
    """tensor_to_voigt(voigt_to_tensor(C)) == C"""
    print("Test 1: Voigt round-trip")
    C = C_cubic_voigt()
    C4 = voigt_to_tensor(C)
    C2 = tensor_to_voigt(C4)
    ok = check("round-trip matches", np.allclose(C, C2, atol=1e-8),
               f"max diff = {np.max(np.abs(C - C2)):.2e}")
    # Spot-check known values
    ok &= check("C[0,0] == C11", abs(C[0, 0] - C11) < 1, f"{C[0,0]}")
    ok &= check("C[3,3] == C44", abs(C[3, 3] - C44) < 1, f"{C[3,3]}")
    ok &= check("C[0,1] == C12", abs(C[0, 1] - C12) < 1, f"{C[0,1]}")
    # Full 4th-order tensor symmetry: C[i,j,k,l] = C[k,l,i,j] = C[j,i,k,l]
    sym1 = np.allclose(C4, C4.transpose(2, 3, 0, 1), atol=1e-8)
    sym2 = np.allclose(C4, C4.transpose(1, 0, 2, 3), atol=1e-8)
    ok &= check("C4 minor symmetry (ij)", sym2)
    ok &= check("C4 major symmetry (ijkl = klij)", sym1)
    return ok


def test_rotation_orthogonality():
    """Miscut rotation matrix must be a proper rotation."""
    print("Test 2: Rotation matrix orthogonality")
    R = miscut_rotation()
    RtR = R.T @ R
    ok  = check("R^T R = I", np.allclose(RtR, np.eye(3), atol=1e-12),
                f"max off-diag = {np.max(np.abs(RtR - np.eye(3))):.2e}")
    det = np.linalg.det(R)
    ok &= check("det(R) = 1", abs(det - 1.0) < 1e-12, f"det = {det:.10f}")
    return ok


def test_C_lab_properties():
    """C_lab must be symmetric, positive definite, and differ from C_cubic."""
    print("Test 3: C_lab symmetry and positive definiteness")
    C4_lab, C_v_lab, R = build_C_lab()
    ok = check("C_voigt_lab symmetric", np.allclose(C_v_lab, C_v_lab.T, atol=1e-6),
               f"max diff = {np.max(np.abs(C_v_lab - C_v_lab.T)):.2e}")
    eigvals = np.linalg.eigvalsh(C_v_lab)
    ok &= check("C_voigt_lab positive definite", np.all(eigvals > 0),
                f"min eigval = {eigvals.min():.1f} GPa")
    # C_lab must differ from C_cubic (because (111)-cut ≠ cubic axes)
    C_cubic = C_cubic_voigt()
    diff = np.max(np.abs(C_v_lab - C_cubic))
    ok &= check("C_lab differs from C_cubic", diff > 1.0,
                f"max diff = {diff:.1f} GPa")
    print(f"    C_voigt_lab diagonal: {np.round(np.diag(C_v_lab), 1)}")
    return ok


def test_nv_axes():
    """NV axes must be unit vectors."""
    print("Test 4: NV axis unit vectors")
    nvs = build_nv_axes()
    norms = np.linalg.norm(nvs, axis=1)
    ok = check("all NV axes are unit vectors",
               np.allclose(norms, 1.0, atol=1e-12),
               f"norms = {np.round(norms, 8)}")
    print(f"    NV axes (lab frame):")
    for g, n in enumerate(nvs):
        print(f"      NV{g}: [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]")
    return ok


def test_isotropic_compression():
    """
    Under isotropic compression sigma = P * I (P > 0 = tension in standard convention):
      D_g = alpha * tr(sigma) + Delta * (n_g @ n_g) * P  ... wait
      D_g = M_g : sigma = (alpha*I + Delta*n_g n_g) : (P*I)
           = alpha * 3P + Delta * |n_g|^2 * P
           = (3*alpha + Delta) * P
           = (3*ALPHA + DELTA) * P
           = BETA + 2*ALPHA * P  ... let me verify numerically

    But in our physics, compression means sigma = -P*I (negative because
    compressive is negative in solid mechanics sign convention).
    NV data shows D > 0 under compression, and the SPEC says we drop the
    leading minus sign so compressive stress gives positive D.
    So we test with sigma = +P*I to represent compression.
    """
    print("Test 5: D_g under isotropic compression")
    _, M_list = build_coupling_matrices()
    P = 1.0  # GPa
    sigma_v = np.array([P, P, P, 0.0, 0.0, 0.0])  # isotropic, Voigt

    # Analytic: D_g = sum_ij M_g[i,j] * P * delta_ij = P * tr(M_g)
    #          = P * (3*alpha + Delta * |n_g|^2)
    #          = P * (3*ALPHA + DELTA)
    expected = P * (3 * ALPHA + DELTA)
    print(f"    Expected D_g = {expected:.5f} GHz  (all 4 orientations equal)")
    print(f"    = P * (3*alpha + Delta) = {P:.1f} * ({3*ALPHA:.5f} + {DELTA:.5f})")

    ok = True
    for g, M in enumerate(M_list):
        D = dg_from_stress_voigt(sigma_v, M)
        match = abs(D - expected) < 1e-10
        ok &= check(f"D_{g} = {D:.5f} GHz", match,
                    f"expected {expected:.5f}, diff={abs(D-expected):.2e}")
    return ok


def test_coupling_constants():
    """Verify the derived coupling constants from SPEC.md."""
    print("Test 6: Coupling constants")
    A1, A2 = 0.00486, -0.0037
    alpha = A1 - A2
    beta  = A1 + 2 * A2
    delta = beta - alpha

    ok  = check("ALPHA = a1 - a2", abs(ALPHA - alpha) < 1e-10,
                f"{ALPHA:.5f} vs {alpha:.5f}")
    ok &= check("BETA  = a1 + 2*a2", abs(BETA - beta) < 1e-10,
                f"{BETA:.5f} vs {beta:.5f}")
    ok &= check("DELTA = beta - alpha", abs(DELTA - delta) < 1e-10,
                f"{DELTA:.5f} vs {delta:.5f}")
    return ok


def main():
    print("=" * 60)
    print("NV-diamond stress inversion — physics unit tests")
    print("=" * 60)

    tests = [
        test_cubic_voigt_roundtrip,
        test_rotation_orthogonality,
        test_C_lab_properties,
        test_nv_axes,
        test_isotropic_compression,
        test_coupling_constants,
    ]

    results = []
    for t in tests:
        print()
        results.append(t())

    print()
    print("=" * 60)
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"Results: {n_pass}/{len(results)} passed"
          + (f"  ({n_fail} FAILED)" if n_fail else "  — all OK"))
    print("=" * 60)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
