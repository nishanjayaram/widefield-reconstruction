"""
Crystal orientation and stiffness tensor rotation for (111)-cut diamond.

Rotates C_cubic (cubic crystal frame, Voigt) to C_lab (lab frame, full tensor)
using the miscut rotation matrix R defined in SPEC.md Sections 2-3.

Units: GPa throughout.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Diamond elastic constants (GPa, cubic crystal frame)
# ---------------------------------------------------------------------------
C11 = 1079.0
C12 =  125.0
C44 =  578.0

# Voigt index mapping: I -> (i, j)  (0-indexed)
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


def C_cubic_voigt() -> np.ndarray:
    """Return 6×6 Voigt stiffness matrix for cubic diamond (GPa)."""
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    for I in range(3):
        for J in range(3):
            if I != J:
                C[I, J] = C12
    return C


def voigt_to_tensor(C_voigt: np.ndarray) -> np.ndarray:
    """
    Convert 6×6 Voigt stiffness to full 3×3×3×3 tensor (physical shear,
    no factor-of-2 in strain).

    The Voigt matrix relates stress to *engineering* shear strain; the tensor
    C4[i,j,k,l] relates stress to *physical* strain eps[k,l].
    """
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT):
        for J, (k, l) in enumerate(VOIGT):
            val = C_voigt[I, J]
            for ii, jj in [(i, j), (j, i)]:
                for kk, ll in [(k, l), (l, k)]:
                    C4[ii, jj, kk, ll] = val
    return C4


def tensor_to_voigt(C4: np.ndarray) -> np.ndarray:
    """Convert full 3×3×3×3 tensor back to 6×6 Voigt matrix."""
    C = np.zeros((6, 6))
    for I, (i, j) in enumerate(VOIGT):
        for J, (k, l) in enumerate(VOIGT):
            C[I, J] = C4[i, j, k, l]
    return C


def miscut_rotation(theta_misc_deg: float = -3.5,
                    phi_misc_deg: float = 174.4) -> np.ndarray:
    """
    Build rotation matrix R that maps crystal-frame vectors to lab-frame vectors.

    Row i of R is the i-th axis of the lab frame expressed in crystal coordinates,
    so:  v_lab = R @ v_crystal.

    Formulae from SPEC.md Section 2:
      phi_misc = phi_CCD + correction = 180° + (-5.6°) = 174.4°
      theta_misc = -3.5°

    Parameters
    ----------
    theta_misc_deg : miscut polar angle (degrees)
    phi_misc_deg   : effective miscut azimuthal angle (degrees)

    Returns
    -------
    R : (3, 3) ndarray  — rotation matrix, rows = x1, x2, x3
    """
    theta = np.radians(theta_misc_deg)
    phi   = np.radians(phi_misc_deg)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi),   np.sin(phi)

    x1 = np.array([
        ct * np.sqrt(2 / 3) * cp - st * np.sqrt(1 / 3),
        ct * (-cp * np.sqrt(1 / 6) + sp * np.sqrt(1 / 2)) - st * np.sqrt(1 / 3),
        ct * (-cp * np.sqrt(1 / 6) - sp * np.sqrt(1 / 2)) - st * np.sqrt(1 / 3),
    ])
    x2 = np.array([
        -np.sqrt(2 / 3) * sp,
         sp * np.sqrt(1 / 6) + cp * np.sqrt(1 / 2),
         sp * np.sqrt(1 / 6) - cp * np.sqrt(1 / 2),
    ])
    x3 = np.array([
        st * np.sqrt(2 / 3) * cp + ct * np.sqrt(1 / 3),
        st * (-cp * np.sqrt(1 / 6) + sp * np.sqrt(1 / 2)) + ct * np.sqrt(1 / 3),
        st * (-cp * np.sqrt(1 / 6) - sp * np.sqrt(1 / 2)) + ct * np.sqrt(1 / 3),
    ])
    return np.array([x1, x2, x3])


def rotate_stiffness(C_voigt: np.ndarray,
                     R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate stiffness tensor from crystal frame to lab frame.

    C_lab_{ijkl} = R_{ip} R_{jq} R_{kr} R_{ls} C_crystal_{pqrs}

    Parameters
    ----------
    C_voigt : (6, 6) ndarray  — Voigt stiffness in crystal frame (GPa)
    R       : (3, 3) ndarray  — rotation matrix (rows = lab axes in crystal frame)

    Returns
    -------
    C4_lab      : (3, 3, 3, 3) ndarray  — full tensor in lab frame (GPa)
    C_voigt_lab : (6, 6) ndarray        — Voigt form of C4_lab (GPa)
    """
    C4_crystal = voigt_to_tensor(C_voigt)
    C4_lab = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, C4_crystal)
    return C4_lab, tensor_to_voigt(C4_lab)


def build_C_lab(theta_misc_deg: float = -3.5,
                phi_misc_deg: float = 174.4
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline: miscut rotation → rotated stiffness tensor.

    Returns
    -------
    C4_lab      : (3, 3, 3, 3) ndarray  [GPa]
    C_voigt_lab : (6, 6) ndarray        [GPa]
    R           : (3, 3) ndarray        — rotation matrix crystal→lab
    """
    R = miscut_rotation(theta_misc_deg, phi_misc_deg)
    C_voigt = C_cubic_voigt()
    C4_lab, C_voigt_lab = rotate_stiffness(C_voigt, R)
    return C4_lab, C_voigt_lab, R


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    C4_lab, C_voigt_lab, R = build_C_lab()
    print("R (crystal → lab):")
    print(np.round(R, 4))
    print("\nC_voigt_lab (GPa):")
    print(np.round(C_voigt_lab, 2))
    # Check symmetry
    assert np.allclose(C_voigt_lab, C_voigt_lab.T, atol=1e-6), "C not symmetric!"
    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(C_voigt_lab)
    print(f"\nEigenvalues of C_voigt_lab: {np.round(eigvals, 1)}")
    assert np.all(eigvals > 0), "C not positive definite!"
    print("OK — C_lab is symmetric and positive definite.")
