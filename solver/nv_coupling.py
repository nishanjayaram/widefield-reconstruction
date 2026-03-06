"""
NV center stress coupling for (111)-cut diamond with miscut.

Computes:
  - NV axis unit vectors in the lab frame (4 orientations)
  - Coupling matrices M_g^{ij} = alpha*delta_{ij} + Delta*ng_i*ng_j
  - Zero-field splitting shift D_g = M_g : sigma  (GHz from GPa stress)

Formulae from SPEC.md Section 4.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Coupling constants (GHz / GPa)
# ---------------------------------------------------------------------------
A1    =  0.00486          # GHz/GPa
A2    = -0.0037           # GHz/GPa
ALPHA = A1 - A2           #  0.00856  GHz/GPa  (isotropic part)
BETA  = A1 + 2 * A2       # -0.00254  GHz/GPa
DELTA = BETA - ALPHA      # -0.01110  GHz/GPa  (anisotropic part)


def build_nv_axes(theta_misc_deg: float = -3.5,
                  phi_misc_deg: float = 174.4) -> np.ndarray:
    """
    Compute the 4 NV unit vectors in the lab frame.

    Formulae from SPEC.md Section 4 (NV0–NV3 expressions).

    Parameters
    ----------
    theta_misc_deg : miscut polar angle (degrees), default -3.5°
    phi_misc_deg   : effective miscut azimuthal angle (degrees), default 174.4°

    Returns
    -------
    nvs : (4, 3) ndarray — unit vectors along each NV [111]-family direction
          in the lab frame. Rows index NV0–NV3.
    """
    theta = np.radians(theta_misc_deg)
    phi   = np.radians(phi_misc_deg)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi),   np.sin(phi)

    # NV0 — along approximate culet normal
    n0 = np.array([-st, 0.0, ct])

    # NV1
    n1 = np.array([
         (2 * np.sqrt(2) / 3) * ct * cp + st / 3,
        -(2 * np.sqrt(2) / 3) * sp,
        -(1 / 3) * ct + (2 * np.sqrt(2) / 3) * st * cp,
    ])

    # NV2
    q2 = -(np.sqrt(2) * cp + np.sqrt(6) * sp) / 3
    n2 = np.array([
        ct * q2 + st / 3,
        (np.sqrt(2) * sp - np.sqrt(6) * cp) / 3,
        -ct / 3 + st * q2,
    ])

    # NV3
    q3 = (-np.sqrt(2) * cp + np.sqrt(6) * sp) / 3
    n3 = np.array([
        ct * q3 + st / 3,
        (np.sqrt(2) * sp + np.sqrt(6) * cp) / 3,
        -ct / 3 + st * q3,
    ])

    nvs = np.array([n0, n1, n2, n3])
    # Normalise (should already be unit vectors, but be safe against fp errors)
    nvs /= np.linalg.norm(nvs, axis=1, keepdims=True)
    return nvs


def coupling_matrix(n_g: np.ndarray,
                    alpha: float = ALPHA,
                    delta: float = DELTA) -> np.ndarray:
    """
    Build the 3×3 coupling matrix for one NV orientation.

    M_g^{ij} = alpha * delta_{ij} + Delta * ng_i * ng_j

    Parameters
    ----------
    n_g   : (3,) unit vector along NV axis (lab frame)
    alpha : isotropic coupling constant (GHz/GPa)
    delta : anisotropic coupling constant (GHz/GPa)

    Returns
    -------
    M_g : (3, 3) ndarray  [GHz/GPa]
    """
    return alpha * np.eye(3) + delta * np.outer(n_g, n_g)


def build_coupling_matrices(theta_misc_deg: float = -3.5,
                             phi_misc_deg: float = 174.4
                             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Build NV axes and all 4 coupling matrices.

    Returns
    -------
    nvs    : (4, 3) ndarray
    M_list : list of 4 (3, 3) ndarrays  [GHz/GPa]
    """
    nvs = build_nv_axes(theta_misc_deg, phi_misc_deg)
    M_list = [coupling_matrix(nvs[g]) for g in range(4)]
    return nvs, M_list


def dg_from_stress_voigt(sigma_voigt: np.ndarray,
                          M_g: np.ndarray) -> np.ndarray:
    """
    Compute D_g (GHz) from Voigt stress and coupling matrix M_g.

    D_g = M_g^{ij} sigma_{ij}
        = M[0,0]*s11 + M[1,1]*s22 + M[2,2]*s33
          + 2*M[0,1]*s12 + 2*M[1,2]*s23 + 2*M[0,2]*s13

    Parameters
    ----------
    sigma_voigt : (..., 6) ndarray
        Stress in Voigt order [s11, s22, s33, s12, s23, s13] (GPa).
        Leading dimensions are arbitrary (e.g., N measurement points).
    M_g : (3, 3) ndarray
        Coupling matrix for orientation g [GHz/GPa].

    Returns
    -------
    D_g : (...,) ndarray  [GHz]
    """
    m = M_g
    # Voigt contraction: factor of 2 for off-diagonal because sigma is symmetric
    # and we listed only the upper triangle in Voigt (s12, s23, s13).
    coeffs = np.array([
        m[0, 0], m[1, 1], m[2, 2],
        2 * m[0, 1], 2 * m[1, 2], 2 * m[0, 2],
    ])
    return sigma_voigt @ coeffs


def dg_all(sigma_voigt: np.ndarray,
           M_list: list[np.ndarray]) -> np.ndarray:
    """
    Compute D_g for all 4 NV orientations at once.

    Parameters
    ----------
    sigma_voigt : (N, 6) ndarray — stress at N points (GPa)
    M_list      : list of 4 (3, 3) coupling matrices

    Returns
    -------
    D : (4, N) ndarray  [GHz]  — row g is D_g at each measurement point
    """
    return np.array([dg_from_stress_voigt(sigma_voigt, M) for M in M_list])


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    nvs, M_list = build_coupling_matrices()
    print("NV axes (lab frame):")
    for g, n in enumerate(nvs):
        print(f"  NV{g}: {np.round(n, 4)}  |n|={np.linalg.norm(n):.6f}")

    # Test: isotropic pressure sigma = -P * I  (P > 0 = compression in our sign convention)
    P = 1.0  # GPa
    sigma_v = np.array([-P, -P, -P, 0.0, 0.0, 0.0])
    print(f"\nIsotropic pressure P={P} GPa => sigma = -P*I")
    for g, M in enumerate(M_list):
        D = dg_from_stress_voigt(sigma_v, M)
        # Expected: D_g = alpha*tr(sigma) = alpha*(-3P) = -3*ALPHA*P
        # With our sign convention (compressive = positive), the data has +ve D,
        # so in practice we will negate sigma or use sigma = +P*I for compression.
        print(f"  D_{g} = {D:.5f} GHz  (expected {-3*ALPHA*P:.5f} GHz)")
