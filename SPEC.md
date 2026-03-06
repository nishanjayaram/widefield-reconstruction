# NV-Diamond Stress Inversion: Specification

## Overview

This project reconstructs the full Cauchy stress tensor inside a diamond anvil cell (DAC)
from nitrogen-vacancy (NV) center spectroscopy data measured on the culet surface.
NV centers embedded in the diamond culet act as local stress sensors: the zero-field
splitting shift $\Delta D_g$ of each of the 4 NV orientations is a known linear functional
of the stress tensor. Given spatial maps of $\Delta D_g$ over the sample chamber region of
the culet, we solve an inverse problem to determine the traction boundary condition on
the culet, and thereby the full 3D stress field in the diamond.

**Key principle:** We do NOT add penalty terms to the elastic energy. Instead, we build
a basis of elastostatic solutions (each satisfying equilibrium, constitutive law, and all
known BCs exactly), then find the linear combination that best fits the NV data via
regularized least squares.

---

## 1. Geometry

The diamond anvil is a frustum (truncated cone) with:

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Culet radius | `r1` | 0.25 mm = 250 µm |
| Table radius | `r2` | 1.75 mm = 1750 µm |
| Sample chamber semi-major axis | `ell_a` | 72 µm |
| Sample chamber semi-minor axis | `ell_b` | 50 µm |
| Fine mesh cap thickness | `tcap` | 1 µm |

**Facet profile:** Straight cone (linear interpolation between culet and table radii).

**Height:** Not explicitly given in COMSOL params. Must be determined from the cone angle
or set as a parameter. Typical Boehler-Almax anvils have a half-angle of ~8–10°. The
height can be computed as:
```
h = (r2 - r1) / tan(half_angle)
```
**TODO:** Confirm the anvil height or half-angle with the user.

**Sample chamber:** An ellipse centered on the culet face with semi-axes `ell_a` × `ell_b`.
The culet face is the smaller circular face (radius `r1`). The sample chamber is the
interior of the ellipse; the gasket region is the annulus between the ellipse and the
culet edge (circle of radius `r1`).

### Geometry construction in Gmsh

1. Build the frustum as a cone (straight sidewall) from culet to table.
2. The culet face (z = 0 plane) is a disk of radius `r1`.
3. Mark the elliptical sample chamber boundary on the culet.
4. The table face is at z = -h (or z = +h depending on convention). We choose
   **z = 0 at the culet, z < 0 going into the diamond toward the table**.

### Meshing strategy

- **Culet, sample chamber region:** Fine mesh, element size ≤ 1 µm (matching the
  1 µm pixel resolution of the NV data).
- **Culet, gasket annulus:** Slightly coarser, ~2–5 µm.
- **Fine cap layer (0 ≤ z ≤ tcap = 1 µm below culet):** Fine mesh matching the
  culet surface mesh, structured or semi-structured.
- **Bulk (z > tcap toward table):** Progressively coarser mesh. The table region
  can be very coarse since we only need displacement = 0 there.

---

## 2. Crystal Orientation and Miscut

The diamond is (111)-cut with a small miscut. The lab frame has:
- **z-axis:** Normal to the culet surface (pointing out of the diamond, toward the sample)
- **x, y axes:** In the culet plane

The crystal [100], [010], [001] axes are related to the lab frame by a rotation
that accounts for the (111) orientation plus the miscut.

### Miscut parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Miscut polar angle | `theta_misc` | −3.5° |
| Miscut azimuthal angle (raw) | `phi_CCD` | 180° |
| Effective miscut azimuthal | `phi_misc` | `phi_CCD + (-5.6°)` = 174.4° |

### Rotation matrix: crystal frame → lab frame

COMSOL defines three orthonormal vectors (x1, x2, x3) that express the lab axes
in terms of the crystal axes. Let `ct = cos(theta_misc)`, `st = sin(theta_misc)`,
`cp = cos(phi_misc)`, `sp = sin(phi_misc)`. Then:

```
x1 = ( ct*sqrt(2/3)*cp - st*sqrt(1/3),
        ct*(-cp*sqrt(1/6) + sp*sqrt(1/2)) - st*sqrt(1/3),
        ct*(-cp*sqrt(1/6) - sp*sqrt(1/2)) - st*sqrt(1/3) )

x2 = ( -sqrt(2/3)*sp,
         sp*sqrt(1/6) + cp*sqrt(1/2),
         sp*sqrt(1/6) - cp*sqrt(1/2) )

x3 = ( st*sqrt(2/3)*cp + ct*sqrt(1/3),
        st*(-cp*sqrt(1/6) + sp*sqrt(1/2)) + ct*sqrt(1/3),
        st*(-cp*sqrt(1/6) - sp*sqrt(1/2)) + ct*sqrt(1/3) )
```

The rotation matrix R from crystal to lab frame is:
```
R = [x1 | x2 | x3]^T
```
i.e., row i of R is xi. If v_crystal is a vector in crystal coordinates, then
v_lab = R @ v_crystal.

For the elasticity tensor rotation, we need this R to transform the 4th-order
stiffness tensor from the cubic frame to the lab frame (see Section 3).

---

## 3. Elasticity

Diamond has cubic symmetry. The elastic constants in the **cubic crystal frame**
(Voigt notation, standard ordering 11, 22, 33, 12, 23, 13):

```
C_cubic = | 1079   125   125     0     0     0  |   (GPa)
          |  125  1079   125     0     0     0  |
          |  125   125  1079     0     0     0  |
          |    0     0     0   578     0     0  |
          |    0     0     0     0   578     0  |
          |    0     0     0     0     0   578  |
```

Note: C11 = 1079 GPa, C12 = 125 GPa, C44 = 578 GPa.

### Rotation to lab frame

The Voigt matrix above is in the cubic crystal frame. To use it in FEniCSx (which
works in the lab frame), we must rotate the full 4th-order tensor:

```
C_lab_{ijkl} = R_{ip} R_{jq} R_{kr} R_{ls} C_crystal_{pqrs}
```

where R is the rotation matrix from Section 2.

**Implementation:** Convert the 6×6 Voigt matrix to a 3×3×3×3 tensor, apply the
rotation, then convert back to 6×6 (or use the tensor form directly in the
variational formulation).

### Voigt conventions

COMSOL uses the standard ordering: (11, 22, 33, 12, 23, 13) with engineering
shear strain (factor of 2). FEniCSx uses symmetric tensors directly, so be
careful with the 2× factors when converting between Voigt and tensor notation.

---

## 4. NV Stress Coupling

### NV axis orientations

The 4 NV centers point along the [111]-family directions of the diamond crystal.
In the lab frame (accounting for miscut), the unit vectors along the 4 NV axes are:

**NV0** (pointing approximately along the culet normal):
```
n0 = (-st, 0, ct)
```

**NV1** (no y-projection in the zero-miscut limit):
```
n1 = ( (2√2/3)*ct*cp + st/3,
       -(2√2/3)*sp,
        -(1/3)*ct + (2√2/3)*st*cp )
```

**NV2:**
```
q2 = -(√2*cp + √6*sp)/3
n2 = ( ct*q2 + st/3,
        (√2*sp - √6*cp)/3,
       -ct/3 + st*q2 )
```

**NV3:**
```
q3 = (-√2*cp + √6*sp)/3
n3 = ( ct*q3 + st/3,
        (√2*sp + √6*cp)/3,
       -ct/3 + st*q3 )
```

### Coupling constants

| Parameter | Value |
|-----------|-------|
| `a1` | 0.00486 GHz/GPa |
| `a2` | −0.0037 GHz/GPa |
| `alpha = a1 - a2` | 0.00856 GHz/GPa |
| `Delta = (a1 + 2*a2) - (a1 - a2) = 3*a2` | −0.0111 GHz/GPa |

Wait — let me recompute. From the COMSOL variables:
```
alpha = a1 - a2 = 0.00486 - (-0.0037) = 0.00856 GHz/GPa
beta  = a1 + 2*a2 = 0.00486 + 2*(-0.0037) = -0.00254 GHz/GPa
Delta = beta - alpha = -0.00254 - 0.00856 = -0.01110 GHz/GPa
```

### D_g formula

The NV zero-field splitting shift for orientation g, with the sign convention
that **positive ΔD corresponds to positive (compressive) stress**:

```
D_g(x) = alpha * tr(sigma) + Delta * (n_g ⊗ n_g) : sigma
```

Expanded:
```
D_g = alpha * (σ_xx + σ_yy + σ_zz)
    + Delta * (ng_x² σ_xx + ng_y² σ_yy + ng_z² σ_zz
               + 2 ng_x ng_y σ_xy + 2 ng_x ng_z σ_xz + 2 ng_y ng_z σ_yz)
```

where sigma is the Cauchy stress in the lab frame (GPa), and D_g is in GHz.

**Note on signs:** The original COMSOL had a leading negative sign. We drop it so
that compressive stress (positive sigma in our convention) gives positive Delta D.
The measured data (`nv*_roi_centered_minusZFS.txt`) has values ~0.3–0.7 GHz, all
positive, consistent with compression.

### Tensor form of the coupling

For each NV orientation g, define the 3×3 symmetric coupling tensor:
```
M_g^{ij} = alpha * delta_{ij} + Delta * ng_i * ng_j
```

Then: `D_g = M_g^{ij} sigma_{ij}` (Einstein summation).

---

## 5. Boundary Conditions

| Surface | Condition |
|---------|-----------|
| **Table** (large face, z = −h) | Clamped: u = 0 (Dirichlet) |
| **Facets** (conical sidewall) | Traction-free: σ·n = 0 (Neumann) |
| **Culet, gasket annulus** | Unknown — to be determined by inversion |
| **Culet, sample chamber** | Unknown — to be determined by inversion, constrained by NV data |

The entire culet traction is the unknown. We parameterize it with basis functions
and let the NV data (plus regularization) determine the coefficients.

---

## 6. Algorithm: Basis Function Inversion

### Step 1: Build traction basis functions

On the culet surface (z = 0), identify all mesh nodes. At each node i, define
3 basis tractions (one per component: normal and two shear):

```
t^{(i,alpha)}(x) = delta(x - x_i) * e_alpha    (alpha = x, y, z)
```

In practice, with finite elements, this is a unit traction concentrated at node i
in direction alpha (implemented as a point load or a narrow Gaussian if needed).

For each basis traction, solve the forward elasticity problem:
- PDE: div(sigma) = 0 in the diamond
- Constitutive: sigma = C_lab : epsilon(u)
- BCs: u = 0 on table, t = 0 on facets, t = t^{(i,alpha)} on culet

This gives basis displacement fields u^{(k)}(x) and basis stress fields
sigma^{(k)}(x), where k indexes the (node, direction) pair.

**Optimization:** The stiffness matrix K is the same for all basis functions.
Factor K once (sparse LU via MUMPS), then do N_B back-substitutions with
different RHS vectors. Each RHS differs only in the surface load vector.

### Step 2: Build the influence matrix A

For each basis function k and each NV measurement point (x_m, y_m) on the
culet surface, compute:

```
A_{(g,m), k} = M_g^{ij} * sigma_{ij}^{(k)}(x_m, y_m, z=0)
```

This gives a matrix A of shape (4 * N_meas) × N_B, where N_meas = 11593
measurement points and N_B = 3 * N_culet_nodes.

### Step 3: Regularized least squares

The data vector d has entries d_{(g,m)} = Delta_D_g(x_m) from the NV measurements.
Solve:

```
min_c || d - A c ||^2 + lambda * || L c ||^2
```

where:
- c is the vector of N_B traction coefficients
- L is a regularization operator (surface Laplacian for smoothness, or identity)
- lambda is the regularization parameter (chosen by L-curve or GCV)

The solution is:
```
c = (A^T A + lambda L^T L)^{-1} A^T d
```

### Step 4: Reconstruct the full stress field

The full traction on the culet is:
```
t(x) = sum_k c_k * t^{(k)}(x)
```

The full stress field is:
```
sigma(x) = sum_k c_k * sigma^{(k)}(x)
```

Evaluate this at any point in the diamond bulk.

---

## 7. Data Format

### NV data files

Four files: `nv0_roi_centered_minusZFS.txt` through `nv3_roi_centered_minusZFS.txt`

Format: CSV with header `x_um_centered_to_ellipse,y_um_centered_to_ellipse,delta_f_GHz`

| Property | Value |
|----------|-------|
| Number of points | 11593 per NV orientation |
| Grid | 142 × 104 (not all pixels filled — elliptical mask) |
| Pixel spacing | 1.0 µm |
| x range | [−70.7, 70.3] µm |
| y range | [−52.0, 51.0] µm |
| D range (NV0) | [0.584, 0.667] GHz |
| D range (NV1) | [0.335, 0.476] GHz |
| D range (NV2) | [0.389, 0.527] GHz |
| D range (NV3) | [0.398, 0.535] GHz |

Coordinates are centered on the elliptical sample chamber, in the culet plane (µm).
Values are ΔD = D − 2.87 GHz (shift from zero-field splitting), in GHz.

---

## 8. Implementation Stack

- **Meshing:** Gmsh (Python API) — build frustum geometry, mark boundaries,
  generate tetrahedral mesh with refinement zones
- **FEM forward solves:** FEniCSx (DOLFINx) — assemble stiffness matrix, solve
  with MUMPS, extract surface stresses
- **Inversion:** NumPy/SciPy — build influence matrix, regularized least squares
- **Visualization:** PyVista or Matplotlib for stress maps

### Key implementation notes

1. **Element order:** Use P2 (quadratic) tetrahedra. This gives piecewise-linear
   stress, which is adequate for the 1 µm data resolution.

2. **Stiffness matrix reuse:** Assemble K once. For each basis traction, only the
   RHS load vector changes. Use a direct solver (MUMPS) and store the factorization.

3. **Basis function implementation:** Rather than literal delta-function loads,
   apply a unit traction as a surface load on the FEM test function at each node.
   This is equivalent to setting one entry of the surface load vector to 1.

4. **Stress extraction on the culet:** After each forward solve, evaluate the
   stress tensor at the NV measurement points (which lie on the culet surface).
   Use FEniCSx interpolation or projection to get stress at specific coordinates.

5. **Scaling:** With ~250 µm culet radius and ~1 µm element size on the culet,
   expect O(1000) culet surface nodes → N_B ~ 3000. The influence matrix A is
   (4 × 11593) × 3000 ≈ 46000 × 3000 — easily fits in memory. The 3000 forward
   solves are the bottleneck but are embarrassingly parallel (same K, different RHS).

---

## 9. Open Questions / TODO

- [ ] **Anvil height:** Need to confirm the height of the frustum (or equivalently
  the pavilion half-angle). Not in the COMSOL params file.
- [ ] **Sign convention validation:** Run a simple test case (uniform pressure on
  sample chamber) and verify that D_g comes out positive with our sign convention.
- [ ] **Regularization operator L:** Start with identity (Tikhonov), then try
  surface Laplacian if results are too rough.
- [ ] **Gasket treatment:** Initially parameterize traction on the full culet
  (sample + gasket). If too many DOFs, can restrict to sample chamber only and
  set gasket to traction-free as an approximation.
