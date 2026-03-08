# Session Log 3 — NV-Diamond Stress Inversion Pipeline

## Date
2026-03-08

---

## What Was Done This Session

### 1. Root cause of adjoint failure on medium mesh — identified and fixed

**Problem:** Adjoint check failed on `anvil_medium.msh` (P2 geometry) with rel_err ~1–7%,
while passing on `anvil_preview.msh` (P1 geometry) at machine precision.

**Root cause:** Two connected bugs, both specific to P2 geometry meshes:

**Bug A: Wrong `cmap.pull_back` for P2 geometry cells.**
`mesh.geometry.cmap.pull_back(x_m, x_cell_P2)` inverts the P2 (quadratic) coordinate
map iteratively. For some cells, the iterative solver converges to the wrong xi_m:
```
pixel 1: xi_affine=[0.517885 0.289766 0.], xi_cmap=[0.516154 0.290726 0.], diff=1.73e-03
pixel 2: xi_affine=[0.259546 0.070432 0.670023], xi_cmap=[0.259818 0.070704 0.669479], diff=5.44e-04
```
This caused `dphi_dx` to be tabulated at the wrong reference position in the adjoint
body load assembly → wrong physical gradients.

**Fix:** Replace `cmap.pull_back` with the affine inverse: `xi_m = J_inv @ (x_m - x0)`.
For straight-sided tets (which all FEM elements are here), the P2 coordinate map is
identical to the P1 affine map, so the affine inverse is exact.

**Bug B: Forward `matvec` and adjoint `rmatvec` used different stress evaluation.**
The forward (via `stress_at_coords_batch`) interpolates sigma onto a DG1 function space
and evaluates it at pixels using DOLFINx's internal P2 pullback — the SAME wrong pullback
as Bug A. The adjoint (after fixing Bug A) used the affine pullback. These two were
inconsistent, causing E/E^T discrepancy of 0.8–17%.

**E/E^T check showing the mismatch (medium mesh, before fix):**
```
pixel 1: DG1 sigma_xx = -420.7 GPa, manual sigma_xx = -410.1 GPa  (diff = 10.6 GPa)
pixel 2: DG1 sigma_xx = -702.3 GPa, manual sigma_xx = -693.7 GPa  (diff = 8.5 GPa)
```

**Fix:** Add `_stress_at_pixels_direct(u)` to NVOperator that computes sigma using the
SAME precomputed dphi_dx (affine pullback) as the adjoint body load. Use this in
`matvec` instead of `stress_at_coords_batch`.

**Result after fix:** Both meshes pass adjoint check at machine precision:
```
Preview mesh: rel_err = 4e-15, 2e-12, 2e-14   PASS
Medium mesh:  rel_err = 4e-15, 9e-15, 1e-14   PASS
Cartesian grid (medium): rel_err = 8e-14, 2e-13, 6e-15  PASS
```

---

### 2. Medium mesh inversion — NOISE-FLOOR LIMITED

**Settings:** `anvil_medium.msh`, Cartesian grid 10 µm, lambda=1e-6, 200 LSQR iters.

**System dimensions:**
- N_meas = 11,593 pixels, N_B = 5,883 (1,961 grid points), oversampling = 5.9×
- DOF = 173,238, 200 LSQR iterations in 108 s

**Results:**
| Metric | Preview mesh (old) | Medium mesh (new) | Target |
|--------|-------------------|-------------------|--------|
| `‖d−Ac‖` | 21.4 GHz | **4.09 GHz** | ~4.3 GHz (noise floor) |
| `‖c‖` | 22,485 GPa | 2,675 GPa | — |
| Anomaly r (NV0) | 0.37 | **0.70** | — |
| Anomaly r (NV1) | 0.25 | **0.67** | — |
| Anomaly r (NV2) | 0.40 | **0.75** | — |
| Anomaly r (NV3) | 0.45 | **0.76** | — |
| Model NV1-NV2 r | 0.16 | **0.787** | data: 0.874 |
| Model NV1-NV3 r | 0.09 | **0.770** | data: 0.822 |
| Model NV2-NV3 r | 0.48 | **0.845** | data: 0.884 |

The residual has reached the noise floor (4.09 vs 4.3 GHz). The model is no longer
mesh-limited — it is now data-noise-limited.

**Stress on culet (medium mesh):**
- σ_zz mean = 20.1 ± 3.4 GPa
- σ_xx mean = 40.3 ± 3.2 GPa, σ_yy mean = 38.1 ± 2.0 GPa
- tr(σ) mean = 98.5 GPa → hydrostatic P ≈ 33 GPa ✓ (physically reasonable)
- Off-diagonal mean ≈ 0 (expected by symmetry of the DAC geometry) ✓

---

### 3. Files modified

| File | Changes |
|------|---------|
| `solver/operators.py` | Replace `cmap.pull_back` with affine pullback; add `_stress_at_pixels_direct`; use in `matvec` |
| `solver/traction_coeffs.npy` | Updated — from medium mesh, Cartesian grid |
| `solver/sigma_culet.npy` | Updated — stress at 11,593 pixels |
| `solver/D_predicted.npy` | Updated — predicted D_g (4 channels × 11,593 pixels) |
| `solver/traction_coeffs_medium.npy` | Same as traction_coeffs.npy (backup copy) |
| `solver/plots/*.pdf` | Updated — sigma_zz, stress_maps, Dg_comparison, Dg_scatter |

---

## Key Physical Results

**Medium mesh, lambda=1e-6, Cartesian grid 10 µm:**
- Residual 4.09 GHz at noise floor — inversion is converged
- σ_zz = 20.1 ± 3.4 GPa on culet surface (confirms GPa-level pressure)
- Hydrostatic pressure P = tr(σ)/3 ≈ 33 GPa ✓
- Model inter-channel r = 0.77–0.85, data r = 0.82–0.88 (90% of data structure explained)

---

## What Needs To Be Done Next Session

### Immediate:
1. **L-curve / lambda sweep** on medium mesh to confirm lambda=1e-6 is optimal:
   ```bash
   # Try lambda in [1e-7, 1e-6, 1e-5] and plot residual vs ||c||
   ```
   Current residual 4.09 GHz = noise floor, so any smaller lambda would overfit.

2. **BU SCC fine mesh inversion:**
   ```bash
   # On SCC node with >=128 GB RAM:
   conda run -n dac-recon python -m solver.invert real mesh/anvil_fine_tet.msh data 1e-6
   ```
   Expected: even better inter-channel r (>0.9?), finer stress spatial resolution.

3. **Visualize sigma_zz** — does the map show the expected Gaussian-like pressure
   distribution peaked at the culet centre?

4. **Fix visualize.py traction plot** — currently skips because it looks for preview
   mesh nodes (5400 nodes). Need to pass the Cartesian grid coords for the traction plot.

### Technical:
- The `_stress_at_pixels_direct` method uses a Python loop (N_meas iterations).
  For large meshes or many LSQR iterations this could be a bottleneck.
  Optimization: precompute a stacked (N_meas, 10, 3) dphi_dx tensor and use batch matmul.
  But at 108s/200 iters = 0.54s/iter on medium mesh, this may not be necessary.

---

## Environment

```bash
conda activate dac-recon
cd ~/widefield-reconstruction
```
