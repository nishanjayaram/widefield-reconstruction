# Session Log 4 — NV-Diamond Stress Inversion Pipeline

## Date
2026-03-08 (continued from Session 3)

---

## What Was Done This Session

### 1. Two-zone (non-uniform) traction grid — implemented

**Problem:** 10µm uniform Cartesian traction grid limits stress resolution to ~20µm.
NV data pixels are at 1µm spacing; traction grid should match.

**Fix:** Non-uniform two-zone grid in `NVOperator.__init__`:
- Fine spacing `sample_grid_spacing` inside sample ellipse (ELL_A=72, ELL_B=50)
- Coarse spacing `traction_grid_spacing` in gasket annulus
- No overlap: gasket grid points filtered to (x/ELL_A)²+(y/ELL_B)² > 1

New parameter: `sample_grid_spacing: float = 0.0`

`invert.py` CLI updated: `python -m solver.invert real mesh data lam s_gasket s_sample`
Default: s_gasket=10.0, s_sample=3.0.

---

### 2. Medium mesh (3µm traction grid) — first run

**Settings:** `anvil_medium.msh`, s_sample=3µm, s_gasket=10µm, lambda=1e-6

Grid: 1,252 sample pts + 1,848 gasket pts = 3,100 total → N_B=9,300
Oversampling = 46,372 / 9,300 = 5.0× (sample-only: 12.0×)

**Results:**
| Metric | 10µm grid | 3µm grid |
|--------|-----------|----------|
| Residual | 4.09 GHz | 2.81 GHz |
| NV1-NV2 r | 0.787 | 0.832 |
| NV1-NV3 r | 0.770 | 0.801 |
| NV2-NV3 r | 0.845 | 0.873 |
| σ_zz range | ~14–27 GPa | 2.5–38.1 GPa |

Residual 2.81 GHz < 4.3 GHz noise floor → slightly overfitting.

---

### 3. Resolution analysis

97% of pixels are inside sample ellipse. All pixels at 1µm spacing.
- 3µm traction grid: 9 pixels per traction node → washes out features
- Bottleneck: medium mesh LC_SAMPLE=3µm → stress cannot resolve features < 3µm
  (P2 interpolation between 3µm nodes; evaluating at 1µm pixels gives no new info)

Maximum useful traction density (for 3× oversampling): N_B ≤ 46372/3 ≈ 15000 nodes.
→ Maximum in-sample spacing: sqrt(11310/5000) ≈ 1.5µm

To go finer: need finer mesh. Rebuilt medium mesh with LC_SAMPLE=1.5µm.

---

### 4. Rebuilt medium mesh — `anvil_medium2.msh`

Built with: LC_SAMPLE=1.5, LC_GASKET=5.0, LC_TABLE=500, TCAP=5.0
Result: 57,707 nodes (same total), but culet_sample nodes: 2,758 → **3,188** at **1.81µm spacing**.

Command:
```bash
conda run -n dac-recon python mesh/build_mesh.py \
  -o mesh/anvil_medium2.msh --lc-sample 1.5 --lc-gasket 5.0 --lc-table 500 --tcap 5.0
```

---

### 5. Medium2 mesh (2µm traction grid) — best result so far

**Settings:** `anvil_medium2.msh`, s_sample=2µm, s_gasket=10µm, lambda=1e-6

Grid: 2,815 sample pts + 1,848 gasket pts = 4,663 total → N_B=13,989
Oversampling = 46,372 / 13,989 = 3.3× (sample nodes/grid pt = 1.13)
LSQR: 195 iterations in 134.3s

**Results:**
| Metric | 3µm (medium) | **2µm (medium2)** | Data |
|--------|-------------|-------------------|------|
| Residual | 2.81 GHz | **2.52 GHz** | ~4.3 GHz |
| NV1-NV2 r | 0.832 | **0.862** | 0.874 |
| NV1-NV3 r | 0.801 | **0.826** | 0.822 |
| NV2-NV3 r | 0.873 | **0.885** | 0.884 |
| NV2 r | 0.871 | **0.900** | — |
| NV3 r | 0.877 | **0.900** | — |

NV1-NV3 and NV2-NV3 model r now essentially equals data r.
NV2 and NV3 fit at r=0.900.

**NV0 outlier:** NV0 inter-channel model r = 0.52–0.57 vs data r = 0.77–0.80.
This has been consistent across all mesh/grid choices → likely physical/calibration,
not a resolution issue. NV0 has a different coupling geometry.

**Stress stats:**
- σ_zz = 19.97 ± 2.74 GPa, range [-0.9, 36.1] GPa
- σ_xx = 40.66 ± 3.15 GPa, σ_yy = 37.75 ± 2.33 GPa
- Hydrostatic P = 32.79 GPa ✓
- Off-diagonals: means ~0–1 GPa (expected by symmetry) ✓

---

### 6. Files modified

| File | Changes |
|------|---------|
| `solver/operators.py` | Added `sample_grid_spacing` param; two-zone grid implementation |
| `solver/invert.py` | CLI updated: s_gasket, s_sample args; NVOperator call updated |
| `mesh/anvil_medium2.msh` | New mesh: LC_SAMPLE=1.5µm, 57k nodes, 3188 culet_sample nodes |
| `solver/traction_coeffs.npy` | Updated — medium2 mesh, 2µm grid |
| `solver/sigma_culet.npy` | Updated — stress at 11,593 pixels |
| `solver/D_predicted.npy` | Updated — predicted D_g (4 × 11,593) |
| `solver/plots/*.pdf` | Updated — sigma_zz, tau, stress_maps, Dg_comparison, Dg_scatter |

---

## What Needs To Be Done Next Session

1. **BU SCC fine mesh** (`anvil_fine_tet.msh`, LC_SAMPLE=1µm):
   ```bash
   conda run -n dac-recon python -m solver.invert real mesh/anvil_fine_tet.msh data 1e-6 10.0 1.5
   ```
   Expected: ~5,000 sample traction pts, N_B~20k, ~2.5× oversampling.
   Need 128 GB RAM for MUMPS factor.

2. **Investigate NV0:** Why does NV0 consistently under-correlate with NV1-NV3?
   - Check coupling matrix for NV0 vs others
   - Check if NV0 data has systematic offset or different noise level
   - Is there a sign error in the NV0 coupling matrix?

3. **L-curve check:** Current residual (2.52 GHz) is below noise floor (4.3 GHz).
   Try lambda=3e-6 or 1e-5 to avoid overfitting — may slightly reduce NV r but
   will give more physically stable stress field.

4. **Visualize sigma_zz spatial structure:** Does it show the expected Gaussian-like
   pressure distribution peaked at culet centre?

---

## Environment

```bash
conda activate dac-recon
cd ~/widefield-reconstruction
```
