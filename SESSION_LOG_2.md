# Session Log 2 — NV-Diamond Stress Inversion Pipeline

## Date
2026-03-07

---

## What Was Done This Session

### 1. Adjoint operator — three bugs fixed (from previous session, confirmed working)

All three bugs in `solver/operators.py` were fixed and verified:
- **Bug 1** (off-diagonal S_m factor-of-2 in `_adjoint_body_load`)
- **Bug 2** (P^T read from pixel coord instead of nearest DOF in `rmatvec`)
- **Bug 3** (Jacobian transpose: `J^{-T}` → `J^{-1}` in `_build_pixel_geometry`)

Adjoint check result (preview mesh):
```
trial 1:  <Ac,r>=-2.448775e-05  <c,A^Tr>=-2.448775e-05  rel_err=1.82e-13  OK
trial 2:  <Ac,r>=2.797455e-04   <c,A^Tr>=2.797455e-04   rel_err=4.46e-15  OK
PASS (5/5 trials)
```

---

### 2. Real data inversion — preview mesh, lambda sweep

Ran inversion with `mesh/anvil_preview.msh` (7,579 nodes, 1,800 culet nodes):

| Lambda | Iterations | Residual ||d-Ac|| | ||c|| | Per-pixel RMS |
|--------|-----------|----------------------|------|--------------|
| 1e-3   | 5         | 84.3 GHz (80% rel)   | 1,288 GPa | 0.39 GHz |
| 1e-5   | ~5        | 28.2 GHz             | 7,906 GPa | 0.10 GHz |
| 1e-8   | 200 (max) | 6.86 GHz             | 44,128 GPa | 0.032 GHz |

Data norm: ||d|| = 104.9 GHz; expected noise floor: sqrt(46,372) × 0.02 ≈ 4.3 GHz (per-pixel ~0.02 GHz).

**Spatial quality** (all lambdas gave same anomaly correlation — key finding):
- Anomaly correlation r ≈ 0.37 (NV0), 0.25 (NV1), 0.40 (NV2), 0.45 (NV3)
- Model inter-channel correlations: NV1-NV2 r=0.16, NV1-NV3 r=0.09 vs data r=0.87, 0.82
- Predicted spatial std 3–10× larger than measured std (noise amplification)

**Root cause identified:** Preview mesh has only 1,800 unique culet nodes for 11,593 pixels.
Old parameterization (N_B = 3×11,593 = 34,779) has **84% null space** — 29,000+ directions
in which c can change without affecting D_predicted. This is why all lambdas give the
same spatial pattern.

---

### 3. Node-based parameterization — implemented and verified

**Fix:** Index traction coefficients by unique culet mesh nodes, not pixels.

- Old: N_B = 3 × N_meas = 34,779 (84% null space on preview mesh)
- New: N_B = 3 × N_unique_nodes = 5,400 (oversampling 8.6×)

Key fields added to `NVOperator`:
- `op.N_nodes` — number of unique traction nodes
- `op._culet_node_list` — sorted array of unique culet block DOFs
- `op._node_idx_of_pixel` — maps pixel m → node index k

Adjoint verified exact after change (rel_err ~1e-13 to 1e-16).

**Result on preview mesh (lambda=1e-6, node-based):**
```
LSQR: shape=(51772, 5400), lambda=1e-6, 27 iterations, 150s
residual = 21.4 GHz, ||c|| = 22,485 GPa
anomaly r ≈ 0.37–0.45 (same as before)
```

The correlation did NOT improve — because the problem is the mesh, not the parameterization.
The preview mesh is simply too coarse to resolve the stress field.

---

### 4. Irreducible model error confirmed

The residual plateaus at ~21 GHz regardless of lambda or parameterization, vs expected
noise floor of ~4.3 GHz. This 5× gap is **irreducible model error** from the preview mesh.

Evidence:
- Data inter-channel correlations: r = 0.77–0.88 (all channels see same physical field)
- Model inter-channel correlations: r = 0.09–0.48 (model predicts decorrelated fields)
- Both channels should be dominated by the same isotropic alpha*tr(sigma) term
- Decorrelated predictions = spatially noisy stress from coarse FEM

Per-pixel RMS budget: model gives 0.1 GHz per pixel, noise floor is 0.02 GHz.
Need ~5× improvement in stress resolution → fine mesh.

---

### 5. Medium mesh built

Built `mesh/anvil_medium.msh` using `build_mesh.py` with:
- LC_SAMPLE = 3.0 µm, LC_GASKET = 5.0 µm, TCAP = 5.0 µm, LC_TABLE = 500 µm

Result: 102,765 nodes, 24,999 tet10 elements, DOF = 173,238. File size: 5.1 MB.
- culet_sample nodes: 2,758
- culet_gasket nodes: 39,678 (total culet: 42,436)

CLI args added to `build_mesh.py`: `--lc-sample`, `--lc-gasket`, `--lc-table`, `--tcap`

**Problem:** Medium mesh has P2 geometry (10 nodes/cell vs 4 for P1). Jacobian code
assumed 4 → `LinAlgError` on medium mesh. **Fixed** by using `x_cell[:4]` (vertex nodes only).

---

### 6. Cartesian grid parameterization — implemented, adjoint NOT yet verified

**Problem:** Node-based parameterization degenerates on finer meshes. When mesh has more
culet nodes than NV pixels, every pixel gets a unique nearest node → N_B = N_meas → back
to the null-space problem.

**Solution implemented:** Cartesian grid with fixed spacing (independent of mesh):
- `NVOperator(..., traction_grid_spacing=10.0)` → 1,961 grid points → N_B = 5,883
- Oversampling = 46,372 / 5,883 = 7.9× (good, matches node-based on preview mesh)
- Works for any mesh resolution

Forward P: each culet mesh node assigned traction of nearest grid point.
Adjoint P^T: accumulates v over all culet mesh nodes per grid point.

**STATUS: Adjoint check FAILS (rel_err ~3.9e-3 to 6.9e-2). NOT YET DEBUGGED.**

Verified correct on preview mesh (node-based mode). Medium mesh, Cartesian grid mode
needs adjoint debugging next session.

---

### 7. Visualizer written

`solver/visualize.py` — generates 5 PDF figures in `solver/plots/`:
- `sigma_zz.pdf` — normal stress on culet
- `stress_maps.pdf` — all 6 Voigt components
- `Dg_comparison.pdf` — measured vs predicted per channel + residual maps
- `Dg_scatter.pdf` — scatter plot predicted vs measured
- `traction_field.pdf` — recovered traction components

Run: `conda run -n dac-recon python -m solver.visualize`

---

### 8. Mesh conversion (convert_to_tet.py) — fixed and verified

Rewrote prism→tet conversion to use `removeElements` + `addElementsByType` instead of
broken `SubdivisionAlgorithm + refine()`. Also splits quad4 surface elements → 2 tri3.
`anvil_fine_tet.msh` produced and verified (no invalid node refs, no duplicate tags).
DOLFINx crashes on fine mesh locally due to memory (~40 GB MUMPS factor). Use BU SCC.

---

## Key Physical Results

**From lambda=1e-6 node-based inversion on preview mesh:**
- sigma_zz mean = 27.7 ± 3.2 GPa on culet surface (physically reasonable for DAC)
- tr(sigma) mean = 91.8 GPa → hydrostatic pressure ~30 GPa ✓
- D_g mean predicted: 0.41–0.48 GHz vs measured 0.39–0.61 GHz (NV0 has 0.13 GHz offset)
- Sign convention: model uses tensile-positive; data D_g > 0 corresponds to compressive
  loading. LSQR handles automatically by finding appropriate sign of c.

**Noise floor:** sqrt(46,372) × 0.02 GHz ≈ 4.3 GHz total, 0.02 GHz per pixel.
Current residual 21 GHz = 5× noise floor = model-limited (not data-noise-limited).

---

## What Needs To Be Done Next Session

### Immediate (start of next session):
1. **Debug Cartesian grid adjoint** — the P^T accumulation may have an off-by-one or
   wrong index. Test with isolated P/P^T check before full adjoint.

2. **Run medium mesh + Cartesian grid inversion:**
   ```bash
   conda run -n dac-recon python -m solver.invert real mesh/anvil_medium.msh data 1e-6
   ```
   Expected: better inter-channel correlation (r > 0.6), residual < 10 GHz.

3. **Check spatial correlation improvement** — key metric is model inter-channel r:
   should rise from 0.09–0.48 (preview) to >0.7 (medium) if mesh is the bottleneck.

### BU SCC (fine mesh):
```bash
# Request 128 GB RAM node
python -m solver.invert real mesh/anvil_fine_tet.msh data 1e-6
```
Expected timing: MUMPS ~15 min, 200 LSQR iterations ~2–5 hrs.

### Lambda selection:
- Run L-curve: lambda in [1e-8, 1e-6, 1e-4] for each mesh
- Optimal lambda: residual ≈ noise floor (4.3 GHz total, 0.02 GHz per pixel)

### Visualization / analysis (post-inversion):
- Plot sigma_zz map (should show Gaussian-like pressure distribution)
- Compare D_predicted vs D_measured per channel
- Compute residual maps to identify systematic errors

---

## File Map

| File | Status | Notes |
|------|--------|-------|
| `solver/operators.py` | Modified | 3 adjoint bugs fixed; node-based + Cartesian grid parameterization; P2 geometry Jacobian fix |
| `solver/invert.py` | Modified | Updated for node-based c (N_nodes not N_meas); t_nodes key in result dict |
| `solver/visualize.py` | New | 5-panel visualization script |
| `mesh/build_mesh.py` | Modified | Added CLI args: --lc-sample, --lc-gasket, --lc-table, --tcap |
| `mesh/convert_to_tet.py` | Modified | Fixed prism6→tet4 conversion; added quad4→tri3 |
| `mesh/anvil_medium.msh` | New | 102k nodes, LC_GASKET=5µm, DOF=173k |
| `mesh/anvil_fine_tet.msh` | Exists | 2.66M nodes; too large for local (use BU SCC) |
| `solver/traction_coeffs.npy` | Exists | From lambda=1e-6 node-based (preview mesh) |
| `solver/sigma_culet.npy` | Exists | Stress at 11,593 pixels from above |
| `solver/D_predicted.npy` | Exists | Predicted D_g from above |

---

## Environment

```bash
conda activate dac-recon
cd ~/widefield-reconstruction
```
