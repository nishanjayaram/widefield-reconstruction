"""
Matrix-free forward (A) and adjoint (A^T) operators for LSQR inversion.

Problem structure (SPEC_v2.md Section 6)
-----------------------------------------
Unknown: traction coefficients c of shape (N_meas, 3), flattened to (N_B,)
  where N_B = 3 * N_meas = 34,779  (3 traction components × 11,593 pixels)

  c[3*m + alpha] = t_alpha at pixel m   (alpha in {0,1,2} = x, y, z)

Data: d of shape (4 * N_meas,) = (46,372,)
  d[g * N_meas + m] = Delta_D_g at pixel m

Forward operator  A @ c  (one MUMPS back-substitution)
---------------------------------------------------------
  1. Reshape c -> (N_meas, 3) pixel tractions.
  2. For each pixel m, add t_m to the DOF of the nearest culet surface node
     (nearest-node surface load assembly, matrix P).
  3. One back-substitution:  K u = f.
  4. Evaluate sigma(u) at all pixel coords -> (N_meas, 6) Voigt.
  5. Contract: D[m, g] = M_g_voigt @ sigma[m] -> flatten to (4*N_meas,).

Adjoint operator  A^T @ r  (one MUMPS back-substitution with same factored K)
-------------------------------------------------------------------------------
  A = M E K^{-1} P,  so  A^T = P^T (K^T)^{-1} E^T M^T = P^T K^{-1} E^T M^T
  (K is symmetric, so K^{-T} = K^{-1}.)

  Step-by-step:
  1. M^T r:  unpack r -> R (4, N_meas), then
             S_voigt[m] = sum_g M_voigt[g] * R[g, m]       (N_meas, 6)
             M_g is symmetric so M_g^T = M_g.

  2. E^T S:  adjoint of "evaluate sigma at pixels with weight S".
             sigma(u)(x_m) = C_lab : eps(u)(x_m)
             The adjoint maps stress weights S -> a body load vector f_adj:

               f_adj[scalar_dof(n, k)] = sum_m G_m[k, l] * dphi_n/dx_l(x_m)

             where  G_m = C_lab : S_m  (contracting C on the stress indices i,j)
             and    phi_n is the n-th P2 scalar basis function.

             "The adjoint of stress evaluation at a point is a point load in
              stress-conjugate directions" (user's hint), where the conjugate
              direction is G_m = C : S_m.

  3. K^{-1}:  solve K v = f_adj  (same MUMPS factorisation).

  4. P^T v:   read off v_alpha(nearest_node(m)) for each pixel m and direction alpha.
              This gives (N_meas, 3) -> flatten to (N_B,).
"""

from __future__ import annotations

import numpy as np
import basix
from scipy.sparse.linalg import LinearOperator
from petsc4py import PETSc

from .forward import ForwardSolver
from .nv_coupling import build_coupling_matrices

# Voigt index -> (i, j) mapping
_VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


class NVOperator:
    """
    Matrix-free operator A and adjoint A^T for the NV stress inversion.

    Traction parameterisation: N_B = 3 * N_meas scalar DOFs
      c[3*m + alpha] = t_alpha(x_m)  [GPa].

    Parameters
    ----------
    solver       : ForwardSolver — mesh loaded, K factored.
    pixel_coords : (N_meas, 2) or (N_meas, 3) in µm. z=0 assumed if 2-D.
    theta_misc_deg, phi_misc_deg : miscut angles for coupling matrices.
    """

    def __init__(
        self,
        solver: ForwardSolver,
        pixel_coords: np.ndarray,
        theta_misc_deg: float = -3.5,
        phi_misc_deg: float   = 174.4,
        traction_grid_spacing: float = 0.0,
    ) -> None:
        """
        traction_grid_spacing : µm — if > 0, use a Cartesian traction grid
            with this spacing instead of culet mesh nodes. Gives N_B independent
            of mesh refinement; spacing ~10 µm → ~1963 grid points → N_B=5889.
            If 0 (default), use unique culet mesh nodes nearest to pixels.
        """
        self.solver = solver

        coords = np.asarray(pixel_coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        if coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros((len(coords), 1))])
        self.pixel_coords = coords          # (N_meas, 3)

        self.N_meas = len(self.pixel_coords)
        self.N_B    = 3 * self.N_meas
        self.N_data = 4 * self.N_meas

        _, self.M_list = build_coupling_matrices(theta_misc_deg, phi_misc_deg)
        # (4, 6) array: M_voigt[g] gives the Voigt contraction row for D_g
        self._M_voigt = np.array([
            [M[0,0], M[1,1], M[2,2], 2*M[0,1], 2*M[1,2], 2*M[0,2]]
            for M in self.M_list
        ])  # (4, 6)

        self._traction_grid_spacing = traction_grid_spacing

        # Precompute geometry for both P (nearest-node) and E^T (shape grads)
        self._build_pixel_geometry()

    # ---------------------------------------------------------------------- #
    # Geometry precomputation                                                  #
    # ---------------------------------------------------------------------- #

    def _build_pixel_geometry(self) -> None:
        """
        For each pixel:
          - nearest culet surface node block-DOF index  (for P and P^T)
          - containing cell index and reference coordinates (for E^T)
          - P2 basis function gradients in physical coords (for E^T body load)

        Stored as:
          self._nearest_block_dof : (N_meas,) int  — block DOF of nearest node
          self._cell_of_pixel     : (N_meas,) int  — cell index (-1 if outside)
          self._dphi_dx_pixels    : list of (10, 3) arrays — physical P2 grad per pixel
        """
        import dolfinx.geometry as geo
        from dolfinx.fem import locate_dofs_topological
        from .forward import TAG_CULET_SAMPLE, TAG_CULET_GASKET

        mesh  = self.solver.mesh
        V     = self.solver.V
        tree  = self.solver._bb_tree

        # ---- Nearest culet node for P / P^T ----------------------------
        # Culet DOF block indices
        culet_facets = np.concatenate([
            self.solver.facet_tags.find(TAG_CULET_SAMPLE),
            self.solver.facet_tags.find(TAG_CULET_GASKET),
        ])
        mesh.topology.create_connectivity(2, 3)
        culet_block_dofs = locate_dofs_topological(V, 2, culet_facets)
        # Block DOF i -> x-component scalar DOF = i*3+0; coordinate = tabulate_dof_coordinates()[i*3]
        # tabulate_dof_coordinates returns (N_block_dofs, 3) — one coord per node
        dof_coords_full = V.tabulate_dof_coordinates()  # (N_block_dofs, 3)
        culet_coords = dof_coords_full[culet_block_dofs]  # (N_culet, 3)

        self._nearest_block_dof = np.empty(self.N_meas, dtype=np.intp)
        for m in range(self.N_meas):
            px, py = self.pixel_coords[m, 0], self.pixel_coords[m, 1]
            dx = culet_coords[:, 0] - px
            dy = culet_coords[:, 1] - py
            idx = int(np.argmin(dx*dx + dy*dy))
            self._nearest_block_dof[m] = culet_block_dofs[idx]

        if self._traction_grid_spacing > 0.0:
            # ---- Cartesian grid parameterization (mesh-independent) --------
            # Build a regular grid with spacing s over the culet disk (radius R1).
            # Each culet mesh node is assigned to the nearest grid point.
            # N_B = 3 * N_grid is fixed regardless of mesh refinement.
            s = self._traction_grid_spacing
            R1 = 250.0   # culet radius (µm) — must match geometry
            xs = np.arange(-R1, R1 + s, s)
            ys = np.arange(-R1, R1 + s, s)
            gx, gy = np.meshgrid(xs, ys)
            mask = gx**2 + gy**2 <= R1**2
            grid_xy = np.column_stack([gx[mask], gy[mask]])  # (N_grid, 2)

            # Map each culet mesh node to nearest grid point
            # culet_coords: (N_culet, 3) — z≈0 for all
            grid_of_culet = np.empty(len(culet_block_dofs), dtype=np.intp)
            for j in range(len(culet_block_dofs)):
                dx2 = culet_coords[j, 0] - grid_xy[:, 0]
                dy2 = culet_coords[j, 1] - grid_xy[:, 1]
                grid_of_culet[j] = int(np.argmin(dx2**2 + dy2**2))

            self._culet_node_list = culet_block_dofs      # ALL culet mesh nodes
            self._all_culet_to_traction_idx = grid_of_culet   # mesh node → grid idx
            self._traction_grid_xy = grid_xy
            N_grid = len(grid_xy)
            self.N_nodes = N_grid
            self.N_B = 3 * N_grid
            # For check_adjoint — pixel → grid point
            self._node_idx_of_pixel = np.array(
                [int(grid_of_culet[np.argmin(
                    (culet_coords[:, 0] - self.pixel_coords[m, 0])**2 +
                    (culet_coords[:, 1] - self.pixel_coords[m, 1])**2)])
                 for m in range(self.N_meas)], dtype=np.intp)
        else:
            # ---- Node-based parameterization (unique nearest culet mesh nodes) ----
            # Build the unique set of culet nodes that pixels actually map to.
            # On coarse meshes N_nodes << N_meas; on fine meshes N_nodes ≈ N_meas.
            self._culet_node_list = np.unique(self._nearest_block_dof)
            self._all_culet_to_traction_idx = None
            N_nodes = len(self._culet_node_list)
            self.N_nodes = N_nodes
            self.N_B = 3 * N_nodes
            _node_lookup = {int(nd): k for k, nd in enumerate(self._culet_node_list)}
            self._node_idx_of_pixel = np.array(
                [_node_lookup[int(self._nearest_block_dof[m])] for m in range(self.N_meas)],
                dtype=np.intp,
            )  # (N_meas,) — pixel m maps to culet node index k

        # ---- Cell + reference coords + P2 shape function gradients -----
        # These are used in the adjoint body load assembly (E^T)
        cell_type = getattr(basix.CellType, mesh.topology.cell_name())
        fem_el = basix.create_element(
            basix.ElementFamily.P, cell_type, 2, basix.LagrangeVariant.gll_warped
        )

        coord_x      = mesh.geometry.x        # (N_geom, 3)
        coord_dofmap = mesh.geometry.dofmap   # cell -> geom node indices

        cell_cands = geo.compute_collisions_points(tree, self.pixel_coords)
        colliding  = geo.compute_colliding_cells(mesh, cell_cands, self.pixel_coords)

        self._cell_of_pixel  = np.full(self.N_meas, -1, dtype=np.intp)
        self._dphi_dx_pixels = [None] * self.N_meas   # (10, 3) per pixel

        for m in range(self.N_meas):
            cells_m = colliding.links(m)
            if len(cells_m) == 0:
                continue
            cell = int(cells_m[0])
            self._cell_of_pixel[m] = cell

            # Geometry: tet with constant Jacobian (straight-sided element).
            # coord_dofmap returns 4 nodes for P1 geometry, 10 for P2 geometry.
            # We only use the 4 vertex nodes (first 4 rows) for both Jacobian
            # and pullback, since P2 midpoints are at exact edge midpoints and
            # all elements are straight-sided.  Using cmap.pull_back with the
            # full P2 x_cell gives wrong xi_m on some cells (up to ~1e-3 error)
            # because the P2 coordinate map's iterative inverse converges
            # inconsistently.  The affine inverse J^{-1}(x-x0) is exact.
            x_cell  = coord_x[coord_dofmap[cell]]              # (4 or 10, 3)
            x_verts = x_cell[:4]                               # (4, 3) vertex nodes

            # Jacobian: columns = edge vectors from vertex 0
            J     = (x_verts[1:] - x_verts[0]).T              # (3, 3)
            J_inv = np.linalg.inv(J)

            # Affine pullback: exact for straight-sided tets
            xi_m  = (J_inv @ (self.pixel_coords[m] - x_verts[0])).reshape(1, 3)

            # P2 basis derivatives in reference coords
            table    = fem_el.tabulate(1, xi_m)               # (4, 1, 10, 1)
            dphi_dxi = table[1:, 0, :, 0].T                   # (10, 3)

            # Row-vector form: (grad_x phi)^T = (grad_xi phi)^T @ J^{-1}

            dphi_dx = dphi_dxi @ J_inv                        # (10, 3) physical gradients
            self._dphi_dx_pixels[m] = dphi_dx

    # ---------------------------------------------------------------------- #
    # Surface load assembly  (P : c -> f)                                    #
    # ---------------------------------------------------------------------- #

    def _traction_to_load_vector(self, t_nodes: np.ndarray) -> PETSc.Vec:
        """
        Node-based surface load assembly.
        t_nodes : (N_nodes, 3) traction in GPa — one value per unique culet node.
        Returns : PETSc.Vec f  (load vector, BCs applied).
        """
        from dolfinx.fem.petsc import set_bc

        V  = self.solver.V
        bs = V.dofmap.index_map_bs   # = 3
        n_local = V.dofmap.index_map.size_local * bs

        f_arr = np.zeros(n_local, dtype=np.float64)
        if self._all_culet_to_traction_idx is not None:
            # Cartesian grid mode: each culet mesh node gets the traction of its grid point
            for j, block_dof in enumerate(self._culet_node_list):
                k = int(self._all_culet_to_traction_idx[j])
                for alpha in range(3):
                    scalar_dof = int(block_dof) * bs + alpha
                    if scalar_dof < n_local:
                        f_arr[scalar_dof] = float(t_nodes[k, alpha])
        else:
            # Node-based mode: direct assignment, one traction value per culet node
            for k, block_dof in enumerate(self._culet_node_list):
                for alpha in range(3):
                    scalar_dof = int(block_dof) * bs + alpha
                    if scalar_dof < n_local:
                        f_arr[scalar_dof] = float(t_nodes[k, alpha])

        b = self.solver.K.createVecRight()
        b.zeroEntries()
        b.setArray(f_arr)
        b.assemblyBegin()
        b.assemblyEnd()
        set_bc(b, [self.solver.bc])
        return b

    # ---------------------------------------------------------------------- #
    # Adjoint body load assembly  (E^T S -> f_adj)                          #
    # ---------------------------------------------------------------------- #

    def _adjoint_body_load(self, S_voigt: np.ndarray) -> PETSc.Vec:
        """
        Assemble the adjoint body load vector:

          f_adj[scalar_dof(n, k)] = sum_m  G_m[k, l] * dphi_n/dx_l(x_m)

        where G_m = C_lab : S_m  (contraction on stress indices i,j).

        This is the exact adjoint of "evaluate sigma(u) at pixel m with weight S_m".

        S_voigt : (N_meas, 6) stress weights [Voigt order: s11,s22,s33,s12,s23,s13]
        Returns : PETSc.Vec f_adj  (BCs applied)
        """
        from dolfinx.fem.petsc import set_bc

        C4  = self.solver.C4_lab       # (3,3,3,3) [GPa]
        V   = self.solver.V
        dm  = V.dofmap
        bs  = V.dofmap.index_map_bs    # = 3
        n_local = V.dofmap.index_map.size_local * bs

        f_arr = np.zeros(n_local, dtype=np.float64)

        for m in range(self.N_meas):
            cell = self._cell_of_pixel[m]
            if cell < 0:
                continue

            # Reconstruct S_m (3,3) from Voigt.
            # M_voigt stores 2*M[i,j] for off-diagonal, so S_voigt[off-diag] = 2*S_m[i,j].
            S_m = np.zeros((3, 3))
            for I, (i, j) in enumerate(_VOIGT):
                if i == j:
                    S_m[i, j] = S_voigt[m, I]
                else:
                    S_m[i, j] = S_m[j, i] = S_voigt[m, I] / 2.0

            # G_m[k,l] = C_lab[i,j,k,l] * S_m[i,j]  using C major symmetry
            G_m = np.einsum("ijkl,ij->kl", C4, S_m)   # (3,3)

            dphi_dx  = self._dphi_dx_pixels[m]          # (10, 3)
            cell_dofs = dm.cell_dofs(cell)               # (10,) block DOFs

            for i in range(10):                          # P2 nodes in cell
                block_dof = int(cell_dofs[i])
                for k in range(3):
                    scalar_dof = block_dof * bs + k
                    if 0 <= scalar_dof < n_local:
                        # f_adj[n,k] += G_m[k,:] . dphi_n/dx
                        f_arr[scalar_dof] += float(np.dot(G_m[k], dphi_dx[i]))

        # Build PETSc vector
        b = self.solver.K.createVecRight()
        b.zeroEntries()
        b.setArray(f_arr)
        b.assemblyBegin()
        b.assemblyEnd()
        set_bc(b, [self.solver.bc])
        return b

    # ---------------------------------------------------------------------- #
    # Forward operator  A @ c                                                 #
    # ---------------------------------------------------------------------- #

    def _stress_at_pixels_direct(self, u) -> np.ndarray:
        """
        Evaluate sigma = C : eps(u) at pixel positions using the same precomputed
        P2 basis function gradients (dphi_dx) used by the adjoint body load.

        This is the ONLY correct forward for the adjoint consistency check:
        DOLFINx's DG1 stress evaluation uses the P2 coordinate-map pullback
        internally, which gives wrong reference coords on some cells of P2-geometry
        meshes.  Using dphi_dx (computed from the exact affine pullback) makes
        forward and adjoint perfectly consistent.

        Returns (N_meas, 6) Voigt stress [GPa].
        """
        u_arr = u.x.array
        bs    = self.solver.V.dofmap.index_map_bs
        dm    = self.solver.V.dofmap
        C4    = self.solver.C4_lab
        n_loc = len(u_arr)

        sigma_v = np.zeros((self.N_meas, 6))
        for m in range(self.N_meas):
            cell = self._cell_of_pixel[m]
            if cell < 0:
                continue
            cell_dofs = dm.cell_dofs(cell)   # (10,) block DOFs
            dphi_dx   = self._dphi_dx_pixels[m]   # (10, 3)

            # u_cell[i, a] = u_a at the i-th P2 node of this cell
            u_cell = np.zeros((10, 3))
            for i in range(10):
                base = int(cell_dofs[i]) * bs
                for a in range(3):
                    sd = base + a
                    if sd < n_loc:
                        u_cell[i, a] = u_arr[sd]

            # grad_u[a, l] = sum_i dphi_dx[i, l] * u_cell[i, a]
            grad_u = u_cell.T @ dphi_dx   # (3, 3)
            eps    = (grad_u + grad_u.T) / 2.0
            sigma  = np.einsum("ijkl,kl->ij", C4, eps)
            for I, (i, j) in enumerate(_VOIGT):
                sigma_v[m, I] = sigma[i, j]

        return sigma_v

    def matvec(self, c: np.ndarray) -> np.ndarray:
        """
        A @ c  — one MUMPS back-substitution.

        c : (N_B,) traction coefficients [GPa]
        Returns: (N_data,) predicted D_g values [GHz]
        """
        c = np.asarray(c, dtype=np.float64)
        t_nodes = c.reshape(self.N_nodes, 3)

        b = self._traction_to_load_vector(t_nodes)
        u = self.solver.solve_load_vector(b)

        sigma_v = self._stress_at_pixels_direct(u)
        sigma_v = np.where(np.isnan(sigma_v), 0.0, sigma_v)

        # D[m, g] = sigma_v[m] @ M_voigt[g]  ->  shape (N_meas, 4)
        D = sigma_v @ self._M_voigt.T   # (N_meas, 4)
        return D.T.ravel()              # (4*N_meas,), order: all NV0, NV1, NV2, NV3

    # ---------------------------------------------------------------------- #
    # Adjoint operator  A^T @ r                                               #
    # ---------------------------------------------------------------------- #

    def rmatvec(self, r: np.ndarray) -> np.ndarray:
        """
        A^T @ r  — one MUMPS back-substitution with same factored K.

        r : (N_data,) residual vector
        Returns: (N_B,) adjoint output (sensitivity of misfit to each traction DOF)
        """
        r = np.asarray(r, dtype=np.float64)
        R = r.reshape(4, self.N_meas)   # (4, N_meas)

        # Step 1: M^T r  ->  S_voigt (N_meas, 6)
        #   S_voigt[m] = sum_g M_voigt[g] * R[g, m]
        S_voigt = R.T @ self._M_voigt   # (N_meas, 6)

        # Step 2: E^T S  ->  f_adj  (body load via shape function gradients)
        f_adj = self._adjoint_body_load(S_voigt)

        # Step 3: K^{-1} f_adj  ->  v
        v = self.solver.solve_load_vector(f_adj)

        # Step 4: P^T v  ->  map displacement back to traction parameters.
        bs    = self.solver.V.dofmap.index_map_bs
        c_adj = np.zeros(self.N_B, dtype=np.float64)
        v_arr = v.x.array  # (n_local_scalar_dofs,)

        if self._all_culet_to_traction_idx is not None:
            # Cartesian grid mode: accumulate v over all culet mesh nodes per grid point
            for j, block_dof in enumerate(self._culet_node_list):
                k = int(self._all_culet_to_traction_idx[j])
                c_adj[3*k : 3*k+3] += v_arr[int(block_dof)*bs : int(block_dof)*bs+3]
        else:
            # Node-based mode: read v at each traction node directly
            for k, block_dof in enumerate(self._culet_node_list):
                c_adj[3*k : 3*k+3] = v_arr[int(block_dof)*bs : int(block_dof)*bs+3]

        return c_adj

    # ---------------------------------------------------------------------- #
    # Wrap as scipy LinearOperator                                            #
    # ---------------------------------------------------------------------- #

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            shape=(self.N_data, self.N_B),
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )

    # ---------------------------------------------------------------------- #
    # Adjoint consistency check                                               #
    # ---------------------------------------------------------------------- #

    def check_adjoint(self, n_trials: int = 3, seed: int = 42,
                      tol: float = 1e-4) -> bool:
        """
        Verify <A c, r> == <c, A^T r> for random c and r.
        Relative error should be < tol for a correct adjoint.
        """
        rng = np.random.default_rng(seed)
        print("Adjoint consistency check  (<Ac, r>  vs  <c, A^T r>):")
        all_ok = True
        for i in range(n_trials):
            # Use small random vectors to keep solve cost low
            c = rng.standard_normal(self.N_B)
            r = rng.standard_normal(self.N_data)
            Ac   = self.matvec(c)
            ATr  = self.rmatvec(r)
            lhs  = float(np.dot(Ac, r))
            rhs  = float(np.dot(c, ATr))
            rel  = abs(lhs - rhs) / (0.5 * (abs(lhs) + abs(rhs)) + 1e-30)
            ok   = rel < tol
            all_ok &= ok
            print(f"  trial {i+1}:  <Ac,r>={lhs:.6e}  <c,A^Tr>={rhs:.6e}  "
                  f"rel_err={rel:.2e}  {'OK' if ok else 'FAIL'}")
        return all_ok
