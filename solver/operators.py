"""
Matrix-free forward (A) and adjoint (A^T) operators for LSQR inversion.

Problem structure (SPEC_v2.md Section 6)
-----------------------------------------
Unknown: traction coefficients c of shape (N_meas, 3), flattened to (N_B,)
  where N_B = 3 * N_meas = 34,779 (3 components × 11,593 pixels).

  c[3*m + alpha] = t_alpha at pixel m  (alpha in {0,1,2} = x,y,z)

Data: d of shape (4 * N_meas,) = (46,372,)
  d[g * N_meas + m] = Delta_D_g at pixel m

Forward operator  A @ c
-----------------------
1. Reshape c -> (N_meas, 3) pixel tractions.
2. Assemble one FEM surface load vector by distributing each pixel traction
   to the nearest surface node(s) via the surface projection (described below).
3. One MUMPS back-substitution: K u = f.
4. Evaluate sigma(u) at all N_meas pixel coords -> (N_meas, 6) Voigt.
5. Contract with each M_g -> (4, N_meas) -> flatten to (4*N_meas,).

Adjoint operator  A^T @ r
--------------------------
Given r of shape (4*N_meas,):
1. Reshape r -> (4, N_meas).
2. At each pixel m, sum over g: M_g^T r_g[m] -> a (3,3) "stress residual"
   tensor S[m].   (M_g is symmetric so M_g^T = M_g.)
3. The adjoint of "evaluate sigma at pixel m" is a point load at pixel m
   in the stress-conjugate directions. Specifically, the Cauchy stress
   sigma = C : eps(u), so the adjoint of sigma_ij(x_m) applied to weight
   w_{ij} is the body load:
       f^adj_alpha(x) = sum_{i,j} w_{ij} * C_lab_{ij alpha beta} * d/dx_beta [delta(x - x_m)]
   which in weak form gives nodal loads at x_m:
       F_n = sum_{i,j} S_{ij} * C_lab_{ij alpha beta} * d phi_n / dx_beta |_{x_m}
   Equivalently: the adjoint load in direction alpha at node n is
       F_{n, alpha} = C_lab_{alpha beta ij} * S_{ij} * d phi_n / dx_beta (x_m)
   This is equivalent to applying a stress sigma_adj = C : S (symmetric,
   via the constitutive operator) as a body force, which translates to
   consistent nodal loads via the virtual work principle.

   Practical implementation: apply S as a "traction" on a virtual test
   problem. Because C is the same operator as in the forward problem,
   the adjoint load equals:  f_adj = K^T * u_adj,  where u_adj satisfies
   K v = loads from S.  Since K = K^T (symmetric stiffness), the adjoint
   solve uses the same factored K.

   Concretely: the adjoint of "evaluate sigma(u) at point x_m with weight
   w_{ij}" is a point traction load at x_m:
       f^adj contribution: add w_{ij} * (C_lab_{ij 2 k} e_k) as a nodal load at x_m
   where the "2" index is z (the culet normal direction) -- this is because
   sigma·n at the surface with n=e_z gives the traction.

   Simpler direct formulation (used here):
   The adjoint of sigma evaluation at x_m with weight S[m] is a surface
   point load in the direction  t_adj[m] = S[m] @ n_culet  where n_culet = e_z.
   This converts the (3,3) stress-weight to a (3,) traction load at x_m.

4. Distribute the adjoint traction loads t_adj[m, :] for m=1..N_meas
   onto the FEM mesh surface nodes (same as in the forward step).
5. Solve K v = f_adj (one back-substitution with same MUMPS factorisation).
6. Read off the surface traction DOF values from v at the pixel locations.
   The adjoint output c_adj[3m + alpha] = v_alpha(x_m) restricted to the
   sample chamber surface.

Why this works
--------------
Let F(u)(x_m) = sigma(u)(x_m) be the stress-evaluation operator.
Its adjoint F^T maps a (3,3) weight S at x_m to a body load via the
principle of virtual work:
    <F(u), S> = integral_Omega sigma(u):S dx_m  (schematically at a point)
The functional on u is: v -> sigma(v)(x_m):S = C:eps(v)(x_m):S
which by the FEM test-function formulation equals:
    a(v, w) where w solves: a(w, phi) = C:eps(phi)(x_m):S for all test phi
This adjoint problem has the same stiffness K and the RHS is the virtual
work of the stress S applied at x_m.

In practice, we project S through the culet normal: t_adj = S @ n  (n=e_z),
making it a surface traction load at x_m on the culet, then solve normally.

The output of the adjoint is the DOF values of v on the culet surface,
which represent the sensitivities of the misfit to each traction DOF.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .forward import ForwardSolver
from .nv_coupling import build_coupling_matrices, dg_all


# Culet outward normal (pointing out of the diamond, toward the sample)
_N_CULET = np.array([0.0, 0.0, 1.0])


class NVOperator:
    """
    Matrix-free operator A and adjoint A^T for NV-stress LSQR inversion.

    The traction is parameterized at the NV pixel locations (sample chamber only).
    The parameterization is: N_B = 3 * N_meas scalar DOFs, where
      c[3*m + alpha] = t_alpha(x_m)  in GPa.

    Parameters
    ----------
    solver     : ForwardSolver — mesh loaded, K factored.
    pixel_coords : (N_meas, 3) array — (x, y, 0) pixel locations in µm.
    theta_misc_deg, phi_misc_deg : miscut parameters for coupling matrices.
    """

    def __init__(
        self,
        solver: ForwardSolver,
        pixel_coords: np.ndarray,
        theta_misc_deg: float = -3.5,
        phi_misc_deg: float   = 174.4,
    ) -> None:
        self.solver = solver
        self.pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
        if self.pixel_coords.shape[1] == 2:
            z = np.zeros((len(self.pixel_coords), 1))
            self.pixel_coords = np.hstack([self.pixel_coords, z])

        self.N_meas = len(self.pixel_coords)
        self.N_B    = 3 * self.N_meas     # traction DOFs (x,y,z per pixel)
        self.N_data = 4 * self.N_meas     # data DOFs (4 NV orientations)

        _, self.M_list = build_coupling_matrices(theta_misc_deg, phi_misc_deg)
        # M_g as (4, 6) Voigt-contraction rows (precomputed for speed)
        self._M_voigt = self._precompute_M_voigt()

        # Precompute surface mass projection:
        # _pixel_to_node[m] maps pixel m's traction to DOF-space loads.
        # We use a simple nearest-node approach here (correct for fine meshes
        # where pixels sit almost on top of mesh nodes).
        # This is filled lazily on first matvec call.
        self._surf_proj = None  # (N_meas, N_dof_local) sparse, built on first use

    # ---------------------------------------------------------------------- #

    def _precompute_M_voigt(self) -> np.ndarray:
        """
        Precompute (4, 6) array where row g gives the Voigt contraction
        coefficients for D_g = M_g_voigt @ sigma_voigt.
        Voigt order: [s11, s22, s33, s12, s23, s13].
        """
        M_voigt = np.zeros((4, 6))
        for g, M in enumerate(self.M_list):
            M_voigt[g] = [
                M[0, 0], M[1, 1], M[2, 2],
                2 * M[0, 1], 2 * M[1, 2], 2 * M[0, 2],
            ]
        return M_voigt

    # ---------------------------------------------------------------------- #
    # Surface load assembly helper
    # ---------------------------------------------------------------------- #

    def _traction_to_load_vector(self, t_pixels: np.ndarray) -> "PETSc.Vec":
        """
        Convert per-pixel traction array (N_meas, 3) to a PETSc load vector
        by distributing point tractions to the mesh surface.

        Strategy: build a UFL-free load vector by setting one DOF = 1 per pixel
        component, summing. This is the point-load (delta-function traction)
        approximation — valid for mesh elements smaller than pixel spacing.

        For sub-pixel mesh resolution, using the surface mass matrix for
        consistent distribution is more accurate; that upgrade is noted below.
        """
        from petsc4py import PETSc
        import dolfinx.fem as fem
        from .forward import TAG_CULET_SAMPLE

        # We convert pixel tractions to a UFL expression for the Neumann form
        # by directly assembling the load vector from a sum of point loads.
        # For each pixel m and direction alpha, contribute t[m, alpha] to the
        # DOF closest to x_m in direction alpha on the culet surface.

        # Locate the nearest surface node for each pixel
        if self._surf_proj is None:
            self._build_surface_projection()

        # t_pixels: (N_meas, 3)
        b = self.solver.K.createVecRight()
        b.zeroEntries()

        # For each pixel, add traction contributions to nearby DOF(s)
        for m in range(self.N_meas):
            for alpha in range(3):
                val = float(t_pixels[m, alpha])
                if abs(val) < 1e-30:
                    continue
                # Get DOF indices and weights for this pixel
                dofs_m, weights_m = self._surf_proj[m]
                for dof, w in zip(dofs_m, weights_m):
                    # DOF encodes both node and direction; dof is the flattened
                    # vector DOF index. We stored (node_dof_alpha, weight) pairs.
                    actual_dof = dof * 3 + alpha
                    b.setValueLocal(int(actual_dof), b.getValueLocal(actual_dof) + val * w)

        b.assemblyBegin()
        b.assemblyEnd()
        from dolfinx.fem.petsc import set_bc
        set_bc(b, [self.solver.bc])
        return b

    def _build_surface_projection(self) -> None:
        """
        For each pixel, find the nearest culet surface node and store its
        node index + weight (=1 for nearest-node projection).
        Stored as self._surf_proj: list of (dof_indices, weights) per pixel.

        For a fine mesh (node spacing < 1 µm), this is essentially exact.
        """
        mesh = self.solver.mesh
        V    = self.solver.V

        # Get coordinates of all DOFs in the function space
        # V is a P2 vector space; dof coordinates come from the DOF map
        dof_coords = V.tabulate_dof_coordinates()  # (N_dof_total, 3)

        # Culet surface DOFs (from solver)
        culet_dofs = self.solver.culet_dofs  # local DOF indices in V
        # These are block DOF indices (one per node per component block).
        # In a P2 vector space with 3 components, DOF i corresponds to
        # node i//3, direction i%3.

        # Get the (x,y,z) coordinate of each culet node DOF
        culet_node_indices  = culet_dofs // 3          # which node
        culet_dof_component = culet_dofs % 3           # which direction (0,1,2)

        # Only use x-component DOFs to get unique node positions
        # (avoid triple-counting the same node)
        x_dofs_mask = culet_dof_component == 0
        culet_x_dofs   = culet_dofs[x_dofs_mask]      # DOF indices of x-component
        culet_node_idx = culet_node_indices[x_dofs_mask]

        culet_node_coords = dof_coords[culet_x_dofs]   # (N_culet_nodes, 3)

        # For each pixel, find nearest culet node
        self._surf_proj = []
        for m in range(self.N_meas):
            px, py, pz = self.pixel_coords[m]
            dx = culet_node_coords[:, 0] - px
            dy = culet_node_coords[:, 1] - py
            dist2 = dx * dx + dy * dy
            nearest = int(np.argmin(dist2))
            node_dof_idx = int(culet_node_idx[nearest])
            # Store node DOF index (x-component), weight=1
            self._surf_proj.append(([node_dof_idx], [1.0]))

    # ---------------------------------------------------------------------- #
    # Forward operator: A @ c
    # ---------------------------------------------------------------------- #

    def matvec(self, c: np.ndarray) -> np.ndarray:
        """
        Forward operator: A @ c

        Parameters
        ----------
        c : (N_B,) = (3 * N_meas,) array — traction coefficients

        Returns
        -------
        d : (N_data,) = (4 * N_meas,) array — predicted D_g values [GHz]
        """
        c = np.asarray(c, dtype=np.float64)
        t_pixels = c.reshape(self.N_meas, 3)  # (N_meas, 3) traction in GPa

        # 1. Assemble load vector
        b = self._traction_to_load_vector(t_pixels)

        # 2. Forward solve: K u = b
        u = self.solver.solve_load_vector(b)

        # 3. Evaluate stress at pixel locations: (N_meas, 6)
        sigma_v = self.solver.stress_at_coords_batch(u, self.pixel_coords)
        sigma_v = np.where(np.isnan(sigma_v), 0.0, sigma_v)

        # 4. NV contraction: (4, N_meas) -> flatten to (4 * N_meas,)
        # D[g, m] = M_voigt[g] @ sigma_v[m]
        D = sigma_v @ self._M_voigt.T   # (N_meas, 4)
        return D.T.ravel()              # (4 * N_meas,) order: all NV0, then NV1, ...

    # ---------------------------------------------------------------------- #
    # Adjoint operator: A^T @ r
    # ---------------------------------------------------------------------- #

    def rmatvec(self, r: np.ndarray) -> np.ndarray:
        """
        Adjoint operator: A^T @ r

        The adjoint reverses the chain:
          r (4*N_meas) -> unpack to (4, N_meas)
          -> M_g^T = M_g (symmetric) contracts to stress-space weights: (N_meas, 3, 3)
          -> project through culet normal n=e_z to get adjoint tractions: (N_meas, 3)
          -> assemble adjoint load vector
          -> solve K v = f_adj  (same factored K, K is symmetric)
          -> read off culet surface traction DOF values: (3 * N_meas,)

        Parameters
        ----------
        r : (N_data,) = (4 * N_meas,) residual vector

        Returns
        -------
        c_adj : (N_B,) = (3 * N_meas,) adjoint output
        """
        r = np.asarray(r, dtype=np.float64)
        R = r.reshape(4, self.N_meas)   # (4, N_meas)

        # 1. Adjoint of NV contraction: map r -> stress weights S[m] (N_meas, 3, 3)
        #    D_g[m] = M_g : sigma[m]  =>  adjoint: S_ij[m] = sum_g M_g[i,j] * r[g,m]
        #    Because M_g is symmetric, M_g^T = M_g.
        #    In Voigt: S_voigt[m] = sum_g M_voigt[g] * R[g, m]
        #    S_voigt has shape (N_meas, 6).
        S_voigt = R.T @ self._M_voigt   # (N_meas, 6)

        # 2. Adjoint of stress evaluation at pixel m with weight S[m]:
        #    "The adjoint of stress evaluation at a point is a point load
        #     in stress-conjugate directions."
        #    Concretely: the adjoint load at x_m is the traction
        #       t_adj[m] = S[m] @ n_culet
        #    where n_culet = (0,0,1) is the culet outward normal.
        #    S[m] is a (3,3) tensor; S @ n gives a (3,) traction vector.
        #    In Voigt [s11,s22,s33,s12,s23,s13], with n = e_z (index 2):
        #       t_adj[m, 0] = S[m][0,2] = s13 = S_voigt[m, 5]  (with factor 1, not 2)
        #       t_adj[m, 1] = S[m][1,2] = s23 = S_voigt[m, 4]
        #       t_adj[m, 2] = S[m][2,2] = s33 = S_voigt[m, 2]
        # NOTE: Voigt stores s12, s23, s13 without the factor of 2 in the tensor.
        # The contraction above with M_voigt used 2*M[0,1] etc., giving S_voigt
        # entries that already include the double counting. But S_voigt is a
        # Voigt stress tensor, so:
        #   S[0,2] = S_voigt[5]   (= s13, off-diagonal, stored once)
        #   S[1,2] = S_voigt[4]   (= s23)
        #   S[2,2] = S_voigt[2]   (= s33, diagonal)
        # The adjoint traction t = S @ e_z reads column 2 of S tensor:
        t_adj = np.column_stack([
            S_voigt[:, 5],   # S_{0,2} = s13
            S_voigt[:, 4],   # S_{1,2} = s23
            S_voigt[:, 2],   # S_{2,2} = s33
        ])  # (N_meas, 3)

        # 3. Assemble adjoint load vector
        b_adj = self._traction_to_load_vector(t_adj)

        # 4. Adjoint solve: K v = f_adj  (K is symmetric so same factored K)
        v = self.solver.solve_load_vector(b_adj)

        # 5. Read off displacement values at pixel locations (N_meas, 3)
        #    The adjoint output is the traction sensitivity = displacement at pixels.
        #    In the adjoint formulation, the sensitivity of the misfit to
        #    traction DOF c[3m + alpha] is v_alpha(x_m).
        v_pixels = self._eval_displacement_at_pixels(v)  # (N_meas, 3)
        return v_pixels.ravel()   # (N_B,)

    def _eval_displacement_at_pixels(self, u) -> np.ndarray:
        """
        Evaluate displacement field u at the pixel locations.
        Returns (N_meas, 3) array.
        """
        import dolfinx.geometry as geo
        coords = self.pixel_coords
        tree   = self.solver._bb_tree

        cell_candidates = geo.compute_collisions_points(tree, coords)
        colliding = geo.compute_colliding_cells(self.solver.mesh, cell_candidates, coords)

        vals = np.zeros((self.N_meas, 3))
        for m, pt in enumerate(coords):
            cells_m = colliding.links(m)
            if len(cells_m) == 0:
                continue
            v_val = u.eval(pt.reshape(1, 3), cells_m[:1])  # (1, 3)
            vals[m] = v_val[0]
        return vals

    # ---------------------------------------------------------------------- #
    # Build scipy LinearOperator
    # ---------------------------------------------------------------------- #

    def as_linear_operator(self) -> LinearOperator:
        """Return a scipy LinearOperator wrapping matvec and rmatvec."""
        return LinearOperator(
            shape=(self.N_data, self.N_B),
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )

    # ---------------------------------------------------------------------- #
    # Adjoint consistency check (finite difference)
    # ---------------------------------------------------------------------- #

    def check_adjoint(self, n_trials: int = 3, seed: int = 42) -> bool:
        """
        Verify <A c, r> == <c, A^T r> for random c and r.

        Passes if relative error < 1e-6 for all trials.
        """
        rng = np.random.default_rng(seed)
        print("Adjoint consistency check (<A c, r> vs <c, A^T r>):")
        all_ok = True
        for i in range(n_trials):
            c = rng.standard_normal(self.N_B)
            r = rng.standard_normal(self.N_data)
            Ac    = self.matvec(c)
            ATr   = self.rmatvec(r)
            lhs   = float(np.dot(Ac, r))
            rhs   = float(np.dot(c, ATr))
            rel   = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30)
            ok    = rel < 1e-6
            all_ok &= ok
            print(f"  trial {i+1}: <Ac,r>={lhs:.6e}  <c,A^Tr>={rhs:.6e}  "
                  f"rel_err={rel:.2e}  {'OK' if ok else 'FAIL'}")
        return all_ok
