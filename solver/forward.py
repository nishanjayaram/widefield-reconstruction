"""
FEniCSx forward elasticity solver for the diamond anvil cell.

Physics
-------
  div(sigma) = 0          in diamond bulk
  sigma = C_lab : eps(u)  constitutive law (cubic diamond, rotated to lab frame)
  u = 0                   on table (z = -h, physical group tag 3)  [Dirichlet]
  sigma·n = 0             on facets (conical sidewall, tag 4)      [natural]
  sigma·n = t             on culet (tags 1 + 2)                    [Neumann]

Units: coordinates in µm, stiffness in GPa → stress in GPa, displacement in µm.

Key design
----------
- The stiffness matrix K is assembled and LU-factored (MUMPS) once.
- Subsequent solves for different culet tractions require only a back-substitution.
- For the basis-function inversion, each traction basis vector is a unit vector
  in the global DOF space restricted to culet surface DOFs.
- Stress is evaluated at NV measurement points by interpolating sigma onto a DG1
  tensor function space and using dolfinx's bounding-box tree.

Usage
-----
    solver = ForwardSolver("mesh/anvil_preview.msh")
    dofs = solver.culet_dofs          # array of culet DOF local indices
    u    = solver.solve_unit_load(dofs[0])
    sig  = solver.stress_at_coords(u, points_xyz)  # (N, 6) Voigt [GPa]
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Sequence

from mpi4py import MPI
import dolfinx
import dolfinx.fem as fem
import dolfinx.geometry as geo
from dolfinx.io import gmsh as gmshio
import dolfinx.fem.petsc as fem_petsc
import ufl
from petsc4py import PETSc
import basix.ufl as bufl

from .crystal import build_C_lab

# Physical group tags (must match mesh/build_mesh.py)
TAG_CULET_SAMPLE = 1
TAG_CULET_GASKET = 2
TAG_TABLE        = 3
TAG_FACET        = 4

# Default miscut parameters from SPEC.md
THETA_MISC = -3.5    # degrees
PHI_MISC   = 174.4   # degrees


class ForwardSolver:
    """
    FEniCSx elasticity solver for the diamond anvil.

    Parameters
    ----------
    mesh_path : str or Path
        Path to the .msh file (Gmsh MSH format, physical groups tagged).
    theta_misc_deg : float
        Miscut polar angle in degrees (default: -3.5°).
    phi_misc_deg : float
        Effective miscut azimuthal angle in degrees (default: 174.4°).
    verbose : bool
        Print assembly and solver timings.
    """

    def __init__(
        self,
        mesh_path: str | Path,
        theta_misc_deg: float = THETA_MISC,
        phi_misc_deg: float   = PHI_MISC,
        verbose: bool = True,
    ) -> None:
        self.mesh_path = Path(mesh_path)
        self.verbose   = verbose
        self.comm      = MPI.COMM_WORLD

        # ------------------------------------------------------------------ #
        # 1. Load mesh                                                        #
        # ------------------------------------------------------------------ #
        if verbose:
            print(f"Loading mesh: {self.mesh_path}")
        mesh_data = gmshio.read_from_msh(
            str(self.mesh_path), self.comm, rank=0, gdim=3
        )
        self.mesh       = mesh_data.mesh
        self.cell_tags  = mesh_data.cell_tags
        self.facet_tags = mesh_data.facet_tags
        if verbose:
            num_cells = self.mesh.topology.index_map(
                self.mesh.topology.dim).size_global
            print(f"  {num_cells:,} cells, gdim=3")

        # ------------------------------------------------------------------ #
        # 2. Build C_lab (3×3×3×3 stiffness tensor in lab frame)            #
        # ------------------------------------------------------------------ #
        self.C4_lab, self.C_voigt_lab, self.R = build_C_lab(
            theta_misc_deg, phi_misc_deg
        )
        if verbose:
            print(f"  C_lab built (theta={theta_misc_deg}°, phi={phi_misc_deg}°)")

        # ------------------------------------------------------------------ #
        # 3. Function space — P2 vector (quadratic Lagrange, 3 components)  #
        # ------------------------------------------------------------------ #
        el = bufl.element("Lagrange", self.mesh.topology.cell_name(), 2, shape=(3,))
        self.V = fem.functionspace(self.mesh, el)
        if verbose:
            total_dofs = self.V.dofmap.index_map.size_global * \
                         self.V.dofmap.index_map_bs
            print(f"  DOF count: {total_dofs:,}")

        # ------------------------------------------------------------------ #
        # 4. Dirichlet BC: u = 0 on table (tag 3)                           #
        # ------------------------------------------------------------------ #
        table_facets = self.facet_tags.find(TAG_TABLE)
        self.mesh.topology.create_connectivity(2, 3)
        table_dofs = fem.locate_dofs_topological(self.V, 2, table_facets)
        u_zero = fem.Function(self.V)
        u_zero.x.array[:] = 0.0
        self.bc = fem.dirichletbc(u_zero, table_dofs)

        # ------------------------------------------------------------------ #
        # 5. UFL variational form                                             #
        # ------------------------------------------------------------------ #
        u_trial = ufl.TrialFunction(self.V)
        v_test  = ufl.TestFunction(self.V)

        C4 = self.C4_lab   # numpy (3,3,3,3) used as float constants in UFL

        def _sigma(u):
            """Cauchy stress σ = C_lab : ε(u)."""
            eps = ufl.sym(ufl.grad(u))
            return ufl.as_tensor([
                [
                    sum(
                        float(C4[i, j, k, l]) * eps[k, l]
                        for k in range(3) for l in range(3)
                    )
                    for j in range(3)
                ]
                for i in range(3)
            ])

        self._sigma = _sigma  # keep for later use with solved u

        a_form = fem.form(
            ufl.inner(_sigma(u_trial), ufl.sym(ufl.grad(v_test))) * ufl.dx
        )
        self.a_form = a_form

        # ------------------------------------------------------------------ #
        # 6. Assemble stiffness matrix K with Dirichlet BC                  #
        # ------------------------------------------------------------------ #
        if verbose:
            print("Assembling stiffness matrix K ...")
        self.K = fem_petsc.assemble_matrix(a_form, bcs=[self.bc])
        self.K.assemble()
        if verbose:
            print("  Done.")

        # ------------------------------------------------------------------ #
        # 7. LU factorisation via MUMPS (single factorisation, reused)      #
        # ------------------------------------------------------------------ #
        if verbose:
            print("Factorising K (MUMPS) ...")
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.K)
        self.ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = self.ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        self.ksp.setFromOptions()
        self.ksp.setUp()
        if verbose:
            print("  Factorisation complete.")

        # ------------------------------------------------------------------ #
        # 8. Identify culet surface DOFs (tags 1 and 2)                     #
        # ------------------------------------------------------------------ #
        culet_facets = np.concatenate([
            self.facet_tags.find(TAG_CULET_SAMPLE),
            self.facet_tags.find(TAG_CULET_GASKET),
        ])
        self.mesh.topology.create_connectivity(2, 3)
        self._culet_dofs = fem.locate_dofs_topological(self.V, 2, culet_facets)
        if verbose:
            print(f"  Culet DOFs: {len(self._culet_dofs):,}")

        # ------------------------------------------------------------------ #
        # 9. Stress function space (DG1 tensor) for point evaluation        #
        # ------------------------------------------------------------------ #
        el_S = bufl.element(
            "DG", self.mesh.topology.cell_name(), 1, shape=(3, 3)
        )
        self.V_S = fem.functionspace(self.mesh, el_S)

        # Build bounding-box tree for point evaluation (built once, reused)
        self._bb_tree = geo.bb_tree(self.mesh, self.mesh.topology.dim)

    # ---------------------------------------------------------------------- #
    # Public properties                                                       #
    # ---------------------------------------------------------------------- #

    @property
    def culet_dofs(self) -> np.ndarray:
        """Array of local DOF indices on the culet surface (tags 1 + 2)."""
        return self._culet_dofs

    # ---------------------------------------------------------------------- #
    # Solve                                                                   #
    # ---------------------------------------------------------------------- #

    def solve_load_vector(self, b: PETSc.Vec) -> fem.Function:
        """
        Solve K u = b and return u as a dolfinx Function.

        The input vector b should already have Dirichlet rows zeroed
        (or set to the prescribed value). For the basis-traction approach,
        b is a unit vector at one culet DOF — the Dirichlet rows are
        automatically zero since the culet and table are disjoint.

        Parameters
        ----------
        b : PETSc.Vec  — right-hand side (modified in-place for BCs)

        Returns
        -------
        u : dolfinx.fem.Function  — displacement in V
        """
        # Ensure Dirichlet DOFs in b have their prescribed value (0 for table)
        fem_petsc.set_bc(b, [self.bc])

        u = fem.Function(self.V)
        self.ksp.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        return u

    def solve_unit_load(self, dof_idx: int) -> fem.Function:
        """
        Solve K u = e_{dof_idx}: unit point load at one DOF.

        This is the k-th traction basis solve.  dof_idx is a local DOF
        index (as returned by locate_dofs_topological and stored in
        self.culet_dofs).

        Parameters
        ----------
        dof_idx : int  — local DOF index in V

        Returns
        -------
        u : dolfinx.fem.Function  — displacement in V
        """
        b = self.K.createVecRight()
        b.zeroEntries()
        b.setValueLocal(int(dof_idx), 1.0)
        b.assemblyBegin()
        b.assemblyEnd()
        return self.solve_load_vector(b)

    def solve_traction(
        self,
        traction_fn,
        culet_tags: Sequence[int] = (TAG_CULET_SAMPLE, TAG_CULET_GASKET),
    ) -> fem.Function:
        """
        Solve for a given traction field on the culet surface.

        Parameters
        ----------
        traction_fn : ufl expression of shape (3,)
            Surface traction t(x) in GPa applied on the culet.  Must be a
            UFL expression compatible with the mesh function space.
        culet_tags : sequence of int
            Physical facet tags to apply the traction on.

        Returns
        -------
        u : dolfinx.fem.Function
        """
        v_test = ufl.TestFunction(self.V)
        ds = ufl.Measure("ds", domain=self.mesh,
                         subdomain_data=self.facet_tags)
        L_form = fem.form(
            sum(ufl.inner(traction_fn, v_test) * ds(tag) for tag in culet_tags)
        )
        b = fem_petsc.assemble_vector(L_form)
        fem_petsc.apply_lifting(b, [self.a_form], bcs=[[self.bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        return self.solve_load_vector(b)

    # ---------------------------------------------------------------------- #
    # Stress evaluation                                                       #
    # ---------------------------------------------------------------------- #

    def _interpolate_stress(self, u: fem.Function) -> fem.Function:
        """
        Interpolate σ = C_lab : ε(u) onto the DG1 tensor function space.

        Returns
        -------
        sigma_h : fem.Function in V_S  (tensor (3,3), DG1)
        """
        sigma_expr = fem.Expression(
            self._sigma(u),
            self.V_S.element.interpolation_points,
        )
        sigma_h = fem.Function(self.V_S)
        sigma_h.interpolate(sigma_expr)
        return sigma_h

    def stress_at_coords(
        self,
        u: fem.Function,
        coords: np.ndarray,
        tol: float = 1e-3,
    ) -> np.ndarray:
        """
        Evaluate the Cauchy stress tensor at arbitrary coordinates.

        Parameters
        ----------
        u      : displacement Function (from solve_unit_load / solve_traction)
        coords : (N, 3) ndarray  — evaluation points in µm
        tol    : bounding-box search tolerance (µm)

        Returns
        -------
        sigma_voigt : (N, 6) ndarray  — stress in Voigt order
                      [s11, s22, s33, s12, s23, s13] (GPa).
                      Points outside the mesh return NaN.
        """
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords[np.newaxis, :]
        N = len(coords)

        sigma_h = self._interpolate_stress(u)

        # Find containing cells
        cell_candidates = geo.compute_collisions_points(self._bb_tree, coords)
        colliding = geo.compute_colliding_cells(
            self.mesh, cell_candidates, coords
        )

        # Evaluate: sigma_h is a (3,3) tensor → flattened to 9 values per point
        sigma_voigt = np.full((N, 6), np.nan)
        for i, pt in enumerate(coords):
            cells_i = colliding.links(i)
            if len(cells_i) == 0:
                continue
            # Use first colliding cell
            val = sigma_h.eval(pt.reshape(1, 3), cells_i[:1])
            s = np.reshape(val, 9).reshape(3, 3)  # normalise shape
            # Pack into Voigt order: [s11, s22, s33, s12, s23, s13]
            sigma_voigt[i] = [
                s[0, 0], s[1, 1], s[2, 2],
                s[0, 1], s[1, 2], s[0, 2],
            ]
        return sigma_voigt

    def stress_at_coords_batch(
        self,
        u: fem.Function,
        coords: np.ndarray,
    ) -> np.ndarray:
        """
        Faster batch evaluation: interpolates stress once and evaluates
        at all points simultaneously.

        Parameters
        ----------
        u      : displacement Function
        coords : (N, 3) ndarray  — evaluation points in µm

        Returns
        -------
        sigma_voigt : (N, 6) ndarray  [GPa]  (NaN for points outside mesh)
        """
        coords = np.asarray(coords, dtype=np.float64)
        N = len(coords)

        sigma_h = self._interpolate_stress(u)

        cell_candidates = geo.compute_collisions_points(self._bb_tree, coords)
        colliding = geo.compute_colliding_cells(
            self.mesh, cell_candidates, coords
        )

        sigma_voigt = np.full((N, 6), np.nan)

        # Group points by their colliding cell for batch eval
        cell_to_pts: dict[int, list[int]] = {}
        for i in range(N):
            cells_i = colliding.links(i)
            if len(cells_i) > 0:
                c = int(cells_i[0])
                cell_to_pts.setdefault(c, []).append(i)

        for cell, pt_idxs in cell_to_pts.items():
            pts_batch = coords[pt_idxs]
            cells_arr = np.full(len(pt_idxs), cell, dtype=np.int32)
            vals = sigma_h.eval(pts_batch, cells_arr)
            vals = np.reshape(vals, (-1, 9))  # normalise: always (M, 9)
            for local_j, global_i in enumerate(pt_idxs):
                s = vals[local_j].reshape(3, 3)
                sigma_voigt[global_i] = [
                    s[0, 0], s[1, 1], s[2, 2],
                    s[0, 1], s[1, 2], s[0, 2],
                ]
        return sigma_voigt


# ---------------------------------------------------------------------------
# Convenience: uniform pressure test
# ---------------------------------------------------------------------------
def _test_uniform_pressure(mesh_path: str = "mesh/anvil_preview.msh") -> None:
    """
    Apply a uniform normal pressure P=1 GPa on the sample chamber and
    print the resulting D_g values at the mesh centroid of the culet.
    """
    import sys
    from .nv_coupling import build_coupling_matrices, dg_from_stress_voigt

    solver = ForwardSolver(mesh_path, verbose=True)
    _, M_list = build_coupling_matrices()

    # Build a uniform traction: t = P * (-e_z) on sample chamber
    # (compressive, pushing inward toward the diamond)
    P = 1.0  # GPa
    x = ufl.SpatialCoordinate(solver.mesh)
    t_uniform = ufl.as_vector([0.0, 0.0, -P])

    print("\nSolving for uniform pressure 1 GPa on sample chamber ...")
    u = solver.solve_traction(t_uniform, culet_tags=(TAG_CULET_SAMPLE,))
    print("Solved.")

    # Evaluate stress at the culet centre
    pts = np.array([[0.0, 0.0, 0.0]])
    sig = solver.stress_at_coords_batch(u, pts)
    print(f"sigma at (0,0,0) [GPa]: {np.round(sig[0], 4)}")
    for g, M in enumerate(M_list):
        D = dg_from_stress_voigt(sig, M)[0]
        print(f"  D_{g} = {D:.4f} GHz")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "mesh/anvil_preview.msh"
    _test_uniform_pressure(path)
