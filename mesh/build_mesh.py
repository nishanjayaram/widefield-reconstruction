#!/usr/bin/env python3
"""
Build the diamond anvil cell frustum geometry and mesh using Gmsh.

All dimensions in micrometres (µm).

Geometry
--------
  Culet face  :  z = 0,    radius r1 = 250 µm  (smaller face, NV surface)
  Table face  :  z = -h,   radius r2 = 1750 µm (clamped face)
  Sidewall    :  linear cone between culet and table
  h           :  2500 µm = 2.5 mm (specified directly)

  Sample chamber: ellipse on culet, semi-axes ell_a × ell_b (centred at origin)

Physical groups written to the mesh
-------------------------------------
  Volume  tag 1  "bulk"          — entire frustum interior
  Surface tag 1  "culet_sample"  — elliptic sample chamber (z = 0, inside ellipse)
  Surface tag 2  "culet_gasket"  — annular gasket region   (z = 0, outside ellipse)
  Surface tag 3  "table"         — clamped table face (z = -h)
  Surface tag 4  "facet"         — traction-free conical sidewall

Mesh sizing strategy
--------------------
  Sample chamber (cap)       :  lc_sample  = 1.0 µm
  Gasket annulus (cap)       :  lc_gasket  = 1.2 µm  (targets ~300k total vertices)
  Cap thickness              :  tcap = 2 µm  (z = 0 to -2 µm)
  Bulk below cap             :  lc_table = 500 µm (very coarse)
  Element order              :  P2 quadratic tetrahedra (set after meshing)
"""

import gmsh
import numpy as np
import os

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
R1        = 250.0        # culet radius (µm)
R2        = 1750.0       # table radius (µm)
ELL_A     = 72.0         # sample chamber semi-major axis (µm)
ELL_B     = 50.0         # sample chamber semi-minor axis (µm)
H = 2500.0               # frustum height (µm) = 2.5 mm

# Mesh characteristic lengths
# Target: ~300k total vertices
# LC_GASKET drives most of the node count (dominates the cap volume).
# Empirically calibrated; adjust if node count is off.
LC_SAMPLE  = 1.0         # sample chamber surface (matches NV pixel size)
LC_GASKET  = 1.2         # gasket annulus + cap interior — fine throughout cap
LC_TABLE   = 500.0       # bulk below cap — deliberately coarse

# Cap layer: fine mesh from z=0 to z=-TCAP; bulk below is coarse
TCAP = 2.0               # cap thickness (µm)


def build_mesh(output_path: str = "mesh/anvil.msh", verbose: bool = True) -> None:
    print(f"Frustum height h = {H:.1f} µm")
    print(f"Output: {output_path}")

    gmsh.initialize()
    gmsh.model.add("diamond_anvil")
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

    occ = gmsh.model.occ

    # ------------------------------------------------------------------
    # 1. Frustum solid
    #    addCone(x, y, z,  dx, dy, dz,  r1_base, r2_tip)
    #    The first circular face is at (x,y,z) with radius r1_base.
    #    The second face is at (x+dx, y+dy, z+dz) with radius r2_tip.
    #    We put the table (large face) at z=-H and the culet (small) at z=0.
    # ------------------------------------------------------------------
    cone_tag = occ.addCone(0, 0, -H,   # centre of table face
                            0, 0,  H,   # axis vector pointing to culet
                            R2,         # table radius
                            R1)         # culet radius

    # ------------------------------------------------------------------
    # 2. Elliptic disk at z=0 to split the culet face into two regions.
    #    addDisk(xc, yc, zc, rx, ry) creates an elliptic surface in the
    #    z=zc plane.
    # ------------------------------------------------------------------
    ell_disk_tag = occ.addDisk(0, 0, 0, ELL_A, ELL_B)

    # ------------------------------------------------------------------
    # 3. BooleanFragments: embed the elliptic disk into the culet face.
    #    This splits the z=0 boundary face of the cone into:
    #      - inner elliptic surface  (sample chamber)
    #      - outer annular surface   (gasket)
    #    The cone volume is preserved as one entity.
    # ------------------------------------------------------------------
    out_dim_tags, out_map = occ.fragment(
        [(3, cone_tag)],
        [(2, ell_disk_tag)]
    )
    occ.synchronize()

    # ------------------------------------------------------------------
    # 4. Identify and tag surfaces
    # ------------------------------------------------------------------
    all_surfs = gmsh.model.getEntities(dim=2)

    culet_sample_tags = []
    culet_gasket_tags  = []
    table_tags         = []
    facet_tags         = []

    tol = 0.5  # µm tolerance for plane detection

    for (dim, tag) in all_surfs:
        bb = gmsh.model.getBoundingBox(dim, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bb

        if abs(zmin - 0.0) < tol and abs(zmax - 0.0) < tol:
            # On the culet plane (z = 0).
            # Both surfaces are centred at (0,0,0) so CoM won't distinguish them.
            # Use bounding-box extent instead:
            #   sample chamber  → bounded by ellipse, extent ≈ 2*ELL_A = 144 µm
            #   gasket annulus  → bounded by full culet circle, extent ≈ 2*R1 = 500 µm
            bbox_extent = max(xmax - xmin, ymax - ymin)
            if bbox_extent < 2.0 * max(ELL_A, ELL_B) + tol:
                culet_sample_tags.append(tag)
            else:
                culet_gasket_tags.append(tag)

        elif abs(zmin - (-H)) < tol and abs(zmax - (-H)) < tol:
            # On the table plane (z = -H)
            table_tags.append(tag)

        else:
            # Conical sidewall
            facet_tags.append(tag)

    if not culet_sample_tags:
        raise RuntimeError("Could not identify sample-chamber surface. Check geometry.")
    if not culet_gasket_tags:
        raise RuntimeError("Could not identify gasket-annulus surface. Check geometry.")
    if not table_tags:
        raise RuntimeError("Could not identify table surface. Check geometry.")

    # All volumes
    all_vols = [tag for (dim, tag) in gmsh.model.getEntities(dim=3)]

    # ------------------------------------------------------------------
    # 5. Physical groups
    # ------------------------------------------------------------------
    gmsh.model.addPhysicalGroup(3, all_vols,          tag=1, name="bulk")
    gmsh.model.addPhysicalGroup(2, culet_sample_tags, tag=1, name="culet_sample")
    gmsh.model.addPhysicalGroup(2, culet_gasket_tags, tag=2, name="culet_gasket")
    gmsh.model.addPhysicalGroup(2, table_tags,        tag=3, name="table")
    gmsh.model.addPhysicalGroup(2, facet_tags,        tag=4, name="facet")

    print(f"  culet_sample surfaces : {culet_sample_tags}")
    print(f"  culet_gasket surfaces : {culet_gasket_tags}")
    print(f"  table surfaces        : {table_tags}")
    print(f"  facet surfaces        : {facet_tags}")

    # ------------------------------------------------------------------
    # 6. Mesh size fields
    #
    #  Strategy:
    #    F1  Box field — LC_GASKET inside the cap layer (z ∈ [−TCAP, 0]),
    #        LC_TABLE outside.  Transition band = 5 µm below the cap.
    #        This makes the bulk coarse without the gradation artefacts
    #        that a Distance-based field produces.
    #    F2  Distance from the sample-chamber surface
    #    F3  Threshold on F2 — refines sample chamber to LC_SAMPLE.
    #        SizeMax = LC_TABLE so this field does NOT force a fine mesh
    #        outside the sample-chamber zone.
    #    F4  Min(F1, F3) — take the finest applicable size everywhere.
    # ------------------------------------------------------------------

    # F1: Box — fine cap, coarse bulk
    f1 = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f1, "VIn",       LC_GASKET)
    gmsh.model.mesh.field.setNumber(f1, "VOut",      LC_TABLE)
    gmsh.model.mesh.field.setNumber(f1, "XMin",     -(R1 + 20))
    gmsh.model.mesh.field.setNumber(f1, "XMax",      (R1 + 20))
    gmsh.model.mesh.field.setNumber(f1, "YMin",     -(R1 + 20))
    gmsh.model.mesh.field.setNumber(f1, "YMax",      (R1 + 20))
    gmsh.model.mesh.field.setNumber(f1, "ZMin",     -(TCAP + 1))
    gmsh.model.mesh.field.setNumber(f1, "ZMax",      1.0)
    gmsh.model.mesh.field.setNumber(f1, "Thickness", 5.0)

    # F2: distance from sample-chamber surface
    f2 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f2, "SurfacesList", culet_sample_tags)

    # F3: refine sample chamber to LC_SAMPLE; SizeMax = LC_TABLE so it
    #     doesn't force fine mesh away from the sample chamber
    f3 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f3, "InField",  f2)
    gmsh.model.mesh.field.setNumber(f3, "SizeMin",  LC_SAMPLE)
    gmsh.model.mesh.field.setNumber(f3, "SizeMax",  LC_TABLE)
    gmsh.model.mesh.field.setNumber(f3, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(f3, "DistMax",  30.0)

    # F4: take minimum of Box and sample-chamber Threshold
    f4 = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f4, "FieldsList", [f1, f3])

    gmsh.model.mesh.field.setAsBackgroundMesh(f4)

    # Prevent Gmsh from overriding the background field with entity sizes
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",    0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # ------------------------------------------------------------------
    # 7. Generate mesh
    # ------------------------------------------------------------------
    gmsh.option.setNumber("Mesh.Algorithm",   5)   # Delaunay (2-D)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay (3-D)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    print("Generating 3-D mesh (P1 first, then promoted to P2)...")
    gmsh.model.mesh.generate(3)

    # Promote to quadratic (P2) after generation for better quality
    gmsh.model.mesh.setOrder(2)

    # ------------------------------------------------------------------
    # 8. Write output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    gmsh.write(output_path)
    print(f"Mesh written to {output_path}")

    # Report element counts
    elem_types, _, _ = gmsh.model.mesh.getElements(dim=3)
    print(f"3-D element type tags: {elem_types}")

    gmsh.finalize()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build diamond anvil mesh with Gmsh")
    parser.add_argument("-o", "--output", default="mesh/anvil.msh",
                        help="Output .msh file path (default: mesh/anvil.msh)")
    parser.add_argument("--lc-sample", type=float, default=LC_SAMPLE,
                        help=f"Mesh size in sample chamber (default: {LC_SAMPLE} µm)")
    parser.add_argument("--lc-gasket", type=float, default=LC_GASKET,
                        help=f"Mesh size in gasket/cap region (default: {LC_GASKET} µm)")
    parser.add_argument("--lc-table", type=float, default=LC_TABLE,
                        help=f"Mesh size in bulk (default: {LC_TABLE} µm)")
    parser.add_argument("--tcap", type=float, default=TCAP,
                        help=f"Cap layer thickness (default: {TCAP} µm)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress Gmsh terminal output")
    args = parser.parse_args()

    LC_SAMPLE = args.lc_sample
    LC_GASKET = args.lc_gasket
    LC_TABLE  = args.lc_table
    TCAP      = args.tcap

    build_mesh(output_path=args.output, verbose=not args.quiet)
