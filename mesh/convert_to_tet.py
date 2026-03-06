"""
Convert the COMSOL mixed-element mesh (prism15 + tet10) to an all-tet P2 mesh
that DOLFINx can read.

The COMSOL NAS file uses 15-node quadratic triangular prisms (wedge15) in the
near-culet cap layer and 10-node tetrahedra (tet10) in the bulk.  DOLFINx does
not support wedge15 elements.

Strategy
--------
1. Open anvil_fine.msh (already tagged with physical groups).
2. Downgrade all elements to linear order (setOrder(1)).
3. Split every prism6 into 3 tetrahedra using Gmsh's subdivision algorithm
   (option Mesh.SubdivisionAlgorithm = 1).
4. Re-promote to quadratic order (setOrder(2)) so midpoint nodes are added.
5. Write the result as anvil_fine_tet.msh (MSH 2.2 format).

Output
------
  mesh/anvil_fine_tet.msh  — all-tet P2 mesh with the same physical groups.

Usage
-----
    python mesh/convert_to_tet.py [--input mesh/anvil_fine.msh] [--output mesh/anvil_fine_tet.msh]
"""

import argparse
import os
import gmsh


def convert(input_path: str, output_path: str, verbose: bool = True) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

    print(f"Reading {input_path} ...")
    gmsh.open(input_path)

    # Check element types before conversion
    elem_types_before = set()
    for dim in (2, 3):
        for etype, _, _ in [gmsh.model.mesh.getElements(dim=dim)]:
            elem_types_before.update(etype)
    print(f"  Element type tags before conversion: {sorted(elem_types_before)}")

    # Step 1: Downgrade to linear (order 1) — removes midpoint nodes
    print("Downgrading to linear elements (order 1) ...")
    gmsh.model.mesh.setOrder(1)

    # Step 2: Subdivide all prisms (wedge6) into tetrahedra (tet4).
    # SubdivisionAlgorithm=1 splits hexes and prisms into tets.
    # This must be applied via a mesh refine step after setting the option.
    print("Subdividing prisms → tetrahedra ...")
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.refine()   # This triggers subdivision

    # Step 3: Promote back to quadratic (order 2) — adds midpoint nodes
    print("Promoting to quadratic elements (order 2) ...")
    gmsh.model.mesh.setOrder(2)

    # Verify
    elem_types_after = []
    for etype, _, _ in [gmsh.model.mesh.getElements(dim=3)]:
        elem_types_after.extend(etype)
    print(f"  3-D element type tags after conversion: {sorted(set(elem_types_after))}")
    # Expected: only tag 11 (tet10)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_path)
    print(f"Written: {output_path}")

    # Node count
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(-1, -1, includeBoundary=True)
    print(f"  Total nodes: {len(node_tags):,}")

    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mixed mesh to all-tet P2")
    parser.add_argument("--input",  default="mesh/anvil_fine.msh",
                        help="Input .msh file (default: mesh/anvil_fine.msh)")
    parser.add_argument("--output", default="mesh/anvil_fine_tet.msh",
                        help="Output .msh file (default: mesh/anvil_fine_tet.msh)")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    convert(args.input, args.output, verbose=not args.quiet)
