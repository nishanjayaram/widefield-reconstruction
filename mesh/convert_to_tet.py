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

    # Step 2: Split every prism6 into 3 tet4 by direct element manipulation.
    # Gmsh's SubdivisionAlgorithm+refine does not reliably convert prisms —
    # it only refines elements in-place. We do it manually per entity using
    # removeElements + addElementsByType.
    print("Splitting prism6 → 3 tet4 per 3D entity ...")

    import numpy as np

    def _prism6_to_3tet4(node6):
        """Split one prism6 [n0..n5] into 3 tet4 using diagonal decomposition."""
        n0, n1, n2, n3, n4, n5 = node6
        return [[n0, n1, n2, n3],
                [n1, n2, n3, n4],
                [n2, n3, n4, n5]]

    # Each 3D entity may contain prism6 (type=6) and/or tet4 (type=4).
    entities_3d = [tag for (dim, tag) in gmsh.model.getEntities(3)]
    max_tag = int(gmsh.model.mesh.getMaxElementTag())

    for vol_tag in entities_3d:
        etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(3, vol_tag)

        total_prisms = 0
        total_new_tets = 0
        for etype, elem_tags, elem_nodes in zip(etypes, etags_list, enodes_list):
            if etype != 6:
                continue   # not a prism6, leave it

            total_prisms += len(elem_tags)
            pnodes = elem_nodes.reshape(len(elem_tags), 6)

            # Build new tet4 connectivity
            new_t, new_n = [], []
            for pn in pnodes:
                for tet in _prism6_to_3tet4(pn.tolist()):
                    max_tag += 1
                    new_t.append(max_tag)
                    new_n.extend(tet)
            total_new_tets += len(new_t)

            # Remove old prism6 elements
            gmsh.model.mesh.removeElements(3, vol_tag, elem_tags.tolist())

            # Add new tet4 elements (type=4, 4 nodes each) to same entity
            gmsh.model.mesh.addElementsByType(
                vol_tag, 4,
                new_t,
                new_n,
            )

        if total_prisms > 0:
            print(f"  entity {vol_tag}: split {total_prisms} prisms → {total_new_tets} tets")

    # Also split any quad4 (type=3) surface elements into 2 tri3 (type=2).
    # These arise from the rectangular prism side-faces on external boundaries.
    # DOLFINx requires a single element type per topological dimension.
    print("Splitting quad4 surface elements → 2 tri3 ...")
    total_quads = 0
    for surf_tag in [tag for (dim, tag) in gmsh.model.getEntities(2)]:
        etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(2, surf_tag)
        for etype, elem_tags, elem_nodes in zip(etypes, etags_list, enodes_list):
            if etype != 3:   # 3 = quad4
                continue
            n_quad = len(elem_tags)
            total_quads += n_quad
            qnodes = elem_nodes.reshape(n_quad, 4)
            new_t, new_n = [], []
            for qn in qnodes:
                n0, n1, n2, n3 = qn.tolist()
                for tri in [[n0, n1, n2], [n0, n2, n3]]:
                    max_tag += 1
                    new_t.append(max_tag)
                    new_n.extend(tri)
            gmsh.model.mesh.removeElements(2, surf_tag, elem_tags.tolist())
            gmsh.model.mesh.addElementsByType(surf_tag, 2, new_t, new_n)
    if total_quads > 0:
        print(f"  split {total_quads} quad4 → {2*total_quads} tri3")

    # Step 3: Promote to quadratic (order 2) — Gmsh adds midpoint nodes on all edges
    print("Promoting to quadratic elements (order 2) ...")
    gmsh.model.mesh.setOrder(2)

    # Verify
    etypes_after, etags_after, _ = gmsh.model.mesh.getElements(dim=3)
    print(f"  3-D element types after conversion: {sorted(set(etypes_after.tolist()))}")
    # Expected: only [11] (tet10)

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
