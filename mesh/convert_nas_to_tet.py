"""
Convert a COMSOL NAS (Nastran) mesh to a DOLFINx-compatible all-tet P2 .msh file.

Steps
-----
1. Open the .nas file with Gmsh.
2. Identify boundary surfaces by bounding box and assign physical groups:
     Volume tag 1  "bulk"
     Surface tag 1 "culet_sample"  — elliptic sample chamber (z≈0, small extent)
     Surface tag 2 "culet_gasket"  — annular gasket          (z≈0, full culet extent)
     Surface tag 3 "table"         — clamped table face
     Surface tag 4 "facet"         — traction-free conical sidewall
3. Split every prism15 (CPENTA15) → 3 tet10 (CTETRA10) via linear downgrade,
   prism6→3×tet4 split, then re-promotion to P2.
4. Split any quad surface elements → 2 tri3 (same as convert_to_tet.py).
5. Write output as MSH 2.2 format.

Usage
-----
    conda run -n dac-recon python mesh/convert_nas_to_tet.py \\
        --input  mesh/medium_mesh_COMSOL.nas \\
        --output mesh/medium_mesh_COMSOL_tet.msh
"""

import argparse
import os
import gmsh
import numpy as np


# ---------------------------------------------------------------------------
# Physical group tags — must match forward.py TAG_* constants
# ---------------------------------------------------------------------------
TAG_BULK          = 1
TAG_CULET_SAMPLE  = 1   # surface
TAG_CULET_GASKET  = 2
TAG_TABLE         = 3
TAG_FACET         = 4


def _classify_surfaces(tol: float = 1.0):
    """
    Classify all 2-D entities into culet_sample / culet_gasket / table / facet
    by bounding-box geometry.

    Returns dict mapping category -> list of entity tags.
    """
    ELL_A = 72.0    # sample chamber semi-major axis (µm)
    ELL_B = 50.0

    culet_sample, culet_gasket, table, facet = [], [], [], []

    for (dim, tag) in gmsh.model.getEntities(dim=2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bb

        if abs(zmin) < tol and abs(zmax) < tol:
            # On the culet plane (z = 0).
            extent = max(xmax - xmin, ymax - ymin)
            # sample chamber is bounded by ~2*ELL_A=144 µm; gasket by ~2*R1=500 µm.
            if extent < 2.0 * max(ELL_A, ELL_B) + tol:
                culet_sample.append(tag)
            else:
                culet_gasket.append(tag)

        elif abs(zmax - zmin) < tol:
            # Horizontal face not at z=0 → table or internal.
            # Table is the most negative z face; internal cap interfaces are ignored.
            table.append((zmin, tag))   # sort by z later

        else:
            # Non-horizontal, non-culet → conical sidewall
            facet.append(tag)

    # Table = the most-negative horizontal face
    if table:
        table.sort()        # sort by z ascending → most negative first
        z_table = table[0][0]
        table_tags = [t for z, t in table if abs(z - z_table) < tol]
    else:
        table_tags = []

    return {
        "culet_sample": culet_sample,
        "culet_gasket": culet_gasket,
        "table":        table_tags,
        "facet":        facet,
    }


def _prism6_to_3tet4(node6):
    """Split one prism6 [n0..n5] into 3 tet4 using diagonal decomposition."""
    n0, n1, n2, n3, n4, n5 = node6
    return [[n0, n1, n2, n3],
            [n1, n2, n3, n4],
            [n2, n3, n4, n5]]


def convert(input_path: str, output_path: str, verbose: bool = True) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

    print(f"Reading {input_path} ...")
    gmsh.open(input_path)

    nodes, _, _ = gmsh.model.mesh.getNodes(-1, -1, includeBoundary=True)
    print(f"  Nodes before conversion: {len(nodes):,}")

    # ------------------------------------------------------------------
    # Step 1: Classify surfaces and assign physical groups
    # ------------------------------------------------------------------
    print("Classifying surfaces by bounding box ...")
    groups = _classify_surfaces()

    for name, tags in groups.items():
        print(f"  {name}: {tags}")

    if not groups["culet_sample"]:
        raise RuntimeError("Could not identify culet_sample surface")
    if not groups["culet_gasket"]:
        raise RuntimeError("Could not identify culet_gasket surface")
    if not groups["table"]:
        raise RuntimeError("Could not identify table surface")

    # Assign physical groups to surfaces
    gmsh.model.addPhysicalGroup(2, groups["culet_sample"], tag=TAG_CULET_SAMPLE,
                                name="culet_sample")
    gmsh.model.addPhysicalGroup(2, groups["culet_gasket"], tag=TAG_CULET_GASKET,
                                name="culet_gasket")
    gmsh.model.addPhysicalGroup(2, groups["table"],        tag=TAG_TABLE,
                                name="table")
    gmsh.model.addPhysicalGroup(2, groups["facet"],        tag=TAG_FACET,
                                name="facet")

    # All volumes → bulk
    all_vols = [tag for (dim, tag) in gmsh.model.getEntities(dim=3)]
    gmsh.model.addPhysicalGroup(3, all_vols, tag=TAG_BULK, name="bulk")

    # ------------------------------------------------------------------
    # Step 2: Downgrade to linear (order 1) — removes midpoint nodes
    # ------------------------------------------------------------------
    print("Downgrading to linear elements (order 1) ...")
    gmsh.model.mesh.setOrder(1)

    etypes, _, _ = gmsh.model.mesh.getElements(dim=3)
    print(f"  3-D element types after downgrade: {sorted(set(etypes.tolist()))}")
    # Expected: 4 (tet4) and/or 6 (prism6)

    # ------------------------------------------------------------------
    # Step 3: Split prism6 → 3 tet4
    # ------------------------------------------------------------------
    print("Splitting prism6 → 3 tet4 ...")
    max_tag = int(gmsh.model.mesh.getMaxElementTag())
    for vol_tag in all_vols:
        etypes_v, etags_list, enodes_list = gmsh.model.mesh.getElements(3, vol_tag)
        for etype, elem_tags, elem_nodes in zip(etypes_v, etags_list, enodes_list):
            if etype != 6:
                continue
            pnodes = elem_nodes.reshape(len(elem_tags), 6)
            new_t, new_n = [], []
            for pn in pnodes:
                for tet in _prism6_to_3tet4(pn.tolist()):
                    max_tag += 1
                    new_t.append(max_tag)
                    new_n.extend(tet)
            gmsh.model.mesh.removeElements(3, vol_tag, elem_tags.tolist())
            gmsh.model.mesh.addElementsByType(vol_tag, 4, new_t, new_n)
            print(f"  entity {vol_tag}: split {len(elem_tags)} prisms → {len(new_t)} tets")

    # ------------------------------------------------------------------
    # Step 4: Split quad4 surface elements → 2 tri3
    # ------------------------------------------------------------------
    print("Splitting quad4 → 2 tri3 (if any) ...")
    total_quads = 0
    for surf_tag in [tag for (dim, tag) in gmsh.model.getEntities(2)]:
        etypes_s, etags_list_s, enodes_list_s = gmsh.model.mesh.getElements(2, surf_tag)
        for etype, elem_tags, elem_nodes in zip(etypes_s, etags_list_s, enodes_list_s):
            if etype != 3:
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
    if total_quads:
        print(f"  split {total_quads} quad4 → {2*total_quads} tri3")
    else:
        print("  none found")

    # ------------------------------------------------------------------
    # Step 5: Promote back to quadratic P2
    # ------------------------------------------------------------------
    print("Promoting to P2 (order 2) ...")
    gmsh.model.mesh.setOrder(2)

    etypes_after, _, _ = gmsh.model.mesh.getElements(dim=3)
    print(f"  3-D element types after P2: {sorted(set(etypes_after.tolist()))}")
    # Expected: only [11] (tet10)

    nodes_after, _, _ = gmsh.model.mesh.getNodes(-1, -1, includeBoundary=True)
    print(f"  Total nodes after conversion: {len(nodes_after):,}")

    # ------------------------------------------------------------------
    # Step 6: Write output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(output_path)
    print(f"Written: {output_path}")

    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert COMSOL NAS mesh to all-tet P2 .msh for DOLFINx")
    parser.add_argument("--input",  default="mesh/medium_mesh_COMSOL.nas",
                        help="Input COMSOL .nas file")
    parser.add_argument("--output", default="mesh/medium_mesh_COMSOL_tet.msh",
                        help="Output all-tet P2 .msh file")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    convert(args.input, args.output, verbose=not args.quiet)
