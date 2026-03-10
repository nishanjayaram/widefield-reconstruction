"""
Convert a Gmsh .msh file (MSH 4.1, all-tet P2) to XDMF+HDF5 for DOLFINx.

DOLFINx's gmshio.read_from_msh fails on meshes converted from COMSOL NAS files
(MPI_ERR_RANK crash due to incomplete entity topology in the Gmsh model).
This script bypasses Gmsh entirely: reads the .msh with meshio (which handles
the raw file format fine), then writes two XDMF files DOLFINx can load natively.

Output
------
  <stem>_mesh.xdmf / .h5   — volume mesh (tet10) + cell tags
  <stem>_facets.xdmf / .h5 — surface mesh (tri6) + facet tags

Physical group tags (must match TAG_* in forward.py):
  Volume  tag 1  "bulk"
  Surface tag 1  "culet_sample"
  Surface tag 2  "culet_gasket"
  Surface tag 3  "table"
  Surface tag 4  "facet"

Usage
-----
    conda run -n dac-recon python mesh/convert_msh_to_xdmf.py \\
        --input  mesh/medium_mesh_COMSOL_tet.msh \\
        --output mesh/medium_mesh_COMSOL_tet.xdmf

Then load in forward.py with XDMFFile instead of gmshio.read_from_msh.
"""

import argparse
from pathlib import Path

import meshio
import numpy as np


def convert(input_path: str, output_path: str) -> None:
    import os
    input_path  = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    stem = output_path.with_suffix("")  # strip .xdmf if given

    # -----------------------------------------------------------------------
    # Read MSH
    # -----------------------------------------------------------------------
    print(f"Reading {input_path} ...")
    m = meshio.read(str(input_path))
    print(f"  {len(m.points):,} nodes")

    if "gmsh:physical" not in m.cell_data:
        raise RuntimeError("No 'gmsh:physical' tags in mesh — physical groups missing.")

    # -----------------------------------------------------------------------
    # Split into volume (tet10) and surface (tri6) blocks
    # -----------------------------------------------------------------------
    tet_cells, tet_tags = [], []
    tri_cells, tri_tags = [], []

    for block, tag_block in zip(m.cells, m.cell_data["gmsh:physical"]):
        if block.type == "tetra10":
            tet_cells.append(block.data)
            tet_tags.append(tag_block)
        elif block.type == "triangle6":
            tri_cells.append(block.data)
            tri_tags.append(tag_block)
        else:
            print(f"  Skipping element type: {block.type}")

    if not tet_cells:
        raise RuntimeError("No tet10 volume elements found.")

    tet_cells = np.vstack(tet_cells)
    tet_tags  = np.concatenate(tet_tags).astype(np.int32)
    print(f"  tet10 cells: {len(tet_cells):,}, tags: {np.unique(tet_tags).tolist()}")

    if tri_cells:
        tri_cells = np.vstack(tri_cells)
        tri_tags  = np.concatenate(tri_tags).astype(np.int32)
        print(f"  tri6  cells: {len(tri_cells):,}, tags: {np.unique(tri_tags).tolist()}")
    else:
        print("  WARNING: no tri6 surface elements — facet tags will be empty.")
        tri_cells = np.empty((0, 6), dtype=np.int64)
        tri_tags  = np.empty((0,),   dtype=np.int32)

    # -----------------------------------------------------------------------
    # Write volume and facet XDMF
    # Change into the output directory so meshio writes the HDF5 reference as
    # a plain filename.  DOLFINx resolves the HDF5 path relative to CWD when
    # loading, so mesh + h5 must be in the same directory as CWD at load time
    # (or use absolute output paths and cd before loading).
    # -----------------------------------------------------------------------
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_dir = Path.cwd()
    os.chdir(out_dir)

    mesh_xdmf  = Path(str(stem.name) + "_mesh.xdmf")
    facet_xdmf = Path(str(stem.name) + "_facets.xdmf")

    vol_mesh = meshio.Mesh(
        points=m.points,
        cells=[("tetra10", tet_cells)],
        cell_data={"cell_tags": [tet_tags]},
    )
    meshio.write(str(mesh_xdmf), vol_mesh)
    print(f"Written: {out_dir / mesh_xdmf}")

    fac_mesh = meshio.Mesh(
        points=m.points,
        cells=[("triangle6", tri_cells)],
        cell_data={"facet_tags": [tri_tags]},
    )
    meshio.write(str(facet_xdmf), fac_mesh)
    print(f"Written: {out_dir / facet_xdmf}")

    os.chdir(orig_dir)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all-tet P2 .msh to XDMF+HDF5 for DOLFINx")
    parser.add_argument("--input",  required=True,  help="Input .msh file")
    parser.add_argument("--output", required=True,  help="Output .xdmf path (stem used)")
    args = parser.parse_args()
    convert(args.input, args.output)
