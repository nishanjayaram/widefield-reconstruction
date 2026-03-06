#!/usr/bin/env python3
"""
Visualize diamond anvil mesh. Saves mesh/mesh_preview.png.

Usage:
    python mesh/visualize_mesh.py [path/to/mesh.msh]
    (defaults to mesh/anvil_preview.msh if it exists, else mesh/anvil.msh)
"""

import pyvista as pv
import meshio
import numpy as np
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_mesh():
    for name in ("anvil_preview.msh", "anvil.msh"):
        path = os.path.join(REPO_ROOT, "mesh", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No mesh file found in mesh/. Run build_mesh.py first.")

MESH_PATH = sys.argv[1] if len(sys.argv) > 1 else find_mesh()

# Physical tag → colour / label
SURF_COLOURS = {1: "#4C9BE8", 2: "#F5A623", 3: "#7ED321", 4: "#CCCCCC"}
SURF_LABELS  = {1: "culet_sample", 2: "culet_gasket", 3: "table", 4: "facet"}


def build_surface_meshes(m: meshio.Mesh) -> dict:
    """Return {phys_tag: pv.PolyData} for each tagged triangle surface."""
    pts = m.points
    phys_data = m.cell_data.get("gmsh:physical", [])
    tag_cells: dict[int, list] = {}
    for block, tags in zip(m.cells, phys_data):
        if block.type not in ("triangle", "triangle6"):
            continue
        for cell, tag in zip(block.data, tags):
            tag_cells.setdefault(int(tag), []).append(cell[:3])  # linear corners
    result = {}
    for tag, cells in tag_cells.items():
        arr = np.array(cells)
        faces = np.hstack([np.full((len(arr), 1), 3), arr]).flatten()
        result[tag] = pv.PolyData(pts, faces)
    return result


def build_volume_grid(m: meshio.Mesh):
    """Return pv.UnstructuredGrid of tetrahedral cells."""
    pts = m.points
    phys_data = m.cell_data.get("gmsh:physical", [])
    all_cells, all_tags = [], []
    for block, tags in zip(m.cells, phys_data):
        if block.type not in ("tetra", "tetra10"):
            continue
        for cell, tag in zip(block.data, tags):
            all_cells.append(cell[:4])
            all_tags.append(int(tag))
    if not all_cells:
        return None
    n = len(all_cells)
    cells_vtk = np.hstack([np.full((n, 1), 4), np.array(all_cells)]).flatten()
    celltypes = np.full(n, pv.CellType.TETRA)
    grid = pv.UnstructuredGrid(cells_vtk, celltypes, pts)
    grid.cell_data["phys_tag"] = np.array(all_tags)
    return grid


def main():
    pv.set_plot_theme("document")
    print(f"Loading {MESH_PATH} ...")
    m = meshio.read(MESH_PATH)
    print(f"  {len(m.points):,} nodes")

    surfs = build_surface_meshes(m)
    vol   = build_volume_grid(m)
    if vol:
        print(f"  {vol.n_cells:,} tetrahedra")
    print(f"  Surface groups found: { {SURF_LABELS.get(t,t): surfs[t].n_cells for t in sorted(surfs)} }")

    # ------------------------------------------------------------------ #
    # Figure: 2 × 2 panels                                               #
    # ------------------------------------------------------------------ #
    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1800, 1400))
    pl.set_background("white")

    # ── 0,0  Full 3D exterior, coloured by BC group ────────────────────
    pl.subplot(0, 0)
    pl.add_text("Exterior BCs\n(blue=culet_sample  orange=gasket  green=table  grey=facet)",
                font_size=8, color="black")
    for tag in sorted(surfs):
        pl.add_mesh(surfs[tag], color=SURF_COLOURS[tag],
                    opacity=0.9 if tag != 4 else 0.45,
                    show_edges=False, smooth_shading=True)
    pl.camera_position = [
        (3000, -5000, -500),   # camera
        (0, 0, -1250),          # focal
        (0, 0, 1),              # up
    ]

    # ── 0,1  Culet face close-up (top-down, z=0 plane) ────────────────
    pl.subplot(0, 1)
    pl.add_text("Culet face (z = 0)\nBlue = sample chamber (72×50 µm ellipse)\nOrange = gasket annulus",
                font_size=8, color="black")
    for tag in (2, 1):   # gasket first (background), sample on top
        if tag in surfs:
            pl.add_mesh(surfs[tag], color=SURF_COLOURS[tag],
                        show_edges=True, edge_color="#555555", line_width=0.4)
    pl.view_xy()
    pl.camera.position   = (0, 0, 600)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up         = (0, 1, 0)
    pl.reset_camera(bounds=(-260, 260, -260, 260, -1, 1))

    # ── 1,0  Y=0 axial cross-section (whole anvil) ────────────────────
    pl.subplot(1, 0)
    pl.add_text("Axial cross-section (y ≈ 0)\nshows frustum shape and mesh grading",
                font_size=8, color="black")
    if vol is not None:
        clipped = vol.clip("y", origin=(0, 0, 0), invert=False)
        pl.add_mesh(clipped, scalars="phys_tag", cmap="tab10",
                    show_edges=True, edge_color="#888888", line_width=0.15,
                    clim=[1, 4], show_scalar_bar=False)
    pl.view_xz()
    pl.camera.position    = (5000, -1, -1250)
    pl.camera.focal_point = (0, 0, -1250)
    pl.camera.up          = (0, 0, 1)
    pl.reset_camera()

    # ── 1,1  Near-culet zoom cross-section (z = 0 to −200 µm) ─────────
    pl.subplot(1, 1)
    pl.add_text("Near-culet cross-section (z = 0 to −200 µm)\nshows mesh refinement near sample chamber",
                font_size=8, color="black")
    if vol is not None:
        near = vol.clip_box((-300, 300, -0.5, 0.5, -200, 5), invert=False)
        pl.add_mesh(near, scalars="phys_tag", cmap="tab10",
                    show_edges=True, edge_color="#666666", line_width=0.4,
                    clim=[1, 4], show_scalar_bar=False)
    for tag in (1, 2):
        if tag in surfs:
            pl.add_mesh(surfs[tag], color=SURF_COLOURS[tag],
                        show_edges=True, edge_color="k", line_width=0.6,
                        opacity=0.6)
    pl.view_xz()
    pl.camera.position    = (700, -1, -100)
    pl.camera.focal_point = (0, 0, -100)
    pl.camera.up          = (0, 0, 1)
    pl.reset_camera(bounds=(-300, 300, -1, 1, -205, 5))

    out = os.path.join(REPO_ROOT, "mesh", "mesh_preview.png")
    pl.screenshot(out, transparent_background=False)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
