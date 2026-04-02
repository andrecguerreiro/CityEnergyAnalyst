"""
Interactive workflow1 building viewer with dropdown selection.

Loads workflow1 geometry pickles and lets the user switch buildings in a GUI.
"""

from __future__ import annotations

import argparse
import os
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from py4design.py3dmodel.fetch import points_frm_occface

from cea.resources.radiation.geometry_generator import BuildingGeometry


WORKFLOW_1 = "workflow1_geometry_generator"
DEFAULT_SCENARIO = r"C:\Users\Andre\cea-scenarios\test-case"
DEFAULT_COMPARISON_ROOT = r"C:\Users\Andre\cea-scenarios\test-case\outputs\data\roof-workflow-comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an interactive workflow1 3D building viewer with dropdown selection."
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="Path to CEA scenario. Used for default comparison-root if omitted.",
    )
    parser.add_argument(
        "--comparison-root",
        default="",
        help=(
            "Path to one workflow comparison folder containing workflow1_geometry_generator. "
            "Default: <scenario>/outputs/data/roof-workflow-comparison."
        ),
    )
    parser.add_argument(
        "--zone-pickle-dir",
        default="",
        help=(
            "Optional direct path to workflow1 zone geometry pickle directory. "
            "If provided, this takes precedence over --comparison-root/--scenario."
        ),
    )
    return parser.parse_args()


def resolve_comparison_root(args: argparse.Namespace) -> str:
    if args.comparison_root.strip():
        return os.path.abspath(args.comparison_root.strip())
    if args.scenario.strip():
        return os.path.join(
            os.path.abspath(args.scenario.strip()),
            "outputs",
            "data",
            "roof-workflow-comparison",
        )
    return DEFAULT_COMPARISON_ROOT


def workflow1_zone_pickle_dir(comparison_root: str) -> str:
    return os.path.join(comparison_root, WORKFLOW_1, "radiance_geometry_pickle", "zone")


def resolve_zone_pickle_dir(args: argparse.Namespace) -> str:
    if args.zone_pickle_dir.strip():
        return os.path.abspath(args.zone_pickle_dir.strip())
    comparison_root = resolve_comparison_root(args)
    return workflow1_zone_pickle_dir(comparison_root)


def list_buildings(zone_pickle_dir: str) -> list[str]:
    if not os.path.isdir(zone_pickle_dir):
        raise NotADirectoryError(f"Workflow1 zone pickle folder does not exist: {zone_pickle_dir}")
    names = []
    for name in os.listdir(zone_pickle_dir):
        path = os.path.join(zone_pickle_dir, name)
        if os.path.isfile(path):
            names.append(name)
    if not names:
        raise FileNotFoundError(f"No building geometry pickles found in: {zone_pickle_dir}")
    return sorted(names)


def face_to_points_xyz(face: Any) -> np.ndarray | None:
    if isinstance(face, np.ndarray):
        points = np.asarray(face, dtype=float)
    elif isinstance(face, (list, tuple)) and len(face) > 0:
        points = np.asarray(face, dtype=float)
    else:
        points = np.asarray(points_frm_occface(face), dtype=float)

    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 3:
        return None
    return points[:, :3]


def occ_faces_to_polygons_xyz(faces: list[Any] | None) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    if not faces:
        return polygons
    for face in faces:
        points = face_to_points_xyz(face)
        if points is not None:
            polygons.append(points)
    return polygons


def polygon_centroid(points_xyz: np.ndarray) -> np.ndarray:
    if len(points_xyz) >= 2 and np.allclose(points_xyz[0], points_xyz[-1]):
        return points_xyz[:-1].mean(axis=0)
    return points_xyz.mean(axis=0)


def add_surfaces(ax, polygons: list[np.ndarray], color: str, alpha: float) -> None:
    if not polygons:
        return
    collection = Poly3DCollection(
        polygons, facecolor=color, edgecolor="black", linewidth=0.4, alpha=alpha
    )
    ax.add_collection3d(collection)


def collect_bounds(polygons: list[np.ndarray], roof_centroids: np.ndarray) -> dict[str, tuple[float, float] | float]:
    sets: list[np.ndarray] = []
    if polygons:
        sets.append(np.vstack(polygons))
    if len(roof_centroids):
        sets.append(roof_centroids)

    if not sets:
        return {
            "xlim": (0.0, 1.0),
            "ylim": (0.0, 1.0),
            "zlim": (0.0, 1.0),
            "arrow_length": 1.0,
        }

    xyz = np.vstack(sets)
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    centre = 0.5 * (min_xyz + max_xyz)
    span = float(np.max(max_xyz - min_xyz))
    if span <= 0:
        span = 1.0
    half = 0.6 * span
    return {
        "xlim": (float(centre[0] - half), float(centre[0] + half)),
        "ylim": (float(centre[1] - half), float(centre[1] + half)),
        "zlim": (float(centre[2] - half), float(centre[2] + half)),
        "arrow_length": max(0.08 * span, 1.0),
    }


def load_building_geometry(zone_pickle_dir: str, building: str) -> BuildingGeometry:
    path = os.path.join(zone_pickle_dir, building)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Building geometry pickle not found: {path}")
    return BuildingGeometry.load(path)


def render_building(ax, zone_pickle_dir: str, building: str) -> None:
    geometry = load_building_geometry(zone_pickle_dir, building)
    walls = occ_faces_to_polygons_xyz(getattr(geometry, "walls", []))
    roofs = occ_faces_to_polygons_xyz(getattr(geometry, "roofs", []))
    windows = occ_faces_to_polygons_xyz(getattr(geometry, "windows", []))

    roof_centroids = np.array([polygon_centroid(p) for p in roofs], dtype=float) if roofs else np.empty((0, 3))
    roof_normals = np.asarray(getattr(geometry, "normals_roofs", []), dtype=float)
    if roof_normals.ndim == 1 and len(roof_normals) > 0:
        roof_normals = roof_normals.reshape(1, -1)
    if roof_normals.ndim != 2 or roof_normals.shape[1] < 3:
        roof_normals = np.empty((0, 3), dtype=float)
    else:
        roof_normals = roof_normals[:, :3]

    n = min(len(roof_centroids), len(roof_normals))
    if n > 0:
        roof_centroids = roof_centroids[:n]
        norms = np.linalg.norm(roof_normals[:n], axis=1)
        norms[norms == 0.0] = 1.0
        roof_normals = roof_normals[:n] / norms[:, None]
    else:
        roof_centroids = np.empty((0, 3), dtype=float)
        roof_normals = np.empty((0, 3), dtype=float)

    ax.clear()
    add_surfaces(ax, walls, color="lightgrey", alpha=0.25)
    add_surfaces(ax, roofs, color="steelblue", alpha=0.5)
    add_surfaces(ax, windows, color="gold", alpha=0.35)

    bounds = collect_bounds(walls + roofs + windows, roof_centroids)
    if len(roof_centroids) and len(roof_normals):
        ax.quiver(
            roof_centroids[:, 0],
            roof_centroids[:, 1],
            roof_centroids[:, 2],
            roof_normals[:, 0],
            roof_normals[:, 1],
            roof_normals[:, 2],
            length=float(bounds["arrow_length"]),
            color="royalblue",
            normalize=True,
            linewidth=1.2,
        )

    ax.set_xlim(*bounds["xlim"])
    ax.set_ylim(*bounds["ylim"])
    ax.set_zlim(*bounds["zlim"])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_box_aspect((1.0, 1.0, 0.8))
    ax.view_init(elev=24, azim=-60)
    ax.set_title(f"Workflow1 - {building}")


def open_viewer(zone_pickle_dir: str) -> None:
    buildings = list_buildings(zone_pickle_dir)

    root = tk.Tk()
    root.title("CEA Workflow1 3D Building Viewer")
    root.geometry("1200x850")

    controls = ttk.Frame(root, padding=8)
    controls.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(controls, text="Building:").pack(side=tk.LEFT, padx=(0, 8))
    selected_building = tk.StringVar(value=buildings[0])

    combo = ttk.Combobox(
        controls, textvariable=selected_building, values=buildings, state="readonly", width=30
    )
    combo.pack(side=tk.LEFT)

    figure = Figure(figsize=(11, 7), dpi=100)
    ax = figure.add_subplot(111, projection="3d")
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    status_text = tk.StringVar(value=f"Zone pickle dir: {zone_pickle_dir}")
    ttk.Label(root, textvariable=status_text, padding=(8, 4)).pack(side=tk.BOTTOM, fill=tk.X)

    def refresh(*_):
        building = selected_building.get().strip()
        try:
            render_building(ax, zone_pickle_dir, building)
            status_text.set(f"Workflow1 building loaded: {building}")
            canvas.draw_idle()
        except Exception as exc:
            status_text.set(f"Failed to load {building}: {exc}")

    combo.bind("<<ComboboxSelected>>", refresh)
    refresh()
    root.mainloop()


def main() -> None:
    args = parse_args()
    zone_pickle_dir = resolve_zone_pickle_dir(args)
    print(f"Opening workflow1 viewer from: {zone_pickle_dir}")
    open_viewer(zone_pickle_dir)


if __name__ == "__main__":
    main()
