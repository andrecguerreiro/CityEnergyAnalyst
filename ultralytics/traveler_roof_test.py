import sys
import csv
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QStackedWidget, QPushButton, QLineEdit,
                             QLabel, QFrame)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pyvista as pv
from pyvistaqt import QtInteractor
from pathlib import Path
from typing import Tuple, List, Optional
import math
import cv2
import tifffile

from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay

from utils import rqst_satelite_image
from ultralytics import YOLO

import re
from pyproj import Transformer

_WGS84_TO_TM06 = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)

def parse_lonlat_polygon(s: str) -> List[Tuple[float, float]]:
    """
    Parse string like:
    "(-9.136993,38.708859),(-9.137898,38.709030),..."
    Returns list of (lon, lat).
    """
    # Find all "(lon,lat)" pairs
    pairs = re.findall(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", s)
    if not pairs:
        raise ValueError("Could not parse polygon string. Expected '(lon,lat),(lon,lat),...'")

    pts = [(float(lon), float(lat)) for lon, lat in pairs]

    # If polygon isn't closed, close it
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    # Need at least 4 points including closure (triangle + closure)
    if len(pts) < 4:
        raise ValueError("Polygon must have at least 3 vertices (plus closure).")

    return pts


def polygon_centroid_lonlat(lonlat: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Compute centroid of a simple polygon using the standard planar formula.
    For small areas (a building roof), treating lon/lat as planar is OK.
    Input must be closed (first == last).
    Returns (lon_c, lat_c).
    """
    # Using x=lon, y=lat
    A = 0.0
    Cx = 0.0
    Cy = 0.0

    for i in range(len(lonlat) - 1):
        x0, y0 = lonlat[i]
        x1, y1 = lonlat[i + 1]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross

    A *= 0.5
    if abs(A) < 1e-15:
        # Degenerate polygon; fall back to average
        xs = [p[0] for p in lonlat[:-1]]
        ys = [p[1] for p in lonlat[:-1]]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return (Cx, Cy)



def lonlat_to_tm06(lon: float, lat: float) -> Tuple[float, float]:
    x, y = _WGS84_TO_TM06.transform(lon, lat)
    return x, y

def parse_wkt_polygon(wkt: str) -> List[List[Tuple[float, float]]]:
    if not isinstance(wkt, str):
        return []

    text = wkt.strip()
    if not text or not text.upper().startswith("POLYGON"):
        return []

    start = text.find("((")
    end = text.rfind("))")
    if start < 0 or end <= start + 1:
        return []

    body = text[start + 2:end]
    ring_tokens = re.split(r"\)\s*,\s*\(", body)
    rings = []
    for ring_token in ring_tokens:
        ring = []
        for point_token in ring_token.split(","):
            coords = point_token.strip().split()
            if len(coords) < 2:
                continue
            try:
                x = float(coords[0])
                y = float(coords[1])
            except ValueError:
                continue
            ring.append((x, y))
        if len(ring) >= 3:
            rings.append(ring)

    return rings

def load_zone_attributes(csv_path: Path):
    zones = []
    if not csv_path.exists():
        print(f"Zone attributes file not found: {csv_path}")
        return zones

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rings = parse_wkt_polygon(row.get("geometry", ""))
            if not rings:
                continue

            zone_id = str(row.get("name") or row.get("reference") or f"zone_{idx}")
            zones.append({"id": zone_id, "rings": rings})

    print(f"Loaded {len(zones)} zone polygons from {csv_path.name}")
    return zones

def plot_yolo_boxes(img, predictions, confidence_threshold=-1.0):
    dh, dw, _ = img.shape
    for i in range(predictions.shape[0]):  
        x1, y1, x2, y2, c, classid  = predictions[i,:]
        if c < confidence_threshold:
            continue
        color = (0, 255, 0)
        # Ensure coordinates are within image bounds
        l, r = max(0, int(x1)), min(dw - 1, int(x2))
        t, b = max(0, int(y1)), min(dh - 1, int(y2))

        cv2.rectangle(img, (l, t), (r, b), color, 2)
    return img

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def visualize_building_prediction(boxes, roof, satelite_image, plotter):
    # --- 1. Extract Data ---
    x, y, w, h = boxes[0]

    prediction = roof
    # Unpack prediction (Area, Inc, Ori)
    probh, probtop, probright, probbottom, probleft = prediction[0:5]
    ah, atop, aright, abottom, aleft = prediction[5:10]
    inc_top, inc_right, inc_bottom, inc_left = prediction[10:14]
    ori_top, ori_right, ori_bottom, ori_left = prediction[14:18]

    # --- 2. Coordinate Scaling (Pixels to Scene Units) ---
    img_h, img_w = satelite_image.shape[:2]

    # Center X, Y in 3D space
    center_x = x * img_w - (img_w / 2)
    center_y = (img_h / 2) - y * img_h
    center_z = 25.0  # Base height

    # Dimensions in 3D space
    build_w = w * img_w
    build_h = h * img_h

    # --- 3. Define the Rigid Base Rectangle (The "Attic Floor") ---
    dx = build_w / 4
    dy = build_h / 4

    # Corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    p_tl = np.array([-dx, dy, 0]) + [center_x, center_y, center_z]
    p_tr = np.array([dx, dy, 0]) + [center_x, center_y, center_z]
    p_br = np.array([dx, -dy, 0]) + [center_x, center_y, center_z]
    p_bl = np.array([-dx, -dy, 0]) + [center_x, center_y, center_z]

    # --- 4. Helper to create Side Planes ---
    def add_roof_flap(p_start, p_end, prob, inc, ori, base_ori_rad, color):
        if prob < 0.5:
            return

        mid = (p_start + p_end) / 2

        # Convert model outputs to angles
        tilt = inc * (math.pi / 2)
        azimuth = (ori - 0.5) * (math.pi / 2) + base_ori_rad

        # Calculate Normal Vector (Spherical to Cartesian)
        nz = math.cos(tilt)
        nx = math.sin(tilt) * math.cos(azimuth)
        ny = math.sin(tilt) * math.sin(azimuth)
        normal = np.array([nx, ny, nz], dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            return
        normal /= normal_norm

        # Visualize the normal projection on XY
        normalproj = np.array([normal[0], normal[1], 0.0], dtype=float)
        normalproj_norm = np.linalg.norm(normalproj)
        if normalproj_norm > 1e-12:
            normalproj /= normalproj_norm
            plotter.add_arrows(mid, normalproj, mag=100, color=color)

        # Tangent of edge
        edge_vec = p_end - p_start
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            return
        edge_unit = edge_vec / edge_len

        # The slope vector is perpendicular to edge and normal
        slope_vec = np.cross(edge_unit, normal)

        # Ensure slope points down
        if slope_vec[2] > 0:
            slope_vec = -slope_vec

        slope_len = max(build_w, build_h) * 0.2

        p_start_bot = p_start + (slope_vec * slope_len)
        p_end_bot = p_end + (slope_vec * slope_len)

        points = np.vstack([p_start, p_end, p_end_bot, p_start_bot])
        faces = [4, 0, 1, 2, 3]
        mesh = pv.PolyData(points, faces)

        plotter.add_mesh(mesh, color=color, opacity=0.8)

    # --- 5. Draw Elements ---
    if probh > 0.05:
        points = np.vstack([p_tl, p_tr, p_br, p_bl])
        faces = [4, 0, 1, 2, 3]
        center_mesh = pv.PolyData(points, faces)
        plotter.add_mesh(center_mesh, color="cyan", opacity=0.6, show_edges=True)

    # Side Flaps
    add_roof_flap(p_tl, p_tr, probtop, inc_top, ori_top, math.pi / 2, "red")
    add_roof_flap(p_tr, p_br, probright, inc_right, ori_right, 0, "blue")
    add_roof_flap(p_br, p_bl, probbottom, inc_bottom, ori_bottom, -math.pi / 2, "green")
    add_roof_flap(p_bl, p_tl, probleft, inc_left, ori_left, math.pi, "yellow")

class MainWindow(QMainWindow):
    def __init__(self, polygon_str=None):
        super().__init__()

        self.setWindowTitle("Building Inspector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top layout with left and right panels
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout, 1)  # 1 is stretch factor
        
        # Left panel for PyVista
        self.left_panel = QFrame()
        self.left_panel.setFrameStyle(QFrame.Shape.Box)
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        
        # Create PyVista widget
        self.plotter = QtInteractor(self.left_panel)
        self.left_panel_layout.addWidget(self.plotter)
        
        # Right panel for Matplotlib
        self.right_panel = QFrame()
        self.right_panel.setFrameStyle(QFrame.Shape.Box)
        self.right_panel_layout = QVBoxLayout(self.right_panel)
        
        # Create Matplotlib figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.right_panel_layout.addWidget(self.canvas)
        
        # Add panels to top layout
        top_layout.addWidget(self.left_panel, 1)
        top_layout.addWidget(self.right_panel, 1)
        
        # Bottom layout for navigation
        bottom_layout = QVBoxLayout()
        main_layout.addLayout(bottom_layout)
        
        # Stacked widget for different states
        self.stacked_widget = QStackedWidget()
        bottom_layout.addWidget(self.stacked_widget)
        self.press = False
        self.canvas.setMouseTracking(True)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        x_default = -86631.16455079315
        y_default = -102426.60522455335
        # ---- If polygon provided, override center ----
        if polygon_str:
            pts = parse_lonlat_polygon(polygon_str)
            lon_c, lat_c = polygon_centroid_lonlat(pts)
            x_c, y_c = lonlat_to_tm06(lon_c, lat_c)

            print(f"Polygon centroid (lon,lat): ({lon_c:.6f},{lat_c:.6f})")
            print(f"Centroid in EPSG:3763 (x,y): ({x_c:.3f},{y_c:.3f})")

            self.x_geolocation = x_c
            self.y_geolocation = y_c
        else:
            self.x_geolocation = x_default
            self.y_geolocation = y_default

        # IMPORTANT: set drag origin AFTER final geolocation is set
        self.x0 = self.x_geolocation
        self.y0 = self.y_geolocation

        self.setWindowTitle(f"Building Topology ({self.x_geolocation},{self.y_geolocation})")

        self.model = YOLO("best.pt")
        self.model.to("cpu")
        self.model.eval()

        zone_csv_path = Path(__file__).resolve().parent / "zone_attributes.csv"
        self.zone_features = load_zone_attributes(zone_csv_path)

        self.update_view()

    def on_press(self, event):
        self.press = True
        self.x0, self.y0 = event.xdata, event.ydata
        #print(f"press: ({self.x0},{self.y0})")

    def on_move(self, event):
        print("update_view center:", self.x_geolocation, self.y_geolocation)
        if not self.press:
            return
        #print(f"move: ({event.xdata},{event.ydata})")
        dx = (event.xdata - self.x0)*0.14662742614744
        dy = (event.ydata - self.y0)*0.14662742614744
        #print(f"move: ({dx},{dy})")
        self.x_geolocation += dx
        self.y_geolocation += dy
        self.x0, self.y0 = event.xdata, event.ydata
        # Update the view based on dx, dy
        self.update_view()

    def on_release(self, event):
        self.press = False

    def draw_zone_overlays(self, conversion, satelite_image):
        if not self.zone_features:
            return

        img_h, img_w = satelite_image.shape[:2]

        for zone in self.zone_features:
            rings_px = []
            for ring in zone["rings"]:
                ring_px = []
                for x_world, y_world in ring:
                    col, row = conversion(x_world, y_world)
                    ring_px.append((float(col), float(row)))
                if len(ring_px) >= 3:
                    rings_px.append(ring_px)

            if not rings_px:
                continue

            in_view = any((-50 <= x <= img_w + 50 and -50 <= y <= img_h + 50) for ring in rings_px for x, y in ring)
            if not in_view:
                continue

            for ring_px in rings_px:
                xs = np.array([p[0] for p in ring_px], dtype=float)
                ys = np.array([p[1] for p in ring_px], dtype=float)
                self.ax.plot(xs, ys, color="magenta", linewidth=2)

                points_3d = np.column_stack((xs - img_w / 2, img_h / 2 - ys, np.full(xs.shape[0], 1.0)))
                polyline = pv.lines_from_points(points_3d, close=True)
                self.plotter.add_mesh(polyline, color="magenta", line_width=4)

            outer_ring = rings_px[0]
            cx = float(np.mean([p[0] for p in outer_ring]))
            cy = float(np.mean([p[1] for p in outer_ring]))
            self.ax.text(
                cx,
                cy,
                zone["id"],
                color="magenta",
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
            )

    def update_view(self):

        region_size = 50
        coordinates = (self.x_geolocation,self.y_geolocation)
        self.setWindowTitle(f"Building Topology ({self.x_geolocation},{self.y_geolocation})")
        satelite_image, _, conversion = rqst_satelite_image(coordinates,region_size)
        
        #cv2.imwrite("img0.jpg",cv2.cvtColor(satelite_image, cv2.COLOR_RGB2BGR))  
        #im = self.model.preprocess(satelite_image)
        #preds = self.model.inference(im)
        img = cv2.cvtColor(satelite_image, cv2.COLOR_RGB2BGR)
        #results = self.model("img0.jpg")
        results2 = self.model(img,conf=0.4)[0]
        img = results2.plot()
        #img = plot_yolo_boxes(img,results2[0].roof.data,confidence_threshold=0.7)
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_title("Image/Plot Display")
        self.ax.axis("off")

        texture = pv.numpy_to_texture(satelite_image)

        plane = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, -1),
            i_size=satelite_image.shape[1],  # width
            j_size=satelite_image.shape[0],  # height
        )
        plane.texture_map_to_plane(inplace=True)

        # Clear the plotter before re-adding (important when refreshing view)
        self.plotter.clear()

        # Add the plane with texture
        self.plotter.add_mesh(plane, texture=texture, show_edges=False)
        self.draw_zone_overlays(conversion, satelite_image)
        for i in range(results2.boxes.shape[0]):
            box_xywhn = results2.boxes.xywhn[i].detach().cpu().numpy().reshape(1, 4)
            roof_pred = sigmoid(results2.roof.data[i, :].detach().cpu().numpy())
            visualize_building_prediction(box_xywhn, roof_pred, satelite_image, self.plotter)

        self.canvas.draw()

def read_polygon_from_cli() -> Optional[str]:
    # Usage:
    # python traveler.py "(-9.13,38.70),(-9.14,38.70),..."
    if len(sys.argv) >= 2:
        return sys.argv[1]
    return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    polygon_str = read_polygon_from_cli()
    window = MainWindow(polygon_str=polygon_str)
    window.show()
    sys.exit(app.exec())

