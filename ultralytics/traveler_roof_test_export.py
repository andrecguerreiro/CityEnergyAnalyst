import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QStackedWidget, QPushButton, QLineEdit,
                             QLabel, QFrame)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pyvista as pv
from pyvistaqt import QtInteractor
import json
from pathlib import Path
import os
from typing import Any, Dict, Tuple, List, Optional
import math
import cv2
import tifffile

from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
from datetime import datetime
from scipy.spatial import ConvexHull, Delaunay

from utils import rqst_satelite_image
from ultralytics import YOLO

import re
from typing import List, Tuple, Optional
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
        
        # Default center (projected coordinates used by rqst_satelite_image)
        x_default = -86631.16455079315
        y_default = -102426.60522455335

        # If polygon provided, override center (polygon is lon/lat EPSG:4326)
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

        # Drag origin must be set AFTER geolocation is finalized
        self.x0 = self.x_geolocation
        self.y0 = self.y_geolocation

        # Session accumulator: each update_view appends detected roofs here
        self.session_roofs = []

        self.setWindowTitle(f"Building Topology ({self.x_geolocation},{self.y_geolocation})")

        self.model = YOLO("best.pt")
        self.model.to("cpu")
        self.model.eval()

        self.update_view()

    def on_press(self, event):
        self.press = True
        self.x0, self.y0 = event.xdata, event.ydata
        #print(f"press: ({self.x0},{self.y0})")

    def on_move(self, event):
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

    def update_view(self):

        region_size = 50
        coordinates = (self.x_geolocation,self.y_geolocation)
        self.setWindowTitle(f"Building Topology ({self.x_geolocation},{self.y_geolocation})")
        satelite_image,resolution,transform = rqst_satelite_image(coordinates,region_size)
        
        #cv2.imwrite("img0.jpg",cv2.cvtColor(satelite_image, cv2.COLOR_RGB2BGR))  
        #im = self.model.preprocess(satelite_image)
        #preds = self.model.inference(im)
        img = cv2.cvtColor(satelite_image, cv2.COLOR_RGB2BGR)
        #results = self.model("img0.jpg")
        results2 = self.model(img)[0]
        # ---- Capture detections for export-on-close ----
        # Store enough info to compute areas and group roofs into buildings later.
        view_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "center_xy": (float(self.x_geolocation), float(self.y_geolocation)),
            "region_size": float(region_size),
            "resolution_m_per_px": float(resolution),
            "img_shape": tuple(int(v) for v in satelite_image.shape),
            "roofs": []
        }

        # Extract bboxes and confidences (if available)
        xyxy = results2.boxes.xyxy.detach().cpu().numpy()
        if getattr(results2.boxes, "conf", None) is not None:
            confs = results2.boxes.conf.detach().cpu().numpy()
        else:
            confs = np.full((xyxy.shape[0],), np.nan, dtype=float)

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            w_px = abs(x2 - x1)
            h_px = abs(y2 - y1)

            # Approx planar area from bbox in m²
            planar_area_m2 = (w_px * h_px) * (resolution ** 2)

            # Keep raw "roof head" for later decoding (tilt/orientation bins etc.)
            roof_head_raw = results2.roof.data[i, :].detach().cpu().numpy().tolist()

            view_record["roofs"].append({
                "roof_i": int(i),
                "conf": float(confs[i]),
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "cx_px": cx, "cy_px": cy,
                "w_px": w_px, "h_px": h_px,
                "planar_area_m2": float(planar_area_m2),
                "roof_head_raw": roof_head_raw,
            })

        self.session_roofs.append(view_record)
        # ----------------------------------------------
        img = results2.plot()
        #img = plot_yolo_boxes(img,results2[0].roof.data,confidence_threshold=0.7)
        plt.imshow(img)
        self.ax.set_title("Image/Plot Display")
        self.canvas.draw()

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
        for i in range(results2.boxes.xyxy.shape[0]):
            x1, y1, x2, y2 = results2.boxes.xyxy[i, :]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            prediction = results2.roof.data[i,:]
            dh,dtop,dright,dbottom,dleft = prediction[:5].sigmoid()
            # print(f"{ah},{atop},{aright},{abottom},{aleft}")
            ah,atop,aright,abottom,aleft,inctop,incright,incbottom,incleft,oritop,oriright,oribottom,orileft = prediction[5:]
            # Map to 3D coordinates
            x_3d = cx - satelite_image.shape[1] / 2
            y_3d = satelite_image.shape[0] / 2 - cy
            z_3d = 25
            cube = pv.Cube(
                center=(x_3d, y_3d, z_3d),
                x_length=w,
                y_length=h,
                z_length=50,
            )
            self.plotter.add_mesh(cube, color="red", opacity=0.7)
    def export_session_roofs_per_building(self, out_csv: str = "roofs_per_building.csv") -> None:
        """Export all detected roofs grouped into buildings for the *last viewed image*.
        We take the last view in `self.session_roofs` (i.e., what you were looking at when closing).
        Building grouping is done by clustering roof centers in meter space (DBSCAN).
        """
        if not getattr(self, "session_roofs", None):
            print("Nothing to export: no views captured.")
            return

        last_view = self.session_roofs[-1]
        roofs = last_view.get("roofs", [])
        if not roofs:
            print("Nothing to export: no roofs detected in the last view.")
            return

        res = float(last_view["resolution_m_per_px"])
        H, W = last_view["img_shape"][:2]

        # Convert pixel centers to local meters (origin at image center).
        pts_m = []
        for r in roofs:
            x_m = (r["cx_px"] - W / 2.0) * res
            y_m = (H / 2.0 - r["cy_px"]) * res
            pts_m.append([x_m, y_m])
        pts_m = np.asarray(pts_m, dtype=float)

        # Cluster into buildings. eps is in meters.
        # Tune eps if you see one building split into many (increase) or multiple buildings merged (decrease).
        labels = DBSCAN(eps=8.0, min_samples=1).fit(pts_m).labels_

        rows = []
        for r, b_id in zip(roofs, labels):
            rows.append({
                "building_id": int(b_id),
                "roof_i": r["roof_i"],
                "conf": r["conf"],
                "planar_area_m2": r["planar_area_m2"],
                "bbox_x1": r["bbox_x1"], "bbox_y1": r["bbox_y1"],
                "bbox_x2": r["bbox_x2"], "bbox_y2": r["bbox_y2"],
                "cx_px": r["cx_px"], "cy_px": r["cy_px"],
                "center_x_geolocation": last_view["center_xy"][0],
                "center_y_geolocation": last_view["center_xy"][1],
                "resolution_m_per_px": res,
                "timestamp": last_view["timestamp"],
                # Keep raw head so you can decode tilt/orientation later
                "roof_head_raw": json.dumps(r["roof_head_raw"]),
            })

        df = pd.DataFrame(rows)

        # Add per-building totals (planar)
        totals = df.groupby("building_id")["planar_area_m2"].sum().rename("building_planar_area_m2")
        df = df.merge(totals, on="building_id", how="left")

        df.to_csv(out_csv, index=False)
        print(f"Exported {len(df)} roofs grouped into {df['building_id'].nunique()} buildings -> {out_csv}")

    def closeEvent(self, event):
        # Export when the window closes
        try:
            self.export_session_roofs_per_building("roofs_per_building.csv")
        except Exception as e:
            print("Export failed on close:", repr(e))
        event.accept()
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