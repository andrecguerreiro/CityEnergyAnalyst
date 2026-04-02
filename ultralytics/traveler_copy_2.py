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
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay

from utils import rqst_satelite_image
from ultralytics import YOLO

import yaml, torch
from ultralytics.data import build_dataloader
from ultralytics.data.augment import v8_transforms
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import (
        DEFAULT_CFG,
         GIT,
        LOCAL_RANK,
        LOGGER,
        RANK,
        TQDM,
        YAML,
        callbacks,
        clean_url,
        colorstr,
        emojis,
)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

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

def visualize_building_training(inst, satelite_image, plotter):
    # --- 1. Extract Data ---
    x, y, w, h = inst.bboxes[0]
    prediction = inst.extras
    # Unpack prediction (Area, Inc, Ori)
    ah, atop, aright, abottom, aleft = prediction[0:5]
    inc_top, inc_right, inc_bottom, inc_left = prediction[5:9]
    ori_top, ori_right, ori_bottom, ori_left = prediction[9:13]

    # --- 2. Coordinate Scaling (Pixels to Scene Units) ---
    # Centering the building in the scene
    img_h, img_w = satelite_image.shape[:2]
    
    # Center X, Y in 3D space
    center_x = x * img_w - (img_w / 2)
    center_y = (img_h / 2) - y * img_h
    center_z = 25.0 # Base height
    
    # Dimensions in 3D space
    build_w = w * img_w
    build_h = h * img_h

    # --- 3. Define the Rigid Base Rectangle (The "Attic Floor") ---
    # We do NOT modify these corners based on orientation. 
    # This ensures the building is watertight at the seams.
    # Local coordinates relative to center
    dx = build_w / 4
    dy = build_h / 4
    
    # Corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # specific to image coords (Top is usually low Y in images, but High Y in 3D? 
    # Assuming standard math plot: Top is +Y)
    p_tl = np.array([-dx,  dy, 0]) + [center_x, center_y, center_z]
    p_tr = np.array([ dx,  dy, 0]) + [center_x, center_y, center_z]
    p_br = np.array([ dx, -dy, 0]) + [center_x, center_y, center_z]
    p_bl = np.array([-dx, -dy, 0]) + [center_x, center_y, center_z]

    # --- 4. Helper to create Side Planes ---
    def add_roof_flap(p_start, p_end, area_score, inc, ori, base_ori_rad, color):
        if area_score < 0.05: return

        # Midpoint of the edge (for visualizing normal)
        mid = (p_start + p_end) / 2

        # --- Geometry Math ---
        # Convert model outputs to angles
        # Assuming: inc 0-1 maps to 0-90 degrees (0 is flat, 1 is vertical wall)
        tilt = inc * (math.pi / 2)
        
        # Assuming: ori 0-1 maps to -45 to +45 relative to the face normal
        # We add 'base_ori_rad' to rotate it to the correct side of the building
        azimuth = (ori - 0.5) * (math.pi / 2) + base_ori_rad

        # Calculate Normal Vector (Spherical to Cartesian)
        # Z is Up. 
        nz = math.cos(tilt)
        nx = math.sin(tilt) * math.cos(azimuth)
        ny = math.sin(tilt) * math.sin(azimuth)
        normal = np.array([nx, ny, nz])
        normal /= np.linalg.norm(normal)

        # --- Visualize the Normal Vector (The "Debug Stick") ---
        # This shows you exactly where the model thinks the roof is facing
        ##### 
        normalproj = np.array([normal[0],normal[1],0])
        normalproj /= np.linalg.norm(normalproj)
        plotter.add_arrows(mid, normalproj, mag=100, color=color)

        # --- Create the Flap ---
        # We calculate the "Down" vector along the roof slope
        # Tangent of edge
        edge_vec = p_end - p_start
        edge_len = np.linalg.norm(edge_vec)
        edge_unit = edge_vec / edge_len
        
        # The slope vector is perpendicular to the edge and perpendicular to the normal
        # slope_vec = cross(edge, normal)
        slope_vec = np.cross(edge_unit, normal)
        
        # Ensure slope points "down" (z should be negative if it's a roof)
        if slope_vec[2] > 0: slope_vec = -slope_vec

        # Length of the roof slope. 
        # User code used Area/Length. We can default to a fixed length or use Area scaling.
        # Here we assume the roof extends outwards by a fixed amount or based on area score.
        # Fixed length for visualization stability:
        slope_len = max(build_w, build_h) * 0.2 # 40% of building width
        
        # Calculated bottom points
        p_start_bot = p_start + (slope_vec * slope_len)
        p_end_bot   = p_end + (slope_vec * slope_len)

        # Mesh construction
        points = np.vstack([p_start, p_end, p_end_bot, p_start_bot])
        faces = [4, 0, 1, 2, 3]
        mesh = pv.PolyData(points, faces)
        
        plotter.add_mesh(mesh, color=color, opacity=0.8)

    # --- 5. Draw Elements ---
    
    # A. Center Flat Roof (Cyan)
    if ah > 0.05:
        points = np.vstack([p_tl, p_tr, p_br, p_bl])
        faces = [4, 0, 1, 2, 3]
        center_mesh = pv.PolyData(points, faces)
        plotter.add_mesh(center_mesh, color="cyan", opacity=0.6, show_edges=True)

    # B. Side Flaps
    # Note: We pass a "Base Orientation" so the math knows which way is "out" 
    # for that specific wall (0, 90, 180, 270 degrees)
    
    # Top Edge (p_tl -> p_tr) - Base Ori: 90 deg (Pi/2)
    add_roof_flap(p_tl, p_tr, atop, inc_top, ori_top, math.pi/2, "red")
    
    # Right Edge (p_tr -> p_br) - Base Ori: 0 deg
    add_roof_flap(p_tr, p_br, aright, inc_right, ori_right, 0, "blue")
    
    # Bottom Edge (p_br -> p_bl) - Base Ori: 270 deg (-Pi/2)
    add_roof_flap(p_br, p_bl, abottom, inc_bottom, ori_bottom, -math.pi/2, "green")
    
    # Left Edge (p_bl -> p_tl) - Base Ori: 180 deg (Pi)
    add_roof_flap(p_bl, p_tl, aleft, inc_left, ori_left, math.pi, "yellow")

    # C. Draw the Base Cube (Context)
    cube = pv.Cube(center=(center_x, center_y, center_z/2), 
                   x_length=build_w, y_length=build_h, z_length=center_z)
    plotter.add_mesh(cube, color="white", style='wireframe', opacity=0.3)

def visualize_building_prediction(boxes,roof,satelite_image,plotter):
    # --- 1. Extract Data ---
    x, y, w, h = boxes[0]

    prediction = roof
    # Unpack prediction (Area, Inc, Ori)
    probh, probtop, probright, probbottom, probleft = prediction[0:5]
    ah, atop, aright, abottom, aleft = prediction[5:10]
    inc_top, inc_right, inc_bottom, inc_left = prediction[10:14]
    ori_top, ori_right, ori_bottom, ori_left = prediction[14:18]

    # --- 2. Coordinate Scaling (Pixels to Scene Units) ---
    # Centering the building in the scene
    img_h, img_w = satelite_image.shape[:2]
    
    # Center X, Y in 3D space
    center_x = x * img_w - (img_w / 2)
    center_y = (img_h / 2) - y * img_h
    center_z = 25.0 # Base height
    
    # Dimensions in 3D space
    build_w = w * img_w
    build_h = h * img_h

    # --- 3. Define the Rigid Base Rectangle (The "Attic Floor") ---
    # We do NOT modify these corners based on orientation. 
    # This ensures the building is watertight at the seams.
    # Local coordinates relative to center
    dx = build_w / 4
    dy = build_h / 4
    
    # Corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # specific to image coords (Top is usually low Y in images, but High Y in 3D? 
    # Assuming standard math plot: Top is +Y)
    p_tl = np.array([-dx,  dy, 0]) + [center_x, center_y, center_z]
    p_tr = np.array([ dx,  dy, 0]) + [center_x, center_y, center_z]
    p_br = np.array([ dx, -dy, 0]) + [center_x, center_y, center_z]
    p_bl = np.array([-dx, -dy, 0]) + [center_x, center_y, center_z]

    # --- 4. Helper to create Side Planes ---
    def add_roof_flap(p_start, p_end, prob, inc, ori, base_ori_rad, color):
        if prob < 0.5: return

        # Midpoint of the edge (for visualizing normal)
        mid = (p_start + p_end) / 2

        # --- Geometry Math ---
        # Convert model outputs to angles
        # Assuming: inc 0-1 maps to 0-90 degrees (0 is flat, 1 is vertical wall)
        tilt = inc * (math.pi / 2)
        
        # Assuming: ori 0-1 maps to -45 to +45 relative to the face normal
        # We add 'base_ori_rad' to rotate it to the correct side of the building
        azimuth = (ori - 0.5) * (math.pi / 2) + base_ori_rad

        # Calculate Normal Vector (Spherical to Cartesian)
        # Z is Up. 
        nz = math.cos(tilt)
        nx = math.sin(tilt) * math.cos(azimuth)
        ny = math.sin(tilt) * math.sin(azimuth)
        normal = np.array([nx, ny, nz])
        normal /= np.linalg.norm(normal)

        # --- Visualize the Normal Vector (The "Debug Stick") ---
        # This shows you exactly where the model thinks the roof is facing
        ##### 
        normalproj = np.array([normal[0],normal[1],0])
        normalproj /= np.linalg.norm(normalproj)
        plotter.add_arrows(mid, normalproj, mag=100, color=color)

        # --- Create the Flap ---
        # We calculate the "Down" vector along the roof slope
        # Tangent of edge
        edge_vec = p_end - p_start
        edge_len = np.linalg.norm(edge_vec)
        edge_unit = edge_vec / edge_len
        
        # The slope vector is perpendicular to the edge and perpendicular to the normal
        # slope_vec = cross(edge, normal)
        slope_vec = np.cross(edge_unit, normal)
        
        # Ensure slope points "down" (z should be negative if it's a roof)
        if slope_vec[2] > 0: slope_vec = -slope_vec

        # Length of the roof slope. 
        # User code used Area/Length. We can default to a fixed length or use Area scaling.
        # Here we assume the roof extends outwards by a fixed amount or based on area score.
        # Fixed length for visualization stability:
        slope_len = max(build_w, build_h) * 0.2 # 40% of building width
        
        # Calculated bottom points
        p_start_bot = p_start + (slope_vec * slope_len)
        p_end_bot   = p_end + (slope_vec * slope_len)

        # Mesh construction
        points = np.vstack([p_start, p_end, p_end_bot, p_start_bot])
        faces = [4, 0, 1, 2, 3]
        mesh = pv.PolyData(points, faces)
        
        plotter.add_mesh(mesh, color=color, opacity=0.8)

    # --- 5. Draw Elements ---
    
    # A. Center Flat Roof (Cyan)
    if probh > 0.05:
        points = np.vstack([p_tl, p_tr, p_br, p_bl])
        faces = [4, 0, 1, 2, 3]
        center_mesh = pv.PolyData(points, faces)
        plotter.add_mesh(center_mesh, color="cyan", opacity=0.6, show_edges=True)

    # B. Side Flaps
    # Note: We pass a "Base Orientation" so the math knows which way is "out" 
    # for that specific wall (0, 90, 180, 270 degrees)
    
    # Top Edge (p_tl -> p_tr) - Base Ori: 90 deg (Pi/2)
    add_roof_flap(p_tl, p_tr, probtop, inc_top, ori_top, math.pi/2, "red")
    
    # Right Edge (p_tr -> p_br) - Base Ori: 0 deg
    add_roof_flap(p_tr, p_br,probright, inc_right, ori_right, 0, "blue")
    
    # Bottom Edge (p_br -> p_bl) - Base Ori: 270 deg (-Pi/2)
    add_roof_flap(p_br, p_bl,probbottom, inc_bottom, ori_bottom, -math.pi/2, "green")
    
    # Left Edge (p_bl -> p_tl) - Base Ori: 180 deg (Pi)
    add_roof_flap(p_bl, p_tl,probleft, inc_left, ori_left, math.pi, "yellow")

    # C. Draw the Base Cube (Context)
    cube = pv.Cube(center=(center_x, center_y, center_z/2), 
                   x_length=build_w, y_length=build_h, z_length=center_z)
    plotter.add_mesh(cube, color="white", style='wireframe', opacity=0.3)

class MainWindow(QMainWindow):
    def __init__(self):
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
        
        # Add panels to top layout
        top_layout.addWidget(self.left_panel, 1)
        
        # Bottom layout for navigation
        bottom_layout = QVBoxLayout()
        main_layout.addLayout(bottom_layout)
        
        # Stacked widget for different states

        button_layout = QHBoxLayout()

        # 2. Create the buttons
        self.btn_switch = QPushButton("Switch View")
        self.btn_next = QPushButton("Next")

        # 3. Connect buttons to functions (we will define these next)
        self.btn_switch.clicked.connect(self.toggle_view_mode)
        self.btn_next.clicked.connect(self.load_next_image)

        # 4. Add buttons to the layout
        button_layout.addWidget(self.btn_switch)
        button_layout.addWidget(self.btn_next)
        bottom_layout.addLayout(button_layout)

        self.current_idx = 0  # Track which image index we are on
        self.show_ground_truth = False # Track which view mode we are in

        #self.x_geolocation = -86631.16455079315
        #self.y_geolocation = -102426.60522455335

        #self.x0 = self.x_geolocation
        #self.y0 = self.y_geolocation

        self.setWindowTitle(f"Building Topology")
        self.model = YOLO("best.pt")
        self.model.to("cpu")
        self.model.eval()

        overrides = {
            "data": "datasets/roof_dataset.yaml",
            "model": "yolo11n-roof.yaml",   # or path to model config / weights
            "epochs": 1,
            "batch": 2,
            "task" : "roof"
        }
        args = get_cfg(DEFAULT_CFG, overrides)  # or a custom dict with required hyperparameters
        args.rooftop = True
        data = check_det_dataset("datasets/roof_dataset.yaml")
        self.dataset = build_yolo_dataset(args , img_path= data['train'], batch = 2 , data = data, mode="train", rect=False, stride=32)

        self.update_view()

    def toggle_view_mode(self):
        self.show_ground_truth = not self.show_ground_truth
        self.update_view()

    def load_next_image(self):
        self.current_idx = self.current_idx + 1
        self.update_view()

    def update_view(self):

        region_size = 50
        #coordinates = (self.x_geolocation,self.y_geolocation)
        #self.setWindowTitle(f"Building Topology ({self.x_geolocation},{self.y_geolocation})")
        #satelite_image,resolution,transform = rqst_satelite_image(coordinates,region_size)
        
        input = self.dataset.get_image_and_label(self.current_idx)
        #cv2.imwrite("img0.jpg",cv2.cvtColor(satelite_image, cv2.COLOR_RGB2BGR))  
        #im = self.model.preprocess(satelite_image)
        #preds = self.model.inference(im)
        satelite_image = image = cv2.imread(input['im_file'])
        img = cv2.cvtColor(satelite_image, cv2.COLOR_RGB2BGR)
        #results = self.model("img0.jpg")
        results = self.model(image)[0]

        texture1 = pv.numpy_to_texture(img)
        texture2 = pv.numpy_to_texture(img)

        plane1 = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, -1),
            i_size=satelite_image.shape[1],  # width
            j_size=satelite_image.shape[0],  # height
        )
        plane1.texture_map_to_plane(inplace=True)

        plane2 = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, -1),
            i_size=satelite_image.shape[1],  # width
            j_size=satelite_image.shape[0],  # height
        )
        plane2.texture_map_to_plane(inplace=True)

        # Clear the plotter before re-adding (important when refreshing view)
        self.plotter.clear()
        #self.prediction_plotter.clear()


        # Add the plane with texture
        self.plotter.add_mesh(plane1, texture=texture1, show_edges=False)
        #self.prediction_plotter.add_mesh(plane2, texture=texture2, show_edges=False)

        if self.show_ground_truth:
            for inst in input['instances']:
                visualize_building_training(inst,satelite_image,self.plotter)
        else:
            for i in range(results.boxes.shape[0]):
                print(f"orientations: {results.roof.data[i,14:18]} transformed into {sigmoid(results.roof.data[i,14:18])}")
                visualize_building_prediction(np.array(results.boxes[i,:].xywhn),np.array(sigmoid(results.roof.data[i,:])),satelite_image,self.plotter)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())