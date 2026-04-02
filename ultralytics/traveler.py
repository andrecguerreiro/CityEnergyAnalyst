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

from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay

from utils import rqst_satelite_image
from ultralytics import YOLO

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
        
        self.x_geolocation = -86631.16455079315
        self.y_geolocation = -102426.60522455335

        self.x0 = self.x_geolocation
        self.y0 = self.y_geolocation

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
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())