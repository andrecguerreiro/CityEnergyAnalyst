import numpy as np
from ultralytics import YOLO
from owslib.wmts import WebMapTileService
from PIL import Image
import io
import cv2
import folium
import math
from pyproj import Transformer
import requests
import json
import gradio as gr
import os
import hashlib

## Listing of logic in the project
'''
We download high resolution satellite images. 
From this images we retrieve predictions with our model.
Then we retrieve the OSM buildings from openstreetmap.
Then we match these buildings.
We recreate a rooftop based on the building outline and the model prediction 
We render it
'''

def wmts_tile_to_array(tile_data):
    img = Image.open(io.BytesIO(tile_data.read()))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

def get_tile_indices(xmin, ymin, xmax, ymax, matrix):
    tile_size = matrix.tilewidth * matrix.scaledenominator * 0.28e-3
    origin_x, origin_y = matrix.topleftcorner
    col_min = int((xmin - origin_x) // tile_size)
    col_max = int((xmax - origin_x) // tile_size)
    row_min = int((origin_y - ymax) // tile_size)
    row_max = int((origin_y - ymin) // tile_size)
    return col_min, col_max, row_min, row_max

def retrieve_satelite_image(top_left_corner, bottom_right_corner,progress_cb = None):
    wmts_url = (
        "https://cartografia.dgterritorio.gov.pt/ortos2018/service"
        "?service=WMTS&request=GetCapabilities"
    )
    wmts = WebMapTileService(wmts_url)

    xmin, ymax = top_left_corner
    xmax, ymin = bottom_right_corner

    layer          = "Ortos2018-RGB"
    tile_matrix_set = "PTTM_06"
    zoom_level     = "14"
    matrix   = wmts.tilematrixsets[tile_matrix_set].tilematrix[zoom_level]
    res      = matrix.scaledenominator * 0.28e-3
    tile_size_m = matrix.tilewidth * res

    col_min, col_max, row_min, row_max = get_tile_indices(xmin, ymin, xmax, ymax, matrix)
    n_rows = row_max + 1 - row_min
    n_cols = col_max + 1 - col_min
    total_tiles = n_rows * n_cols
    processed_blocks = np.empty((n_rows, n_cols), dtype=object)

    os.makedirs("cache/tiles", exist_ok=True)

    done = 0
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            tile_filename = f"cache/tiles/tile_{zoom_level}_{row}_{col}.png"
            if os.path.exists(tile_filename):
                img_array = np.array(Image.open(tile_filename).convert("RGB"))
            else:
                tile = wmts.gettile(
                    layer=layer, tilematrixset=tile_matrix_set,
                    tilematrix=zoom_level, row=row, column=col, format="image/png",
                )
                img = Image.open(io.BytesIO(tile.read())).convert("RGB")
                img.save(tile_filename) # Save to cache
                img_array = np.array(img)
            processed_blocks[row - row_min, col - col_min] = {"img": img_array}
            done += 1
            if progress_cb:
                progress_cb(done, total_tiles)

    block_h, block_w = processed_blocks[0, 0]["img"].shape[:2]
    stitched = np.zeros((n_rows * block_h, n_cols * block_w, 3), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            img = processed_blocks[row, col]["img"]
            stitched[row*block_h:(row+1)*block_h, col*block_w:(col+1)*block_w] = img

    origin_x = matrix.topleftcorner[0] + col_min * tile_size_m
    origin_y = matrix.topleftcorner[1] - row_min * tile_size_m

    col_start = int(round((xmin - origin_x) / res))
    row_start = int(round((origin_y - ymax) / res))
    col_end   = int(round((xmax - origin_x) / res))
    row_end   = int(round((origin_y - ymin) / res))
    satellite_image = stitched[row_start:row_end, col_start:col_end, :]

    def conversion(x, y):
        return int((x - xmin) / res), int((ymax - y) / res)

    return satellite_image, res, conversion

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class CachedModel:
    def __init__(self):
        self.model = YOLO("best.pt")
        self.model.to("cpu")
        self.model.eval()

class RoofPlane:
    def __init__(self, face, probability, corners, normal, inclination, orientation):
        self.face = face;  self.probability = probability
        self.corners = corners;  self.normal = normal
        self.inclination = inclination;  self.orientation = orientation

class EstimatedBuilding:
    def __init__(self):
        self.box_coords_in_epsg_3763: list[float] = []
        self.planes_in_physical_dimensions: list[RoofPlane] = []
        self.raw_roof_data = []

    def convert_tensor_prediction_to_building(
        self, boxes_xywhn, roof_prediction, top_left_corner, res, image_shape
    ):
        self.raw_roof_data = roof_prediction
        img_h, img_w = image_shape[:2]
        xmin_map, ymax_map = top_left_corner
        cx_norm, cy_norm, w_norm, h_norm = boxes_xywhn
        cx_map = xmin_map + cx_norm * img_w * res
        cy_map = ymax_map - cy_norm * img_h * res
        w_map  = w_norm * img_w * res
        h_map  = h_norm * img_h * res

        self.box_coords_in_epsg_3763 = [
            cx_map - w_map/2, cy_map - h_map/2,
            cx_map + w_map/2, cy_map + h_map/2,
        ]

        probh, probtop, probright, probbottom, probleft = roof_prediction[0:5]
        ah, atop, aright, abottom, aleft = roof_prediction[5:10]
        inc_top, inc_right, inc_bottom, inc_left = roof_prediction[10:14]
        ori_top, ori_right, ori_bottom, ori_left = roof_prediction[14:18]

        dx, dy = w_map/2, h_map/2
        p_tl = np.array([cx_map-dx, cy_map+dy, 0.])
        p_tr = np.array([cx_map+dx, cy_map+dy, 0.])
        p_br = np.array([cx_map+dx, cy_map-dy, 0.])
        p_bl = np.array([cx_map-dx, cy_map-dy, 0.])

        for face_name, prob, inc, ori, base_az, ps, pe in [
            ("top",    probtop,    inc_top,    ori_top,     math.pi/2,  p_tl, p_tr),
            ("right",  probright,  inc_right,  ori_right,   0.,         p_tr, p_br),
            ("bottom", probbottom, inc_bottom, ori_bottom, -math.pi/2,  p_br, p_bl),
            ("left",   probleft,   inc_left,   ori_left,    math.pi,    p_bl, p_tl),
        ]:
            if prob < 0.5: continue
            tilt    = inc * (math.pi/2)
            azimuth = (ori - 0.5) * (math.pi/2) + base_az
            normal  = np.array([math.sin(tilt)*math.cos(azimuth),
                                 math.sin(tilt)*math.sin(azimuth),
                                 math.cos(tilt)])
            normal /= np.linalg.norm(normal)
            edge_unit = (pe - ps) / np.linalg.norm(pe - ps)
            slope_vec = np.cross(edge_unit, normal)
            if slope_vec[2] > 0: slope_vec = -slope_vec
            sl = max(w_map, h_map) * 0.2
            self.planes_in_physical_dimensions.append(RoofPlane(
                face=face_name, probability=float(prob),
                corners=[ps, pe, pe+slope_vec*sl, ps+slope_vec*sl],
                normal=normal, inclination=float(inc), orientation=float(ori),
            ))

def retrieve_prediction_list(satellite_image, top_left_corner, res,
                              building_threshold, overlap_threshold, cached_model):
    results = cached_model.model(satellite_image, conf=building_threshold,
                                  iou=overlap_threshold)[0]
    predictions = []
    for i in range(results.boxes.shape[0]):
        b = EstimatedBuilding()
        b.convert_tensor_prediction_to_building(
            boxes_xywhn=np.array(results.boxes[i].xywhn[0]),
            roof_prediction=np.array(sigmoid(results.roof.data[i, :])),
            top_left_corner=top_left_corner, res=res,
            image_shape=satellite_image.shape,
        )
        predictions.append(b)
    return predictions

def get_osm_buildings(top_left_corner, bottom_right_corner):
    tr = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)
    tl_lon, tl_lat = tr.transform(*top_left_corner)
    br_lon, br_lat = tr.transform(*bottom_right_corner)
    s, n = min(tl_lat, br_lat), max(tl_lat, br_lat)
    w, e = min(tl_lon, br_lon), max(tl_lon, br_lon)
    q = f"""[out:json];(way["building"]({s},{w},{n},{e});
    relation["building"]({s},{w},{n},{e}););out body;>;out skel qt;"""
    r = requests.post("https://overpass-api.de/api/interpreter", data=q)
    r.raise_for_status()
    data  = r.json()
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"]=="node"}
    features = []
    for el in data["elements"]:
        if el["type"] != "way": continue
        coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
        if len(coords) < 3: continue
        features.append({"type":"Feature",
                          "properties":{"osm_id":el["id"], **el.get("tags",{})},
                          "geometry":{"type":"Polygon","coordinates":[coords]}})
    return {"type":"FeatureCollection","features":features}

def get_osm_buildings_cached(top_left, bottom_right):
    os.makedirs("cache/osm", exist_ok=True)
    
    # Create a unique ID based on the coordinates
    coord_str = f"{top_left}_{bottom_right}"
    cache_key = hashlib.md5(coord_str.encode()).hexdigest()
    cache_path = f"cache/osm/{cache_key}.json"

    if os.path.exists(cache_path):
        print(f"📦 Loading OSM data from cache: {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    # If not cached, fetch it
    data = get_osm_buildings(top_left, bottom_right)
    
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data

import numpy as np
import math
from pyproj import Transformer
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import cv2

def planenormal(face_id, inc, ori,axis_aligned_bounding_box_rotation):
    # user of this code! be aware, it took me a billion years to catch this discusting error
    # caused by the axis_aligned_bounding_box_rotation. Appreciate my sacrifice for your benefit!
    if face_id == "T":
        base_ori_rad = math.pi / 2
    elif face_id == "B":
        base_ori_rad = -math.pi / 2 
    elif face_id == "R":
        base_ori_rad = 0 
    elif face_id == "L":
        base_ori_rad = math.pi 
    elif face_id == "H":
        return [0, 0, 1]
    else:
        return None
 
    # inc 0-1 -> 0-90 degrees
    tilt = inc * (math.pi / 2)
    # ori 0-1 -> -45 to +45 degrees relative to base_ori
    global_azimuth = (ori - 0.5) * (math.pi / 2) + base_ori_rad
    azimuth = global_azimuth + axis_aligned_bounding_box_rotation
    nz = math.cos(tilt)
    nx = math.sin(tilt) * math.cos(azimuth)
    ny = math.sin(tilt) * math.sin(azimuth)
    
    return [nx, ny, nz]
 
def split_with_lines(corners, lines):
    """Split a polygon defined by corners using a list of lines"""
    polys = [Polygon(corners)]
 
    for line in lines:
        new_polys = []
        splitter = LineString(line)
 
        for poly in polys:
            result = split(poly, splitter)
 
            if len(result.geoms) > 1:
                new_polys.extend(result.geoms)
            else:
                new_polys.append(poly)
 
        polys = new_polys
 
    return [np.array(p.exterior.coords[:-1]) for p in polys]
 
def determine_face_for_polygon(poly_centroid, corners, code):
    """
    Determine which face a polygon belongs to based on its centroid position.
    Returns the face identifier (e.g., 'T', 'R', 'B', 'L', 'H')
    
    corners: [[x_min_l, y_min_l], [x_max_l, y_min_l], [x_max_l, y_max_l], [x_min_l, y_max_l]]
    """
    x_min_l, y_min_l = corners[0]
    x_max_l, y_max_l = corners[2]
    cx, cy = poly_centroid
    
    # For cases with H (horizontal/flat), check if centroid is near center
    if 'H' in code:
        # Define a central region
        center_threshold = 0.3  # 30% of dimension from center
        x_center_min = x_min_l + (x_max_l - x_min_l) * (0.5 - center_threshold)
        x_center_max = x_min_l + (x_max_l - x_min_l) * (0.5 + center_threshold)
        y_center_min = y_min_l + (y_max_l - y_min_l) * (0.5 - center_threshold)
        y_center_max = y_min_l + (y_max_l - y_min_l) * (0.5 + center_threshold)
        
        if (x_center_min <= cx <= x_center_max and y_center_min <= cy <= y_center_max):
            return 'H'
    
    # Otherwise, determine by position relative to edges
    # Calculate distances to each edge
    dist_to_top = abs(cy - y_max_l)
    dist_to_bottom = abs(cy - y_min_l)
    dist_to_right = abs(cx - x_max_l)
    dist_to_left = abs(cx - x_min_l)
    
    # Find the nearest edge
    distances = {
        'T': dist_to_top,
        'B': dist_to_bottom,
        'R': dist_to_right,
        'L': dist_to_left
    }
    
    # Only consider faces that are in the code
    valid_distances = {face: dist for face, dist in distances.items() if face in code}
    
    if valid_distances:
        nearest_face = min(valid_distances, key=valid_distances.get)
        return nearest_face
    
    # Fallback: use quadrant-based logic
    if cx >= 0 and cy >= 0:
        return 'T' if 'T' in code else ('R' if 'R' in code else 'H')
    elif cx >= 0 and cy < 0:
        return 'R' if 'R' in code else ('B' if 'B' in code else 'H')
    elif cx < 0 and cy < 0:
        return 'B' if 'B' in code else ('L' if 'L' in code else 'H')
    else:  # cx < 0 and cy >= 0
        return 'L' if 'L' in code else ('T' if 'T' in code else 'H')
 
def rooftile(corners, plane_point, plane_normal, height=1000.0):
    """
    This function should use PyVista to compute the intersection.
    Since we're providing example logic, we'll return a placeholder.
    """
    import pyvista as pv
    
    pts = np.array(corners)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    
    # Create a closed loop polygon
    n_pts = len(pts)
    faces = [n_pts] + list(range(n_pts))
    footprint = pv.PolyData(pts, faces=faces)
    
    # Extrude to create a cuboid
    cuboid = footprint.extrude((0, 0, height), capping=True)
    
    # Slice with the plane
    intersection_polygon = cuboid.slice(normal=plane_normal, origin=plane_point)
    
    return intersection_polygon
 
def topology_converter_mine(roof_prediction, osm_building):
    """
    Convert roof predictions to topology with plane intersections.
    
    Returns:
        outline: Building outline in 3D
        rect: Oriented bounding box (center, size, angle)
        lines_world: Dividing lines in world coordinates
        code: Roof topology code (e.g., "HTRBL")
        face_data: Dictionary of face data including intersections
    """
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    ring = osm_building["geometry"]["coordinates"][0]
    xs, ys = zip(*[tr.transform(lon, lat) for lon, lat in ring])
 
    outline = np.concatenate((np.array(xs), np.array(ys), np.zeros_like(np.array(xs))), axis=0)
    outline = outline.reshape(3, -1)
 
    probh, probtop, probright, probbottom, probleft = roof_prediction[0:5]
    ah, atop, aright, abottom, aleft = roof_prediction[5:10]
    inc_top, inc_right, inc_bottom, inc_left = roof_prediction[10:14]
    ori_top, ori_right, ori_bottom, ori_left = roof_prediction[14:18]
 
    points = outline[:2, :].T.astype(np.float32)
    rect = cv2.minAreaRect(points)
    center, (width, height), angle = rect
    print(f"Code angle: {angle}°")
 
    x_min_l = -width / 2.0
    x_max_l = width / 2.0
    y_min_l = -height / 2.0
    y_max_l = height / 2.0
 
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
 
    global_dirs = {
        "T": np.array([0., 1.]),
        "R": np.array([1., 0.]),
        "B": np.array([0., -1.]),
        "L": np.array([-1., 0.]),
    }
    global_probs = {"T": probtop, "R": probright, "B": probbottom, "L": probleft}
    global_inc = {"T": inc_top, "R": inc_right, "B": inc_bottom, "L": inc_left}
    global_ori = {"T": ori_top, "R": ori_right, "B": ori_bottom, "L": ori_left}
 
    local_faces = {
        "T": np.array([-sin_t, cos_t]),
        "R": np.array([cos_t, sin_t]),
        "L": np.array([-cos_t, -sin_t]),
        "B": np.array([sin_t, -cos_t]),
    }
 
    remapped = {}
    remapped_inc = {}
    remapped_ori = {}
    for local_name, local_vec in local_faces.items():
        best_face = max(global_dirs, key=lambda g: np.dot(local_vec, global_dirs[g]))
        remapped[local_name] = global_probs[best_face]
        remapped_inc[local_name] = global_inc[best_face]
        remapped_ori[local_name] = global_ori[best_face]
 
    face_data = {
        "T": {"active": remapped["T"] > 0.5, "inclination": remapped_inc["T"], "orientation": remapped_ori["T"]},
        "R": {"active": remapped["R"] > 0.5, "inclination": remapped_inc["R"], "orientation": remapped_ori["R"]},
        "B": {"active": remapped["B"] > 0.5, "inclination": remapped_inc["B"], "orientation": remapped_ori["B"]},
        "L": {"active": remapped["L"] > 0.5, "inclination": remapped_inc["L"], "orientation": remapped_ori["L"]},
        "H": {"active": probh > 0.5, "inclination": 0.0, "orientation": 0.0},
    }
 
    code = ""
    if probh > 0.5: code += "H"
    if remapped["T"] > 0.5: code += "T"
    if remapped["R"] > 0.5: code += "R"
    if remapped["B"] > 0.5: code += "B"
    if remapped["L"] > 0.5: code += "L"
    print(f"Roof code (local frame): {code}")
 
    lines = []
    corners = np.array([
        [x_min_l, y_min_l],
        [x_max_l, y_min_l],
        [x_max_l, y_max_l],
        [x_min_l, y_max_l],
    ])
    
    # Dictionary to store computed intersections
    intersections = {}
    
    # Base height for plane intersections
    base_height = 10.0
    
    # Process each topology case
    match code:
        case "L":
            normal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            intersections["L"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "B":
            normal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            intersections["B"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "R":
            normal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            intersections["R"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "T":
            normal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            intersections["T"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "H":
            normal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            intersections["H"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "BL":
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "B":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Bnormal)
                elif face == "L":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Lnormal)
                    
        case "RL":
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "R":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Rnormal)
                elif face == "L":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Lnormal)
                    
        case "RB":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "R":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Rnormal)
                elif face == "B":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Bnormal)
                    
        case "HL":
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "H":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Hnormal)
                elif face == "L":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Lnormal)
                    
        case "HB":
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "H":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Hnormal)
                elif face == "B":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Bnormal)
                    
        case "TL":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "T":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Tnormal)
                elif face == "L":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Lnormal)
                    
        case "TB":
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "T":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Tnormal)
                elif face == "B":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Bnormal)
                    
        case "TR":
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "T":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Tnormal)
                elif face == "R":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Rnormal)
                    
        case "HR":
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "H":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Hnormal)
                elif face == "R":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Rnormal)
                    
        case "HT":
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "H":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Hnormal)
                elif face == "T":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Tnormal)
                    
        case "RBL":
            lines.append([[0, y_min_l], [x_min_l, y_max_l]])
            lines.append([[0, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"R": Rnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "TBL":
            lines.append([[x_max_l, 0], [x_min_l, y_min_l]])
            lines.append([[x_max_l, 0], [x_min_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"T": Tnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "TRL":
            lines.append([[0, y_max_l], [x_min_l, y_min_l]])
            lines.append([[0, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"T": Tnormal, "R": Rnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "TRB":
            lines.append([[x_min_l, 0], [x_max_l, y_min_l]])
            lines.append([[x_min_l, 0], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"T": Tnormal, "R": Rnormal, "B": Bnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTL":
            lines.append([[0, 0], [x_min_l, y_min_l]])
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTB":
            lines.append([[0, y_min_l], [0, y_max_l]])
            lines.append([[0, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "B": Bnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HBL":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            lines.append([[0, 0], [x_min_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HRL":
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            lines.append([[0, 0], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "R": Rnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HRB":
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "R": Rnormal, "B": Bnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTR":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            lines.append([[0, 0], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "R": Rnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "TRBL":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"T": Tnormal, "R": Rnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HRBL":
            lines.append([[0, 0], [x_min_l, y_max_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "R": Rnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTBL":
            lines.append([[0, 0], [x_min_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTRL":
            lines.append([[0, 0], [x_min_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_min_l]])
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "R": Rnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTRB":
            lines.append([[0, 0], [x_max_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "R": Rnormal, "B": Bnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTRBL":
            # Complex hip roof with ridge
            lines.append([[0.5*x_min_l, 0.5*y_min_l], [0.5*x_min_l, 0.5*y_max_l]])
            lines.append([[0.5*x_max_l, 0.5*y_min_l], [0.5*x_max_l, 0.5*y_max_l]])
            lines.append([[0.5*x_min_l, 0.5*y_min_l], [0.5*x_max_l, 0.5*y_min_l]])
            lines.append([[0.5*x_min_l, 0.5*y_max_l], [0.5*x_max_l, 0.5*y_max_l]])
            lines.append([[0.5*x_min_l, 0.5*y_min_l], [x_min_l, y_min_l]])
            lines.append([[0.5*x_max_l, 0.5*y_min_l], [x_max_l, y_min_l]])
            lines.append([[0.5*x_min_l, 0.5*y_max_l], [x_min_l, y_max_l]])
            lines.append([[0.5*x_max_l, 0.5*y_max_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for i, part in enumerate(parts):
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "R": Rnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
    
    # Convert lines to world coordinates
    cx, cy = center
    for face in intersections:
        # Double the count of every intersection
        intersection = intersections[face]
        intersection.rotate_z(angle, inplace=True)
        intersection.translate([cx, cy, 0], inplace=True)

    # Store intersections in face_data
    face_data["intersections"] = intersections

    def local_to_world(pt):
        xl, yl = pt
        xw = cx + xl * cos_t - yl * sin_t
        yw = cy + xl * sin_t + yl * cos_t
        return [xw, yw]
 
    lines_world = [[local_to_world(p0), local_to_world(p1)] for p0, p1 in lines]
 
    return outline, rect, lines_world, code, face_data

from pyproj import Transformer
import numpy as np

def compute_area_overlap(osm_building, prediction_building) -> float:
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)

    xs, ys = zip(*[tr.transform(lon, lat)
                   for lon, lat in osm_building["geometry"]["coordinates"][0]])
    osm_box = [min(xs), min(ys), max(xs), max(ys)]
    pred_box = prediction_building.box_coords_in_epsg_3763

    ix0, iy0 = max(osm_box[0], pred_box[0]), max(osm_box[1], pred_box[1])
    ix1, iy1 = min(osm_box[2], pred_box[2]), min(osm_box[3], pred_box[3])
    inter_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    area_osm  = (osm_box[2]  - osm_box[0])  * (osm_box[3]  - osm_box[1])
    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    union_area = area_osm + area_pred - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def compute_iou_matrix(osm_buildings, predictions):
    iou_matrix = np.zeros((len(osm_buildings), len(predictions)))
    for i in range(len(osm_buildings)):
        for j in range(len(predictions)):
            iou_matrix[i, j] = compute_area_overlap(osm_buildings[i], predictions[j])
    return iou_matrix

def build_map_html(osm_geojson, predicted_buildings, satellite_image,
                   top_left_corner, bottom_right_corner, matched_data=None):
    tr = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)

    def box_poly(box):
        x0,y0,x1,y1 = box
        return [tr.transform(x,y)[::-1] for x,y in [(x0,y1),(x1,y1),(x1,y0),(x0,y0),(x0,y1)]]

    lats, lons = [], []
    for f in osm_geojson["features"]:
        for lon, lat in f["geometry"]["coordinates"][0]:
            lats.append(lat); lons.append(lon)
    center = [sum(lats)/len(lats), sum(lons)/len(lons)]

    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap")

    tl_lon, tl_lat = tr.transform(*top_left_corner)
    br_lon, br_lat = tr.transform(*bottom_right_corner)
    s,n = min(tl_lat,br_lat), max(tl_lat,br_lat)
    w,e = min(tl_lon,br_lon), max(tl_lon,br_lon)

    sat = folium.FeatureGroup(name="Satellite Image")
    folium.raster_layers.ImageOverlay(image=satellite_image,
        bounds=[[s,w],[n,e]], opacity=0.85,
        interactive=False, cross_origin=False).add_to(sat)
    sat.add_to(m)

    osm = folium.FeatureGroup(name="OSM Buildings")
    folium.GeoJson(osm_geojson,
        style_function=lambda _: {"fillColor":"#3388ff","color":"#1a55cc",
                                   "weight":2,"fillOpacity":0.3},
        tooltip=folium.GeoJsonTooltip(fields=["osm_id","building"],
                                       aliases=["OSM ID","Type"], localize=True)
    ).add_to(osm)
    osm.add_to(m)

    pred = folium.FeatureGroup(name="Model Predictions")
    for i, building in enumerate(predicted_buildings):
        if not building.box_coords_in_epsg_3763: continue
        n_planes = len(building.planes_in_physical_dimensions)
        folium.Polygon(locations=box_poly(building.box_coords_in_epsg_3763),
            color="#cc0000", fill_color="#ff4444", fill_opacity=0.3, weight=2,
            tooltip=f"Building {i} — {n_planes} roof plane(s)").add_to(pred)
        for plane in building.planes_in_physical_dimensions:
            cx = sum(p[0] for p in plane.corners)/4
            cy = sum(p[1] for p in plane.corners)/4
            lon, lat = tr.transform(cx, cy)
            folium.CircleMarker(location=[lat,lon], radius=4,
                color="#ff9900", fill=True, fill_opacity=0.8,
                tooltip=(f"Face: {plane.face}<br>Prob: {plane.probability:.2f}<br>"
                         f"Inc: {plane.inclination:.2f}<br>Ori: {plane.orientation:.2f}")
            ).add_to(pred)
    pred.add_to(m)

    # Minimum bounding boxes (rotated rectangles from cv2.minAreaRect)
    if matched_data:
        mbb = folium.FeatureGroup(name="Min Bounding Boxes")
        for entry in matched_data:
            oriented_rect = entry.get("roof_planes")
            if oriented_rect is None:
                continue
            # cv2.boxPoints returns the 4 corners of the rotated rectangle in EPSG:3763
            corners_3763 = cv2.boxPoints(oriented_rect).astype(float)  # shape (4, 2)
            # Close the ring by appending the first point again
            corners_latlon = []
            for x, y in corners_3763:
                lon, lat = tr.transform(x, y)
                corners_latlon.append([lat, lon])
            corners_latlon.append(corners_latlon[0])  # close polygon
            osm_id = entry.get("osm_id", "?")
            _, (w_rect, h_rect), angle = oriented_rect
            folium.Polygon(
                locations=corners_latlon,
                color="#00cc66", fill_color="#00ff88", fill_opacity=0.25, weight=2,
                dash_array="6",
                tooltip=(f"OSM {osm_id} — Min Bounding Box<br>"
                         f"W: {w_rect:.1f} m  H: {h_rect:.1f} m<br>"
                         f"Angle: {angle:.1f}°"
                         f"Name: {entry.get('code')}"),
            ).add_to(mbb)
        mbb.add_to(m)


    # Roof topology lines (rotated into world coordinates)
    if matched_data:
        topo_layer = folium.FeatureGroup(name="Roof Topology Lines")
        n_lines_total = 0
        for entry in matched_data:
            lines_world = entry.get("lines_world", [])
            osm_id = entry.get("osm_id", "?")
            print(f"OSM {osm_id}: {len(lines_world)} topology line(s)")
            for line in lines_world:
                p0, p1 = line
                lon0, lat0 = tr.transform(p0[0], p0[1])
                lon1, lat1 = tr.transform(p1[0], p1[1])
                print(f"  line latlon: ({lat0:.6f},{lon0:.6f}) → ({lat1:.6f},{lon1:.6f})")
                folium.PolyLine(
                    locations=[[lat0, lon0], [lat1, lon1]],
                    color="#ffffff", weight=2, opacity=0.9,
                    tooltip=f"OSM {osm_id} — roof line",
                ).add_to(topo_layer)
                n_lines_total += 1
        print(f"Total topology lines drawn: {n_lines_total}")
        topo_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

model = CachedModel()
jsonbox = "{\"north\":38.71287282568031,\"south\":38.71177693488299,\"east\":-9.141268730163576,\"west\":-9.142599105834963}"

bbox  = json.loads(jsonbox)
north = bbox["north"]; south = bbox["south"]
east  = bbox["east"];  west  = bbox["west"]

print("Converting coordinates…")
t = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
tl_x, tl_y = t.transform(west, north)
br_x, br_y = t.transform(east, south)
area_m2 = abs(br_x - tl_x) * abs(tl_y - br_y)
print(f"📐  Area: {area_m2/1e6:.4f} km²")
print(f"🔁  EPSG:4326 → EPSG:3763 done")

print("Connecting to WMTS…")
print("📡  Connecting to DGT WMTS satellite service…")
wmts_url = ("https://cartografia.dgterritorio.gov.pt/ortos2018/service"
            "?service=WMTS&request=GetCapabilities")
wmts   = WebMapTileService(wmts_url)
matrix = wmts.tilematrixsets["PTTM_06"].tilematrix["14"]
col_min, col_max, row_min, row_max = get_tile_indices(tl_x, br_y, br_x, tl_y, matrix)
total_tiles = (col_max+1-col_min) * (row_max+1-row_min)
print(f"🛰   Service ready — {total_tiles} tile(s) to download")

def tile_cb(done, total):
    print(f"Downloading tiles… {done}/{total}")

satellite_image, res, _ = retrieve_satelite_image(
    (tl_x, tl_y), (br_x, br_y), progress_cb=tile_cb)
h, w = satellite_image.shape[:2]
print(f"✅  Satellite image ready  ({w}×{h} px, {res:.3f} m/px)")

print("Fetching OSM buildings…")
print("🗺   Querying Overpass API…")
geojson = get_osm_buildings_cached((tl_x, tl_y), (br_x, br_y))
print(f"✅  OSM done — {len(geojson['features'])} footprint(s)")


building_threshold = 0.1
overlap_threshold = 0.3
print("Running YOLO inference…")
print(f"🤖  Running detector  (conf≥{building_threshold:.2f}, iou≤{overlap_threshold:.2f})…")
img_bgr   = cv2.cvtColor(satellite_image, cv2.COLOR_RGB2BGR)
buildings = retrieve_prediction_list(img_bgr, (tl_x, tl_y), res,
                                        building_threshold, overlap_threshold, model)
total_planes = sum(len(b.planes_in_physical_dimensions) for b in buildings)
print(f"✅  {len(buildings)} building(s), {total_planes} roof plane(s)")

# here is where we need to assing all rooftop predictions to the OSM buildings
iou_mat = compute_iou_matrix(geojson['features'], buildings)
matched_data = []
 
for i, osm_feat in enumerate(geojson['features']):
    best_match_idx = np.argmax(iou_mat[i, :])
    if iou_mat[i, best_match_idx] > 0.3:
        pred = buildings[best_match_idx]
        outline_coordinates = []
        outline,orientedbox,lines_world,code,face_data = topology_converter_mine(pred.raw_roof_data, osm_feat) 
        matched_data.append({
            "osm_id": osm_feat["properties"].get("osm_id"),
            "footprint": outline,
            "roof_planes": orientedbox,
            "lines_world": lines_world,
            "code":code,
            "face_data":face_data,
        })


# then we need to generate the 3D view for each building

# Stage 6 — Render
print("Rendering map…")
print("🗾  Compositing layers…")
map_html = build_map_html(geojson, buildings, satellite_image,
                            (tl_x, tl_y), (br_x, br_y), matched_data=matched_data)
with open("building_map.html", "w", encoding="utf-8") as f:
    f.write(map_html)

print(f"✅ Map saved successfully to: building_map.html")
print("Double-click the file to open it in your browser.")

print("Converting rooftops to real planes…")


import pyvista as pv
import numpy as np
import math
def render_3d_tiles_old(matched_data, base_height=10.0, max_roof_extension=20.0):
    """
    Renders buildings by taking a solid cuboid and 'carving' it with 
    intersecting roof planes based on the model predictions.
    """
    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background("white")

    for entry in matched_data:
        # 1. Get the oriented bounding box details from the matched data
        # 'roof_planes' in your matched_data is the 'rect' from cv2.minAreaRect
        oriented_rect = entry.get("roof_planes") 
        if oriented_rect is None:
            continue
            
        center, (width, height), angle = oriented_rect
        face_data = entry.get("face_data", {})
        osm_id = entry.get("osm_id", "Unknown")

        # 2. Create the initial solid Cuboid in LOCAL coordinates
        # We start with a block taller than the building to allow for clipping
        z_min = 0.0
        z_max = base_height + max_roof_extension
        local_box = pv.Box(bounds=[-width/2, width/2, -height/2, height/2, z_min, z_max])

        # 3. Define the cutting planes for each face (T, R, B, L)
        # Normal points 'up and out'. Clipping removes geometry in the direction of the normal.
        face_configs = {
            "T": {"origin": [0,  height/2, base_height], "normal_2d": [0,  1]}, # Top (+y)
            "B": {"origin": [0, -height/2, base_height], "normal_2d": [0, -1]}, # Bottom (-y)
            "R": {"origin": [ width/2, 0, base_height], "normal_2d": [ 1,  0]}, # Right (+x)
            "L": {"origin": [-width/2, 0, base_height], "normal_2d": [-1,  0]}, # Left (-x)
        }

        fused_mesh = local_box
        sloped_active = False

        for side, config in face_configs.items():
            data = face_data.get(side, {})
            if data.get("active"):
                sloped_active = True
                # Inclination is 0-1 (mapped to 0 to ~90 degrees)
                # We cap at 85 degrees to prevent infinite planes/errors
                alpha = min(data.get("inclination", 0.0) * (math.pi / 2), math.radians(85))
                
                nx, ny = config["normal_2d"]
                # The 3D normal vector of the roof plane
                normal_3d = [nx * math.sin(alpha), ny * math.sin(alpha), math.cos(alpha)]
                
                # Clip the mesh: keep the part 'behind' the plane
                fused_mesh = fused_mesh.clip(normal=normal_3d, origin=config["origin"], invert=False)

        # 4. Handle Flat surfaces (H code)
        # If 'H' is active, or no slopes were defined, slice the top horizontally
        if face_data.get("H", {}).get("active") or not sloped_active:
            # If there are slopes, the flat part is usually at the ridge height
            # If not, it's at the base_height
            h_level = base_height + (2.0 if sloped_active else 0.0) 
            fused_mesh = fused_mesh.clip(normal=[0, 0, 1], origin=[0, 0, h_level], invert=False)

        # 5. Transform from Local to World Coordinates
        # Rotate by the cv2.minAreaRect angle, then translate to the global center
        fused_mesh.rotate_z(angle, inplace=True)
        fused_mesh.translate([center[0], center[1], 0], inplace=True)

        # 6. Add to plotter
        pl.add_mesh(fused_mesh, color="lightblue", opacity=0.7, 
                    show_edges=True, edge_color="steelblue", line_width=1.5)

    pl.add_axes()
    pl.show_grid(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    pl.show()

def render_3d_tiles(matched_data):
    """
    Renders buildings by iterating through the pre-computed and pre-rotated
    3D roof polygons generated by the topology converter.
    """
    pl = pv.Plotter(window_size=[1400, 900])
    pl.set_background("white")
    mappedcolors = {}
    mappedcolors["T"] = "red"
    mappedcolors["R"] = "blue"
    mappedcolors["B"] = "green"
    mappedcolors["L"] = "yellow"
    mappedcolors["H"] = "black"
    buildings_rendered = 0

    for entry in matched_data:
        face_data = entry.get("face_data", {})
        intersections = face_data.get("intersections", {})
        osm_id = entry.get("osm_id", "Unknown")

        if not intersections:
            print(f"⚠️ Skipping OSM {osm_id} — no intersection tiles found.")
            continue

        buildings_rendered += 1

        # Iterate through each ready-made 3D roof tile and add it to the scene
        for face_name, tile_mesh in intersections.items():
            if tile_mesh is not None:
                pl.add_mesh(
                    tile_mesh.delaunay_2d(), 
                    style="surface",
                    color=mappedcolors[face_name], 
                )

    print(f"🗺️ Rendering {buildings_rendered} buildings in PyVista...")

    pl.add_axes()
    # Since these are in real world coordinates (EPSG:3763), the grid will reflect actual map coordinates
    pl.show_grid(xlabel="X (m - EPSG:3763)", ylabel="Y (m - EPSG:3763)", zlabel="Z (m)")
    pl.show()

# Execute the new renderer
render_3d_tiles(matched_data)