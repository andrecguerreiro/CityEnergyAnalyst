325476i98p0'    f jimport numpy as np
from ultralytics import YOLO
from owslib.wmts import WebMapTileService
from PIL import Image
import io
import cv2
import folium

def wmts_tile_to_array(tile_data):
    '''
    the function converts the tiled images into RGB channels
    '''
    img = Image.open(io.BytesIO(tile_data.read()))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    array = np.array(img)
    return array

def get_tile_indices(xmin:float, ymin:float, xmax:float, ymax:float, matrix):
    ''' 
    the function receies the coordinates of the retrieved image and computes the offset to matrix indices
    '''
    tile_size = matrix.tilewidth * matrix.scaledenominator * 0.28e-3
    origin_x, origin_y = matrix.topleftcorner

    col_min = int((xmin - origin_x) // tile_size)
    col_max = int((xmax - origin_x) // tile_size)
    row_min = int((origin_y - ymax) // tile_size)
    row_max = int((origin_y - ymin) // tile_size)

    return col_min, col_max, row_min, row_max

def retrieve_satelite_image(top_left_corner : list[float] ,bottom_right_corner: list[float]):
    '''
    the function receives a set of coordinates [the upper and the lower corner coordinates of a region]
    and stiches the predictions into a common single image. Because image retrival is not perfect, we then
    crop the image so that the corners match exactly to the physical coordinates that we received
    '''
    wmts_url = "https://cartografia.dgterritorio.gov.pt/ortos2018/service?service=WMTS&request=GetCapabilities"
    wmts = WebMapTileService(wmts_url)
    
    xmin = top_left_corner[0]
    ymax = top_left_corner[1]
    xmax = bottom_right_corner[0]
    ymin = bottom_right_corner[1] 
    layer = "Ortos2018-RGB" 
    tile_matrix_set = "PTTM_06"
    zoom_level = "14"
    matrix = wmts.tilematrixsets[tile_matrix_set].tilematrix[zoom_level]
    res = matrix.scaledenominator * 0.28e-3 
    tile_size_m = matrix.tilewidth * res
    col_min, col_max, row_min, row_max = get_tile_indices(xmin, ymin, xmax, ymax, matrix)
    n_rows = row_max + 1 - row_min
    n_cols = col_max + 1 - col_min
    processed_blocks = np.empty((n_rows, n_cols), dtype=object)
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            tile = wmts.gettile(
                layer=layer,
                tilematrixset=tile_matrix_set,
                tilematrix=zoom_level,
                row=row,
                column=col,
                format="image/png"
            )
            processed_blocks[row - row_min, col - col_min] = {
                'img': wmts_tile_to_array(tile)
            }
    block_h, block_w = processed_blocks[0, 0]['img'].shape[:2]
    stitched = np.zeros((n_rows * block_h, n_cols * block_w, 3), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            block_img = processed_blocks[row, col]['img']
            y0, y1 = row * block_h, (row + 1) * block_h
            x0, x1 = col * block_w, (col + 1) * block_w
            stitched[y0:y1, x0:x1, :] = block_img
    origin_x = matrix.topleftcorner[0] + col_min * tile_size_m
    origin_y = matrix.topleftcorner[1] - row_min * tile_size_m
    
    col_start = int(round((xmin - origin_x) / res))
    row_start = int(round((origin_y - ymax) / res)) 
    col_end   = int(round((xmax - origin_x) / res))
    row_end   = int(round((origin_y - ymin) / res))
    satellite_image = stitched[row_start:row_end, col_start:col_end, :]
    
    def conversion(x, y):
        col = int((x - xmin) / res)
        row = int((ymax - y) / res)
        return col, row
    
    return satellite_image, res, conversion

import math
import numpy as np
from ultralytics import YOLO

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CachedModel:
    def __init__(self):
        self.model = YOLO("best.pt")
        self.model.to("cpu")
        self.model.eval()


class RoofPlane:
    """One roof face (top/right/bottom/left) in EPSG:3763 + Z (meters)."""
    def __init__(self, face: str, probability: float,
                 corners: list,   # [p_start, p_end, p_end_bot, p_start_bot] — each np.array([x,y,z])
                 normal: np.ndarray,
                 inclination: float,
                 orientation: float):
        self.face        = face          # "top" | "right" | "bottom" | "left"
        self.probability = probability
        self.corners     = corners       # 4 x (x, y, z) in EPSG:3763 + elevation metres
        self.normal      = normal        # unit normal vector
        self.inclination = inclination   # raw model output [0-1], 0=flat, 1=vertical
        self.orientation = orientation   # raw model output [0-1]

class EstimatedBuilding:
    def __init__(self):
        # [x_min, y_min, x_max, y_max] in EPSG:3763 metres
        self.box_coords_in_epsg_3763: list[float] = []
        # One RoofPlane per active face
        self.planes_in_physical_dimensions: list[RoofPlane] = []

    def convert_tensor_prediction_to_building(
        self,
        boxes_xywhn: np.ndarray,   # shape (4,) — normalised cx,cy,w,h from YOLO
        roof_prediction: np.ndarray,  # shape (18,) — already sigmoid-ed
        top_left_corner: tuple,     # (xmin, ymax) in EPSG:3763  ← from retrieve_satelite_image
        res: float,                 # metres/pixel                ← from retrieve_satelite_image
        image_shape: tuple          # (img_h, img_w, ...)         ← satellite_image.shape
    ):
        img_h, img_w = image_shape[:2]
        xmin_map, ymax_map = top_left_corner   # map origin of the cropped image

        # ── 1. Bounding box → EPSG:3763 ──────────────────────────────────────
        cx_norm, cy_norm, w_norm, h_norm = boxes_xywhn

        # Pixel coords of box centre and extents
        cx_px = cx_norm * img_w
        cy_px = cy_norm * img_h
        w_px  = w_norm  * img_w
        h_px  = h_norm  * img_h

        # Inverse of conversion():  x = xmin + col*res,  y = ymax - row*res
        cx_map = xmin_map + cx_px * res
        cy_map = ymax_map - cy_px * res
        w_map  = w_px * res
        h_map  = h_px * res

        self.box_coords_in_epsg_3763 = [
            cx_map - w_map / 2,   # x_min
            cy_map - h_map / 2,   # y_min
            cx_map + w_map / 2,   # x_max
            cy_map + h_map / 2,   # y_max
        ]

        # ── 2. Roof planes → EPSG:3763 + Z ───────────────────────────────────
        probh, probtop, probright, probbottom, probleft = roof_prediction[0:5]
        # (areas prediction[5:10] available if needed for slope length)
        inc_top, inc_right, inc_bottom, inc_left = roof_prediction[10:14]
        ori_top, ori_right, ori_bottom, ori_left = roof_prediction[14:18]

        base_z = 0.0          # attach planes at ground; add building height if known
        dx = w_map / 2
        dy = h_map / 2

        # Four corners of the building footprint in map space.
        # EPSG:3763 is a standard projected CRS: x=Easting (right), y=Northing (up).
        p_tl = np.array([cx_map - dx,  cy_map + dy, base_z])
        p_tr = np.array([cx_map + dx,  cy_map + dy, base_z])
        p_br = np.array([cx_map + dx,  cy_map - dy, base_z])
        p_bl = np.array([cx_map - dx,  cy_map - dy, base_z])

        face_params = [
            # (name,     prob,        inc,        ori,        base_azimuth, p_start, p_end)
            ("top",    probtop,    inc_top,    ori_top,    math.pi/2,   p_tl, p_tr),
            ("right",  probright,  inc_right,  ori_right,  0.0,         p_tr, p_br),
            ("bottom", probbottom, inc_bottom, ori_bottom, -math.pi/2,  p_br, p_bl),
            ("left",   probleft,   inc_left,   ori_left,   math.pi,     p_bl, p_tl),
        ]

        for face_name, prob, inc, ori, base_azimuth, p_start, p_end in face_params:
            if prob < 0.5:
                continue

            # Model outputs → angles  (same logic as visualize_building_prediction)
            tilt    = inc * (math.pi / 2)                          # 0=flat, π/2=vertical
            azimuth = (ori - 0.5) * (math.pi / 2) + base_azimuth  # direction face points

            # Unit normal in map-space (x=East, y=North, z=Up)
            nz = math.cos(tilt)
            nx = math.sin(tilt) * math.cos(azimuth)
            ny = math.sin(tilt) * math.sin(azimuth)
            normal = np.array([nx, ny, nz])
            normal /= np.linalg.norm(normal)

            # Slope vector: perpendicular to the edge AND to the normal, pointing "downhill"
            edge_unit  = (p_end - p_start) / np.linalg.norm(p_end - p_start)
            slope_vec  = np.cross(edge_unit, normal)
            if slope_vec[2] > 0:
                slope_vec = -slope_vec   # ensure it goes downward

            # Slope length in metres (mirrors the visualisation heuristic)
            slope_len = max(w_map, h_map) * 0.2

            p_start_bot = p_start + slope_vec * slope_len
            p_end_bot   = p_end   + slope_vec * slope_len

            self.planes_in_physical_dimensions.append(
                RoofPlane(
                    face        = face_name,
                    probability = float(prob),
                    corners     = [p_start, p_end, p_end_bot, p_start_bot],
                    normal      = normal,
                    inclination = float(inc),
                    orientation = float(ori),
                )
            )


def retrieve_prediction_list(
    satellite_image: np.ndarray,
    top_left_corner: tuple,    # (xmin, ymax) in EPSG:3763 — from retrieve_satelite_image
    res: float,                # metres/pixel               — from retrieve_satelite_image
    building_threshold: float,
    overlap_threshold: float,
    cached_model: CachedModel,
) -> list[EstimatedBuilding]:

    results = cached_model.model(
        satellite_image,
        conf=building_threshold,
        iou=overlap_threshold
    )[0]

    predictions = []
    for i in range(results.boxes.shape[0]):
        building = EstimatedBuilding()
        building.convert_tensor_prediction_to_building(
            boxes_xywhn    = np.array(results.boxes[i].xywhn[0]),        # (4,)
            roof_prediction= np.array(sigmoid(results.roof.data[i, :])), # (18,)
            top_left_corner= top_left_corner,
            res            = res,
            image_shape    = satellite_image.shape,
        )
        predictions.append(building)

    return predictions

from pyproj import Transformer
import requests
import json

def get_osm_buildings(top_left_corner, bottom_right_corner):
    """
    top_left_corner, bottom_right_corner in EPSG:3763
    Returns a GeoJSON FeatureCollection of buildings.
    """
    transformer = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)

    tl_lon, tl_lat = transformer.transform(top_left_corner[0],     top_left_corner[1])
    br_lon, br_lat = transformer.transform(bottom_right_corner[0], bottom_right_corner[1])

    south = min(tl_lat, br_lat)
    north = max(tl_lat, br_lat)
    west  = min(tl_lon, br_lon)
    east  = max(tl_lon, br_lon)

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    response = requests.post(overpass_url, data=query)
    response.raise_for_status()
    osm_data = response.json()

    nodes = {
        el["id"]: (el["lon"], el["lat"])
        for el in osm_data["elements"]
        if el["type"] == "node"
    }

    features = []
    for el in osm_data["elements"]:
        if el["type"] != "way":
            continue
        coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
        if len(coords) < 3:
            continue

        features.append({
            "type": "Feature",
            "properties": {
                "osm_id": el["id"],
                **el.get("tags", {})   # building name, height, etc. if present
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]  # GeoJSON polygons are [lon, lat]
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }


def visualize_on_map(
    osm_geojson: dict,
    predicted_buildings: list,
    satellite_image: np.ndarray,       # ← new
    top_left_corner: tuple,            # ← new (EPSG:3763)
    bottom_right_corner: tuple,        # ← new (EPSG:3763)
    output_path: str = "map.html"
):
    transformer = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)

    def box_to_wgs84_polygon(box_coords):
        xmin, ymin, xmax, ymax = box_coords
        corners_3763 = [
            (xmin, ymax), (xmax, ymax),
            (xmax, ymin), (xmin, ymin),
            (xmin, ymax),
        ]
        return [transformer.transform(x, y)[::-1] for x, y in corners_3763]

    # ── Map centre ────────────────────────────────────────────────────────────
    all_lats, all_lons = [], []
    for f in osm_geojson["features"]:
        for lon, lat in f["geometry"]["coordinates"][0]:
            all_lats.append(lat)
            all_lons.append(lon)
    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap")

    # ── Layer 0: Satellite image ──────────────────────────────────────────────
    # Convert the image corners from EPSG:3763 → WGS84
    tl_lon, tl_lat = transformer.transform(top_left_corner[0],     top_left_corner[1])
    br_lon, br_lat = transformer.transform(bottom_right_corner[0], bottom_right_corner[1])

    # ImageOverlay bounds format: [[south, west], [north, east]]
    south, north = min(tl_lat, br_lat), max(tl_lat, br_lat)
    west,  east  = min(tl_lon, br_lon), max(tl_lon, br_lon)

    sat_layer = folium.FeatureGroup(name="Satellite Image")
    folium.raster_layers.ImageOverlay(
        image=satellite_image,
        bounds=[[south, west], [north, east]],
        opacity=0.85,
        interactive=False,
        cross_origin=False,
    ).add_to(sat_layer)
    sat_layer.add_to(m)

    # ── Layer 1: OSM buildings (blue) ─────────────────────────────────────────
    osm_layer = folium.FeatureGroup(name="OSM Buildings")
    folium.GeoJson(
        osm_geojson,
        style_function=lambda _: {
            "fillColor":   "#3388ff",
            "color":       "#1a55cc",
            "weight":       2,
            "fillOpacity": 0.3,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["osm_id", "building"],
            aliases=["OSM ID", "Type"],
            localize=True
        )
    ).add_to(osm_layer)
    osm_layer.add_to(m)

    # ── Layer 2: Model predictions (red) ─────────────────────────────────────
    pred_layer = folium.FeatureGroup(name="Model Predictions")
    for i, building in enumerate(predicted_buildings):
        if not building.box_coords_in_epsg_3763:
            continue

        polygon_coords = box_to_wgs84_polygon(building.box_coords_in_epsg_3763)
        n_planes = len(building.planes_in_physical_dimensions)

        folium.Polygon(
            locations=polygon_coords,
            color="#cc0000",
            fill_color="#ff4444",
            fill_opacity=0.3,
            weight=2,
            tooltip=f"Building {i} — {n_planes} roof plane(s)"
        ).add_to(pred_layer)

        for plane in building.planes_in_physical_dimensions:
            cx = sum(p[0] for p in plane.corners) / 4
            cy = sum(p[1] for p in plane.corners) / 4
            lon, lat = transformer.transform(cx, cy)
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#ff9900",
                fill=True,
                fill_opacity=0.8,
                tooltip=(
                    f"Face: {plane.face}<br>"
                    f"Prob: {plane.probability:.2f}<br>"
                    f"Inc: {plane.inclination:.2f}<br>"
                    f"Ori: {plane.orientation:.2f}"
                )
            ).add_to(pred_layer)

    pred_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_path)
    print(f"Map saved to {output_path}")
    return m

def convert_prediction_list_to_geojson(predictions,top_left_corner,bottom_right_corner,filename="file.json"):
    for predict in predictions:
        print("i am parsing a prediction")
        
def visualize_geojson(filename):
    print("i am seeing the geojson")
    
from pyproj import Transformer
    
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
tl_lon, tl_lat = -9.137569523316204, 38.71035316799024
br_lon, br_lat = -9.135325465227648, 38.7092395515382

tl_x, tl_y = transformer.transform(tl_lon, tl_lat)
br_x, br_y = transformer.transform(br_lon, br_lat)
    
print(f"top_left_corner     = ({tl_x}, {tl_y})")
print(f"bottom_right_corner = ({br_x}, {br_y})")

satellite_image, res, conversion = retrieve_satelite_image(
    top_left_corner     = (tl_x, tl_y),
    bottom_right_corner = (br_x, br_y),
)

geojson = get_osm_buildings(
    top_left_corner     = (tl_x, tl_y),
    bottom_right_corner = (br_x, br_y),
)

img = cv2.cvtColor(satellite_image, cv2.COLOR_RGB2BGR)

model = CachedModel()

# buildings = retrieve_prediction_list(
#     satellite_image    = img,
#     top_left_corner    = (tl_x, tl_y),
#     res                = res,
#     building_threshold = 0.5,
#     overlap_threshold  = 0.5,
#     cached_model       = model,
# )
# 
# visualize_on_map(
#     osm_geojson         = geojson,
#     predicted_buildings = buildings,
#     satellite_image     = satellite_image,
#     top_left_corner     = (tl_x, tl_y),
#     bottom_right_corner = (br_x, br_y),
#     output_path         = "buildings_map.html"
# )
# 
# import webbrowser
# import os
# 
# webbrowser.open(f"file://{os.path.abspath('buildings_map.html')}")

import gradio as gr
import webbrowser, os

def run_pipeline(building_threshold, overlap_threshold):
    buildings = retrieve_prediction_list(
        satellite_image    = cv2.cvtColor(satellite_image, cv2.COLOR_RGB2BGR),
        top_left_corner    = (tl_x, tl_y),
        res                = res,
        building_threshold = building_threshold,
        overlap_threshold  = overlap_threshold,
        cached_model       = model,
    )

    visualize_on_map(
        osm_geojson         = geojson,
        predicted_buildings = buildings,
        satellite_image     = satellite_image,
        top_left_corner     = (tl_x, tl_y),
        bottom_right_corner = (br_x, br_y),
        output_path         = "buildings_map.html"
    )

    # Open the map in a new browser tab automatically
    webbrowser.open(f"file://{os.path.abspath('buildings_map.html')}")

    return f"Found {len(buildings)} buildings — map opened in browser"


with gr.Blocks(title="Building Detector") as demo:
    gr.Markdown("## Building Detection Threshold Tuner")

    with gr.Row():
        building_slider = gr.Slider(
            minimum=0.1, maximum=0.95, step=0.05,
            value=0.5, label="Building Confidence Threshold"
        )
        overlap_slider = gr.Slider(
            minimum=0.1, maximum=0.95, step=0.05,
            value=0.5, label="Overlap (IoU) Threshold"
        )

    run_btn    = gr.Button("Run", variant="primary")
    status_txt = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn      = run_pipeline,
        inputs  = [building_slider, overlap_slider],
        outputs = [status_txt]
    )

demo.launch()