
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import json

def obtain_mask_from_coordinates_meterbuffer(
    filename,
    coordinates,              # (x,y) in CRS units (e.g. EPSG:3763)
    surrounding_cropping,     # (pixels_y, pixels_x) from satellite crop
    satellite_resolution      # satellite pixel size in CRS units (meters/pixel)
):
    # coordinates: (x, y) in the same CRS as you expect (EPSG:3763)
    x, y = coordinates

    with rasterio.open(filename) as src:
        # If file CRS differs from coordinates' CRS, reproject the point
        expected_crs = "EPSG:3763"
        if src.crs is None:
            raise ValueError("Source has no CRS")
        if src.crs.to_string() != expected_crs:
            # Reproject the point from expected_crs to src.crs
            # rasterio.warp.transform takes lists
            xs, ys = rasterio.warp.transform(expected_crs, src.crs, [x], [y])
            x_src, y_src = xs[0], ys[0]
        else:
            x_src, y_src = x, y

        # Convert satellite pixel counts -> meters (map units)
        buffer_meters_x = surrounding_cropping[1] * satellite_resolution
        buffer_meters_y = surrounding_cropping[0] * satellite_resolution

        left = x_src - buffer_meters_x * 0.5
        right = x_src + buffer_meters_x * 0.5
        bottom = y_src - buffer_meters_y * 0.5
        top = y_src + buffer_meters_y * 0.5

        # make window from bounds using this file's transform
        window = from_bounds(left, bottom, right, top, src.transform)
        crop_data = src.read(window=window)
        crop_transform = rasterio.windows.transform(window, src.transform)

    return crop_data, crop_transform

def obtain_mask_from_coordinates(filename, coordinates, surrounding_cropping,resolution):
    with rasterio.open(filename) as src:
        # Verify CRS
        if src.crs.to_string() != "EPSG:3763":
            print(f"Warning: File CRS is {src.crs}, expected EPSG:3763")
        left = coordinates[0]  - surrounding_cropping[1]*resolution*0.5
        right = coordinates[0]  + surrounding_cropping[1]*resolution*0.5
        bottom = coordinates[1] - surrounding_cropping[0]*resolution*0.5
        top = coordinates[1] + surrounding_cropping[0]*resolution*0.5
        
        # Read window and get transform
        window = from_bounds(left, bottom, right, top, src.transform)
        crop_data = src.read(window=window)
        crop_transform = rasterio.windows.transform(window, src.transform)
        
    return crop_data, crop_transform

# Calculate tile indices for bbox
def get_tile_indices(xmin, ymin, xmax, ymax, matrix):
    tile_size = matrix.tilewidth * matrix.scaledenominator * 0.28e-3
    origin_x, origin_y = matrix.topleftcorner

    col_min = int((xmin - origin_x) // tile_size)
    col_max = int((xmax - origin_x) // tile_size)
    row_min = int((origin_y - ymax) // tile_size)
    row_max = int((origin_y - ymin) // tile_size)

    return col_min, col_max, row_min, row_max

# Download tiles
import os
import requests
from PIL import Image
from io import BytesIO

# Connect to WMTS
from owslib.wmts import WebMapTileService

# Calculate tile indices for bbox
def get_tile_indices(xmin, ymin, xmax, ymax, matrix):
    tile_size = matrix.tilewidth * matrix.scaledenominator * 0.28e-3
    origin_x, origin_y = matrix.topleftcorner

    col_min = int((xmin - origin_x) // tile_size)
    col_max = int((xmax - origin_x) // tile_size)
    row_min = int((origin_y - ymax) // tile_size)
    row_max = int((origin_y - ymin) // tile_size)

    return col_min, col_max, row_min, row_max

from PIL import Image
import io
# Convert the tile to a NumPy array
def wmts_tile_to_array(tile_data):
    # Read the image data into a PIL Image
    img = Image.open(io.BytesIO(tile_data.read()))
    
    # Convert to RGB (in case it's RGBA or other format)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to NumPy array
    array = np.array(img)
    
    return array

def rqst_satelite_image(coordinates,surrounding_cropping):
    wmts_url = "https://cartografia.dgterritorio.gov.pt/ortos2018/service?service=WMTS&request=GetCapabilities"
    # wmts_url = "https://ortos.dgterritorio.gov.pt/wms/ortosat2023?service=wms&request=getcapabilities"
    wmts = WebMapTileService(wmts_url)

    # 200 m buffer
    buffer = surrounding_cropping #meters
    xmin, xmax = coordinates[0] - buffer, coordinates[0] + buffer
    ymin, ymax = coordinates[1] - buffer, coordinates[1] + buffer

    layer = "Ortos2018-RGB"  #"Ortho_2018_RBG"
    tile_matrix_set = "PTTM_06"  #"EPSG:3763"  # or check what tile matrix sets are available
    zoom = "14"
    matrix = wmts.tilematrixsets[tile_matrix_set].tilematrix[str(zoom)]
    col_min, col_max, row_min, row_max = get_tile_indices(xmin,ymin,xmax,ymax,matrix)
    scale_denominator = matrix.scaledenominator
    resolution = scale_denominator * 0.00028  # Meters/pixel
    processed_blocks = np.empty((row_max+1-row_min, col_max+1-col_min), dtype=object)
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            tile = wmts.gettile(
                layer=layer,
                tilematrixset=tile_matrix_set,
                tilematrix=zoom,
                row=row,
                column=col,
                format="image/png"
            )
            # Store results
            processed_blocks[row-row_min,col-col_min] = {
                'img': wmts_tile_to_array(tile)
            }
    block_size = processed_blocks[0,0]['img'].shape
    satelite_image = np.zeros((processed_blocks.shape[0]*block_size[0],processed_blocks.shape[1]*block_size[1],3), dtype=np.uint8)
    for row in range(processed_blocks.shape[0]):
        for col in range(processed_blocks.shape[1]):
            block_img = processed_blocks[row, col]['img']
            y_start = row * block_size[1]
            y_end = y_start + block_size[1]
            x_start = col * block_size[0]
            x_end = x_start + block_size[0]
            satelite_image[y_start:y_end, x_start:x_end, :] = block_img

    def conversion(x,y):
        res = matrix.scaledenominator * 0.28e-3  # meters per pixel
        tile_size_m = matrix.tilewidth * res     # tile width in meters

        origin_x = matrix.topleftcorner[0] + col_min * tile_size_m
        origin_y = matrix.topleftcorner[1] - row_min * tile_size_m

        col = int((x - origin_x) / res)
        row = int((origin_y - y) / res)
        return col, row

    return satelite_image,resolution,conversion

def segment_height_image(satelite_image,depth):
    binary_image = np.zeros(satelite_image.shape[0],satelite_image.shape[1])
    segmented_depth = depth
    for row in range():
        for col in range():
            if(binary_image[row,col]>0):
                segmented_depth[row,col] = depth[row,col]
    return segmented_depth

def dummy_segment_height_image(satelite_image,depth):
    binary_image = np.zeros(satelite_image.shape[0],satelite_image.shape[1])
    segmented_depth = depth
    for row in range():
        for col in range():
            if(binary_image[row,col]>0):
                segmented_depth[row,col] = depth[row,col]
    return segmented_depth
            

def convert_to_point_cloud(segmented_depth,colored_image):
    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(segmented_depth.shape[0]), np.arange(segmented_depth.shape[1]))
    x = (x - segmented_depth.shape[0] / 2)
    y = (y - segmented_depth.shape[1] / 2)
    z = np.array(segmented_depth)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(colored_image).reshape(-1, 3) / 255.0

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

# Stitch tiles
def stitch_tiles(out_folder, zoom, col_min, col_max, row_min, row_max, tile_size=256):
    width = (col_max - col_min + 1) * tile_size
    height = (row_max - row_min + 1) * tile_size
    result = Image.new("RGB", (width, height))

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            filename = os.path.join(out_folder, f"tile_{zoom}_{col}_{row}.png")
            if not os.path.exists(filename): continue
            tile = Image.open(filename)
            x = (col - col_min) * tile_size
            y = (row - row_min) * tile_size
            result.paste(tile, (x, y))

    result.save(os.path.join(out_folder, f"stitched_zoom_{zoom}.png"))
    return result

# Convert to GeoTIFF
import rasterio
from rasterio.transform import from_origin
import numpy as np

def save_geotiff(image, output_path, col_min, row_min, matrix, tile_size=256):
    img_array = np.array(image)
    res = matrix.scaledenominator * 0.28e-3
    origin_x = matrix.topleftcorner[0] + col_min * tile_size * res
    origin_y = matrix.topleftcorner[1] - row_min * tile_size * res

    transform = from_origin(origin_x, origin_y, res, res)

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=img_array.shape[0],
        width=img_array.shape[1],
        count=3,
        dtype=img_array.dtype,
        crs="EPSG:3763",
        transform=transform
    ) as dst:
        for i in range(3):
            dst.write(img_array[:, :, i], i + 1)
import cv2
def get_connected_components(binary_mask):
    """
    Find connected components in a binary mask and return their properties.
    
    Parameters:
        binary_mask (np.ndarray): Binary image (0 = background, 255/non-zero = foreground)
    
    Returns:
        tuple: (num_labels, labels, stats, centroids)
        - num_labels: Number of unique labels (including background)
        - labels: Matrix where each pixel is labeled with component ID
        - stats: Statistics for each component [x, y, width, height, area]
        - centroids: (cx, cy) coordinates for each component's centroid
    """
    # Ensure input is uint8 (required by OpenCV)
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, 
        connectivity=8  # 8-connectivity (diagonals count as connected)
    )
    
    return num_labels, labels, stats, centroids

def radial_matrix(shape, center=None):
    rows, cols = shape
    if center is None:
        center = (rows // 2, cols // 2)

    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt(np.sqrt((x - center[1])**2 + (y - center[0])**2))

    # Normalize to [0, 1]
    dist_norm = dist / dist.max()
    return dist_norm

def visualize_depth_rgb(rgb_image, depth_image,cropping):
    """
    Display RGB and depth images side by side with proper colormaps
    
    Args:
        rgb_image: RGB array of shape (H,W,3) with dtype uint8 (0-255)
        depth_image: Depth array of shape (H,W) with any dtype
    """
    plt.figure(figsize=(12, 6))
    
    # RGB Image
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title('RGB Image')
    plt.axis('off')
    
    # Depth Image
    plt.subplot(1, 3, 2)
    # Normalize depth for visualization and apply colormap
    depth_normalized = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    plt.imshow(depth_normalized, cmap='viridis')  # or 'jet', 'plasma', 'inferno'
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.axis('off')

    # Cropping Image
    plt.subplot(1, 3, 3)
    # Normalize cropping for visualization and apply colormap
    cropping = (cropping - cropping.min()) / (cropping.max() - cropping.min())
    plt.imshow(cropping, cmap='viridis')  # or 'jet', 'plasma', 'inferno'
    plt.colorbar(label='Depth')
    plt.title('Croppable Map')
    plt.axis('off')

    plt.tight_layout()
    plt.show()