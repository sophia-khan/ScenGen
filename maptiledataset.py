import os
import requests
from PIL import Image
import math

# MapTiler API key
api_key = 'api_key'

# Base URL for MapTiler tiles
base_url = 'https://api.maptiler.com/maps/topo/{z}/{x}/{y}.png?key={api_key}'

# Output directory
tile_dataset_dir = 'tile_dataset'

# Create output directory if does not exist
os.makedirs(tile_dataset_dir, exist_ok=True)

# Convert latitude/longitude to tile coordinates
def lat_lon_to_tile_coords(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x_tile, y_tile

# Function to download a single tile
def download_tile(z, x, y, label):
    url = base_url.format(z=z, x=x, y=y, api_key=api_key)
    response = requests.get(url)
    if response.status_code == 200:
        tile_path = os.path.join(tile_dataset_dir, f'{label}_{z}_{x}_{y}.png')
        with open(tile_path, 'wb') as f:
            f.write(response.content)
        return tile_path
    else:
        print(f"Failed to download tile {z}/{x}/{y}")
        return None

# Function to combine 4 tiles into a single image
def combine_tiles(z, x, y, label):
    tile_paths = [
        download_tile(z, x, y, label),
        download_tile(z, x + 1, y, label),
        download_tile(z, x, y + 1, label),
        download_tile(z, x + 1, y + 1, label)
    ]

    if None not in tile_paths:
        tiles = [Image.open(tile) for tile in tile_paths]
        tile_width, tile_height = tiles[0].size
        combined_image = Image.new('RGB', (2 * tile_width, 2 * tile_height))

        combined_image.paste(tiles[0], (0, 0))
        combined_image.paste(tiles[1], (tile_width, 0))
        combined_image.paste(tiles[2], (0, tile_height))
        combined_image.paste(tiles[3], (tile_width, tile_height))

        combined_image_path = os.path.join(tile_dataset_dir, f'{label}_combined_{z}_{x}_{y}.png')
        combined_image.save(combined_image_path)
        print(f"Combined image saved as {combined_image_path}")

# Define the locations (latitude, longitude, label)
locations = [
    (39.5501, -105.7821, 'Rocky_Mountains_USA'),  # Rocky Mountains, USA
    (46.8182, 8.2275, 'Swiss_Alps_Switzerland'),  # Swiss Alps, Switzerland
    (-3.4653, -62.2159, 'Amazon_River_Brazil'),   # Amazon River, Brazil
    (40.7128, -74.0060, 'New_York_City_USA'),     # New York City, USA
    (35.6895, 139.6917, 'Tokyo_Japan'),           # Tokyo, Japan
    (-33.8688, 151.2093, 'Sydney_Australia'),     # Sydney, Australia
    (27.1751, 78.0421, 'Taj_Mahal_India'),        # Taj Mahal, India
    (34.0522, -118.2437, 'Los_Angeles_USA'),      # Los Angeles, USA
    (55.7558, 37.6176, 'Moscow_Russia'),          # Moscow, Russia
    (19.4326, -99.1332, 'Mexico_City_Mexico'),    # Mexico City, Mexico
]

# Zoom level
zoom = 10

# Loop through locations, convert to tile coordinates, and download/combine tiles
for lat, lon, label in locations:
    x, y = lat_lon_to_tile_coords(lat, lon, zoom)
    tiles_downloaded = 0
    for i in range(x, x + 100, 2):  
        for j in range(y, y + 100, 2):  
            combine_tiles(zoom, i, j, label)
            tiles_downloaded += 4
            if tiles_downloaded >= 10000:
                break
        if tiles_downloaded >= 10000:
            break

print("Tile generation completed.")
