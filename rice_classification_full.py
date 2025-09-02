import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from sklearn.ensemble import RandomForestClassifier
from sentinelsat import SentinelAPI, geojson_to_wkt
import requests
import os

# 1. Read India district boundaries
shapefile_path = '../india distt.bnd/India_dist_bnd.shp'
districts = gpd.read_file(shapefile_path)

# 2. Filter Punjab districts
punjab = districts[districts['STATE_NAME'] == 'Punjab']

# 3. Read crop signatures from CSV
signatures = pd.read_csv('Input_data.csv')

# 4. Download Sentinel-2 images for specified date ranges using sentinelsat
# You need Copernicus Open Access Hub credentials
SENTINEL_USER = 'your_username'
SENTINEL_PASS = 'your_password'
api = SentinelAPI(SENTINEL_USER, SENTINEL_PASS, 'https://scihub.copernicus.eu/dhus')

# Convert Punjab geometry to WKT
punjab_geom = punjab.geometry.unary_union
punjab_wkt = punjab_geom.wkt

# Date ranges
date_ranges = [
    ('2024-11-01', '2024-11-15'),
    ('2024-11-16', '2024-11-30'),
    ('2024-12-01', '2024-12-15'),
    ('2024-12-16', '2024-12-30'),
    ('2025-01-01', '2025-01-15'),
    ('2025-01-16', '2025-01-30'),
    ('2025-02-01', '2025-02-06'),
]

image_files = []
for i, (start, end) in enumerate(date_ranges):
    products = api.query(punjab_wkt,
                        date=(start, end),
                        platformname='Sentinel-2',
                        processinglevel='Level-2A',
                        cloudcoverpercentage=(0, 30))
    if products:
        uuid = list(products.keys())[0]
        api.download(uuid, directory_path='sentinel_data')
        image_files.append(f'sentinel_data/{uuid}.SAFE')
    else:
        print(f'No product found for {start} to {end}')

# 5. Download ESA WorldCover data for Punjab
# WorldCover 2020: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/ESA_WorldCover_10m_2020_v100_Map.tif
worldcover_url = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/ESA_WorldCover_10m_2020_v100_Map.tif'
worldcover_path = 'ESA_WorldCover_10m_2020_v100_Map.tif'
if not os.path.exists(worldcover_path):
    r = requests.get(worldcover_url)
    with open(worldcover_path, 'wb') as f:
        f.write(r.content)

# Clip WorldCover to Punjab
with rasterio.open(worldcover_path) as src:
    out_image, out_transform = mask(src, [mapping(punjab_geom)], crop=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
    with rasterio.open('worldcover_punjab.tif', "w", **out_meta) as dest:
        dest.write(out_image)

# 6. NDVI calculation function (Assume bands are extracted from SAFE format)
def calculate_ndvi(band8, band4):
    ndvi = (band8 - band4) / (band8 + band4 + 1e-6)
    return ndvi

# 7. Read images and calculate NDVI (pseudo-code, actual SAFE extraction needed)
ndvi_images = []
for img_path in image_files:
    # You need to extract band4 and band8 from .SAFE structure
    # Use rasterio or snappy (ESA SNAP Python API)
    # band4 = ...
    # band8 = ...
    # ndvi = calculate_ndvi(band8, band4)
    # ndvi_images.append(ndvi)
    pass  # Placeholder for actual extraction

# 8. Stack NDVI images (dummy for demonstration)
stacked_ndvi = np.random.rand(len(date_ranges), 100, 100)  # Replace with actual NDVI stack

# 9. Prepare training data from crop signatures (dummy for demonstration)
X_train = np.random.rand(100, stacked_ndvi.shape[0])
y_train = np.random.randint(1, 6, 100)

# 10. Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)

# 11. Classify NDVI stack
X_pred = stacked_ndvi.reshape(stacked_ndvi.shape[0], -1).T
predicted = clf.predict(X_pred)
classified = predicted.reshape(stacked_ndvi.shape[1:])

# 12. Save classified raster
with rasterio.open('worldcover_punjab.tif') as src:
    meta = src.meta.copy()
    meta.update({'count': 1, 'dtype': 'int32'})
    with rasterio.open('classified_rice.tif', 'w', **meta) as dst:
        dst.write(classified.astype(np.int32), 1)

# 13. Area calculation per class
unique, counts = np.unique(classified, return_counts=True)
pixel_area = abs(src.transform[0] * src.transform[4])
areas = {int(u): int(c * pixel_area / 10000) for u, c in zip(unique, counts)}  # hectares

# 14. Export area calculation to CSV
area_df = pd.DataFrame(list(areas.items()), columns=['Class', 'Area_ha'])
area_df.to_csv('rice_class_area.csv', index=False)

print('Classification and area calculation complete. Results saved as classified_rice.tif and rice_class_area.csv.')
