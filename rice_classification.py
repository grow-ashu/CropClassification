import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.mask import mask
from shapely.geometry import shape, Polygon
from sklearn.ensemble import RandomForestClassifier
import os

# 1. Read India district boundaries
shapefile_path = '../india distt.bnd/India_dist_bnd.shp'
districts = gpd.read_file(shapefile_path)

# 2. Filter Punjab districts
punjab = districts[districts['STATE_NAME'] == 'Punjab']

# 3. Read crop signatures from CSV
signatures = pd.read_csv('Input_data.csv')

# 4. Download Sentinel-2 images for specified date ranges (pseudo-code, use sentinelsat/sentinelhub for real download)
# For demonstration, assume images are already downloaded as TIFs in ./sentinel_data/
date_ranges = [
    ('2024-11-01', '2024-11-15'),
    ('2024-11-16', '2024-11-30'),
    ('2024-12-01', '2024-12-15'),
    ('2024-12-16', '2024-12-30'),
    ('2025-01-01', '2025-01-15'),
    ('2025-01-16', '2025-01-30'),
    ('2025-02-01', '2025-02-06'),
]

image_files = [f'sentinel_data/image{i+1}.tif' for i in range(len(date_ranges))]

# 5. NDVI calculation function
def calculate_ndvi(band8, band4):
    ndvi = (band8 - band4) / (band8 + band4 + 1e-6)
    return ndvi

# 6. Read images and calculate NDVI
ndvi_images = []
for img_path in image_files:
    with rasterio.open(img_path) as src:
        band4 = src.read(4)
        band8 = src.read(8)
        ndvi = calculate_ndvi(band8, band4)
        ndvi_images.append(ndvi)

# 7. Stack NDVI images
stacked_ndvi = np.stack(ndvi_images, axis=0)

# 8. Prepare training data from crop signatures
# Convert polygons to raster masks and extract NDVI values (pseudo-code)
# For demonstration, create dummy training data
X_train = np.random.rand(100, stacked_ndvi.shape[0])
y_train = np.random.randint(1, 6, 100)

# 9. Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)

# 10. Classify NDVI stack
X_pred = stacked_ndvi.reshape(stacked_ndvi.shape[0], -1).T
predicted = clf.predict(X_pred)
classified = predicted.reshape(ndvi_images[0].shape)

# 11. Save classified raster
with rasterio.open(image_files[0]) as src:
    meta = src.meta.copy()
    meta.update({'count': 1, 'dtype': 'int32'})
    with rasterio.open('classified_rice.tif', 'w', **meta) as dst:
        dst.write(classified.astype(np.int32), 1)

# 12. Area calculation per class
unique, counts = np.unique(classified, return_counts=True)
pixel_area = abs(src.transform[0] * src.transform[4])
areas = {int(u): int(c * pixel_area / 10000) for u, c in zip(unique, counts)}  # hectares

# 13. Export area calculation to CSV
area_df = pd.DataFrame(list(areas.items()), columns=['Class', 'Area_ha'])
area_df.to_csv('rice_class_area.csv', index=False)

print('Classification and area calculation complete. Results saved as classified_rice.tif and rice_class_area.csv.')
