#!/usr/bin/env python
"""
Simple Rice Classification script to test Earth Engine connectivity
"""

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import json

def initialize_earth_engine():
    """Initialize Google Earth Engine"""
    try:
        # Try to test if EE is already initialized
        ee.Image(1).getInfo()
        print("✅ Earth Engine already initialized")
        return True
    except:
        try:
            print("🔑 Initializing Earth Engine...")
            ee.Initialize(project='sar-automation-439705')
            ee.Image(1).getInfo()
            print("✅ Earth Engine initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Earth Engine initialization failed: {e}")
            return False

def load_district_boundaries():
    """Load India district boundaries"""
    try:
        print("📍 Loading district boundaries...")
        shapefile_path = '../india distt.bnd/India_dist_bnd.shp'
        
        # Read shapefile
        districts = gpd.read_file(shapefile_path)
        print(f"📊 Loaded {len(districts)} districts")
        
        # Filter Punjab districts
        punjab_districts = districts[districts['STATE_NAME'] == 'Punjab']
        print(f"🎯 Filtered to {len(punjab_districts)} Punjab districts")
        
        # Convert to Earth Engine FeatureCollection
        features = []
        for idx, row in punjab_districts.iterrows():
            geom_dict = row.geometry.__geo_interface__
            geom = ee.Geometry(geom_dict)
            feature = ee.Feature(geom, {
                'STATE_NAME': row['STATE_NAME'],
                'DISTRICT': row.get('DISTRICT', 'Unknown')
            })
            features.append(feature)
        
        BB = ee.FeatureCollection(features)
        print("✅ District boundaries converted to Earth Engine FeatureCollection")
        return BB, punjab_districts
        
    except Exception as e:
        print(f"❌ Error loading district boundaries: {e}")
        # Fallback boundary
        coords = [[[74.0, 30.0], [76.5, 30.0], [76.5, 32.5], [74.0, 32.5], [74.0, 30.0]]]
        polygon_geom = ee.Geometry.Polygon(coords)
        feature = ee.Feature(polygon_geom)
        BB = ee.FeatureCollection([feature])
        return BB, None

def load_crop_signatures():
    """Load crop signatures from CSV"""
    try:
        print("🌾 Loading crop signatures...")
        signatures_df = pd.read_csv('Input_data.csv')
        print(f"📊 Loaded {len(signatures_df)} training samples")
        
        # Group by class
        crop_classes = {
            1: "Wheat",
            2: "Potato", 
            3: "Sugarcane",
            4: "Plantation",
            5: "Other"
        }
        
        # Convert to Earth Engine FeatureCollections
        training_collections = {}
        
        for class_id, class_name in crop_classes.items():
            class_data = signatures_df[signatures_df['Class'] == class_id]
            features = []
            
            for idx, row in class_data.iterrows():
                try:
                    # Parse polygon coordinates from string
                    coords_str = row['Polygon']
                    coords = json.loads(coords_str.replace("'", '"'))
                    
                    geom = ee.Geometry.Polygon(coords)
                    feature = ee.Feature(geom, {'Class': class_id})
                    features.append(feature)
                except Exception as e:
                    print(f"Warning: Could not parse polygon for class {class_id}")
                    continue
            
            if features:
                training_collections[class_name] = ee.FeatureCollection(features)
                print(f"  {class_name}: {len(features)} training polygons")
        
        # Merge all training collections
        all_collections = list(training_collections.values())
        if all_collections:
            training_data = all_collections[0]
            for collection in all_collections[1:]:
                training_data = training_data.merge(collection)
            total_samples = training_data.size().getInfo()
            print(f"✅ Total training polygons: {total_samples}")
            return training_data, crop_classes
        else:
            print("❌ No training data available")
            return None, crop_classes
            
    except Exception as e:
        print(f"❌ Error loading crop signatures: {e}")
        raise

def get_sentinel2_data(BB):
    """Get Sentinel-2 data using Earth Engine API (as specified in user request)"""
    try:
        print("🛰️ Loading Sentinel-2 collections...")
        
        # Date ranges from original JS code
        date_ranges = [
            ('2024-11-01', '2024-11-15'),
            ('2024-11-16', '2024-11-30'),
            ('2024-12-01', '2024-12-15'),
            ('2024-12-16', '2024-12-30'),
            ('2025-01-01', '2025-01-15'),
            ('2025-01-16', '2025-01-30'),
            ('2025-02-01', '2025-02-06'),
        ]
        
        collections = []
        for i, (start_date, end_date) in enumerate(date_ranges):
            print(f"  Loading image {i+1}/7: {start_date} to {end_date}")
            
            # Using Earth Engine API call as specified by user
            collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterDate(start_date, end_date)
                        .filterBounds(BB))
            
            collections.append(collection)
            image_count = collection.size().getInfo()
            print(f"    Found {image_count} images")
        
        print("✅ All Sentinel-2 collections loaded")
        return collections
        
    except Exception as e:
        print(f"❌ Error loading Sentinel-2 collections: {e}")
        raise

def get_worldcover_mask(BB):
    """Get ESA WorldCover mask"""
    try:
        print("🌍 Loading ESA WorldCover data...")
        
        # Load WorldCover v100 (same as JS code)
        worldcover_collection = ee.ImageCollection("ESA/WorldCover/v100")
        worldcover = worldcover_collection.first()
        pb_mask = worldcover.clip(BB)
        
        print("✅ WorldCover mask loaded")
        return pb_mask
        
    except Exception as e:
        print(f"❌ Error loading WorldCover: {e}")
        # Return a dummy mask
        return ee.Image(1).clip(BB)

def main():
    """Main function to test the rice classification setup"""
    print("🌾 Rice Classification Test")
    print("=" * 40)
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        print("❌ Cannot proceed without Earth Engine")
        return
    
    # Load boundaries
    BB, punjab_districts = load_district_boundaries()
    
    # Load training data
    training_data, crop_classes = load_crop_signatures()
    
    # Test Sentinel-2 data loading
    s2_collections = get_sentinel2_data(BB)
    
    # Test WorldCover loading
    pb_mask = get_worldcover_mask(BB)
    
    print("\n✅ All components loaded successfully!")
    print(f"📊 Ready to process {len(s2_collections)} time periods")
    
    # Test basic NDVI calculation
    if s2_collections[0].size().getInfo() > 0:
        print("🧪 Testing NDVI calculation...")
        first_image = s2_collections[0].first()
        ndvi = first_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndvi_info = ndvi.getInfo()
        print("✅ NDVI calculation test successful")
    
    print("\n🎉 Rice classification setup completed!")
    print("Run 'conda activate crop && streamlit run rice_classification_streamlit.py' to start the UI")

if __name__ == "__main__":
    main()
