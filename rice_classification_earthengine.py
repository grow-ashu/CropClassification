#!/usr/bin/env python
"""
Rice Classification using Google Earth Engine
Converted from Rice_Aadi.js JavaScript code to Python
"""

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class RiceClassifier:
    def __init__(self):
        """Initialize the Rice Classifier"""
        print("🚀 Initializing Rice Classifier...")
        
        # Initialize Earth Engine
        self.initialize_earth_engine()
        
        # Load district boundaries
        self.load_district_boundaries()
        
        # Load crop signatures
        self.load_crop_signatures()
        
        print("✅ Rice Classifier initialized successfully!")
    
    def initialize_earth_engine(self):
        """Initialize Google Earth Engine"""
        try:
            # Try to initialize if not already done
            try:
                ee.Image(1).getInfo()
                print("✅ Earth Engine already initialized")
            except:
                print("🔑 Authenticating with Google Earth Engine...")
                ee.Authenticate()
                ee.Initialize(project='sar-automation-439705')
                print("✅ Earth Engine initialized successfully")
                
        except Exception as e:
            print(f"❌ Earth Engine initialization failed: {e}")
            print("Please run 'earthengine authenticate' in terminal first")
            raise
    
    def load_district_boundaries(self):
        """Load India district boundaries from shapefile"""
        try:
            print("📍 Loading district boundaries...")
            shapefile_path = '../india distt.bnd/India_dist_bnd.shp'
            
            # Read shapefile
            districts = gpd.read_file(shapefile_path)
            print(f"📊 Loaded {len(districts)} districts")
            
            # Filter Punjab districts (matching the JS code filter)
            self.punjab_districts = districts[districts['STATE_NAME'] == 'Punjab']
            print(f"🎯 Filtered to {len(self.punjab_districts)} Punjab districts")
            
            # Convert to Earth Engine FeatureCollection
            features = []
            for idx, row in self.punjab_districts.iterrows():
                geom = ee.Geometry(row.geometry.__geo_interface__)
                feature = ee.Feature(geom, {
                    'STATE_NAME': row['STATE_NAME'],
                    'DISTRICT': row.get('DISTRICT', 'Unknown')
                })
                features.append(feature)
            
            self.BB = ee.FeatureCollection(features)
            print("✅ District boundaries converted to Earth Engine FeatureCollection")
            
        except Exception as e:
            print(f"❌ Error loading district boundaries: {e}")
            # Fallback to hardcoded Punjab boundary
            print("Using fallback Punjab boundary...")
            coords = [[[74.0, 30.0], [76.5, 30.0], [76.5, 32.5], [74.0, 32.5], [74.0, 30.0]]]
            self.BB = ee.FeatureCollection([ee.Feature(ee.Geometry.Polygon(coords))])
    
    def load_crop_signatures(self):
        """Load crop signatures from CSV"""
        try:
            print("🌾 Loading crop signatures...")
            signatures_df = pd.read_csv('Input_data.csv')
            
            # Group by class
            self.crop_classes = {
                1: "Wheat",
                2: "Potato", 
                3: "Sugarcane",
                4: "Plantation",
                5: "Other"
            }
            
            # Convert to Earth Engine FeatureCollections
            self.training_collections = {}
            
            for class_id, class_name in self.crop_classes.items():
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
                        print(f"Warning: Could not parse polygon for class {class_id}: {e}")
                        continue
                
                if features:
                    self.training_collections[class_name] = ee.FeatureCollection(features)
                    print(f"  {class_name}: {len(features)} training polygons")
                else:
                    print(f"  {class_name}: No valid polygons found")
            
            # Merge all training collections
            all_collections = list(self.training_collections.values())
            if all_collections:
                self.training_data = all_collections[0]
                for collection in all_collections[1:]:
                    self.training_data = self.training_data.merge(collection)
                print(f"✅ Total training polygons: {self.training_data.size().getInfo()}")
            else:
                print("❌ No training data available")
                
        except Exception as e:
            print(f"❌ Error loading crop signatures: {e}")
            raise
    
    def get_worldcover_mask(self):
        """Get ESA WorldCover mask (equivalent to pb_mask in JS code)"""
        try:
            print("🌍 Loading ESA WorldCover data...")
            
            # Load WorldCover v100 (same as JS code)
            worldcover = ee.ImageCollection("ESA/WorldCover/v100").first()
            pb_mask = worldcover.clip(self.BB)
            
            print("✅ WorldCover mask loaded")
            return pb_mask
            
        except Exception as e:
            print(f"❌ Error loading WorldCover: {e}")
            # Return a dummy mask
            return ee.Image(1).clip(self.BB)
    
    def get_sentinel2_collections(self):
        """Get Sentinel-2 collections for all date ranges (matching JS code)"""
        try:
            print("🛰️ Loading Sentinel-2 collections...")
            
            # Date ranges from JS code
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
            for i, (start, end) in enumerate(date_ranges):
                print(f"  Loading image {i+1}: {start} to {end}")
                
                collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                            .filterDate(start, end)
                            .filterBounds(self.BB))
                
                collections.append(collection)
                print(f"    Found {collection.size().getInfo()} images")
            
            print("✅ All Sentinel-2 collections loaded")
            return collections
            
        except Exception as e:
            print(f"❌ Error loading Sentinel-2 collections: {e}")
            raise
    
    def add_ndvi(self, image):
        """Add NDVI band to image (matching JS addNDVI function)"""
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    def create_stacked_bands(self, s2_collections, pb_mask):
        """Create stacked bands (matching JS code logic)"""
        try:
            print("🔨 Creating stacked bands...")
            
            # Add NDVI to all collections
            withNDVI_collections = [collection.map(self.add_ndvi) for collection in s2_collections]
            
            # Create greenest pixel composites for each time period
            greenest_images = []
            for i, collection in enumerate(withNDVI_collections):
                print(f"  Creating composite {i+1}/7...")
                greenest = (collection.qualityMosaic('NDVI')
                          .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
                          .clip(self.BB))
                greenest_images.append(greenest)
            
            # Calculate NDVI for each composite with masking (B2 < 1800)
            ndvi_images = []
            for i, greenest in enumerate(greenest_images):
                ndvi = (greenest.normalizedDifference(['B8', 'B4'])
                       .mask(greenest.select('B2').lt(1800))
                       .rename(f'NDVI{i+1}'))
                
                # Apply focal filter (equivalent to focalMode in JS)
                ndvi_filtered = ndvi.focalMean(radius=3, kernelType='square')
                ndvi_images.append(ndvi_filtered)
            
            # Stack NDVI bands (using indices 2,3,4,6,7 as in JS code)
            selected_ndvi = [ndvi_images[1], ndvi_images[2], ndvi_images[3], 
                           ndvi_images[5], ndvi_images[6]]
            
            if len(selected_ndvi) > 1:
                stacked_ndvi = ee.Image.cat(selected_ndvi)
            else:
                stacked_ndvi = selected_ndvi[0]
            
            # Stack spectral bands (B3, B4, B8) from selected time periods
            spectral_bands = []
            selected_greenest = [greenest_images[1], greenest_images[2], greenest_images[3],
                               greenest_images[5], greenest_images[6]]
            
            for greenest in selected_greenest:
                bands = greenest.select(['B3', 'B4', 'B8'])
                spectral_bands.extend([bands.select('B3'), bands.select('B4'), bands.select('B8')])
            
            if spectral_bands:
                stacked_spectral = ee.Image.cat(spectral_bands)
                stacked_all = stacked_ndvi.addBands(stacked_spectral)
            else:
                stacked_all = stacked_ndvi
            
            # Rename bands to b1, b2, b3... (matching JS code)
            band_count = stacked_all.bandNames().size().getInfo()
            new_band_names = [f'b{i+1}' for i in range(band_count)]
            
            renamed_bands = stacked_all.rename(new_band_names)
            
            # Apply WorldCover mask (class 40 = cropland)
            masked_bands = renamed_bands.updateMask(pb_mask.eq(40))
            
            print(f"✅ Stacked bands created with {band_count} bands")
            return masked_bands
            
        except Exception as e:
            print(f"❌ Error creating stacked bands: {e}")
            raise
    
    def extract_training_data(self, stacked_bands):
        """Extract training data from stacked bands using crop signatures"""
        try:
            print("🎯 Extracting training data...")
            
            # Sample regions using the stacked bands (matching JS sampleRegions)
            training = stacked_bands.sampleRegions(
                collection=self.training_data,
                properties=['Class'],
                scale=10
            )
            
            # Convert to pandas DataFrame
            training_data = training.getInfo()
            
            # Process the training data
            X_train = []
            y_train = []
            
            for feature in training_data['features']:
                properties = feature['properties']
                
                # Extract feature values (band values)
                feature_values = []
                for key, value in properties.items():
                    if key.startswith('b') and key != 'Class':
                        if value is not None:
                            feature_values.append(value)
                
                if len(feature_values) > 0 and 'Class' in properties:
                    X_train.append(feature_values)
                    y_train.append(properties['Class'])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            print(f"✅ Training data extracted: {len(X_train)} samples, {X_train.shape[1]} features")
            
            return X_train, y_train
            
        except Exception as e:
            print(f"❌ Error extracting training data: {e}")
            raise
    
    def train_classifier(self, X_train, y_train):
        """Train Random Forest classifier (matching JS smileRandomForest)"""
        try:
            print("🧠 Training Random Forest classifier...")
            
            # Train Random Forest with 1000 trees (matching JS code)
            classifier = RandomForestClassifier(
                n_estimators=1000,
                random_state=42,
                max_depth=15,
                min_samples_split=5
            )
            
            # Split data for training and validation (70/30 split as in JS)
            split_idx = int(0.7 * len(X_train))
            indices = np.random.permutation(len(X_train))
            
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            X_train_split = X_train[train_indices]
            y_train_split = y_train[train_indices]
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            
            # Train classifier
            classifier.fit(X_train_split, y_train_split)
            
            # Validation
            y_pred = classifier.predict(X_val)
            
            print("📊 Training Results:")
            print(f"  Training samples: {len(X_train_split)}")
            print(f"  Validation samples: {len(X_val)}")
            print(f"  Training accuracy: {classifier.score(X_train_split, y_train_split):.3f}")
            print(f"  Validation accuracy: {classifier.score(X_val, y_val):.3f}")
            
            print("\n📈 Classification Report:")
            print(classification_report(y_val, y_pred, target_names=[self.crop_classes.get(i, f"Class_{i}") for i in np.unique(y_train)]))
            
            return classifier
            
        except Exception as e:
            print(f"❌ Error training classifier: {e}")
            raise
    
    def classify_image(self, stacked_bands, classifier, pb_mask):
        """Classify the stacked bands using trained classifier"""
        try:
            print("🎨 Classifying image...")
            
            # For Earth Engine classification, we need to use EE classifier
            # Convert sklearn model to EE format (simplified approach)
            
            # Alternative: Use Earth Engine's built-in Random Forest
            training = stacked_bands.sampleRegions(
                collection=self.training_data,
                properties=['Class'],
                scale=10
            )
            
            # Train EE classifier
            ee_classifier = ee.Classifier.smileRandomForest(1000).train(
                features=training,
                classProperty='Class'
            )
            
            # Classify the image
            classified = stacked_bands.classify(ee_classifier)
            
            # Apply mask (cropland pixels only)
            classified_masked = classified.updateMask(pb_mask.eq(40))
            
            print("✅ Image classification completed")
            return classified_masked, ee_classifier
            
        except Exception as e:
            print(f"❌ Error classifying image: {e}")
            raise
    
    def calculate_areas(self, classified_image):
        """Calculate area for each crop class"""
        try:
            print("📏 Calculating crop areas...")
            
            areas = {}
            for class_id, class_name in self.crop_classes.items():
                # Create mask for this class
                class_mask = classified_image.eq(class_id)
                
                # Calculate area
                area = class_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=self.BB.geometry(),
                    scale=10,
                    maxPixels=1e9
                )
                
                area_hectares = area.getInfo().get('classification', 0) / 10000
                areas[class_name] = area_hectares
                print(f"  {class_name}: {area_hectares:,.1f} hectares")
            
            return areas
            
        except Exception as e:
            print(f"❌ Error calculating areas: {e}")
            return {}
    
    def export_results(self, classified_image, areas):
        """Export classification results and area calculations"""
        try:
            print("💾 Exporting results...")
            
            # Save area calculations to CSV
            area_df = pd.DataFrame(list(areas.items()), columns=['Crop_Class', 'Area_Hectares'])
            area_df.to_csv('rice_classification_areas.csv', index=False)
            print("✅ Area calculations saved to rice_classification_areas.csv")
            
            # Export classified image (commented out for now - requires Google Drive setup)
            """
            export_task = ee.batch.Export.image.toDrive(
                image=classified_image,
                description='Rice_Classification_Punjab',
                region=self.BB.geometry(),
                scale=10,
                maxPixels=1e13
            )
            export_task.start()
            print("✅ Image export task started - check Google Drive")
            """
            
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
    
    def run_classification(self):
        """Run the complete rice classification workflow"""
        try:
            print("\n🌾 Starting Rice Classification Workflow")
            print("=" * 50)
            
            # 1. Get WorldCover mask
            pb_mask = self.get_worldcover_mask()
            
            # 2. Load Sentinel-2 collections
            s2_collections = self.get_sentinel2_collections()
            
            # 3. Create stacked bands
            stacked_bands = self.create_stacked_bands(s2_collections, pb_mask)
            
            # 4. Extract training data
            X_train, y_train = self.extract_training_data(stacked_bands)
            
            # 5. Train classifier
            classifier = self.train_classifier(X_train, y_train)
            
            # 6. Classify image
            classified_image, ee_classifier = self.classify_image(stacked_bands, classifier, pb_mask)
            
            # 7. Calculate areas
            areas = self.calculate_areas(classified_image)
            
            # 8. Export results
            self.export_results(classified_image, areas)
            
            print("\n🎉 Rice classification completed successfully!")
            print("=" * 50)
            
            return {
                'classified_image': classified_image,
                'classifier': ee_classifier,
                'areas': areas,
                'stacked_bands': stacked_bands,
                'pb_mask': pb_mask
            }
            
        except Exception as e:
            print(f"❌ Classification workflow failed: {e}")
            raise

def main():
    """Main function to run rice classification"""
    try:
        # Initialize classifier
        rice_classifier = RiceClassifier()
        
        # Run classification
        results = rice_classifier.run_classification()
        
        print("\n📊 Final Results Summary:")
        print("-" * 30)
        for crop, area in results['areas'].items():
            print(f"{crop}: {area:,.1f} hectares")
        
        return results
        
    except Exception as e:
        print(f"❌ Main execution failed: {e}")
        raise

if __name__ == "__main__":
    results = main()
