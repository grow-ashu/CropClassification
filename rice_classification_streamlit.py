#!/usr/bin/env python
"""
Rice Classification using Google Earth Engine and Streamlit UI
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
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class RiceClassifier:
    def __init__(self):
        """Initialize the Rice Classifier"""
        st.write("🚀 Initializing Rice Classifier...")
        
        # Initialize Earth Engine
        self.initialize_earth_engine()
        
        # Load district boundaries
        self.load_district_boundaries()
        
        # Load crop signatures
        self.load_crop_signatures()
        
        st.success("✅ Rice Classifier initialized successfully!")
    
    def initialize_earth_engine(self):
        """Initialize Google Earth Engine"""
        try:
            # Try to initialize if not already done
            try:
                # Test if EE is already initialized
                test_img = ee.Image(1)
                test_img.getInfo()
                st.write("✅ Earth Engine already initialized")
            except:
                st.write("🔑 Authenticating with Google Earth Engine...")
                ee.Authenticate()
                ee.Initialize(project='sar-automation-439705')
                st.write("✅ Earth Engine initialized successfully")
                
        except Exception as e:
            st.error(f"❌ Earth Engine initialization failed: {e}")
            st.error("Please run 'earthengine authenticate' in terminal first")
            raise
    
    def load_district_boundaries(self):
        """Load India district boundaries from shapefile"""
        try:
            st.write("📍 Loading district boundaries...")
            shapefile_path = '../india distt.bnd/India_dist_bnd.shp'
            
            # Read shapefile
            districts = gpd.read_file(shapefile_path)
            st.write(f"📊 Loaded {len(districts)} districts")
            
            # Filter Punjab districts (matching the JS code filter)
            self.punjab_districts = districts[districts['STATE_NAME'] == 'Punjab']
            st.write(f"🎯 Filtered to {len(self.punjab_districts)} Punjab districts")
            
            # Convert to Earth Engine FeatureCollection
            features = []
            for idx, row in self.punjab_districts.iterrows():
                geom_dict = row.geometry.__geo_interface__
                geom = ee.Geometry(geom_dict)
                feature = ee.Feature(geom, {
                    'STATE_NAME': row['STATE_NAME'],
                    'DISTRICT': row.get('DISTRICT', 'Unknown')
                })
                features.append(feature)
            
            self.BB = ee.FeatureCollection(features)
            st.success("✅ District boundaries converted to Earth Engine FeatureCollection")
            
        except Exception as e:
            st.error(f"❌ Error loading district boundaries: {e}")
            # Fallback to hardcoded Punjab boundary
            st.warning("Using fallback Punjab boundary...")
            coords = [[[74.0, 30.0], [76.5, 30.0], [76.5, 32.5], [74.0, 32.5], [74.0, 30.0]]]
            polygon_geom = ee.Geometry.Polygon(coords)
            feature = ee.Feature(polygon_geom)
            self.BB = ee.FeatureCollection([feature])
    
    def load_crop_signatures(self):
        """Load crop signatures from CSV"""
        try:
            st.write("🌾 Loading crop signatures...")
            signatures_df = pd.read_csv('Input_data.csv')
            
            # Display signature data
            st.write("Training data overview:")
            st.dataframe(signatures_df.head())
            
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
                        st.warning(f"Could not parse polygon for class {class_id}: {e}")
                        continue
                
                if features:
                    self.training_collections[class_name] = ee.FeatureCollection(features)
                    st.write(f"  {class_name}: {len(features)} training polygons")
                else:
                    st.warning(f"  {class_name}: No valid polygons found")
            
            # Merge all training collections (matching JS code: Wheat.merge(Potato).merge(...)
            all_collections = list(self.training_collections.values())
            if all_collections:
                self.training_data = all_collections[0]
                for collection in all_collections[1:]:
                    self.training_data = self.training_data.merge(collection)
                total_samples = self.training_data.size().getInfo()
                st.write(f"✅ Total training polygons: {total_samples}")
            else:
                st.error("❌ No training data available")
                
        except Exception as e:
            st.error(f"❌ Error loading crop signatures: {e}")
            raise
    
    def get_worldcover_mask(self):
        """Get ESA WorldCover mask (equivalent to pb_mask in JS code)"""
        try:
            st.write("🌍 Loading ESA WorldCover data...")
            
            # Load WorldCover v100 (same as JS code)
            worldcover_collection = ee.ImageCollection("ESA/WorldCover/v100")
            worldcover = worldcover_collection.first()
            pb_mask = worldcover.clip(self.BB)
            
            st.success("✅ WorldCover mask loaded")
            return pb_mask
            
        except Exception as e:
            st.error(f"❌ Error loading WorldCover: {e}")
            # Return a dummy mask
            dummy_mask = ee.Image(1).clip(self.BB)
            return dummy_mask
    
    def get_sentinel2_collections(self):
        """Get Sentinel-2 collections for all date ranges (matching JS code)"""
        try:
            st.write("🛰️ Loading Sentinel-2 collections...")
            
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
            progress_bar = st.progress(0)
            
            for i, (start, end) in enumerate(date_ranges):
                st.write(f"  Loading image {i+1}/7: {start} to {end}")
                
                # Using Earth Engine API call as specified in user request
                collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                            .filterDate(start, end)
                            .filterBounds(self.BB))
                
                collections.append(collection)
                image_count = collection.size().getInfo()
                st.write(f"    Found {image_count} images")
                
                progress_bar.progress((i + 1) / len(date_ranges))
            
            st.success("✅ All Sentinel-2 collections loaded")
            return collections
            
        except Exception as e:
            st.error(f"❌ Error loading Sentinel-2 collections: {e}")
            raise
    
    def add_ndvi(self, image):
        """Add NDVI band to image (matching JS addNDVI function)"""
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    def create_stacked_bands(self, s2_collections, pb_mask):
        """Create stacked bands (matching JS code logic)"""
        try:
            st.write("🔨 Creating stacked bands...")
            progress_bar = st.progress(0)
            
            # Add NDVI to all collections
            withNDVI_collections = [collection.map(self.add_ndvi) for collection in s2_collections]
            
            # Create greenest pixel composites for each time period
            greenest_images = []
            for i, collection in enumerate(withNDVI_collections):
                st.write(f"  Creating composite {i+1}/7...")
                greenest = (collection.qualityMosaic('NDVI')
                          .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
                          .clip(self.BB))
                greenest_images.append(greenest)
                progress_bar.progress((i + 1) / 7 * 0.5)
            
            # Calculate NDVI for each composite with masking (B2 < 1800)
            ndvi_images = []
            for i, greenest in enumerate(greenest_images):
                ndvi = (greenest.normalizedDifference(['B8', 'B4'])
                       .mask(greenest.select('B2').lt(1800))
                       .rename(f'NDVI{i+1}'))
                
                # Apply focal filter (equivalent to focalMode in JS)
                ndvi_filtered = ndvi.focalMean(radius=3, kernelType='square')
                ndvi_images.append(ndvi_filtered)
                progress_bar.progress(0.5 + (i + 1) / 7 * 0.3)
            
            # Stack NDVI bands (using indices 2,3,4,6,7 as in JS code: ndvi2.addBands([ndvi3,ndvi4,ndvi6,ndvi7]))
            selected_ndvi = [ndvi_images[1], ndvi_images[2], ndvi_images[3], 
                           ndvi_images[5], ndvi_images[6]]
            
            if len(selected_ndvi) > 1:
                stacked_ndvi = ee.Image.cat(selected_ndvi)
            else:
                stacked_ndvi = selected_ndvi[0]
            
            # Stack spectral bands (B3, B4, B8) from selected time periods (matching JS stacked_bands logic)
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
            
            # Rename bands to b1, b2, b3... (matching JS code: newBandNames)
            band_count_result = stacked_all.bandNames().size().getInfo()
            if band_count_result and band_count_result > 0:
                band_count = int(band_count_result)
                new_band_names = [f'b{i+1}' for i in range(band_count)]
                renamed_bands = stacked_all.rename(new_band_names)
            else:
                renamed_bands = stacked_all
                band_count = 0
            
            # Apply WorldCover mask (class 40 = cropland, matching JS: updateMask(pb_mask.eq(40)))
            masked_bands = renamed_bands.updateMask(pb_mask.eq(40))
            
            progress_bar.progress(1.0)
            st.success(f"✅ Stacked bands created with {band_count} bands")
            return masked_bands
            
        except Exception as e:
            st.error(f"❌ Error creating stacked bands: {e}")
            raise
    
    def extract_training_data(self, stacked_bands):
        """Extract training data from stacked bands using crop signatures"""
        try:
            st.write("🎯 Extracting training data...")
            
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
            
            st.success(f"✅ Training data extracted: {len(X_train)} samples, {X_train.shape[1]} features")
            
            return X_train, y_train
            
        except Exception as e:
            st.error(f"❌ Error extracting training data: {e}")
            raise
    
    def classify_image(self, stacked_bands, pb_mask):
        """Classify the stacked bands using Earth Engine classifier"""
        try:
            st.write("🎨 Classifying image...")
            
            # Sample training data for EE classifier (matching JS code)
            training = stacked_bands.sampleRegions(
                collection=self.training_data,
                properties=['Class'],
                scale=10
            )
            
            # Train EE classifier (matching JS code: ee.Classifier.smileRandomForest(1000))
            ee_classifier = ee.Classifier.smileRandomForest(1000).train(
                features=training,
                classProperty='Class'
            )
            
            # Get classifier info
            try:
                classifier_info = ee_classifier.explain().getInfo()
                st.write("🔍 Classifier Info:")
                st.json(classifier_info)
            except Exception as info_error:
                st.warning(f"Could not get classifier info: {info_error}")
            
            # Classify the image (matching JS: classify(trainedClassifier))
            classified = stacked_bands.classify(ee_classifier)
            
            # Apply mask (cropland pixels only, matching JS: updateMask(pb_mask.eq(40)))
            classified_masked = classified.updateMask(pb_mask.eq(40))
            
            st.success("✅ Image classification completed")
            return classified_masked, ee_classifier
            
        except Exception as e:
            st.error(f"❌ Error classifying image: {e}")
            raise
    
    def calculate_areas(self, classified_image):
        """Calculate area for each crop class"""
        try:
            st.write("📏 Calculating crop areas...")
            
            areas = {}
            for class_id, class_name in self.crop_classes.items():
                try:
                    # Create mask for this class
                    class_mask = classified_image.eq(class_id)
                    
                    # Calculate area (matching JS area calculation logic)
                    pixel_area_image = ee.Image.pixelArea()
                    area = class_mask.multiply(pixel_area_image).reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=self.BB.geometry(),
                        scale=10,
                        maxPixels=1e9
                    )
                    
                    area_result = area.getInfo()
                    area_hectares = area_result.get('classification', 0) / 10000
                    areas[class_name] = area_hectares
                    st.write(f"  {class_name}: {area_hectares:,.1f} hectares")
                    
                except Exception as class_error:
                    st.warning(f"Could not calculate area for {class_name}: {class_error}")
                    areas[class_name] = 0
            
            return areas
            
        except Exception as e:
            st.error(f"❌ Error calculating areas: {e}")
            return {}
    
    def create_visualization_map(self, classified_image, pb_mask):
        """Create Folium map for visualization"""
        try:
            st.write("🗺️ Creating visualization map...")
            
            # Get Punjab center for map
            punjab_bounds = self.punjab_districts.total_bounds
            center_lat = (punjab_bounds[1] + punjab_bounds[3]) / 2
            center_lon = (punjab_bounds[0] + punjab_bounds[2]) / 2
            
            # Create Folium map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
            
            # Add Punjab districts boundary
            folium.GeoJson(
                self.punjab_districts.to_json(),
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'blue',
                    'weight': 2
                }
            ).add_to(m)
            
            # Try to add classification layer
            try:
                # Create visualization parameters (matching JS palette)
                vis_params = {
                    'min': 0,
                    'max': 5,
                    'palette': ['ff3333', '6bff33', '333cff', '33ffda', 'ff33eb', 'fff833']
                }
                
                # Get map tiles URL
                map_id = classified_image.getMapId(vis_params)
                tile_url = map_id['tile_fetcher'].url_format
                
                # Add classification layer to map
                folium.raster_layers.TileLayer(
                    tiles=tile_url,
                    attr='Google Earth Engine',
                    name='Rice Classification',
                    overlay=True,
                    control=True
                ).add_to(m)
                
                st.success("✅ Classification layer added to map")
                
            except Exception as tile_error:
                st.warning(f"Could not add classification layer to map: {tile_error}")
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"❌ Error creating visualization map: {e}")
            return None
    
    def run_classification(self):
        """Run the complete rice classification workflow"""
        try:
            st.header("🌾 Rice Classification Workflow")
            st.write("=" * 50)
            
            # Progress tracking
            progress = st.progress(0)
            status = st.empty()
            
            # 1. Get WorldCover mask
            status.text("Step 1/5: Loading WorldCover mask...")
            pb_mask = self.get_worldcover_mask()
            progress.progress(1/5)
            
            # 2. Load Sentinel-2 collections
            status.text("Step 2/5: Loading Sentinel-2 collections...")
            s2_collections = self.get_sentinel2_collections()
            progress.progress(2/5)
            
            # 3. Create stacked bands
            status.text("Step 3/5: Creating stacked bands...")
            stacked_bands = self.create_stacked_bands(s2_collections, pb_mask)
            progress.progress(3/5)
            
            # 4. Extract training data
            status.text("Step 4/5: Extracting training data...")
            X_train, y_train = self.extract_training_data(stacked_bands)
            progress.progress(4/5)
            
            # 5. Classify image
            status.text("Step 5/5: Classifying image...")
            classified_image, ee_classifier = self.classify_image(stacked_bands, pb_mask)
            
            # 6. Calculate areas
            areas = self.calculate_areas(classified_image)
            progress.progress(1.0)
            
            status.text("✅ Classification completed!")
            
            return {
                'classified_image': classified_image,
                'classifier': ee_classifier,
                'areas': areas,
                'stacked_bands': stacked_bands,
                'pb_mask': pb_mask
            }
            
        except Exception as e:
            st.error(f"❌ Classification workflow failed: {e}")
            raise

def create_streamlit_ui():
    """Create Streamlit UI for rice classification"""
    st.set_page_config(
        page_title="Rice Classification System",
        page_icon="🌾",
        layout="wide"
    )
    
    st.title("🌾 Rice Classification System")
    st.write("Using Google Earth Engine and Machine Learning for Crop Classification in Punjab")
    
    # Sidebar controls
    st.sidebar.title("🛠️ Controls")
    
    if st.sidebar.button("🚀 Run Classification", type="primary"):
        try:
            # Initialize classifier
            with st.spinner("Initializing classifier..."):
                rice_classifier = RiceClassifier()
            
            # Run classification
            with st.spinner("Running classification workflow..."):
                results = rice_classifier.run_classification()
            
            # Display results
            st.header("📊 Results")
            
            # Area calculations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📏 Crop Areas")
                areas_df = pd.DataFrame(list(results['areas'].items()), 
                                      columns=['Crop', 'Area (hectares)'])
                st.dataframe(areas_df, use_container_width=True)
                
                # Save results
                areas_df.to_csv('rice_classification_areas.csv', index=False)
                st.success("📁 Results saved to rice_classification_areas.csv")
                
                # Download button for CSV
                csv = areas_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name='rice_classification_areas.csv',
                    mime='text/csv'
                )
            
            with col2:
                st.subheader("📈 Area Distribution")
                if results['areas']:
                    # Filter out zero areas for cleaner visualization
                    non_zero_areas = {k: v for k, v in results['areas'].items() if v > 0}
                    if non_zero_areas:
                        fig = px.pie(
                            values=list(non_zero_areas.values()),
                            names=list(non_zero_areas.keys()),
                            title="Crop Area Distribution (hectares)",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No crop areas found")
            
            # Visualization map
            st.header("🗺️ Classification Map")
            try:
                visualization_map = rice_classifier.create_visualization_map(
                    results['classified_image'], 
                    results['pb_mask']
                )
                if visualization_map:
                    st_folium(visualization_map, width=700, height=500)
                else:
                    st.warning("Map visualization not available")
            except Exception as map_error:
                st.error(f"Map creation failed: {map_error}")
            
            # Additional statistics
            st.header("📈 Classification Statistics")
            col3, col4 = st.columns(2)
            
            with col3:
                total_area = sum(results['areas'].values())
                st.metric("Total Classified Area", f"{total_area:,.1f} ha")
                
                # Find dominant crop
                if results['areas']:
                    dominant_crop = max(results['areas'], key=results['areas'].get)
                    dominant_area = results['areas'][dominant_crop]
                    st.metric("Dominant Crop", dominant_crop, f"{dominant_area:,.1f} ha")
            
            with col4:
                # Bar chart of areas
                if results['areas']:
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=list(results['areas'].keys()),
                            y=list(results['areas'].values()),
                            marker_color=['#ff3333', '#6bff33', '#333cff', '#33ffda', '#ff33eb'][:len(results['areas'])]
                        )
                    ])
                    fig_bar.update_layout(
                        title="Crop Areas by Class",
                        xaxis_title="Crop Type",
                        yaxis_title="Area (hectares)"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Classification failed: {e}")
            st.error("Please check your Earth Engine authentication and try again")
            
            # Show error details in expander
            with st.expander("🔍 Error Details"):
                st.code(str(e))
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Information")
    st.sidebar.write("""
    This tool classifies crops in Punjab using:
    - **Sentinel-2** satellite imagery (7 time periods)
    - **ESA WorldCover** land use data
    - **Random Forest** classifier (1000 trees)
    - Ground truth training data from polygons
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Instructions")
    st.sidebar.write("""
    1. Ensure Earth Engine is authenticated
    2. Click '🚀 Run Classification' to start
    3. Wait for processing to complete (~5-10 minutes)
    4. View results and download CSV
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Sources")
    st.sidebar.write("""
    - **Boundaries**: India district shapefile
    - **Training**: Input_data.csv (5 crop classes)
    - **Satellite**: Sentinel-2 SR Harmonized
    - **Land Cover**: ESA WorldCover v100
    """)

def main():
    """Main function"""
    create_streamlit_ui()

if __name__ == "__main__":
    main()
