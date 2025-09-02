
var BB = ee.FeatureCollection('users/nehrasantosh85/India_districts_bnd')
//.filter(ee.Filter.or(ee.Filter.eq('DISTRICT', 'BAHRAICH'), ee.Filter.eq('STATE_NAME', 'GORAKHPUR'), ee.Filter.eq('STATE_NAME', 'GONDA')));
 //.filter(ee.Filter.or(ee.Filter.eq('STATE_NAME', 'Punjab'), ee.Filter.eq('STATE_NAME', 'Haryana')));
 .filter(ee.Filter.eq('STATE_NAME', 'Punjab'));
//  .filter(ee.Filter.eq('DISTRICT', 'MUKTSAR'));
  //Map.centerObject(ST,8);
  print('BB',BB);
//var value = ee.FeatureCollection('users/nehrasantosh85/Grow_indigo/Ragini/A_UP_SS_Rice_Wheat_36');
//Map.centerObject(BB,20);
var pb_mask = ee.Image(ee.FeatureCollection("ESA/WorldCover/v100").first()).clip(BB);
//var pb_mask = ee.Image(ee.FeatureCollection("ESA/WorldCover/v200").first()).clip(BB);
var visualization = {
  bands: ['Map'],
};

//var mask = pb_mask.eq(10).or(pb_mask.eq(50)); // Pixels with 10 or 50 are 1, rest are 0
//Map.addLayer(pb_mask,{},'mask')

var GT = ee.FeatureCollection('projects/ee-nehrasantosh85/assets/Aadi_ZT_CT_GT_2024')
.filter(ee.Filter.eq('Crop', 'Wheat'));
//.filter(ee.Filter.eq('UID', 168));

//Map.centerObject(DD, 16)
 print(GT,'GT')
 
//Map.centerObject(GT,20);
 
 
var image1 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2024-11-01', '2024-11-15')
                   .filterBounds(BB);
var image2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2024-11-16', '2024-11-30')
                   .filterBounds(BB);                   
var image3 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2024-12-01', '2024-12-15')
                   .filterBounds(BB)
var image4 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2024-12-16', '2024-12-30')
                   .filterBounds(BB)
var image5 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2025-01-01', '2025-01-15')
                   .filterBounds(BB)  
var image6 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2025-01-16', '2025-01-30')
                   .filterBounds(BB)                    
var image7 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2025-02-01', '2025-02-06')
                   .filterBounds(BB)
                   
 /*                  
var image8 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2024-02-16', '2024-02-28')
                   .filterBounds(BB) 
var image9 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2024-03-01', '2024-03-15')
                   .filterBounds(BB)
var image10 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
                   .filterDate('2024-03-16', '2024-03-31')
                   .filterBounds(BB)                   
             
 */            
print(image1);
print(image2);
print(image3);
print(image4);
print(image5);
print(image6);
print(image7);
//print(image8);
//print(image9);
//print(image10);

var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};

var withNDVI1 = image1.map(addNDVI);
var withNDVI2 = image2.map(addNDVI);
var withNDVI3 = image3.map(addNDVI);
var withNDVI4 = image4.map(addNDVI);
var withNDVI5 = image5.map(addNDVI);
var withNDVI6 = image6.map(addNDVI);
var withNDVI7 = image7.map(addNDVI);
//var withNDVI8 = image8.map(addNDVI);
//var withNDVI9 = image9.map(addNDVI);
//var withNDVI10 = image10.map(addNDVI);




var greenest1 = withNDVI1.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
var greenest2 = withNDVI2.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
var greenest3 = withNDVI3.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
var greenest4 = withNDVI4.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
var greenest5 = withNDVI5.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
var greenest6 = withNDVI6.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
var greenest7 = withNDVI7.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
//var greenest8 = withNDVI8.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
//var greenest9 = withNDVI9.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
//var greenest10 = withNDVI10.qualityMosaic('NDVI').select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(BB);
//print (greenest3);

var ndvi1 = greenest1.normalizedDifference(['B8', 'B4']).mask(greenest1.select('B2').lt(1800)).rename('NDVI1');
var ndvi2 = greenest2.normalizedDifference(['B8', 'B4']).mask(greenest2.select('B2').lt(1800)).rename('NDVI2');
var ndvi3 = greenest3.normalizedDifference(['B8', 'B4']).mask(greenest3.select('B2').lt(1800)).rename('NDVI3');
var ndvi4 = greenest4.normalizedDifference(['B8', 'B4']).mask(greenest4.select('B2').lt(1800)).rename('NDVI4');
var ndvi5 = greenest5.normalizedDifference(['B8', 'B4']).mask(greenest5.select('B2').lt(1800)).rename('NDVI5');
var ndvi6 = greenest6.normalizedDifference(['B8', 'B4']).mask(greenest6.select('B2').lt(1800)).rename('NDVI6');
var ndvi7 = greenest7.normalizedDifference(['B8', 'B4']).mask(greenest7.select('B2').lt(1800)).rename('NDVI7');

var ndvi1 = ndvi1.focalMode({ radius: 3, kernelType: 'square' });
var ndvi2 = ndvi2.focalMode({ radius: 3, kernelType: 'square' });
var ndvi3 = ndvi3.focalMode({ radius: 3, kernelType: 'square' });
var ndvi4 = ndvi4.focalMode({ radius: 3, kernelType: 'square'});
var ndvi5 = ndvi5.focalMode({ radius: 3, kernelType: 'square'});
var ndvi6 = ndvi6.focalMode({ radius: 3, kernelType: 'square'});
var ndvi7 = ndvi7.focalMode({ radius: 3, kernelType: 'square'});


//var ndvi8 = greenest8.normalizedDifference(['B8', 'B4']).mask(greenest8.select('B2').lt(2500)).rename('NDVI8');
//var ndvi9 = greenest9.normalizedDifference(['B8', 'B4']).mask(greenest9.select('B2').lt(2500)).rename('NDVI9');
//var ndvi10 = greenest10.normalizedDifference(['B8', 'B4']).mask(greenest10.select('B2').lt(2500)).rename('NDVI10');


var stacked_composite = greenest1.addBands([greenest2,greenest3,greenest4,greenest5,greenest6,greenest7]);
var stacked_ndvi = ndvi2.addBands([ndvi3,ndvi4,ndvi6,ndvi7]);



var stacked_bands = greenest2.select(['B3', 'B4', 'B8'])
    .addBands(greenest3.select(['B3', 'B4', 'B8']))
    .addBands(greenest4.select(['B3', 'B4', 'B8']))
    .addBands(greenest6.select(['B3', 'B4', 'B8']))
    .addBands(greenest7.select(['B3', 'B4', 'B8']))
   
var stacked_ndvi_bands = stacked_ndvi.addBands(stacked_bands);

var newBandNames = [
  'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10',
  'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20'
];

// Rename the bands in stacked_ndvi_bands
var renamedStackedBands = stacked_ndvi_bands.rename(newBandNames);
var renamedStackedBands_masked = renamedStackedBands.updateMask(pb_mask.eq(40));


print(renamedStackedBands,'renamedStackedBands')



//var stacked_lswi_masked = stacked_LSWI.updateMask(mask.eq(1));

//Map.addLayer(stacked_ndvi,{min:-1,max:1},"stacked_ndvi")
//Map.addLayer(stacked_LSWI,{min:-1,max:1},"stacked_lswi")

//var aoi = geometry.merge(BB);



//Display the result.
var visParams = {bands: ['B8', 'B4', 'B3'], min: 0, max: 4000};
//Map.addLayer(greenest1, visParams, 'Nov_F1');
Map.addLayer(greenest2, visParams, 'Nov_F2');
Map.addLayer(greenest3, visParams, 'Dec_F1');
Map.addLayer(greenest4, visParams, 'Dec_F2');
//Map.addLayer(greenest5, visParams, 'Jan_F1');
Map.addLayer(greenest6, visParams, 'jan_F2');
Map.addLayer(greenest7, visParams, 'Feb_F1');
//Map.addLayer(greenest8, visParams, 'Feb_F2');
//Map.addLayer(greenest9, visParams, 'Mar_F1');
//Map.addLayer(greenest10, visParams, 'Mar_F2');

Map.addLayer(renamedStackedBands_masked,{},'renamedStackedBands_masked');

var styling = {color: 'Blue', fillColor: '00000000', width: 2};

//Map.addLayer(GT.style(styling),{},'GT')
Map.addLayer(GT.style(styling),{},'GT')


var training_data1 =Wheat.merge(Potato).merge(Sugarcane).merge(Plantation).merge(other);

print(training_data1,'training_data1')

var training = renamedStackedBands_masked.sampleRegions({
  collection: training_data1,
  properties: ['Class'],
  scale: 10
});


print(training,'training');

Export.table.toDrive({description:'training',collection:training,fileFormat:'CSV'});
// Add a random value field to the sample and use it to approximately split 80%
// of the features into a training set and 20% into a validation set.
var sample = training.randomColumn();
var trainingSample = training.filter('random <= 0.7');
var validationSample = training.filter('random > 0.7');



// Train an SVM classifier (C-SVM classification, voting decision procedure,
// linear kernel) from the training sample.

var trainedClassifier = ee.Classifier.smileRandomForest(1000).train({
  features: training,
  classProperty: 'Class',
  //inputProperties: bands1
});

// Get information about the trained classifier.
print('Results of trained classifier', trainedClassifier.explain());

// Get a confusion matrix and overall accuracy for the training sample.
var trainAccuracy = trainedClassifier.confusionMatrix();
print('Training error matrix', trainAccuracy);
print('Training overall accuracy', trainAccuracy.accuracy());

// Get a confusion matrix and overall accuracy for the validation sample.
validationSample = validationSample.classify(trainedClassifier);
var validationAccuracy = validationSample.errorMatrix('Class', 'classification');
print('Validation error matrix', validationAccuracy);
print('Validation accuracy', validationAccuracy.accuracy());

// Classify the reflectance image from the trained classifier.
var imgClassified = renamedStackedBands_masked.classify(trainedClassifier);
var imgclassified_mask= imgClassified.updateMask(pb_mask.eq(40));
                                                             // Red,   Green,     blue    Magenta    Cyan     yellow
Map.addLayer(imgclassified_mask, {min: 0, max: 5, palette: ['ff3333','#6bff33','#333cff','#33ffda','#ff33eb','#fff833',]},'Cropmask');

Map.addLayer(pb_mask,{},'pb_mask')




var styling = {color: 'Blue', fillColor: '00000000', width: 2};
var styling1 = {color: 'Red', fillColor: '00000000', width: 2};
var styling2 = {color: 'Magenta', fillColor: '00000000', width: 2};





// Print the area calculations.
//print('##### CLASS AREA SQ. METERS #####');
//print(areas);



/*
var chart_nd_VH =  ui.Chart.image.regions({
  image: stacked_VH, 
  regions: BB, 
  reducer: ee.Reducer.mean(), 
  scale:10, 
  seriesProperty:'UID',
  });
print('VH',chart_nd_VH);

*/

/*
Export.image.toDrive({
  image: imagclassified_mask,
  description: 'Rice_mask_2019',
  //folder: 'ee_demos',
  region: ST,
  scale: 20,
  maxPixels: 1e13
  //crs: 'EPSG:5070',
  
    
});


*/

//Map.addLayer(imagclassified_mask, {min: 0, max: 7, palette: ['white','Green', 'Orange','Blue','Magenta','Cyan','Pink']},'Cropmask');

/*
//Area calculation at state level
// Define the scale (resolution) in meters

var scale = 50;

// Function to calculate the area of each class
var calculateArea = function(classValue) {
  var singleClass = imagclassified_mask.eq(classValue);
  var area = singleClass.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: ST,
    scale: scale,
    maxPixels: 1e9
  });
  return ee.Number(area.get('classification')).divide(1e4); // Convert to hectare
};

// Calculate the area for each class (adjust the class values as needed)
//var areaClass0 = calculateArea(0);
var areaClass1 = calculateArea(1);
//var areaClass2 = calculateArea(2);
//var areaClass3 = calculateArea(3);

// Print the results
//print('Area of Class 0 (km²):', areaClass0);
print('Area of Class 1 (ha):', areaClass1);
//print('Area of Class 2 (ha):', areaClass2);
//print('Area of Class 3 (km²):', areaClass3);

// Optionally, export the results to Google Drive
 Export.table.toDrive({
  collection: ee.FeatureCollection([
   // ee.Feature(null, {'Class': 0, 'Area_km2': areaClass0}),
    ee.Feature(null, {'Class': 1, 'ha': areaClass1}),
   // ee.Feature(null, {'Class': 2, 'ha': areaClass2}),
    //ee.Feature(null, {'Class': 3, 'Area_km2': areaClass3})
  ]),
  description: 'Class_Areas',
  fileFormat: 'CSV'
});
*/
// district wise area calculation
//var scale = 10;
/*
// Function to calculate the area of each class within a district
var calculateArea = function(ST) {
  var districtName = ST.get('DISTRICT');
  var results = ee.List.sequence(0,1).map(function(classValue) {
    var singleClass = imagclassified_mask.eq(ee.Number(classValue));
    var area = singleClass.multiply(ee.Image.pixelArea()).reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: ST.geometry(),
      scale: scale,
      maxPixels: 1e9
    }).get('classification');
    
    return ee.Feature(null, {
      'District': districtName,
      'Class': classValue,
      'Area_km2': ee.Number(area).divide(1e4) // Convert to square kilometers
    });
  });
  return ee.FeatureCollection(results);
};

// Apply the area calculation function to each district
var districtAreas = ST.map(calculateArea).flatten();

// Print the results
print('District Areas:', districtAreas);

// Optionally, export the results to Google Drive
Export.table.toDrive({
  collection: districtAreas,
  description: 'District_Class_Areas',
  fileFormat: 'CSV'
});


*/



