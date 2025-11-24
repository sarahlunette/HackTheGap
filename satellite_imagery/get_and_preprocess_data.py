import ee
import geemap
import osmnx as ox
import geopandas as gpd
from datetime import datetime, timedelta

# ---- Authenticate to Earth Engine
ee.Authenticate()
ee.Initialize()

# ---- Define Area of Interest (AOI) and Dates
center_lat, center_lon = -1.2585, 36.7374  # Example: Kenya, replace as needed
AOI = ee.Geometry.Polygon(
    [[[36.73, -1.27], [36.75, -1.27], [36.75, -1.25], [36.73, -1.25], [36.73, -1.27]]]
)
event_date = "2023-03-11"
before_date = (datetime.strptime(event_date, "%Y-%m-%d") - timedelta(days=1)).strftime('%Y-%m-%d')
after_date = (datetime.strptime(event_date, "%Y-%m-%d") + timedelta(days=1)).strftime('%Y-%m-%d')

# ---- Sentinel-1 SAR
s1col = ee.ImageCollection('COPERNICUS/S1_GRD')\
    .filterBounds(AOI)\
    .filter(ee.Filter.eq('instrumentMode', 'IW'))\
    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
    .select(['VV', 'VH'])
s1_before = s1col.filterDate(before_date, event_date).mean().clip(AOI)
s1_after = s1col.filterDate(event_date, after_date).mean().clip(AOI)

# ---- Sentinel-2 optical
def cloud_mask(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask)

s2col = ee.ImageCollection('COPERNICUS/S2_SR')\
    .filterBounds(AOI)\
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))\
    .map(cloud_mask)\
    .select(['B2','B3','B4','B8','SCL'])
s2_before = s2col.filterDate(before_date, event_date).median().clip(AOI)
s2_after = s2col.filterDate(event_date, after_date).median().clip(AOI)

# ---- OSM Extraction (save for later rasterization or vector overlay)
bbox = [36.73, -1.27, 36.75, -1.25]  # minx, miny, maxx, maxy
osm_poly = ox.geocode_to_gdf('{},{}'.format(center_lat, center_lon)).unary_union.envelope
osm_buildings = ox.geometries.geometries_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], tags={'building': True})
osm_roads = ox.geometries.geometries_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], tags={'highway': True})
osm_buildings.to_file("osm_buildings.geojson", driver="GeoJSON")
osm_roads.to_file("osm_roads.geojson", driver="GeoJSON")
