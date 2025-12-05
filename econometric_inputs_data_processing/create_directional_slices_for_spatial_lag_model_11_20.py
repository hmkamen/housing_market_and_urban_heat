#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import scipy.stats
import warnings
import mapclassify as mc
import geopandas as gpd
import statsmodels.api as sm
from urllib.request import urlopen
import json
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
from shapely.geometry import shape
from scipy.spatial import cKDTree
from rasterio.mask import mask
import rasterio
from rasterio.features import rasterize
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box, mapping
import rasterio
from rasterio.mask import mask


path='/Users/hannahkamen/Downloads'
import warnings

# Filter or ignore specific warning types
warnings.filterwarnings('ignore')

from shapely.geometry import box, shape, mapping  # Import shape here
from rasterio.features import geometry_mask


# ### Define helper functions

# In[27]:


# Function to clean geometry
def clean_geometry(geom):
    if not geom.is_valid:
        geom = make_valid(geom)
        # Or apply a small buffer if make_valid is not available
        geom = geom.buffer(0)
    return geom

# Define function to categorize bearings
def categorize_bearing(bearing):
    if 360 <= bearing < 45 :
        return "N"
    elif 45 <= bearing < 90:
        return "NE"
    elif 90 <= bearing < 135:
        return "E"
    elif 135 <= bearing < 180:
        return "SE"
    elif 180 <= bearing < 225:
        return "S"
    elif 225<= bearing < 270:
        return "SW"
    elif 270 <= bearing < 315:
        return "W"
    elif 315 <= bearing < 360:
        return "NW"


def calculate_area_weighted_average(raster, slice_geom, nodata_value):
    """
    Calculate the area-weighted average of a raster over a slice geometry.
    """
    try:
        # Convert slice geometry to GeoJSON format
        slice_geom_geojson = [mapping(slice_geom)]
        
        # Mask the raster to the slice geometry
        masked_raster, masked_transform = mask(raster, slice_geom_geojson, crop=True, nodata=nodata_value)
        
        # Flatten the raster values and remove nodata
        data = masked_raster[0].flatten()
        data = data[data != nodata_value]
        
        if len(data) == 0:
            return np.nan
        
        # Get the pixel areas
        pixel_areas = np.ones_like(data)  # Assuming uniform pixels; adjust if non-uniform
        area_weighted_average = np.average(data[data != 0])
        
        return area_weighted_average
    except Exception as e:
        print(f"Error during area-weighted average calculation: {e}")
        return np.nan

# Function to create 8 directional slices as polygons
def create_slices(center, radius):
    slices = {}
    directions = ["NE", "N", "NW", "W", "SW", "S", "SE", "E"]
    angles = np.arange(0, 360, 45)  # Start angles for each slice

    for i, direction in enumerate(directions):
        start_angle = angles[i]
        end_angle = angles[i] + 45

        # Generate boundary points for the slice
        num_points = 100
        boundary_points = [
            (
                center.x + radius * np.cos(np.radians(angle)),
                center.y + radius * np.sin(np.radians(angle))
            )
            for angle in np.linspace(start_angle, end_angle, num_points)
        ]
        slice_polygon = Polygon([center.coords[0]] + boundary_points + [center.coords[0]])
        slices[direction] = slice_polygon

    return slices


# #### Get air temp data

# In[28]:


air_temp=gpd.GeoDataFrame(geometry=[],crs="EPSG:4326")

for year in np.arange(2009,2019,1):
    print(year)
    
    data_year=gpd.GeoDataFrame(geometry=[],crs="EPSG:4326")
    
    for month in [7]:
        for day in np.arange(1,31,1):

            try:
            
                if day <10:
                    day="0"+str(day)
                    
                data_min=gpd.read_file("%s/conus_HUMID_%s0%s%s_t2_min.shp"%(path,year,month,day))
                data_max=gpd.read_file("%s/conus_HUMID_%s0%s%s_t2_max.shp"%(path,year,month,day))
                data_min=data_min[data_min['t2_min_BC']>0]
                data_max=data_max[data_max['t2_max_BC']>0]
                data=data_max.merge(data_min,on=['lon','lat','geometry'])

                data_year=pd.concat([data_year,data],ignore_index=True)

            except Exception as e:
                print(f"Error processing: {e}")
                continue
    data_year['year']=year
    del data_year['geometry']
    data_year_gr=data_year.groupby(['lon','lat'],as_index=False).mean()
    # print(len(data_year_gr))
    data_year_geo = gpd.GeoDataFrame(data_year_gr, geometry=gpd.points_from_xy(data_year_gr['lon'], data_year_gr['lat']),crs="EPSG:4326")
    air_temp=pd.concat([air_temp,data_year_geo],ignore_index=False)
    ## calc mean based on midpoint of max and min
    air_temp['t2_mean']=(air_temp['t2_max_BC']+air_temp['t2_min_BC'])/2


# ### Create air temp raster for each year in sample

# In[29]:


air_temp=air_temp.to_crs('EPSG:26912')

for year in np.arange(2009,2019,1):


    air_temp_lm= air_temp[(air_temp['year']==year)]


    # Define raster properties (1000m resolution)
    xmin, ymin, xmax, ymax = air_temp_lm.total_bounds  # Get bounding box
    resolution = 1100  # 100m grid
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    
    # Define raster transformation
    raster_transform = rasterio.transform.from_origin(xmin, ymax, resolution, resolution)
    
    # Rasterize the temperature values
    air_temp_raster = rasterize(
        [(geom, value) for geom, value in zip(air_temp_lm.geometry, air_temp_lm["t2_min_BC"])],
        out_shape=(height, width),
        transform=raster_transform,
        fill=np.nan,  # Use NaN for missing values
        dtype="float32"
    )
    
    # Save as a GeoTIFF raster file
    with rasterio.open(
        "%s/air_temp_july_min_1k_ncar_%s.tif"%(path,year), "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:26912",
        transform=raster_transform
    ) as dst:
        dst.write(air_temp_raster, 1)
    
    print("✅ Air temperature raster created successfully!")

            


# ### Load home parcel shapefile and limit to parcels in sample

# In[30]:


#### load parcel shapefile and parcelcsv to limit to sold parcels
##parcel shapefile
parcels0 = gpd.read_file(f'{path}/Parcels 2/Parcels_All.shp')

### set crs to meters crs
parcels0=parcels0.to_crs("EPSG:26912")

# sold_parcels=pd.read_stata("%s/hausman_taylor_test_4_10_v2_all_buffers_complete.dta"%path)
sold_parcels=pd.read_csv('%s/urban_iv_paper_input_data_11_20/sold_homes_sample_11_20.csv'%path)
sold_parcels=sold_parcels[['APN','sale_year']]


# In[31]:


len(sold_parcels)


# ### For each buffer of 250m, 250-750m, 750-1250m, 1250-1750m, 1750-2250m, 2250-2750m,
# ### calc contribution of heat from advection from surrounding 500m area 

# In[32]:


# ---------- One-time setup outside the loops ----------

# Get raster CRS from a template file (e.g., 2009)
template_raster_path = f"{path}/air_temp_1k_ncar_2009.tif"
with rasterio.open(template_raster_path) as tmp_r:
    raster_crs = tmp_r.crs

# Reproject parcels once into raster CRS
parcels0 = parcels0.to_crs(raster_crs)

# Predefine directions & angles for slices
DIRECTIONS = ["NE", "N", "NW", "W", "SW", "S", "SE", "E"]
ANGLES = np.arange(0, 360, 45)  # 0, 45, ..., 315
NUM_POINTS = 40  # fewer points -> fewer vertices -> faster

def create_slices(center, radius):
    """
    Create 8 wedge polygons around center with given radius.
    Returns dict: {direction: Polygon}
    """
    cx, cy = center.x, center.y
    slices = {}
    for i, direction in enumerate(DIRECTIONS):
        start_angle = ANGLES[i]
        end_angle = start_angle + 45

        boundary_points = [
            (
                cx + radius * np.cos(np.radians(angle)),
                cy + radius * np.sin(np.radians(angle)),
            )
            for angle in np.linspace(start_angle, end_angle, NUM_POINTS)
        ]
        poly = Polygon([center.coords[0]] + boundary_points + [center.coords[0]])
        slices[direction] = poly
    return slices

# ---------- Main loop ----------

for year in range(2009, 2019):
    print(f"Processing year {year}...")
    
    # Open raster ONCE per year
    raster_path = f"{path}/air_temp_1k_ncar_{year}.tif"
    with rasterio.open(raster_path) as raster_data:
        raster_nodata = raster_data.nodata
        raster_bounds_geom = box(*raster_data.bounds)  # same CRS as parcels0 now

        # list of sold parcels in this year
        sold_apns = sold_parcels.loc[sold_parcels["sale_year"] == year, "APN"].unique()

        # limit parcel shapefile to sold parcels (already in raster CRS)
        base_parcel_gdf = parcels0[parcels0["APN"].isin(sold_apns)].copy()
        base_parcel_gdf["sale_year"] = year

        # Loop over buffer radii
        for buff in [500,1500,2500,3500,4500,5500]:
            print(f"  Buffer radius {buff} m")

            # Work on a fresh copy for this buffer
            parcel_gdf = base_parcel_gdf.copy()

            # Replace geometry with centroid buffers
            parcel_gdf["geometry"] = parcel_gdf.geometry.centroid.buffer(buff)

            # Precompute centroid and inner buffer once per parcel
            parcel_gdf["centroid"] = parcel_gdf.geometry.centroid

            # Initialize columns for directions
            for direction in DIRECTIONS:
                parcel_gdf[f"mean_lst_{direction}"] = np.nan

            # Iterate parcels
            for idx, row in parcel_gdf.iterrows():
                buffer_geom = row["geometry"]
                center = row["centroid"]

                # Outer radius = buff + 500m (your "surrounding 500m" ring)
                outer_radius = buff + 1000

                # Create slices as wedges and subtract inner buffer
                slices = create_slices(center, outer_radius)
                slices_without_ring = {
                    d: poly.difference(buffer_geom) for d, poly in slices.items()
                }

                for direction, slice_geom in slices_without_ring.items():
                    if slice_geom.is_empty or not slice_geom.is_valid:
                        continue

                    # Quick bounding-box intersection test
                    if not slice_geom.intersects(raster_bounds_geom):
                        continue

                    # Mask raster with the slice geometry
                    slice_geojson = [mapping(slice_geom)]
                    try:
                        masked, _ = mask(
                            raster_data,
                            slice_geojson,
                            crop=True,
                            indexes=1,
                            filled=True,
                            all_touched=True,
                            nodata=raster_nodata,
                        )
                    except Exception:
                        # Geometry might be degenerately small or invalid for mask
                        continue

                    # Flatten and drop nodata
                    data = masked[0].ravel()
                    if raster_nodata is not None:
                        data = data[data != raster_nodata]
                    data = data[data != 0]  # if 0 is also non-heat / invalid

                    if data.size == 0:
                        continue

                    # Simple mean (pixels are uniform)
                    slice_mean_lst = float(np.mean(data))

                    parcel_gdf.at[idx, f"mean_lst_{direction}"] = slice_mean_lst

            # After filling parcel_gdf, keep only relevant columns and write to CSV
            out_cols = ["APN", "sale_year"] + [f"mean_lst_{d}" for d in DIRECTIONS]
            out_df = parcel_gdf[out_cols].copy()

            out_csv = f"{path}/spatial_lag_instrument_inputs_{year}m_buffer_{buff}_full_sample_large_buffers.csv"
            out_df.to_csv(out_csv, index=False)
            print(f"  Saved {out_csv}")


# In[ ]:




