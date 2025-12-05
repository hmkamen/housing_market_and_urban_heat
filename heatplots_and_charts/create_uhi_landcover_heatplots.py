#!/usr/bin/env python
# coding: utf-8

# In[503]:


####this file cleans parcel sales information from the file system at:
##### https://www.dropbox.com/sh/0e8wltu2kb9s23y/AAAtlwnfP4bB3pY-Fj80YSE8a/Archived_Maricopa_Parcel_Files?e=1&dl=0
#####then joins parcel sales information with:
####hmda dataparcel lst by year, block imperviousness by year, and vegetative ring of parcel by year


# In[163]:


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
path='/Users/hannahkamen/Downloads'
import warnings
from shapely.geometry import MultiPolygon, Polygon

import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from rasterio.mask import mask
from shapely.wkt import loads
from shapely.geometry import box

from collections import Counter
from rasterstats import zonal_stats
from shapely.geometry import Point
# Load Maricopa County boundary (replace with correct path if needed)
maricopa_gdf = gpd.read_file("%s/maricopa/maricopa.shp"%path)
#get zips
zips=gpd.read_file('%s/Maricopa_County_Zip_Codes/ZipCodes.shp'%path)
pumas=gpd.read_file('%s/tl_2021_04_puma10/tl_2021_04_puma10.shp'%path)
tracts=gpd.read_file('%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp'%path)
subdivisions=gpd.read_file("%s/Subdivisions/Subdivisions.shp" % path)
blocks=gpd.read_file("%s/tl_2010_04013_tabblock10/tl_2010_04013_tabblock10.shp"%path)
# Filter or ignore specific warning types
warnings.filterwarnings('ignore')


# In[178]:


# Read the datasets
burden = pd.read_stata(f"{path}/hausman_taylor_test_4_10_v2_all_buffers_complete.dta")
blocks = gpd.read_file(f"{path}/tl_2010_04013_tabblock10/tl_2010_04013_tabblock10.shp")
blocks['BLOCKID10']=blocks['GEOID10'].astype(float)
tracts = gpd.read_file(f"{path}/tl_2010_04013_tract10/tl_2010_04013_tract10.shp")

# Merge block geometries onto burden by the column "BLOCKCE10"
burden_blocks = burden.merge(blocks[['BLOCKID10', 'geometry']], on='BLOCKID10', how='left')

# Convert the merged DataFrame into a GeoDataFrame
burden_blocks = gpd.GeoDataFrame(burden_blocks, geometry='geometry')

# Make sure coordinate systems match
burden_blocks = burden_blocks.to_crs(tracts.crs)

# Plot the heatmap of gradient_alignment_score by block, with tract boundaries overlaid
fig, ax = plt.subplots(figsize=(10, 10))
burden_blocks.plot(column='gradient_alignment_score_500',
                   cmap='hot',   # Change color scheme as desired
                   legend=True,
                   ax=ax,
                   edgecolor='none')  # Use 'none' to avoid borders if desired

# Overlay the tract boundaries
for coll in ax.collections:
    coll.set_edgecolor("none")
    coll.set_antialiased(False)

##limit tracts
# # Calculate the 90th percentile of geometry areas
area_threshold = tracts.geometry.area.quantile(0.95)

# # Filter out rows with area greater than the threshold
tracts_filtered = tracts[tracts.geometry.area <= area_threshold]
tracts_filtered=tracts_filtered.to_crs(gdf_clipped.crs)
tracts_filtered.plot(ax=ax, facecolor="none", edgecolor="black",linewidth = 0.15)

# Optionally, set a title and remove axis ticks for a cleaner display
# ax.set_title("Heatmap of Gradient Alignment Score by Block with Tract Outlines", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])


plt.show()


# In[145]:


# Convert the merged DataFrame back to a GeoDataFrame.
df_wide_geo=df_wide.merge(blocks[['GEOID10','geometry']],on='GEOID10')
gdf = gpd.GeoDataFrame(df_wide_geo, geometry='geometry',crs=blocks.crs)

xmin, ymin = -112.45, 33.2
xmax, ymax = -111.6, 33.9

# Create a bounding box using shapely.geometry.box
bbox = box(xmin, ymin, xmax, ymax)

# Clip the gdf_filtered GeoDataFrame to the bounding box
gdf_clipped = gpd.clip(gdf, bbox)
# 6. Plot the heat map of the relative change by GEOID.
fig, ax = plt.subplots(figsize=(10, 8))
gdf_clipped.plot(column='Temperature Change Relative to County Mean', ax=ax, legend=True, edgecolor='none', linewidth=0, rasterized=True,
                vmin=-10, vmax=6, cmap="RdYlBu_r")
# The plot returns a GeoAxes with a list of collections (usually one PolyCollection).
for coll in ax.collections:
    coll.set_edgecolor("none")
    coll.set_antialiased(False)

##limit tracts
# # Calculate the 90th percentile of geometry areas
area_threshold = tracts.geometry.area.quantile(0.95)

# # Filter out rows with area greater than the threshold
tracts_filtered = tracts[tracts.geometry.area <= area_threshold]
tracts_filtered=tracts_filtered.to_crs(gdf_clipped.crs)
tracts_filtered.plot(ax=ax, facecolor="none", edgecolor="black",linewidth = 0.15)
plt.xlim(-112.46,-111.59)
plt.ylim(33.2,33.91)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("%s/temp_variation_over_time.png"%path, dpi=300, bbox_inches='tight')


# In[146]:


burden['Temp Max']=burden['t2_mean_500']
burden['Temp Min']=burden['t2_mean_500']
burden_std=burden.groupby('TRACTCE10',as_index=False).agg({'Temp Max':max,'Temp Min':min})
burden_std['Temp. Range'] = burden_std['Temp Max']- burden_std['Temp Min']


# In[16]:


# Load the grid shapefile
block_air_temp=pd.DataFrame()
for year in [2009,2018]:
    ###print which year we're on...
    print(year)
    
    ## set raster lst path
    raster_path = "%s/air_temp_july_min_1k_ncar_%s.tif"%(path,year)
    ## get raster crs
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    ## set parcel gdf to raster crs
    parcel_gdf = blocks.to_crs(raster_crs)

    # Define batch size for processing
    batch_size = 100000  # Adjust as needed based on memory limits

    # Initialize an empty list to store results
    mean_lst_values = []

    # Process parcels in batches
    for i in range(0, len(parcel_gdf), batch_size):

        parcel_batch = parcel_gdf.iloc[i:i + batch_size]

        # Perform zonal statistics in batch
        stats = zonal_stats(
            parcel_batch,
            raster_path,
            stats="mean",
            nodata=0,   # Replace NoData with 0 or adjust as necessary
            geojson_out=False,  # Simplifies results for faster processing
            all_touched=True    # Includes pixels partially covered by the parcel geometry
        )

        # Append batch results
        mean_lst_values.extend([stat['mean'] if stat['mean'] is not None else float('nan') for stat in stats])

    # Add the mean LST values to the parcels GeoDataFrame
    parcel_gdf['t2_mean'] = mean_lst_values
    parcel_gdf['t2_mean']=((parcel_gdf['t2_mean']-273.15) * (9/5)) + 32
    parcel_gdf['sale_year']=year
    parcel_gdf=parcel_gdf[['GEOID10','sale_year','t2_mean','geometry']]
    block_air_temp=pd.concat([block_air_temp,parcel_gdf],ignore_index=True)

block_air_temp=gpd.GeoDataFrame(block_air_temp,geometry='geometry',crs=parcel_gdf.crs)


# In[25]:


mean_by_year=block_air_temp.groupby(['sale_year'],as_index=False).agg({'t2_mean':'mean'})
mean_by_year=mean_by_year.rename(columns={'t2_mean':'annual_mean'})
block_air_temp1=block_air_temp.merge(mean_by_year,on='sale_year')
block_air_temp1['Temperature Relative to County Mean']=block_air_temp1['t2_mean']-block_air_temp1['annual_mean']
df_wide = block_air_temp1.pivot(index='GEOID10', columns='sale_year', values='Temperature Relative to County Mean')
df_wide=df_wide.reset_index()
df_wide['Temperature Change Relative to County Mean']= df_wide[2018]-df_wide[2009]


# In[91]:


# Convert the merged DataFrame back to a GeoDataFrame.
df_wide_geo=df_wide.merge(blocks[['GEOID10','geometry']],on='GEOID10')
gdf = gpd.GeoDataFrame(df_wide_geo, geometry='geometry',crs=blocks.crs)

xmin, ymin = -112.45, 33.2
xmax, ymax = -111.6, 33.9

# Create a bounding box using shapely.geometry.box
bbox = box(xmin, ymin, xmax, ymax)

# Clip the gdf_filtered GeoDataFrame to the bounding box
gdf_clipped = gpd.clip(gdf, bbox)
# 6. Plot the heat map of the relative change by GEOID.
fig, ax = plt.subplots(figsize=(10, 8))
gdf_clipped.plot(column='Temperature Change Relative to County Mean', ax=ax, legend=True, edgecolor='none', linewidth=0, rasterized=True,
                vmin=-10, vmax=6, cmap="RdYlBu_r")
# The plot returns a GeoAxes with a list of collections (usually one PolyCollection).
for coll in ax.collections:
    coll.set_edgecolor("none")
    coll.set_antialiased(False)

##limit tracts
# # Calculate the 90th percentile of geometry areas
area_threshold = tracts.geometry.area.quantile(0.95)

# # Filter out rows with area greater than the threshold
tracts_filtered = tracts[tracts.geometry.area <= area_threshold]
tracts_filtered=tracts_filtered.to_crs(gdf_clipped.crs)
tracts_filtered.plot(ax=ax, facecolor="none", edgecolor="black",linewidth = 0.15)
plt.xlim(-112.46,-111.59)
plt.ylim(33.2,33.91)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("%s/temp_variation_over_time.png"%path, dpi=300, bbox_inches='tight')


# In[95]:


# Load the grid shapefile
block_air_temp=pd.DataFrame()
for year in [2015]:
    ###print which year we're on...
    print(year)
    
    ## set raster lst path
    raster_path = "%s/air_temp_july_min_1k_ncar_%s.tif"%(path,year)
    ## get raster crs
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    ## set parcel gdf to raster crs
    parcel_gdf = blocks.to_crs(raster_crs)

    # Define batch size for processing
    batch_size = 100000  # Adjust as needed based on memory limits

    # Initialize an empty list to store results
    mean_lst_values = []

    # Process parcels in batches
    for i in range(0, len(parcel_gdf), batch_size):

        parcel_batch = parcel_gdf.iloc[i:i + batch_size]

        # Perform zonal statistics in batch
        stats = zonal_stats(
            parcel_batch,
            raster_path,
            stats="mean",
            nodata=0,   # Replace NoData with 0 or adjust as necessary
            geojson_out=False,  # Simplifies results for faster processing
            all_touched=True    # Includes pixels partially covered by the parcel geometry
        )

        # Append batch results
        mean_lst_values.extend([stat['mean'] if stat['mean'] is not None else float('nan') for stat in stats])

    # Add the mean LST values to the parcels GeoDataFrame
    parcel_gdf['t2_mean'] = mean_lst_values
    parcel_gdf['t2_mean']=((parcel_gdf['t2_mean']-273.15) * (9/5)) + 32
    parcel_gdf['sale_year']=year
    parcel_gdf=parcel_gdf[['GEOID10','TRACTCE10','sale_year','t2_mean','geometry']]
    block_air_temp=pd.concat([block_air_temp,parcel_gdf],ignore_index=True)

block_air_temp=gpd.GeoDataFrame(block_air_temp,geometry='geometry',crs=parcel_gdf.crs)


# In[139]:


block_air_temp['Temp Max']=block_air_temp['t2_mean']
block_air_temp['Temp Min']=block_air_temp['t2_mean']
block_air_temp_std=block_air_temp.groupby('TRACTCE10',as_index=False).agg({'Temp Max':max,'Temp Min':min})
block_air_temp_std['Temp. Range'] = block_air_temp_std['Temp Max']- block_air_temp_std['Temp Min']


# In[140]:


block_air_temp_std=block_air_temp_std.merge(tracts,on='TRACTCE10')


# In[142]:


gdf_clipped['Temp. Range'].describe()


# In[141]:


block_air_temp_std=gpd.GeoDataFrame(block_air_temp_std,geometry='geometry',crs=tracts.crs)
block_air_temp_std=block_air_temp_std.loc[block_air_temp_std['Temp. Range']<4.5]
gdf = block_air_temp_std.copy().to_crs(blocks.crs)
xmin, ymin = -112.45, 33.2
xmax, ymax = -111.6, 33.9

# Create a bounding box using shapely.geometry.box
bbox = box(xmin, ymin, xmax, ymax)

# Clip the gdf_filtered GeoDataFrame to the bounding box
gdf_clipped = gpd.clip(gdf, bbox)
# 6. Plot the heat map of the relative change by GEOID.
fig, ax = plt.subplots(figsize=(10, 8))
gdf_clipped.plot(column='Temp. Range', ax=ax, legend=True, edgecolor='none', linewidth=0,rasterized=True )
# The plot returns a GeoAxes with a list of collections (usually one PolyCollection).
for coll in ax.collections:
    coll.set_edgecolor("none")
    coll.set_antialiased(False)

##limit tracts
# # Calculate the 90th percentile of geometry areas
area_threshold = tracts.geometry.area.quantile(0.95)

# # Filter out rows with area greater than the threshold
tracts_filtered = tracts[tracts.geometry.area <= area_threshold]
tracts_filtered=tracts_filtered.to_crs(gdf_clipped.crs)
tracts_filtered.plot(ax=ax, facecolor="none", edgecolor="black",linewidth = 0.15 )   
plt.xlim(-112.46,-111.59)
plt.ylim(33.2,33.91)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("%s/temp_variation_within_tracts.png"%path, dpi=300, bbox_inches='tight')


# In[ ]:


gdf = block_air_temp.copy().to_crs(blocks.crs)
gdf['Relative UHI 2015']=gdf['t2_mean']-gdf['t2_mean'].mean()
xmin, ymin = -112.45, 33.2
xmax, ymax = -111.6, 33.9

# Create a bounding box using shapely.geometry.box
bbox = box(xmin, ymin, xmax, ymax)

# Clip the gdf_filtered GeoDataFrame to the bounding box
gdf_clipped = gpd.clip(gdf, bbox)
# 6. Plot the heat map of the relative change by GEOID.
fig, ax = plt.subplots(figsize=(10, 8))
gdf_clipped.plot(column='Relative UHI 2015', ax=ax, legend=True, edgecolor='none', linewidth=0, cmap="RdYlBu_r",rasterized=True,vmin=-10, vmax=6 )
# The plot returns a GeoAxes with a list of collections (usually one PolyCollection).
for coll in ax.collections:
    coll.set_edgecolor("none")
    coll.set_antialiased(False)

##limit tracts
# # Calculate the 90th percentile of geometry areas
area_threshold = tracts.geometry.area.quantile(0.95)

# # Filter out rows with area greater than the threshold
tracts_filtered = tracts[tracts.geometry.area <= area_threshold]
tracts_filtered=tracts_filtered.to_crs(gdf_clipped.crs)
tracts_filtered.plot(ax=ax, facecolor="none", edgecolor="black",linewidth = 0.15 )   
plt.xlim(-112.46,-111.59)
plt.ylim(33.2,33.91)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("%s/temp_variation_over_space.png"%path, dpi=300, bbox_inches='tight')


# In[90]:


gdf = block_air_temp.copy().to_crs(blocks.crs)
gdf['Relative UHI 2015']=gdf['t2_mean']-gdf['t2_mean'].mean()
xmin, ymin = -112.45, 33.2
xmax, ymax = -111.6, 33.9

# Create a bounding box using shapely.geometry.box
bbox = box(xmin, ymin, xmax, ymax)

# Clip the gdf_filtered GeoDataFrame to the bounding box
gdf_clipped = gpd.clip(gdf, bbox)
# 6. Plot the heat map of the relative change by GEOID.
fig, ax = plt.subplots(figsize=(10, 8))
gdf_clipped.plot(column='Relative UHI 2015', ax=ax, legend=True, edgecolor='none', linewidth=0, cmap="RdYlBu_r",rasterized=True,vmin=-10, vmax=6 )
# The plot returns a GeoAxes with a list of collections (usually one PolyCollection).
for coll in ax.collections:
    coll.set_edgecolor("none")
    coll.set_antialiased(False)

##limit tracts
# # Calculate the 90th percentile of geometry areas
area_threshold = tracts.geometry.area.quantile(0.95)

# # Filter out rows with area greater than the threshold
tracts_filtered = tracts[tracts.geometry.area <= area_threshold]
tracts_filtered=tracts_filtered.to_crs(gdf_clipped.crs)
tracts_filtered.plot(ax=ax, facecolor="none", edgecolor="black",linewidth = 0.15 )   
plt.xlim(-112.46,-111.59)
plt.ylim(33.2,33.91)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("%s/temp_variation_over_space.png"%path, dpi=300, bbox_inches='tight')


# In[908]:


tracts09=gpd.read_file("%s/tl_2010_04013_tract10/natural_land_cover_pct_2009.shp" % path).to_crs('EPSG:26912')
tracts18=gpd.read_file("%s/tl_2010_04013_tract10/natural_land_cover_pct_2018.shp" % path).to_crs('EPSG:26912')


# In[910]:


tracts09['sum']=tracts09[[x for x in tracts09.columns if "area" in x]].sum(axis=1)
# tracts09['sum']=np.where(tracts09['sum']<30,0,tracts09['sum'])
tracts18['sum']=tracts18[[x for x in tracts18.columns if "area" in x]].sum(axis=1)
# tracts18['sum']=np.where(tracts18['sum']<30,0,tracts18['sum'])


# In[911]:


del tracts18['geometry']
tract_merge=tracts09.merge(tracts18,on='TRACTCE10')
tract_merge['diff']=tract_merge['sum_x']-tract_merge['sum_y']


# In[921]:


plt.figure(figsize=(16, 8))

# Plot for 2009
tract_merge[tract_merge['geometry'].area < tract_merge['geometry'].area.quantile(.975)]\
    .plot(column='diff', cmap='Reds', linewidth=0.8, edgecolor='black', legend=True)
plt.title('% Loss of Natural and Ag. Land Cover By Tract 2009-2018')
plt.xlim(345000)
# Remove x-axis labels and ticks
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()


# In[992]:





# In[ ]:


### get temperature plots by block


# In[68]:


import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterstats import zonal_stats
import matplotlib.pyplot as plt

city_outline=gpd.read_file('%s/City_Limit_Light_Outline/City_Limit_Light_Outline.shp'%path)

# Load block shapefile and filter to Maricopa County (COUNTYFP10 == "013")
block_gdf = gpd.read_file(f"{path}/tabblock2010_04_pophu/tabblock2010_04_pophu.shp")
block_gdf = block_gdf[block_gdf["COUNTYFP10"] == "013"]
block_gdf = block_gdf[['BLOCKID10', 'geometry']].copy()


# Reproject to a CRS in meters for distance buffering
block_gdf = block_gdf.to_crs("EPSG:32612")  # UTM zone for Arizona

# Get overall centroid of the block layer
layer_centroid = block_gdf.unary_union.centroid

# Create 20 km buffer around the centroid
buffer_geom = layer_centroid.buffer(80000)  # 20,000 meters

# Keep only blocks that intersect with the buffer
block_gdf = block_gdf[block_gdf.intersects(buffer_geom)].copy()


# # Calculate area in square meters
# block_gdf['area_m2'] = block_gdf.geometry.area

# # Compute 95th percentile
# area_95th = block_gdf['area_m2'].quantile(0.99)

# # Filter out large blocks
# block_gdf = block_gdf[block_gdf['area_m2'] <= area_95th].copy()


# Initialize list to hold yearly stats
all_years_stats = []

for year in [2015]:
    
    raster_path = f"{path}/air_temp_july_min_1k_ncar_{year}.tif"

    # Open raster to get CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # Reproject blocks to match raster CRS
    block_gdf_raster_crs = block_gdf.to_crs(raster_crs)

    # Zonal stats: mean temp per block
    stats = zonal_stats(
        block_gdf_raster_crs,
        raster_path,
        stats="mean",
        nodata=0,
        geojson_out=False,
        all_touched=True
    )

    # Store results
    year_df = pd.DataFrame({
        'BLOCKID10': block_gdf['BLOCKID10'],
        't2_mean': [s['mean'] if s['mean'] is not None else np.nan for s in stats],
        'year': year
    })

    all_years_stats.append(year_df)

# Combine all years and compute average across years
df_all = pd.concat(all_years_stats)
df_mean = df_all.groupby('BLOCKID10', as_index=False)['t2_mean'].mean()

# Merge with block geometries (back in original CRS)
block_avg_temp = block_gdf.merge(df_mean, on='BLOCKID10')


# In[72]:


import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import box

# Step 1: Create bounding box in lon/lat
bbox_ll = box(-112.9, 33.1, -111.3, 33.8)
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_ll], crs="EPSG:4326")

# Step 2: Project to EPSG:32612
bbox_proj = bbox_gdf.to_crs("EPSG:32612")
xmin, ymin, xmax, ymax = bbox_proj.total_bounds

# Convert Kelvin to Fahrenheit
block_avg_temp['t2_mean_f'] = (block_avg_temp['t2_mean'] - 273.15) * (9/5) + 32

# Step 3: Plot with manual colorbar control
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot without automatic legend
plot = block_avg_temp.plot(
    column='t2_mean_f',
    cmap='coolwarm',
    linewidth=0.05,
    edgecolor='black',
    legend=False,
    ax=ax
)

# Set plot bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_axis_off()
ax.set_title("Mean Minimum July Air Temperature (2015) by Census Block – Maricopa County", fontsize=14)

# Create and scale a manual colorbar
norm = Normalize(vmin=block_avg_temp['t2_mean_f'].min(), vmax=block_avg_temp['t2_mean_f'].max())
sm = ScalarMappable(cmap='coolwarm', norm=norm)
sm._A = []  # required for ScalarMappable

# Position colorbar next to the plot and scale it vertically
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label("Temperature (°F)", fontsize=12)

plt.tight_layout()
plt.show()


# In[1020]:


tract_gr=master_final.groupby(['TRACTCE10'],as_index=False).agg({'t2_mean':'mean'})

fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
ax.set_facecolor('lightgrey')
plot_df=tracts.merge(tract_gr,on='TRACTCE10')
plot_df['t2_mean']=(plot_df['t2_mean']-273.15)*(9/5)+32
# tract_merge.to_crs(plot_df.crs).plot(facecolor='white', linewidth=0, edgecolor=None, legend=True,ax=ax)
plot_df.plot(column='t2_mean', cmap='Reds', linewidth=0.8, legend=True,ax=ax)

zips_lm.to_crs(plot_df.crs).plot(facecolor='none', edgecolor='black', linewidth=0.2,ax=ax)
city_outline.to_crs(plot_df.crs).plot(facecolor='none', edgecolor='black', linewidth=2,ax=ax)


plt.title('Mean HUMID Summer Temperature by Census Tract (°F)')
plt.ylim(33.205,33.95)
plt.xlim(-112.7,-111.59)
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()


# In[825]:


# #### import residential completions data

# post1=pd.read_csv('%s/Rescomps_Maricopa_2011-2019(Sheet1).csv'%path,encoding="ISO-8859-1")
# post2=pd.read_excel('%s/Rescomps_Maricopa_2000-2010.xlsx'%path)
post=pd.concat([post1,post2],ignore_index=True)
post=post[post['COMYEAR']>=2009].reset_index(drop=True)

# ###separate into urban and exurb
post['SQFT'] = pd.to_numeric(post['SQFT'], errors='coerce')
post=post[['SQFT','CITY','COMYEAR']].dropna()
post['SQFT']=post['SQFT'].astype(int)
urban=post[post['CITY'].isin(['PH','SC','GL'])]
rural=post[~post['CITY'].isin(['PH','SC','GL'])]

urban_gr=urban.groupby('COMYEAR',as_index=False).agg({'SQFT':sum})
rural_gr=rural.groupby('COMYEAR',as_index=False).agg({'SQFT':sum})


plt.figure(figsize=(10, 6))
plt.plot(urban_gr['COMYEAR'], urban_gr['SQFT'].rolling(window=3).mean(), marker='o', linestyle='-', color='gray',label="Urban")
plt.plot(rural_gr['COMYEAR'], rural_gr['SQFT'].rolling(window=3).mean(), marker='o', linestyle='-', color='green',label="Suburb/Exurb")

plt.legend()
plt.xlabel('Year')
plt.ylabel('Urban - Suburb SqFt Residential Completions')
plt.title('3-Year Rolling Residential Completions: Urban and Suburb/Exurb Zips')
plt.grid(True)
plt.show()


