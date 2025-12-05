#!/usr/bin/env python
# coding: utf-8

# In[22]:


####this file cleans parcel sales information from the file system at:
##### https://www.dropbox.com/sh/0e8wltu2kb9s23y/AAAtlwnfP4bB3pY-Fj80YSE8a/Archived_Maricopa_Parcel_Files?e=1&dl=0
#####then joins parcel sales information with:
####hmda dataparcel lst by year, block imperviousness by year, and vegetative ring of parcel by year


# In[23]:


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
# tracts=gpd.read_file('%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp'%path)
tracts=gpd.read_file('%s/tl_2020_04_tract/tl_2020_04_tract.shp'%path)
subdivisions=gpd.read_file("%s/Subdivisions/Subdivisions.shp" % path)
blocks=gpd.read_file("%s/tl_2010_04013_tabblock10/tl_2010_04013_tabblock10.shp"%path)
# Filter or ignore specific warning types
warnings.filterwarnings('ignore')


# In[24]:


tracts=tracts[tracts['COUNTYFP']=='013'].copy()


# In[28]:


### calculate mean min july temp from ncar data by 2020 tract
# Load the grid shapefile
master_air_temp_tract=pd.DataFrame()

## set raster lst path
raster_path = "%s/air_temp_july_min_1k_ncar_2018.tif"%path

## get raster crs
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

## set parcel gdf to raster crs
tract_gdf = tracts.to_crs(raster_crs)

# Initialize an empty list to store results
mean_lst_values = []

# Perform zonal statistics in batch
stats = zonal_stats(
    tract_gdf,
    raster_path,
    stats="mean",
    nodata=0,   # Replace NoData with 0 or adjust as necessary
    geojson_out=False,  # Simplifies results for faster processing
    all_touched=True    # Includes pixels partially covered by the parcel geometry
)

# Append batch results
mean_lst_values.extend([stat['mean'] if stat['mean'] is not None else float('nan') for stat in stats])

# Add the mean LST values to the parcels GeoDataFrame
tract_gdf['t2_mean'] = mean_lst_values


# In[36]:


tract_gdf_lm=tract_gdf[['TRACTCE','t2_mean','GEOID']]


# In[37]:


tract_gdf['FIP']=tract_gdf['GEOID'].astype(int)


# In[55]:


#### merge 2020 tract temp onto LEADS Data
#### import data
leads=pd.read_csv("%s/AZ-2022-LEAD-data/AZ AMI Census Tracts 2022.csv"%path)

###groupby tract, sum electricity expenditure and units
leads_gr=leads.groupby('FIP',as_index=False).agg({'ELEP*UNITS':sum,'UNITS':sum})

### merge on to tract dataframe

merge1=leads_gr.merge(tract_gdf,on='FIP')


# In[56]:


master_with_locs=gpd.read_file('%s/all_res_homes.shp'%path)


# In[83]:


median_cost


# In[82]:


# 1. Create deciles of the `t2_mean` variable
merge2["t2_decile"] = pd.qcut(merge2["t2_mean"], 10, labels=False)

# 2. Compute median elec_sq_ft across all data
median_cost = merge2["elec_sq_ft"].median()

# 3. Group by decile: calculate mean elec_sq_ft and max t2_mean (for labeling)
grouped = merge2.groupby("t2_decile").agg({
    "elec_sq_ft": lambda x: (x.mean() - median_cost) / median_cost,
    "t2_mean": "max"
}).reset_index()

# 4. Convert max t2_mean in each decile to °F for labeling
grouped["t2_f_label"] = (grouped["t2_mean"]-273.15) * 9/5 + 32
grouped["t2_f_label"] = grouped["t2_f_label"].round(1).astype(str) + "°F"

# 5. Create the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(grouped["t2_f_label"], grouped["elec_sq_ft"], color="steelblue", alpha=0.7)

# # 6. Add trendline
# x = np.arange(len(grouped))
# z = np.polyfit(x, grouped["elec_sq_ft"], 1)
# p = np.poly1d(z)
# plt.plot(grouped["t2_f_label"], p(x), color="darkred", linestyle="--", label="Trendline")

# 7. Final plot formatting
plt.axhline(0, color="gray", linestyle=":")
plt.xlabel("Midpoint of Min. July Temp (°F) Decile",fontsize=12)
plt.ylabel("Electricity Cost per Sq Ft Relative to Median (2020 Dollars)",fontsize=12)
# plt.title("Residential Electricity Cost per Sq Ft by Minimum July Temperature Decile")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()



# In[64]:


import geopandas as gpd

# 2. Ensure both layers use the same CRS (coordinate reference system)
master_with_locs = master_with_locs.to_crs(tracts.crs)

# 3. Perform spatial join: assign tract attributes to each parcel
# 'within' means parcel must fall entirely within the tract polygon
parcels_with_tracts = gpd.sjoin(master_with_locs, tracts[['GEOID','geometry']], how="left", predicate="within")

parcels_with_tracts=parcels_with_tracts[~parcels_with_tracts['GEOID'].isnull()]
parcels_with_tracts['FIP']=parcels_with_tracts['GEOID'].astype(int)

### group by tract
parcels_gr=parcels_with_tracts.groupby(['FIP'],as_index=False).agg({'sq_ft':'median'})

##merge
merge2=parcels_gr.merge(merge1,on="FIP")

###calc energy expenditure per sq foot
merge2['total_sq_ft']=merge2['UNITS']*merge2['sq_ft']
merge2['elec_sq_ft']=merge2['ELEP*UNITS']/merge2['total_sq_ft']


# In[78]:


merge2['elec_unit']=merge2['ELEP*UNITS']/merge2['UNITS']


# In[79]:


merge2['elec_unit'].describe()


# In[69]:


merge2['sq_ft'].describe()


# In[66]:


# 3. Create the scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(merge2["t2_mean"], merge2["elec_sq_ft"], alpha=0.5)
plt.xlabel("Min. July Temperature, Degrees Celsius")
plt.ylabel("Mean Annual Electricity Expenditure per Sq. Ft.")
plt.title("")
plt.grid(True)
plt.show()


# In[38]:


leads['FIP']


# In[178]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'path' with the correct path string

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


# In[21]:


list(df_wide)


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


# In[102]:


block_air_temp_std['Temp. Range'].describe()


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


# In[14]:


master=pd.read_stata("%s/hausman_taylor_test_4_9_v2_all_buffers_complete.dta"%path)
master_lm=master[["TRACTCE10","APN","child_per_household"]].copy()
tracts = tracts[['TRACTCE10', 'geometry']]
tracts['TRACTCE10']=tracts['TRACTCE10'].astype(int)
# Step 1: Aggregate median_income by tract across all years
agg_income = master.groupby("TRACTCE10")["child_per_household"].mean().reset_index()

# Step 2: Merge with tract geometry
merged = tracts.merge(agg_income, on="TRACTCE10", how="left")

# Step 3: Plot
fig, ax = plt.subplots(figsize=(10, 10))
merged.plot(column='child_per_household', cmap='viridis', linewidth=0.1, edgecolor='white', legend=True, ax=ax)
ax.set_title("Average Child Per Household by Tract (All Years)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


# In[17]:


master=pd.read_stata("%s/hausman_taylor_test_3_31_v2_all_buffers_complete.dta"%path)
master_lm=master[["TRACTCE10","APN","t2_mean_500"]].copy()
tracts = tracts[['TRACTCE10', 'geometry']]
tracts['TRACTCE10']=tracts['TRACTCE10'].astype(int)
# Step 1: Aggregate median_income by tract across all years
agg_income = master.groupby("TRACTCE10")["t2_mean_500"].mean().reset_index()

# Step 2: Merge with tract geometry
merged = tracts.merge(agg_income, on="TRACTCE10", how="left")

# Step 3: Plot
fig, ax = plt.subplots(figsize=(10, 10))
merged.plot(column='t2_mean_500', cmap='viridis', linewidth=0.1, edgecolor='white', legend=True, ax=ax)
ax.set_title("Average heat by Tract (All Years)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


# In[10]:


master=pd.read_stata("%s/hausman_taylor_test_3_31_v2_all_buffers_complete.dta"%path)
master_lm=master[["TRACTCE10","APN","median_income"]].copy()
tracts = tracts[['TRACTCE10', 'geometry']]
tracts['TRACTCE10']=tracts['TRACTCE10'].astype(int)
# Step 1: Aggregate median_income by tract across all years
agg_income = master.groupby("TRACTCE10")["median_income"].mean().reset_index()

# Step 2: Merge with tract geometry
merged = tracts.merge(agg_income, on="TRACTCE10", how="left")

# Step 3: Plot
fig, ax = plt.subplots(figsize=(10, 10))
merged.plot(column='median_income', cmap='viridis', linewidth=0.1, edgecolor='white', legend=True, ax=ax)
ax.set_title("Average Median Income by Tract (All Years)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()



# In[ ]:





# In[12]:


master=pd.read_stata("%s/hausman_taylor_test_3_31_v2_all_buffers_complete.dta"%path)
master_lm=master[["TRACTCE10","APN","tract_density"]].copy()
tracts = tracts[['TRACTCE10', 'geometry']]
tracts['TRACTCE10']=tracts['TRACTCE10'].astype(int)
# Step 1: Aggregate median_income by tract across all years
agg_income = master.groupby("TRACTCE10")["tract_density"].mean().reset_index()

# Step 2: Merge with tract geometry
merged = tracts.merge(agg_income, on="TRACTCE10", how="left")

# Step 3: Plot
fig, ax = plt.subplots(figsize=(10, 10))
merged.plot(column='tract_density', cmap='viridis', linewidth=0.1, edgecolor='white', legend=True, ax=ax)
ax.set_title("Average Density by Tract (All Years)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


# In[13]:


master=pd.read_stata("%s/hausman_taylor_test_3_31_v2_all_buffers_complete.dta"%path)
master_lm=master[["TRACTCE10","APN","median_age"]].copy()
tracts = tracts[['TRACTCE10', 'geometry']]
tracts['TRACTCE10']=tracts['TRACTCE10'].astype(int)
# Step 1: Aggregate median_income by tract across all years
agg_income = master.groupby("TRACTCE10")["median_age"].mean().reset_index()

# Step 2: Merge with tract geometry
merged = tracts.merge(agg_income, on="TRACTCE10", how="left")

# Step 3: Plot
fig, ax = plt.subplots(figsize=(10, 10))
merged.plot(column='median_age', cmap='viridis', linewidth=0.1, edgecolor='white', legend=True, ax=ax)
ax.set_title("Average Age by Tract (All Years)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


# In[160]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, FancyArrowPatch
import matplotlib as mpl

# Define 8 directional slices.
directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
# Central angles for each slice (in degrees).
central_angles = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]

# Example anomaly values for each directional slice (from cool to hot).
# Negative values indicate cool anomalies and positive values indicate hot anomalies.
anomalies = {
    'N':  0.8,   # hot anomaly
    'NE': 0.2,
    'E':  -0.1,  # slight cool anomaly
    'SE': -0.5,  # cooler
    'S':  -1.0,  # very cool
    'SW': -0.2,
    'W':  0.3,
    'NW': 0.5
}

# Set up colormap: we use RdYlGn_r so that negative (cool) values become green and positive (hot) become red.
cmap = mpl.cm.RdYlGn_r
# Assume anomaly values range from -1.0 to 1.0.
norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

# Create figure and axis.
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
center = (0,0)
radius = 1  # radius for the circle

# Draw 8 wedge slices.
for d, angle in zip(directions, central_angles):
    # Each wedge spans 45 degrees: 22.5° on either side of the central angle.
    start_angle = angle - 22.5
    end_angle = angle + 22.5
    color = cmap(norm(anomalies[d]))
    wedge = Wedge(center, radius, start_angle, end_angle, facecolor=color,
                  edgecolor='black', lw=1)
    ax.add_patch(wedge)
    
    # Place the direction label at the wedge's center.
    mid_angle_rad = np.deg2rad(angle)
    label_radius = radius * 0.7
    x_text = label_radius * np.cos(mid_angle_rad)
    y_text = label_radius * np.sin(mid_angle_rad)
    ax.text(x_text, y_text, d, ha='center', va='center', fontsize=12)

# Draw an arrow representing the observed airflow trajectory.
# (Set your desired trajectory angle; here we use 120° as an example.)
trajectory_angle = 120  
tra_rad = np.deg2rad(trajectory_angle)
arrow_length = radius * 0.8
arrow_x = arrow_length * np.cos(tra_rad)
arrow_y = arrow_length * np.sin(tra_rad)
arrow = FancyArrowPatch(posA=center, posB=(arrow_x, arrow_y),
                         arrowstyle='->', mutation_scale=20,
                         color='blue', lw=2)
ax.add_patch(arrow)
ax.text(arrow_x, arrow_y, f"{trajectory_angle}°", color='blue',
        fontsize=12, ha='left', va='bottom')

# Draw dotted weight lines from the arrow tip to the midpoint of each wedge.
for d, angle in zip(directions, central_angles):
    # Compute the minimal angular difference between the trajectory angle and slice center.
    angle_diff = abs(angle - trajectory_angle)
    angle_diff = min(angle_diff, 360 - angle_diff)
    # Weight decays linearly from 1 at 0° difference to 0 at 180°.
    weight = 1 - (angle_diff / 180)
    
    # Compute the wedge midpoint coordinates (just inside the outer edge).
    wedge_mid_x = (radius * 0.9) * np.cos(np.deg2rad(angle))
    wedge_mid_y = (radius * 0.9) * np.sin(np.deg2rad(angle))
    
    # Plot a dotted line if the weight is significant.
    if weight > 0.1:
        ax.plot([arrow_x, wedge_mid_x],
                [arrow_y, wedge_mid_y],
                linestyle=':', color='gray', alpha=weight, lw=2)

# Remove axes for a cleaner look.
ax.axis('off')

# Add a colorbar legend for the anomalies.
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Temperature Anomaly', fontsize=12)

plt.title("Infographic: New Instrument Logic", fontsize=14)
plt.show()


# In[159]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Number of time steps
time_steps = 20

# Create a time vector
t = np.linspace(0, 10, time_steps)  # arbitrary time units

# Generate a slightly curved horizontal trajectory:
# x goes from -2 to 2, and y follows a gentle sine curve.
# z remains constant (e.g., near surface, z = 0)
x = np.linspace(-2, 2, time_steps)
y = 0.3 * np.sin(np.linspace(0, np.pi, time_steps))
z = np.zeros_like(x)  # altitude does not change

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot the curved trajectory line
ax.plot(x, y, z, color='gray', linestyle='--', linewidth=2)

# Plot scatter points along the trajectory, colored by time
sc = ax.scatter(x, y, z, c=t, cmap='plasma', s=60, edgecolors='k')

# Place a dark square representing the air parcel at the grid's center (0, 0, 0)
ax.scatter(0, 0.005, 0, marker='s', s=100, color='black', label='Air Parcel')

# Add a colorbar to show time progression
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Time Steps')

# Label axes
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Altitude (km)')

# Set title
ax.set_title("Conceptual HYSPLIT Air Parcel Trajectory")

plt.legend()  # Show legend for the parcel marker
plt.show()


# In[11]:


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Create trajectory subsets based on hour_start
traj_01 = traj[traj['hour_start'] == 1]
traj_07 = traj[traj['hour_start'] == 7]
traj_13 = traj[traj['hour_start'] == 13]
traj_19 = traj[traj['hour_start'] == 19]

# Keep only necessary columns in tracts
tracts = tracts[['TRACTCE10', 'geometry']]

# Compute tract areas and remove tracts larger than the 95th percentile
tracts["area"] = tracts.geometry.area
area_threshold = np.percentile(tracts["area"], 100)
# tracts = tracts[tracts["area"] <= area_threshold]

# Calculate the centroid of all tracts (geometric center)
tracts_center = tracts.unary_union.centroid

# Remove tracts more than 20km from the center
tracts["distance_from_center"] = tracts.geometry.centroid.distance(tracts_center)
tracts = tracts[tracts["distance_from_center"] <= 20000]  # 20 km = 20,000 meters

# List of trajectory datasets and corresponding labels
traj_dfs = {'01': traj_01, '07': traj_07, '13': traj_13, '19': traj_19}

# Dictionary to store mean bearing values
all_bearing_values = []

# First loop: Collect all bearing values to determine global min/max
for traj_df in traj_dfs.values():
    traj_gdf = gpd.GeoDataFrame(traj_df, geometry='geometry_previous', crs="EPSG:4326")
    traj_gdf = traj_gdf.to_crs(tracts.crs)

    joined = gpd.sjoin(traj_gdf, tracts, how='left', predicate='within')
    tract_avg_bearing = joined.groupby('TRACTCE10')['bearing'].mean().reset_index()

    all_bearing_values.extend(tract_avg_bearing["bearing"].dropna().values)

# Get the global min and max for bearing
vmin, vmax = min(all_bearing_values), max(all_bearing_values)

# Second loop: Generate plots with consistent color scales
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, (hour, traj_df) in enumerate(traj_dfs.items()):
    traj_gdf = gpd.GeoDataFrame(traj_df, geometry='geometry_previous', crs="EPSG:4326")
    traj_gdf = traj_gdf.to_crs(tracts.crs)
    
    joined = gpd.sjoin(traj_gdf, tracts, how='left', predicate='within')
    tract_avg_bearing = joined.groupby('TRACTCE10')['bearing'].mean().reset_index()
    
    tracts_with_bearing = tracts.merge(tract_avg_bearing, on='TRACTCE10', how='left')

    # Plot the heatmap with grayscale color scale
    ax = axes[i]
    tracts_with_bearing.plot(
        column='bearing', cmap='Greys', edgecolor='black', linewidth=0.5,
        legend=True, vmin=vmin, vmax=vmax, ax=ax
    )
    
    ax.set_title(f"Mean Bearing of Surface Air Parcel Trajectories by Tract \n Hour {int(hour)-1}")
    ax.set_axis_off()

plt.tight_layout()
plt.show()


# In[4]:


import geopandas as gpd
import matplotlib.pyplot as plt

# Load tracts shapefile
tracts = gpd.read_file(f"{path}/tl_2010_04013_tract10/tl_2010_04013_tract10.shp")
tracts['TRACTCE10'] = tracts['TRACTCE10'].astype(int)

# Merge tract data with the calculated instrument variable
tract_level_instrument_map = tracts.merge(weighted_tract, on='TRACTCE10')

# Create the heatmap
fig, ax = plt.subplots(figsize=(12, 12))
tract_level_instrument_map[tract_level_instrument_map['geometry'].area< tract_level_instrument_map['geometry'].area.quantile(.99)].plot(
    ax=ax,
    column='tract_mean_buffer_lst',  # Color-coded by this variable
    cmap='hot',  # Heatmap color scheme
    linewidth=0.5,  # Thin boundary lines for better visualization
    edgecolor='black',  # Outline tracts
    legend=True
)

# Format plot
ax.set_title('Heatmap of tract_inv_min_angle_w', fontsize=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()


# In[1040]:


import geopandas as gpd
import matplotlib.pyplot as plt

# Load tracts shapefile
tracts = gpd.read_file(f"{path}/tl_2010_04013_tract10/tl_2010_04013_tract10.shp")
tracts['TRACTCE10'] = tracts['TRACTCE10'].astype(int)

# Merge tract data with the calculated instrument variable
tract_level_instrument_map = tracts.merge(weighted_tract, on='TRACTCE10')

# Create the heatmap
fig, ax = plt.subplots(figsize=(12, 12))
tract_level_instrument_map.plot(
    ax=ax,
    column='tract_inv_min_angle_w',  # Color-coded by this variable
    cmap='hot',  # Heatmap color scheme
    linewidth=0.5,  # Thin boundary lines for better visualization
    edgecolor=None,  # Outline tracts
    legend=True
)
zips_lm.to_crs(tract_level_instrument_map.crs).plot(facecolor='none', edgecolor='black', linewidth=0.2,ax=ax)
# Format plot
ax.set_title('Heatmap of tract_inv_min_angle_w', fontsize=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()


# In[524]:


blocks=gpd.read_file('%s/tabblock2010_04_pophu/tabblock2010_04_pophu.shp'%path)
blocks['BLOCKID10']=blocks['BLOCKID10'].astype(int)
grouped=grouped.merge(blocks[['BLOCKID10','geometry']],on="BLOCKID10")
grouped=gpd.GeoDataFrame(grouped,geometry='geometry',crs=blocks.crs)


# In[702]:


# Option 1: Using GeoPandas plot method with legend for a straightforward approach
fig, ax = plt.subplots(figsize=(12, 12))

# [tracts['geometry'].area<0.01]
# Set the geometry to the computed centroids and plot, color-coding by inv_max_angle_w
tracts[tracts['sale_year']==2009].plot(
    ax=ax,
    column='inv_min_angle_w',
    cmap='hot',
    markersize=50,
    marker='o',
    legend=True
)
ax.set_title('Heatplot of PM 2.5 Air Pollution by Tract')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[703]:


# Option 1: Using GeoPandas plot method with legend for a straightforward approach
fig, ax = plt.subplots(figsize=(12, 12))

# [tracts['geometry'].area<0.01]
# Set the geometry to the computed centroids and plot, color-coding by inv_max_angle_w
tracts[tracts['sale_year']==2018].plot(
    ax=ax,
    column='inv_min_angle_w',
    cmap='hot',
    markersize=50,
    marker='o',
    legend=True
)
ax.set_title('Heatplot of PM 2.5 Air Pollution by Tract')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[704]:


# Option 1: Using GeoPandas plot method with legend for a straightforward approach
fig, ax = plt.subplots(figsize=(12, 12))

# [tracts['geometry'].area<0.01]
# Set the geometry to the computed centroids and plot, color-coding by inv_max_angle_w
tracts[tracts['sale_year']==2009].plot(
    ax=ax,
    column='DS_PM_pred',
    cmap='hot',
    markersize=50,
    marker='o',
    legend=True
)
ax.set_title('Heatplot of PM 2.5 Air Pollution by Tract')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[705]:


# Option 1: Using GeoPandas plot method with legend for a straightforward approach
fig, ax = plt.subplots(figsize=(12, 12))

# [tracts['geometry'].area<0.01]
# Set the geometry to the computed centroids and plot, color-coding by inv_max_angle_w
tracts[tracts['sale_year']==2018].plot(
    ax=ax,
    column='DS_PM_pred',
    cmap='hot',
    markersize=50,
    marker='o',
    legend=True
)
ax.set_title('Heatplot of PM 2.5 Air Pollution by Tract')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[688]:


# Option 1: Using GeoPandas plot method with legend for a straightforward approach
fig, ax = plt.subplots(figsize=(12, 12))
# [tracts['geometry'].area<0.01]
# Set the geometry to the computed centroids and plot, color-coding by inv_max_angle_w
tracts.plot(
    ax=ax,
    column='inv_min_angle_w',
    cmap='hot',
    markersize=50,
    marker='o',
    legend=True
)

ax.set_title('Heatplot of Difference Between Frequency weighted Trajectories and  by Geometry Centroid')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[531]:


# Option 1: Using GeoPandas plot method with legend for a straightforward approach
fig, ax = plt.subplots(figsize=(12, 12))

# Set the geometry to the computed centroids and plot, color-coding by inv_max_angle_w
tracts[tracts['geometry'].area<0.01].plot(
    ax=ax,
    column='max_bearing',
    cmap='hot',
    markersize=50,
    marker='o',
    legend=True
)
ax.set_title('Heatplot of Difference Between Frequency weighted Trajectories and  by Geometry Centroid')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[526]:


# Option 1: Using GeoPandas plot method with legend for a straightforward approach
fig, ax = plt.subplots(figsize=(12, 12))

# Set the geometry to the computed centroids and plot, color-coding by inv_max_angle_w
tracts[tracts['geometry'].area<0.01].plot(
    ax=ax,
    column='weighted_bearing',
    cmap='hot',
    markersize=50,
    marker='o',
    legend=True
)
ax.set_title('Heatplot of Difference Between Frequency weighted Trajectories and  by Geometry Centroid')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[ ]:





# In[677]:


tracts


# In[506]:





# In[1025]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assume df is your DataFrame with columns 't2_mean' (temperature) and 'lnp_ann' (sale price)

# 1. Define 10 equally spaced bins for temperature.
num_bins = 10
temp_min, temp_max = df['t2_mean'].min(), df['t2_mean'].max()
bins = np.linspace(temp_min, temp_max, num_bins + 1)

# 2. Bin the temperature data (left-inclusive bins)
df['temp_bin'] = pd.cut(df['t2_mean'], bins=bins, right=False)

# 3. Calculate frequency counts for each temperature bin
bin_counts = df['temp_bin'].value_counts().sort_index()

# 4. Group by temperature bin to compute the mean sale price per bin
bin_means = df.groupby('temp_bin')['lnp_ann'].mean().reset_index()

# 5. For plotting, use the low end (left bound) of each bin as the x-axis value.
bin_means['temp_bin_low'] = bin_means['temp_bin'].apply(lambda x: x.left)

# 6. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Draw a translucent blue background for each bin based on the frequency of observations.
max_count = bin_counts.max()
for interval, count in bin_counts.items():
    alpha_val = (count / max_count) * 0.5  # Scale alpha between 0 and 0.5
    ax.axvspan(interval.left, interval.right, color='blue', alpha=alpha_val)

# Plot the mean sale price for each bin as a scatter plot (with optional line connection)
ax.scatter(bin_means['temp_bin_low'], bin_means['lnp_ann'], color='black', s=100, zorder=5)
ax.plot(bin_means['temp_bin_low'], bin_means['lnp_ann'], color='black', linestyle='--', zorder=5)

ax.set_xlabel('Temperature (Low End of Bin)')
ax.set_ylabel('Mean Sale Price (lnp_ann)')
ax.set_title('Mean Sale Price per Temperature Bin with Bin Frequencies')
plt.show()


# In[1024]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assume df is your DataFrame with columns 'tlc_veg' (temperature) and 'lnp_ann' (sale price)

# 1. Define 10 equally spaced bins for temperature.
num_bins = 10
temp_min, temp_max = df['_sum_y'].min(), df['_sum_y'].max()
bins = np.linspace(temp_min, temp_max, num_bins + 1)

# 2. Bin the temperature data (left-inclusive bins)
df['veg_bin'] = pd.cut(df['_sum_y'], bins=bins, right=False)

# 3. Calculate frequency counts for each temperature bin
bin_counts = df['veg_bin'].value_counts().sort_index()

# 4. Group by temperature bin to compute the mean sale price per bin
bin_means = df.groupby('veg_bin')['lnp_ann'].mean().reset_index()

# 5. For plotting, use the low end (left bound) of each bin as the x-axis value.
bin_means['veg_bin_low'] = bin_means['veg_bin'].apply(lambda x: x.left)

# 6. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Draw a translucent blue background for each bin based on the frequency of observations.
max_count = bin_counts.max()
for interval, count in bin_counts.items():
    alpha_val = (count / max_count) * 0.5  # Scale alpha between 0 and 0.5
    ax.axvspan(interval.left, interval.right, color='blue', alpha=alpha_val)

# Plot the mean sale price for each bin as a scatter plot (with optional line connection)
ax.scatter(bin_means['veg_bin_low'], bin_means['lnp_ann'], color='black', s=100, zorder=5)
ax.plot(bin_means['veg_bin_low'], bin_means['lnp_ann'], color='black', linestyle='--', zorder=5)

ax.set_xlabel('Tract Vegetation Cover (Low End of Bin)')
ax.set_ylabel('Mean Sale Price (lnp_ann)')
ax.set_title('Mean Sale Price per Tract Vegetation Bin with Bin Frequencies')
plt.show()


# In[905]:


tracts09.crs


# In[908]:


tracts09=gpd.read_file("%s/tl_2010_04013_tract10/natural_land_cover_pct_2009.shp" % path).to_crs('EPSG:26912')
tracts18=gpd.read_file("%s/tl_2010_04013_tract10/natural_land_cover_pct_2018.shp" % path).to_crs('EPSG:26912')


# In[909]:


[x for x in tracts09.columns if "area" in x]


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


# In[938]:


plot_df


# In[992]:


city_outline=gpd.read_file('%s/City_Limit_Light_Outline/City_Limit_Light_Outline.shp'%path)


# In[653]:


plot_df


# In[ ]:


### get temperature plots by block


# In[68]:


import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterstats import zonal_stats
import matplotlib.pyplot as plt

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


# In[69]:


block_avg_temp.crs


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


blocks=

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


# In[612]:


tracts['geometry'].area.describe()


# In[633]:


ru_tracts = tracts[tracts['geometry'].area>0.1]


# In[634]:


ru_tracts['TRACTCE10'].unique()


# In[660]:


master_final['tlc_res'].describe()


# In[684]:


ru_tracts


# In[699]:


tract_temp


# In[705]:


tracts=gpd.read_file("%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp" % path)


# In[710]:


tracts['geometry'].area.quantile(.95)


# In[743]:


ru_tracts


# In[756]:


zips_lm


# In[ ]:


zips=gpd.read_file('%s/Maricopa_County_Zip_Codes/ZipCodes.shp'%path)
zips_lm=zips[['geometry','BdVal','USPSCityAl']]
zips_lm=zips_lm.rename(columns={'BdVal':'zipcode','USPSCityAl':'city'})


# In[757]:


zips_air=gpd.read_file('%s/mean_air_temp_by_zip_2009_2018.shp'%path)


# In[766]:


zips_air


# In[767]:


list(zips_air['city'].unique())


# In[1017]:


# 'MESA','TEMPE','CHANDLER','GILBERT','PEORIA','SCOTTDALE','AVONDALE','CAVE CREEK','PARADISE VALLEY','BUCKEYE','QUEEN CREEK','CAREFREE'
# Split the data into rural and urban tracts
rural = zips_air[~zips_air['city'].isin(['PHOENIX','SCOTTSDALE','GLENDALE','GILBERT',"MESA",'PEORIA','PARADISE VALLEY'])]
# rural=rural.loc[(rural['t2_mean'] > rural['t2_mean'].quantile(.10)) &(rural['t2_mean'] < rural['t2_mean'].quantile(.90)) ].copy()
urban = zips_air[zips_air['city'].isin(['PHOENIX','SCOTTSDALE','GLENDALE','GILBERT',"MESA",'PEORIA','PARADISE VALLEY'])]
# urban=urban.loc[(urban['t2_mean'] > urban['t2_mean'].quantile(.10)) &(urban['t2_mean'] < urban['t2_mean'].quantile(.90)) ].copy()

# Define a function to compute the area-weighted average temperature
def weighted_mean(df):
    return (df['t2_mean'] * df['area']).sum() / df['area'].sum()

# Group by sale_year and calculate the area-weighted t2_mean for each group
rural_avg = rural.groupby('sale_year').apply(weighted_mean).reset_index(name='rural_avg')
urban_avg = urban.groupby('sale_year').apply(weighted_mean).reset_index(name='urban_avg')

# rural_avg = rural.groupby('sale_year',as_index=False).agg({'t2_mean':'mean'})
# urban_avg = urban.groupby('sale_year',as_index=False).agg({'t2_mean':'mean'})


# Merge the two datasets on sale_year
avg_diff = pd.merge(urban_avg, rural_avg, on='sale_year')

# Compute the difference in area-weighted average temperature (urban minus rural)
avg_diff['temp_diff'] = avg_diff['urban_avg'] - avg_diff['rural_avg']
avg_diff['rolling'] = avg_diff['temp_diff'].rolling(window=3).mean()*1.8
# Plot the year-over-year temperature difference
plt.figure(figsize=(10, 6))
plt.plot(avg_diff['sale_year'], avg_diff['rolling'], marker='o', linestyle='-', color='red')
# plt.plot(avg_diff['sale_year'], avg_diff['rural_avg'], marker='o', linestyle='-', color='green')
plt.xlabel('Year')
plt.ylabel('Urban - Exurb Area Weighted Avg Temp Difference (°F)')
plt.title('3-Year Rolling HUMID Summer Temperature Difference between Urban/Suburb and Exurb Tracts')
plt.grid(True)
plt.show()


# In[1018]:


# 'MESA','TEMPE','CHANDLER','GILBERT','PEORIA','SCOTTDALE','AVONDALE','CAVE CREEK','PARADISE VALLEY','BUCKEYE','QUEEN CREEK','CAREFREE'
# Split the data into rural and urban tracts
rural = zips_air[~zips_air['city'].isin(['PHOENIX','SCOTTSDALE','GLENDALE','GILBERT',"MESA",'PEORIA','PARADISE VALLEY'])]
# rural=rural.loc[(rural['t2_mean'] > rural['t2_mean'].quantile(.10)) &(rural['t2_mean'] < rural['t2_mean'].quantile(.90)) ].copy()
urban = zips_air[zips_air['city'].isin(['PHOENIX','SCOTTSDALE','GLENDALE','GILBERT',"MESA",'PEORIA','PARADISE VALLEY'])]
# urban=urban.loc[(urban['t2_mean'] > urban['t2_mean'].quantile(.10)) &(urban['t2_mean'] < urban['t2_mean'].quantile(.90)) ].copy()

# Define a function to compute the area-weighted average temperature
def weighted_mean(df):
    return (df['t2_mean'] * df['area']).sum() / df['area'].sum()

# Group by sale_year and calculate the area-weighted t2_mean for each group
rural_avg = rural.groupby('sale_year').apply(weighted_mean).reset_index(name='rural_avg')
urban_avg = urban.groupby('sale_year').apply(weighted_mean).reset_index(name='urban_avg')

# rural_avg = rural.groupby('sale_year',as_index=False).agg({'t2_mean':'mean'})
# urban_avg = urban.groupby('sale_year',as_index=False).agg({'t2_mean':'mean'})


# Merge the two datasets on sale_year
avg_diff = pd.merge(urban_avg, rural_avg, on='sale_year')

# Compute the difference in area-weighted average temperature (urban minus rural)
avg_diff['temp_diff'] = avg_diff['urban_avg'] - avg_diff['rural_avg']
avg_diff['rolling'] = avg_diff['temp_diff'].rolling(window=3).mean()
avg_diff['rolling_ua'] = avg_diff['urban_avg'].rolling(window=3).mean()
avg_diff['rolling_ra'] = avg_diff['rural_avg'].rolling(window=3).mean()
# Plot the year-over-year temperature difference
plt.figure(figsize=(10, 6))
plt.plot(avg_diff['sale_year'], (avg_diff['rolling_ua']-273.15)*(9/5)+32, marker='o', linestyle='-', color='gray')
plt.plot(avg_diff['sale_year'], (avg_diff['rolling_ra']-273.15)*(9/5)+32, marker='o', linestyle='-', color='green')
# plt.plot(avg_diff['sale_year'], avg_diff['rural_avg'], marker='o', linestyle='-', color='green')
plt.xlabel('Year')
plt.ylabel('Urban - Exburb Area Weighted Avg Temp Difference (°F)')
plt.title('3-Year Rolling HUMID Summer Temperature: Urban/Suburb and Exurb Tracts')
plt.grid(True)
plt.show()


# In[773]:


avg_diff


# In[800]:


urban['SQFT']


# In[809]:


post['SQFT']


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



# In[794]:


post['CITY'].unique()


# In[ ]:




