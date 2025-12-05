#!/usr/bin/env python
# coding: utf-8

# In[203]:


####this file cleans parcel sales information from the file system at:
##### https://www.dropbox.com/sh/0e8wltu2kb9s23y/AAAtlwnfP4bB3pY-Fj80YSE8a/Archived_Maricopa_Parcel_Files?e=1&dl=0
#####then joins parcel sales information with:
####hmda dataparcel lst by year, block imperviousness by year, and vegetative ring of parcel by year


# In[204]:


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
from collections import Counter
from rasterstats import zonal_stats
from shapely.geometry import Point
from shapely.geometry import box

# Filter or ignore specific warning types
warnings.filterwarnings('ignore')


# ### Helper functions

# In[205]:


# Function to compute the normalized weight between two directions
def weight_for_direction(ref, cand):
    # Get angles
    angle_ref = direction_angles[ref]
    angle_cand = direction_angles[cand]
    # Compute absolute angular difference, adjusting for wrap-around
    diff = abs(angle_ref - angle_cand)
    diff = min(diff, 360 - diff)
    # Compute raw weight: highest at diff=0 and 0 at diff=180
    # raw_weight = 1 - (diff / 180)
    if angle_ref == angle_cand:
        raw_weight = 1
    else:
        raw_weight = 0
    return raw_weight

# Define function to categorize bearings
def categorize_bearing(bearing):
    if  bearing == "N" :
        return 22.5
    elif bearing == "NE" :
        return 67.5
    elif bearing == "E" :
        return 112.5
    elif bearing == "SE" :
        return 157.5
    elif bearing == "S" :
        return 202.5
    elif bearing == "SW" :
        return 247.5
    elif bearing == "W" :
        return 292.5
    elif bearing == "NW" :
        return 337.5

# Gradient alignment score
def weighted_gradient(row,scale):
    gradient_score = 0
    for dir_label in direction_degrees.keys():
        anomaly_frac = row.get(f'anomaly_frac_{dir_label}_{scale}', 0)
        slice_deg = direction_degrees[dir_label]
        bearing_deg = row['bearing']
        angle_diff = min(abs(slice_deg - bearing_deg), 360 - abs(slice_deg - bearing_deg))
        alignment_weight = 1 - (angle_diff / 180)
        gradient_score += alignment_weight * anomaly_frac
    return gradient_score

# Define function to categorize bearings
def categorize_bearing_reverse(bearing):
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

def calculate_bearing_angle(bearing1, bearing2):
    # Absolute difference between the bearings
    diff = abs(bearing1 - bearing2)
    # Smallest angle accounting for wrap-around at 360 degrees
    angle = min(diff, 360 - diff)
    return angle


# ### Import tract-year estimates of minimum july temp

# In[206]:


tract_air_temp=pd.read_csv('%s/prism_tract_july_min_temp_by_year.csv'%path)
tract_air_temp_ncar=pd.read_csv('%s/ncar_tract_july_min_temp_by_year.csv'%path)
tract_air_temp_ncar=tract_air_temp_ncar.rename(columns={'t2_min':'t2_min_ncar'})
tract_air_temp_ncar=tract_air_temp_ncar[['TRACTCE10','sale_year','t2_min_ncar']]


# In[207]:


### Import sold parcel sample
master=pd.read_csv("%s/urban_iv_paper_input_data_11_20/sold_homes_sample_11_20.csv"%path)

### import parcel shapefile
parcels_master = gpd.read_file('%s/Parcels 2/Parcels_All.shp' % path)


# ### Set buffer designations

# In[208]:


# buffers=[500, 1500, 2500, 3500,4500,5500]
buffers=[500, 1500, 2500, 3500]


# ### Join Parcels with Urban Resolving UHI Data and Instrument data

# In[209]:


# ----------------------------------------------------------
# 0. Setup: base panel and CRS
# ----------------------------------------------------------

# Base frame: same length as master, but only keys we merge on
master_air_temp_dir_slices = master[["APN", "sale_year"]].copy()

# Get raster CRS from a template year (assumes all years share CRS)
template_raster_path = f"{path}/air_temp_july_min_1k_ncar_2009.tif"
with rasterio.open(template_raster_path) as src:
    raster_crs = src.crs

# Reproject parcels once into raster CRS (so we don't do it every iteration)
parcels_master = parcels_master.to_crs(raster_crs)

# ----------------------------------------------------------
# 1. Loop over buffer sizes
# ----------------------------------------------------------

for buff in buffers:
    
    print(f"\n=== Processing buffer {buff} m ===")

    # label for columns:
    label = buff

    # Will collect results across years, then concat once
    temp_list = []
    inst_list = []

    # ------------------------------------------------------
    # 1a. Loop over years
    # ------------------------------------------------------
    for year in range(2009, 2019):
        print(f"  Year {year}")

        # Sold parcels in this year
        sold_apns = master.loc[master["sale_year"] == year, "APN"].unique()
        parcel_gdf = parcels_master[parcels_master["APN"].isin(sold_apns)].copy()

        if parcel_gdf.empty:
            print("    No parcels sold this year; skipping.")
            continue

        # Build buffer / ring geometry around centroids
        centroids = parcel_gdf.geometry.centroid

        if buff == 500:
            parcel_gdf["geometry"] = centroids.buffer(buff)

        else:
            outer = centroids.buffer(buff)
            inner = centroids.buffer(buff - 1000)
            parcel_gdf["geometry"] = outer.difference(inner)

        # Raster path for this year
        raster_path = f"{path}/air_temp_july_min_1k_ncar_{year}.tif"

        # Zonal statistics for mean temp in this buffer/ring
        stats = zonal_stats(
            parcel_gdf["geometry"],
            raster_path,
            stats="mean",
            nodata=src.nodata,       
            geojson_out=False,
            all_touched=True
        )

        parcel_gdf["t2_mean"] = [
            s["mean"] if s["mean"] is not None else np.nan for s in stats
        ]
        parcel_gdf["sale_year"] = year

        # Keep only keys + temperature column
        temp_list.append(
            parcel_gdf[["APN", "sale_year", "t2_mean"]]
        )

        # ------------------------------
        # Instrument data for this year
        # ------------------------------
        inst_path = f"{path}/spatial_lag_instrument_inputs_{year}m_buffer_{buff}_full_sample_large_buffers.csv"
        tmp = pd.read_csv(inst_path)

        # Drop junk column if present
        if "Unnamed: 0" in tmp.columns:
            tmp = tmp.drop(columns=["Unnamed: 0"])

        # Make sure keys exist
        if "sale_year" not in tmp.columns:
            tmp["sale_year"] = year

        inst_list.append(tmp)

    # ------------------------------------------------------
    # 1b. After looping over years: stack & rename columns
    # ------------------------------------------------------

    # Stack temperature results from all years
    if temp_list:
        master_air_temp = pd.concat(temp_list, ignore_index=True)
    else:
        master_air_temp = pd.DataFrame(columns=["APN", "sale_year", "t2_mean"])

    # Rename t2_mean → t2_mean_<label> 
    master_air_temp = master_air_temp.rename(
        columns={"t2_mean": f"t2_mean_{label}"}
    )

    # Stack instrument results from all years
    if inst_list:
        buffer_master = pd.concat(inst_list, ignore_index=True)
    else:
        buffer_master = pd.DataFrame(columns=["APN", "sale_year"])

    # Rename only the directional mean columns (e.g. mean_lst_N → mean_lst_N_<label>)
    mean_cols = [c for c in buffer_master.columns if c.startswith("mean_lst_")]
    rename_map = {c: f"{c}_{label}" for c in mean_cols}
    buffer_master = buffer_master.rename(columns=rename_map)

    # ------------------------------------------------------
    # 1c. Merge this buffer's columns onto the master panel
    # ------------------------------------------------------
    master_air_temp_dir_slices = master_air_temp_dir_slices.merge(
        master_air_temp, on=["APN", "sale_year"], how="left"
    )
    master_air_temp_dir_slices = master_air_temp_dir_slices.merge(
        buffer_master, on=["APN", "sale_year"], how="left"
    )

print("Final shape:", master_air_temp_dir_slices.shape)


# In[210]:


master_air_temp_dir_slices.to_csv("%s/urban_iv_paper_input_data_11_20/master_air_temp_dir_slices_500m-5500m.csv"%path)


# ### Get weighted bearing of backward trajectories originating from parcel centroid

# In[211]:


### import trajecory shares
# # traj_shares=pd.read_csv('%s/parcels_with_weighted_backwards_trajectories_3_25.csv'%path)
# # traj_shares=pd.read_csv("%s/parcels_with_weighted_backwards_trajectories_3_25_low_variance_all_years_combined.csv"%path)
# # traj_shares=pd.read_csv("%s/parcels_with_weighted_backwards_trajectories_3_25_max_days_hours.csv"%path)
traj_shares=pd.read_csv("%s/parcels_with_weighted_backwards_trajectories_3_25_max_days_hours_not_midnight.csv"%path)
del traj_shares['Unnamed: 0']


# In[212]:


# Define the angles (in degrees) for each cardinal direction
direction_angles = {
    'N': 0,
    'NE': 45,
    'E': 90,
    'SE': 135,
    'S': 180,
    'SW': 225,
    'W': 270,
    'NW': 315
}

# Build dictionary of weights for each reference direction
direction_weights = {}
for ref in direction_angles.keys():
    raw_weights = {cand: weight_for_direction(ref, cand) for cand in direction_angles.keys()}
    # Normalize so the sum for each reference direction equals 1
    total = sum(raw_weights.values())
    normalized = {cand: raw_weights[cand] / total for cand in raw_weights}
    direction_weights[ref] = raw_weights


# ### Define buffers and column names / direction degree levels

# In[213]:


# Buffer slice column groups

directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

buffer_columns = {
    str(rad): [f"mean_lst_{d}_{rad}" for d in directions]
    for rad in buffers
}
## mean direction bearing 
direction_degrees = {
    'N': 22.5, 'NE': 67.5, 'E': 112.5, 'SE': 157.5,
    'S': 202.5, 'SW': 247.5, 'W': 292.5, 'NW': 337.5
}


# ### Loop through each buffer and calculate weighted gradient alignment score

# In[214]:


## create copy of laste df
master_instr=master_air_temp_dir_slices.copy()

for scale, columns in buffer_columns.items():

    print(scale)
    
    # Range and mean calculations
    master_air_temp_dir_slices[f'max_temp_{scale}'] = master_air_temp_dir_slices[columns].max(axis=1)
    master_air_temp_dir_slices[f'min_temp_{scale}'] = master_air_temp_dir_slices[columns].min(axis=1)
    master_air_temp_dir_slices[f'max_min_range_{scale}'] = master_air_temp_dir_slices[f'max_temp_{scale}'] - master_air_temp_dir_slices[f'min_temp_{scale}']
    master_air_temp_dir_slices[f'mean_buffer_lst_{scale}'] = master_air_temp_dir_slices[columns].mean(axis=1)

    # Anomaly calculations
    for col in columns:
        direction = col.strip(f"mean_lst_").strip(f"_{scale}")
        anomaly = master_air_temp_dir_slices[col] - master_air_temp_dir_slices[columns].mean(axis=1)
        master_air_temp_dir_slices[f'anomaly_{direction}_{scale}'] = anomaly
        master_air_temp_dir_slices[f'anomaly_frac_{direction}_{scale}'] = anomaly / master_air_temp_dir_slices[f'max_min_range_{scale}']

    # Merge trajectory data
    master_traj = master_air_temp_dir_slices.merge(traj_shares, on='APN')

    # Max/min direction detection
    max_col = master_traj[columns].idxmax(axis=1)
    min_col = master_traj[columns].idxmin(axis=1)
    master_traj[f'max_last_two_{scale}'] = max_col.str[9:].str.strip("_").str.strip(scale).str.strip("_")
    master_traj[f'min_last_two_{scale}'] = min_col.str[9:].str.strip("_").str.strip(scale).str.strip("_")

    master_traj['bearing'] = master_traj['bearing_cat'].apply(categorize_bearing)
    master_traj[f'max_bearing_{scale}'] = master_traj[f'max_last_two_{scale}'].apply(categorize_bearing)
    master_traj[f'min_bearing_{scale}'] = master_traj[f'min_last_two_{scale}'].apply(categorize_bearing)

    # Weighted angular difference
    master_traj[f'max_diff_{scale}'] = abs(master_traj[f'max_bearing_{scale}'] - master_traj['bearing'])
    master_traj[f'min_diff_{scale}'] = abs(master_traj[f'min_bearing_{scale}'] - master_traj['bearing'])

    master_traj[f'inv_max_angle_w_{scale}'] = master_traj[f'max_diff_{scale}'].apply(lambda row: min(row, 360 - row)) * master_traj['weight']
    master_traj[f'inv_min_angle_w_{scale}'] = master_traj[f'min_diff_{scale}'].apply(lambda row: min(row, 360 - row)) * master_traj['weight']


    master_traj[f'gradient_alignment_score_{scale}'] = master_traj.apply(lambda x : weighted_gradient(row=x,scale=scale), axis=1) * master_traj['weight']
    master_traj[f'weighted_bearing_{scale}'] = master_traj['bearing'] * master_traj['weight']

    # Group and aggregate
    weighted = master_traj.groupby(['APN', 'sale_year'], as_index=False).agg({
        f'gradient_alignment_score_{scale}': 'sum',
        f'inv_max_angle_w_{scale}': 'sum',
        f'inv_min_angle_w_{scale}': 'sum',
        f'weighted_bearing_{scale}': 'sum'
    })

    # merge to final df
    master_instr=master_instr.merge(weighted,on=['APN','sale_year'])


# ### Adjust temp variables to degrees celsius

# In[215]:


for buff in buffers:
    master_instr['t2_mean_%s_c'%buff]=master_instr['t2_mean_%s'%buff]-273.15


# ### Create windorized temp and instrument variables

# In[216]:


# base cols
base_cols = (
    [f"t2_mean_{b}_c" for b in buffers] +
    [f"gradient_alignment_score_{b}" for b in buffers]
)

# helper function: Winsorize a Series at central p% (two-sided)
# e.g. p=99 -> clamp at 0.5% and 99.5% quantiles
def winsorize_series(s: pd.Series, p: float) -> pd.Series:
    alpha = (1 - p / 100.0) / 2.0
    non_na = s.dropna()
    if non_na.empty:
        return s  # nothing to do
    lower = non_na.quantile(alpha)
    upper = non_na.quantile(1 - alpha)
    return s.clip(lower, upper)

# Create winsorized versions
for col in base_cols:
    if col not in master_instr.columns:
        # skip if any expected column is missing
        continue

    for p in [99, 95, 90]:
        if "t2" in col:
            new_col = f"{col}_w{p}"   # new name conventions e.g. t2_mean_500_w99

        if "gradient" in col:
            new_col = f"{col.replace("gradient_alignment_score","gas")}_w{p}"   # new name conventions e.g. t2_mean_500_w99
        master_instr[new_col] = winsorize_series(master_instr[col], p)


# ### Get NCAR tract temp

# In[217]:


tract_air_temp_ncar=pd.read_csv('%s/ncar_tract_july_min_temp_by_year.csv'%path)
tract_air_temp_ncar=tract_air_temp_ncar.rename(columns={'t2_min':'t2_min_ncar'})
del tract_air_temp_ncar['Unnamed: 0']
tract_air_temp_ncar=tract_air_temp_ncar.drop_duplicates(subset=['TRACTCE10','sale_year'])


# ### Get Prism tract temp

# In[218]:


prism_tract_temp=pd.read_csv('%s/prism_tract_july_min_temp_by_year.csv'%path)
del prism_tract_temp['Unnamed: 0']
prism_tract_temp=prism_tract_temp.drop_duplicates(subset=['TRACTCE10','sale_year'])


# In[219]:


prism_tract_temp


# In[220]:


### Merge tract temps onto main file
parcel_block_tract=pd.read_csv('/Users/hannahkamen/Downloads/parcel_block_intersection.csv')
parcel_block_tract=parcel_block_tract.drop_duplicates(subset=['APN'])
master_instr_tract_temp=master_instr.merge(parcel_block_tract[['APN','TRACTCE10']],on='APN',how="left").merge(tract_air_temp_ncar,on=['TRACTCE10','sale_year'],how="left"
                                                                ).merge(prism_tract_temp,on=['TRACTCE10','sale_year'],how="left")



# ### Export file

# In[221]:


# master_instr.to_csv("%s/urban_iv_paper_input_data_11_20/temp_and_instr_data_11_20.csv"%path)


# In[222]:


# master_instr_tract_temp.to_csv("%s/urban_iv_paper_input_data_11_20/temp_and_instr_data_incl_500_5500_11_22.csv"%path)
master_instr_tract_temp.to_csv("%s/urban_iv_paper_input_data_11_20/temp_and_instr_data_incl_500_5500_11_30.csv"%path)


# In[ ]:


len(master_instr_tract_temp)


# In[ ]:




