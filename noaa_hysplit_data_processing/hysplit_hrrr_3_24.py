#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import subprocess
from datetime import datetime, timedelta
path='/Users/hannahkamen/Downloads'


# In[9]:


import pandas as pd
import numpy as np
import scipy.stats
import warnings
import geopandas as gpd
from urllib.request import urlopen
import json
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Polygon, box
from scipy.ndimage import label, binary_closing
from scipy.interpolate import griddata
from skimage.morphology import binary_dilation, square
from skimage.measure import label, regionprops
from shapely.geometry import Polygon
import math
from shapely.ops import nearest_points
from tqdm import tqdm 
import rasterio
from shapely.geometry import shape
import os
import subprocess
from shapely.geometry import Point, LineString
from pyproj import CRS
from shapely.strtree import STRtree
import statsmodels.formula.api as smf
from math import atan2, degrees
from shapely.geometry import Point, Polygon, box
from shapely.affinity import rotate
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from shapely.geometry import mapping
from ftplib import FTP
import warnings
# Filter or ignore specific warning types
warnings.filterwarnings('ignore')
path='/Users/hannahkamen/Downloads'


# In[10]:


# Define function to categorize bearings
def categorize_bearing(bearing):
    if bearing <= 45 :
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


# In[11]:


# Helper function to calculate the bearing between two points
def calculate_bearing(point1, point2):
    lon1, lat1 = point1.x, point1.y
    lon2, lat2 = point2.x, point2.y
    dlon = lon2 - lon1
    x = np.sin(np.radians(dlon)) * np.cos(np.radians(lat2))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - (np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dlon)))
    initial_bearing = atan2(x, y)
    return (degrees(initial_bearing) + 360) % 360


# In[12]:


# Paths
hysplit_path = "/Users/hannahkamen/hysplit/exec/hyts_std"  # Path to HYSPLIT executable
working_dir = "/Users/hannahkamen/hysplit/working"  # HYSPLIT working directory
hysplit_data_dir = "/Users/hannahkamen/hysplit/working/hrrr_data"  # Directory containing meteorological data
output_shapefile = "/Users/hannahkamen/Downloads/trajectories_hrrr_1000m_grid_3_24.shp"  # Final shapefile output


# In[13]:


def parse_tdump_to_gdf(tdump_files):
    data = []  # List to hold all parsed data
    
    for tdump_file in tdump_files:
        with open(tdump_file, "r") as f:
            lines = f.readlines()

            print(tdump_file)

        # Print the first line of trajectory endpoints for debugging
        # print(lines[len(grid_master)+3:len(grid_master)+4])
        
        # Parse trajectory points (bulk processing with list comprehension)
        trajectory_points = [
            (
                float(parts[0]),  # grid_id
                float(parts[8]),  # hour
                float(parts[9]),  # lat
                float(parts[10]),  # lon
                float(parts[11])  # alt
                # ,float(parts[13])  # temp
            )
            for line in lines[len(grid_master)+4:]
            for parts in [line.split()]  # Split line into parts once
        ]
        
        # Extend the data list
        data.extend(trajectory_points)
    
    # Convert data to a DataFrame
    df = pd.DataFrame(data, columns=["grid_id","hour", "lats", "lons", "alt"])
    
    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lons"], df["lats"]),
        crs="EPSG:4326"
    )
    
    return gdf


# In[14]:


# Specify your folder path
folder_path = "/Users/hannahkamen/hysplit/working/hrrr"

# List only files (ignoring subdirectories)
filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


# In[35]:


# ### all june july and august dates

# dates=["15 06 30 07",
#        "15 06 30 13",
#        "15 07 01 01",
#        "15 07 01 19",
#        "15 07 15 07",
#        "15 07 15 13",
#        "15 07 15 01",
#        "15 07 15 19",
#        "15 07 30 07",
#        "15 07 30 13",
#        "15 07 30 01",
#        "15 07 30 19",

#        "16 06 30 07",
#        "16 06 30 13",
#        "16 07 01 01",
#        "16 07 01 19",
#        "16 07 15 07",
#        "16 07 15 13",
#        "16 07 15 01",
#        "16 07 15 19",
#        "16 07 30 07",
#        "16 07 30 13",
#        "16 07 30 01",
#        "16 07 30 19",

#        "17 06 30 07",
#        "17 06 30 13",
#        "17 07 01 01",
#        "17 07 01 19",
#        "17 07 15 07",
#        "17 07 15 13",
#        "17 07 15 01",
#        "17 07 15 19",
#        "17 07 30 07",
#        "17 07 30 13",
#        "17 07 30 01",
#        "17 07 30 19",

#        "18 07 01 07",
#        "18 07 01 13",
#        "18 07 01 01",
#        "18 07 01 19",
#        "18 07 15 07",
#        "18 07 15 13",
#        "18 07 15 01",
#        "18 07 15 19",
#        "18 07 30 07",
#        "18 07 30 13",
#        "18 07 30 01",
#        "18 07 30 19"]


# files=['hysplit.20150630.06z.hrrra',       
#  'hysplit.20150630.12z.hrrra',      
#  'hysplit.20150701.00z.hrrra',       
#  'hysplit.20150701.18z.hrrra',     
#  'hysplit.20150715.06z.hrrra',       
#  'hysplit.20150715.12z.hrrra',      
#  'hysplit.20150715.00z.hrrra',       
#  'hysplit.20150715.18z.hrrra',
#  'hysplit.20150730.06z.hrrra',       
#  'hysplit.20150730.12z.hrrra',       
#  'hysplit.20150730.00z.hrrra',      
#  'hysplit.20150730.18z.hrrra',


# 'hysplit.20160630.06z.hrrra',       
#  'hysplit.20160630.12z.hrrra',      
#  'hysplit.20160701.00z.hrrra',       
#  'hysplit.20160701.18z.hrrra',     
#  'hysplit.20160715.06z.hrrra',       
#  'hysplit.20160715.12z.hrrra',      
#  'hysplit.20160715.00z.hrrra',       
#  'hysplit.20160715.18z.hrrra',
#  'hysplit.20160730.06z.hrrra',       
#  'hysplit.20160730.12z.hrrra',       
#  'hysplit.20160730.00z.hrrra',      
#  'hysplit.20160730.18z.hrrra',  


#  'hysplit.20170630.06z.hrrra',       
#  'hysplit.20170630.12z.hrrra',      
#  'hysplit.20170701.00z.hrrra',       
#  'hysplit.20170701.18z.hrrra',     
#  'hysplit.20170715.06z.hrrra',       
#  'hysplit.20170715.12z.hrrra',      
#  'hysplit.20170715.00z.hrrra',       
#  'hysplit.20170715.18z.hrrra',
#  'hysplit.20170730.06z.hrrra',       
#  'hysplit.20170730.12z.hrrra',       
#  'hysplit.20170730.00z.hrrra',      
#  'hysplit.20170730.18z.hrrra', 


#  'hysplit.20180701.06z.hrrra',       
#  'hysplit.20180701.12z.hrrra',      
#  'hysplit.20180701.00z.hrrra',       
#  'hysplit.20180701.18z.hrrra',     
#  'hysplit.20180715.06z.hrrra',       
#  'hysplit.20180715.12z.hrrra',      
#  'hysplit.20180715.00z.hrrra',       
#  'hysplit.20180715.18z.hrrra',
#  'hysplit.20180730.06z.hrrra',       
#  'hysplit.20180730.12z.hrrra',       
#  'hysplit.20180730.00z.hrrra',      
#  'hysplit.20180730.18z.hrrra'        
# ]


# In[54]:


### all june july and august dates

dates=["15 07 07 07",
       "15 07 07 13",
       "15 07 07 01",
       "15 07 07 19",
       "15 07 22 07",
       "15 07 22 13",
       "15 07 22 01",
       "15 07 22 19",

       "16 07 07 07",
       "16 07 07 13",
       "16 07 07 01",
       "16 07 07 19",
       "16 07 22 07",
       "16 07 22 13",
       "16 07 22 01",
       "16 07 22 19",

       "17 07 07 07",
       "17 07 07 13",
       "17 07 07 01",
       "17 07 07 19",
       "17 07 22 07",
       "17 07 22 13",
       "17 07 22 01",
       "17 07 22 19",

       "18 07 07 07",
       "18 07 07 13",
       "18 07 07 01",
       "18 07 07 19",
       "18 07 22 07",
       "18 07 22 13",
       "18 07 22 01",
       "18 07 22 19"]


files=['hysplit.20150707.06z.hrrra',       
 'hysplit.20150707.12z.hrrra',      
 'hysplit.20150707.00z.hrrra',       
 'hysplit.20150707.18z.hrrra',     
 'hysplit.20150722.06z.hrrra',       
 'hysplit.20150722.12z.hrrra',      
 'hysplit.20150722.00z.hrrra',       
 'hysplit.20150722.18z.hrrra',


'hysplit.20160707.06z.hrrra',       
 'hysplit.20160707.12z.hrrra',      
 'hysplit.20160707.00z.hrrra',       
 'hysplit.20160707.18z.hrrra',     
 'hysplit.20160722.06z.hrrra',       
 'hysplit.20160722.12z.hrrra',      
 'hysplit.20160722.00z.hrrra',       
 'hysplit.20160722.18z.hrrra',

'hysplit.20170707.06z.hrrra',       
 'hysplit.20170707.12z.hrrra',      
 'hysplit.20170707.00z.hrrra',       
 'hysplit.20170707.18z.hrrra',     
 'hysplit.20170722.06z.hrrra',       
 'hysplit.20170722.12z.hrrra',      
 'hysplit.20170722.00z.hrrra',       
 'hysplit.20170722.18z.hrrra',


'hysplit.20180707.06z.hrrra',       
 'hysplit.20180707.12z.hrrra',      
 'hysplit.20180707.00z.hrrra',       
 'hysplit.20180707.18z.hrrra',     
 'hysplit.20180722.06z.hrrra',       
 'hysplit.20180722.12z.hrrra',      
 'hysplit.20180722.00z.hrrra',       
 'hysplit.20180722.18z.hrrra',     
]


# In[55]:


#### load parcel shapefile and parcelcsv to limit to sold parcels
##parcel shapefile
parcels0 = gpd.read_file(f'{path}/parcels_by_year/Parcels_-_Maricopa_County%2C_Arizona_(2019).shp')

### set crs to meters crs
parcels0=parcels0.to_crs("EPSG:26912")
sold_parcels=pd.read_csv('%s/parcels_that_sold_2_18.csv'%path)
# sold_parcels=pd.read_csv('%s/parcel_list_no_fringe.csv'%path)


# In[56]:


len(sold_parcels['APN'].unique())


# In[57]:


grid_master=parcels0[parcels0['APN'].isin(sold_parcels['APN'].unique())]


# In[58]:


grid_master=grid_master[['APN','geometry']]
grid_master['geometry']=grid_master['geometry'].centroid.buffer(500)
grid_master['grid_id']=np.arange(1,len(grid_master)+1,1)


# In[59]:


grid_master.head()


# In[61]:


hysplit_path = "/Users/hannahkamen/hysplit/exec/hyts_std"  # Path to HYSPLIT executable
working_dir = "/Users/hannahkamen/hysplit/working"  # HYSPLIT working directory
output_shapefile = "/Users/hannahkamen/Downloads/trajectories_hrrr_1000m_grid_2_23.shp"  # Final shapefile output

grid_master['centroid']=grid_master['geometry'].centroid
grid_master = grid_master.to_crs("EPSG:4326")
grid_master['centroid']=grid_master['centroid'].to_crs("EPSG:4326")

# Make Control File
# Extract latitude and longitude from centroids
grid_master["lat"] = grid_master["centroid"].y
grid_master["lon"] = grid_master["centroid"].x

# Define the height above ground level (e.g., 10 meters)
grid_master["height"] = 10  # Adjust as needed

# Prepare CONTROL file content
starting_points = grid_master[["lat", "lon", "height"]]
num_points = len(starting_points)

### create empty geodataframe
traj_master = gpd.GeoDataFrame(columns=[], geometry=[], crs="EPSG:4326")


for file,date in zip(files,dates):
    if "extract" in file:
        file=file+".bin"
    # Write to CONTROL file
    with open("/Users/hannahkamen/hysplit/working/CONTROL", "w") as f:
        f.write(f"{date}\n")
        f.write(f"{num_points}\n")  # Number of starting points
        for _, row in starting_points.iterrows():
            f.write(f"{row['lat']:.6f} {row['lon']:.6f} {row['height']}\n")
        f.write(f"{-1}\n")
        f.write(f"{0}\n")
        f.write(f"{100}\n")
        f.write(f"{1}\n")
        f.write(f"{"/Users/hannahkamen/hysplit/working/hrrr/"}\n")
        f.write(f"{"%s"}\n" %file)
        f.write(f"{"./"}\n")
        f.write(f"{"tdump_backwards_%s"}\n"%file)
    
    print("CONTROL file created successfully for %s." %date)
    ###now run hysplit
    result = subprocess.run([hysplit_path], cwd=working_dir, capture_output=True, text=True)

    ###parse trajectories endpoint file
    gdf = parse_tdump_to_gdf(["/Users/hannahkamen/hysplit/working/tdump_backwards_%s"%file])
    print(file)

    # Assign trajectory order and process intersections    
    gdf = gdf[["grid_id", "geometry","hour"]]
    gdf["year"]="20"+date[:2]
    gdf["month"]=date[3:5]
    gdf["day"]=date[6:8]
    gdf["hour_start"]=date[9:12]
    gdf["geometry_previous"] = gdf.groupby("grid_id")["geometry"].shift(1)
    gdf["hour_previous"] = gdf.groupby("grid_id")["hour"].shift(1)
    gdf_lm=gdf[~((gdf['geometry_previous'].isnull()) | (gdf['geometry_previous']==None)) ].copy()
    gdf_lm["bearing"]=gdf_lm.apply(lambda row: calculate_bearing(row['geometry_previous'],row['geometry']),axis=1)
    traj_master=pd.concat([traj_master,gdf_lm],ignore_index=True)


# In[62]:


traj_master.head()


# In[63]:


import geopandas as gpd
from shapely.geometry import LineString

# Assuming 'traj_master' is your GeoDataFrame
traj_master["geometry"] = traj_master.apply(
    lambda row: LineString([row["geometry_previous"], row["geometry"]]) if row["geometry_previous"] and row["geometry"] else None,
    axis=1
)

# Drop the old geometry fields if no longer needed
traj_master = traj_master.drop(columns=["geometry_previous"])

# Ensure it remains a GeoDataFrame
traj_master = gpd.GeoDataFrame(traj_master, geometry="geometry", crs=traj_master.crs)


# In[64]:


traj_master=pd.concat([traj_master,traj_master0],ignore_index=True)


# In[221]:


del traj_master0


# In[77]:


traj_master['month']=np.where(traj_master['month']=="06",traj_master['day']=="01",traj_master['day'])
traj_master['month']="07"


# In[235]:


traj_master.to_csv("%s/all_july_trajectories_3_26.csv"%path)


# #### remove year, hour, day anomalies

# In[217]:


# ###get median bearing for that gird_id, hour, and day combo in july
# mean_over_years=traj_master.groupby(['grid_id','day','hour_start'],as_index=False).agg({'bearing':'median'})
# ##rename
# mean_over_years=mean_over_years.rename(columns=({'bearing':'median_bearing'}))
# ###merge onto traj_master
# traj_master_merge=traj_master.merge(mean_over_years,on=['grid_id','day','hour_start'],how="inner")
# ###calculate that year's difference from median
# traj_master_merge['diff_from_median']=abs(traj_master_merge['bearing']-traj_master_merge['median_bearing']).apply(lambda row: min(row,360-row))

# #### remove day hour year combo if 90th percentile of diff from median across all grid ids  is higher than 45
# anomalies = traj_master_merge.groupby(['year','month','day','hour_start'], as_index=False).agg({
#     'diff_from_median': lambda x: x.quantile(0.9)
# }).rename(columns={"diff_from_median":"diff_from_median_90pct"})

# ###limit to combinations where the 90th percentile of deviation is low
# anomalies_lm=anomalies.loc[anomalies['diff_from_median_90pct']<90].copy()

# ###now inner join with traj master to remove high varying combos
# traj_master_merge=traj_master_merge.merge(anomalies_lm,on=['year','month','day','hour_start'],how='inner')

# # ###now remove remaining specific grids where diff_from median is ever above 22.5, ensire that all grids included are caluclated over same mix of hours
# # grid_gr=traj_master_merge.groupby(['grid_id'],as_index=False).agg({'diff_from_median': lambda x: x.quantile(0.9)})
# # grid_gr_lm=grid_gr.loc[grid_gr['diff_from_median']<45].copy()
# # del grid_gr_lm['diff_from_median']

# # ###now inner join with traj master to remove high varying grid_ids
# # traj_master_merge=traj_master_merge.merge(grid_gr_lm,on=['grid_id'],how='inner')


# ### Caluculate share of trajectories that come from each direction, merge on to APN

# In[220]:


# ###categorize bearing
# traj_master_merge['sale_year']=traj_master_merge['year']
# traj_master_merge['frequency']=1
# traj_master_merge['tot_frequency']=1
# traj_master_merge['bearing_cat']=traj_master_merge['bearing'].apply(categorize_bearing)
# # Step 1: Calculate total frequency for each grid_id
# master_freq_ct = traj_master_merge.groupby(["grid_id",'bearing_cat'],as_index=False).agg({'frequency':sum})
# master_hour_ct = traj_master_merge.groupby(["grid_id"],as_index=False).agg({'tot_frequency':sum})

# master_freq_merge=master_freq_ct.merge(master_hour_ct,on=['grid_id'])
# master_freq_merge['weight']=master_freq_merge['frequency']/master_freq_merge['tot_frequency']

# ###now merge back onto APN info

# bearings=master_freq_merge.merge(grid_master[['grid_id','APN']],on='grid_id')[['APN','bearing_cat','weight','tot_frequency']]
# bearings.to_csv('%s/parcels_with_weighted_backwards_trajectories_3_25_low_variance_all_years_combined.csv'%path)


# In[224]:


traj_master['hour_start'].unique()


# In[231]:


len(traj_master_lm)


# In[234]:


###categorize bearing
traj_master_lm=traj_master.loc[~(traj_master['hour_start'].isin(["01"]))].copy()
traj_master_lm['sale_year']=traj_master_lm['year']
traj_master_lm['frequency']=1
traj_master_lm['tot_frequency']=1
traj_master_lm['bearing_cat']=traj_master_lm['bearing'].apply(categorize_bearing)
# Step 1: Calculate total frequency for each grid_id
master_freq_ct = traj_master_lm.groupby(["grid_id",'bearing_cat'],as_index=False).agg({'frequency':sum})
master_hour_ct = traj_master_lm.groupby(["grid_id"],as_index=False).agg({'tot_frequency':sum})

master_freq_merge=master_freq_ct.merge(master_hour_ct,on=['grid_id'])
master_freq_merge['weight']=master_freq_merge['frequency']/master_freq_merge['tot_frequency']

###now merge back onto APN info

bearings=master_freq_merge.merge(grid_master[['grid_id','APN']],on='grid_id')[['APN','bearing_cat','weight','tot_frequency']]
bearings.to_csv('%s/parcels_with_weighted_backwards_trajectories_3_25_max_days_hours_not_midnight.csv'%path)


# In[ ]:


#### remove anomalies


# In[97]:


# traj_master_lm[traj_master_lm['year']=="2015"]


# In[116]:


anomaly_test2=traj_master_lm2.groupby(['grid_id'],as_index=False).agg({'bearing':'std'})


# In[119]:


anomaly_test2['bearing'].quantile(.95)


# In[103]:


anomaly_test['bearing'].describe()


# In[105]:


anomaly_test['bearing'].describe()


# In[101]:


anomaly_test['bearing'].describe()


# In[107]:


anomaly_test['bearing'].describe()


# In[92]:


len(anomaly_test)


# In[91]:


anomaly_test[anomaly_test['bearing']<45]


# In[83]:


anomaly_test[anomaly_test['bearing']>90]['day'].value_counts()


# In[81]:


len(anomaly_test)


# In[ ]:




