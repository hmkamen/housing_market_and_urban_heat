#!/usr/bin/env python
# coding: utf-8

# In[222]:


### this processes  all amenity data used in urban heat iv paper


# In[223]:


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
import warnings
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from rasterio.mask import mask
from collections import Counter
from rasterstats import zonal_stats
from shapely.geometry import Point

# set path
path='/Users/hannahkamen/Downloads'

# Filter or ignore specific warning types
warnings.filterwarnings('ignore')


# ### Import sold parcels and county, and tract shapefile

# In[224]:


# import Maricopa County boundary 
maricopa_gdf = gpd.read_file("%s/maricopa/maricopa.shp"%path)
# import tract shapefile
tracts=gpd.read_file('%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp'%path)

# calc tract area in meters
tracts=tracts[['TRACTCE10','geometry']]
tracts=tracts.to_crs("epsg:26912")
tracts['tract_area']=tracts['geometry'].area
del tracts['geometry']
tracts['TRACTCE10']=tracts['TRACTCE10'].astype(int)
tracts=tracts[['TRACTCE10','tract_area']]

# sold parcels
sold_parcels=pd.read_csv('%s/urban_iv_paper_input_data_11_20/sold_homes_sample_11_20.csv'%path)
sold_parcels=sold_parcels[['APN','sale_year']]

# parcel shapefile
parcels_master = gpd.read_file('%s/Parcels 2/Parcels_All.shp' % path)
# convert parcels shapefile to meters
parcels_master=parcels_master.to_crs("EPSG:26912")

#parcel block and tract intersection
parcel_block_tract=pd.read_csv('/Users/hannahkamen/Downloads/parcel_block_intersection.csv')
parcel_block_tract=parcel_block_tract.drop_duplicates(['APN'])
### clean block var
parcel_block_tract['block_gr']=parcel_block_tract['BLOCKID10'].astype(str).apply(lambda x: x[:12])
### limit dataframe
parcel_block_tract=parcel_block_tract[['APN','TRACTCE10','block_gr','BLOCKID10']]


# ### Merge tract and block IDs with sold homes in sample

# In[225]:


####merge parcels and tract and block id to data
master_block_tract=sold_parcels.merge(parcel_block_tract,on='APN',how='inner')


# ### Make sold parcel df a geodataframe

# In[226]:


parcels_master['parcel_area']=parcels_master.geometry.area
parcels_master=parcels_master[['geometry','parcel_area','APN']]

master_parcel=sold_parcels.merge(parcels_master,on='APN',how='inner')
master_parcel_geo= gpd.GeoDataFrame(master_parcel, geometry="geometry", crs="EPSG:26912")


# ### Calculate distance to Phoenix city center

# In[227]:


# Define the fixed point (Phoenix City Center) as a Point geometry
phx_cc = Point(-112.073795, 33.445592)  # (longitude, latitude)

# Convert the Phoenix City Center point to a GeoDataFrame
phx_gdf = gpd.GeoDataFrame({'geometry': [phx_cc]}, crs="EPSG:26912")

# Extract reprojected Phoenix City Center coordinates
phx_cc_projected = phx_gdf.geometry.iloc[0]

# Compute distance from each parcel to the Phoenix City Center
master_parcel_geo['distance_to_phx_cc'] = master_parcel_geo.to_crs("EPSG:26912").geometry.distance(phx_cc_projected)


# ### Map zip, city and puma onto each parcel

# In[228]:


#import zips and cities file
zips=gpd.read_file('%s/Maricopa_County_Zip_Codes/ZipCodes.shp'%path)
zips=zips[['geometry','BdVal','USPSCityAl']]
zips=zips.rename(columns={'BdVal':'zipcode','USPSCityAl':'city'})
# convert CRS of zips
zips=zips.to_crs("EPSG:26912")

## spatial join with zips
master_with_zips=gpd.sjoin(master_parcel_geo,zips,how='inner',predicate='intersects')

## spatial join with  pumas
## import puma shapefile
pumas=gpd.read_file('%s/tl_2021_04_puma10/tl_2021_04_puma10.shp'%path)
pumas=pumas[['PUMACE10','geometry']].copy()
## convert pumas to meter based CRS
pumas=pumas.to_crs("EPSG:26912")

### delete index col from zip join
del master_with_zips['index_right']
## join parcels with pumas
master_with_puma=gpd.sjoin(master_with_zips,pumas,how='inner',predicate='intersects')


# ### Merge APN file with tract and block ids

# In[229]:


master_with_locs=master_with_puma.merge(master_block_tract[['APN','TRACTCE10','block_gr','BLOCKID10','sale_year']],on=['APN','sale_year'],how='inner')


# ### Assign subdivision

# In[230]:


# import subdivisions and match CRS to parcels
sub = gpd.read_file(f"{path}/Subdivisions/Subdivisions.shp")
sub = sub.to_crs("EPSG:26912")   # master_with_locs already EPSG:26912
sub['sub_area']=sub.to_crs("EPSG:26912").geometry.area
# Just keep the MCR id and geometry
sub = sub[["MCR", "geometry",'sub_area']]

# compute parcel–subdivision intersections
#    This creates a GeoDataFrame where each row is the overlap between one parcel and one subdivision
intersections = gpd.overlay(
    master_with_locs[["APN", "geometry"]],
    sub,
    how="intersection"
)

# compute intersection area
intersections["int_area"] = intersections.geometry.area

# for each parcel (APN), keep the subdivision with the largest overlap
idx_max = intersections.groupby("APN")["int_area"].idxmax()
best_matches = intersections.loc[idx_max, ["APN", "MCR"]].copy()


# In[231]:


# merge back onto the parcel GeoDataFrame
master_with_locs = master_with_locs.merge(best_matches, on="APN", how="left").merge(sub[['MCR','sub_area']],on='MCR')


# ## Begin importing amenity data

# ### Import subdivision greenspace variables

# In[232]:


### get subdivision greenspaces
gs_sub=pd.read_csv("%s/subdivisions_with_green_space.csv"%path)
gs_sub['sale_year_veg']=gs_sub['sale_year']
del gs_sub['sale_year']
gs_sub=gs_sub[['MCR','sub_greenspace','sale_year_veg']].groupby(['MCR','sale_year_veg'],as_index=False).agg({'sub_greenspace':'mean'})

### get distance from parcel to open space
parcel_dist_os=pd.DataFrame()
for year in range(2009,2019):

    tmp=pd.read_csv(f"{path}/parcels_with_distance_to_os_{year}.csv")
    parcel_dist_os=pd.concat([tmp,parcel_dist_os],ignore_index=True)
del parcel_dist_os['Unnamed: 0']        


# ### Distance to nearest features

# In[233]:


###get distance to parks, trailheads, and major roads
dist=pd.read_csv("%s/parcels_with_nearest_feature_distances.csv"%path)


# ### Distance to nearest vegetation

# In[234]:


veg_dist=pd.read_csv("%s/parcels_with_veg_distances.csv"%path)
veg_dist['sale_year_veg']=veg_dist['sale_year']
del veg_dist['sale_year']


# ### Tract land cover 

# In[235]:


##### get tract land cover from CAPLTER
tlc=pd.read_csv('%s/cap_lter_tract_land_cover_pct.csv'%path)
### get tract land cover from NLCD
tract_lc_nlcd=pd.read_csv("%s/tract_lc.csv"%path)
### agg some similar land types
tract_lc_nlcd['area_farm']=tract_lc_nlcd['area_81']+tract_lc_nlcd['area_82']
tract_lc_nlcd['area_high_dev']=tract_lc_nlcd['area_23']+tract_lc_nlcd['area_24']
tract_lc_nlcd['area_low_dev']=tract_lc_nlcd['area_21']+tract_lc_nlcd['area_22']
tract_lc_nlcd['area_soil']=tract_lc_nlcd['area_52']

## delete ectraneous reindexed var
del tract_lc_nlcd['Unnamed: 0']


# ### Import elevation data

# In[236]:


elevation_df=pd.read_csv('%s/elevation_by_parcel.csv'%path)


# ### Import air quality data

# In[237]:


air_q=pd.read_csv('%s/mean_air_q_by_tract.csv'%path)
### clean tract
air_q['TRACTCE10']=air_q['ctfips'].apply(lambda x: int(str(x)[4:]))
air_q=air_q.rename(columns={'year':'sale_year'})


# ### Merge all amenities

# In[238]:


## set vegetation sale year for CAPLTR NAIP based df merges
master_with_locs['sale_year_veg']=np.where(master_with_locs['sale_year']<2014,2010,2015)

master_amenities=master_with_locs.merge(gs_sub,on=['MCR','sale_year_veg']
                                       ).merge(parcel_dist_os,on=['APN','sale_year'],how='inner'
                                       ).merge( dist, on='APN',how='inner'
                                       ).merge(veg_dist,on=['APN','sale_year_veg'],how='inner'
                                       ).merge(tract_lc_nlcd,on=['TRACTCE10','sale_year'],how='inner'
                                        ).merge(elevation_df,on='APN',how='inner'
                                        ).merge(air_q,on=['TRACTCE10','sale_year'])
    


# ### Make some extra variables

# In[239]:


# --- Tag 95th percentile elevation homes (using given cutoff) ---
master_amenities["high_elev"] = (master_amenities["Mean_Elevation"] > 442.1311).astype(int)

# --- Mean elevation within tract and block ---
master_amenities["mean_tract_elevation"] = (
    master_amenities.groupby("TRACTCE10")["Mean_Elevation"].transform("mean")
)

master_amenities["mean_block_elevation"] = (
    master_amenities.groupby("block_gr")["Mean_Elevation"].transform("mean")
)

master_amenities["tract_high_elev"] = (
    master_amenities["mean_tract_elevation"] > 490.5706
).astype(int)


# --- High elevation within ZIP (90th percentile by zip_code) ---
master_amenities["p90_elev"] = (
    master_amenities
    .groupby("zipcode")["Mean_Elevation"]
    .transform(lambda x: x.quantile(0.90))
)

master_amenities["high_elev_zip"] = (
    master_amenities["Mean_Elevation"] >= master_amenities["p90_elev"]
).astype(int)


# --- Adjacency dummies ---

# adjacency to park in meters
master_amenities["park_adjacent"] = (master_amenities["dist_parks"] < 300).astype(int)

# adjacency to trailhead in meters
master_amenities["th_adjacent"] = (master_amenities["dist_th"] < 300).astype(int)

# adjacency to road in meters
master_amenities["road_adjacent"] = (master_amenities["dist_road"] < 300).astype(int)

# adjacency to vegetation in meters
master_amenities["veg_adjacent"] = (master_amenities["dist_to_veg"] < 100).astype(int)


# --- Percent subdivision with vegetation ---
master_amenities["pct_sub_greenspace"] = (
    master_amenities["sub_greenspace"] / master_amenities["sub_area"]
)

# --- Overall green landscape dummy ---
master_amenities["green_landscape"] = (
    master_amenities["pct_sub_greenspace"] > 0.33
).astype(int)

# --- Interaction between parcel size and greenspace ---
master_amenities["area_green_interact"] = (
    master_amenities["parcel_area"] * master_amenities["parcel_greenspace"]
)

# --- Mean distance / elevation variables: by tract × sale_year ---
group_tract_year = master_amenities.groupby(["TRACTCE10", "sale_year"])

master_amenities["mean_dist_road"] = group_tract_year["dist_road"].transform("mean")
master_amenities["mean_dist_th"] = group_tract_year["dist_th"].transform("mean")
master_amenities["mean_dist_parks"] = group_tract_year["dist_parks"].transform("mean")
master_amenities["mean_dist_cc"] = group_tract_year["distance_to_phx_cc"].transform("mean")
master_amenities["mean_dist_veg"] = group_tract_year["dist_to_veg"].transform("mean")

# --- Distance squared terms ---
master_amenities["distance_to_phx_cc_sq"] = master_amenities["distance_to_phx_cc"] ** 2
master_amenities["dist_road_sq"]          = master_amenities["dist_road"] ** 2
master_amenities["dist_th_sq"]            = master_amenities["dist_th"] ** 2
master_amenities["dist_parks_sq"]         = master_amenities["dist_parks"] ** 2
master_amenities["dist_veg_sq"]           = master_amenities["dist_to_veg"] ** 2

master_amenities["mean_dist_road_sq"]   = master_amenities["mean_dist_road"] ** 2
master_amenities["mean_dist_cc_sq"]     = master_amenities["mean_dist_cc"] ** 2
master_amenities["mean_dist_parks_sq"]  = master_amenities["mean_dist_parks"] ** 2
master_amenities["mean_dist_th_sq"]     = master_amenities["mean_dist_th"] ** 2
master_amenities["mean_dist_veg_sq"]    = master_amenities["mean_dist_veg"] ** 2


# ### Export amenities file

# In[240]:


master_amenities.to_csv("%s/urban_iv_paper_input_data_11_20/amenities_data_11_20.csv"%path)


# In[ ]:




