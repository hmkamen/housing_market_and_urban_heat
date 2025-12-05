#!/usr/bin/env python
# coding: utf-8

# In[ ]:


####this file cleans parcel sales information from the file system at:
##### https://www.dropbox.com/sh/0e8wltu2kb9s23y/AAAtlwnfP4bB3pY-Fj80YSE8a/Archived_Maricopa_Parcel_Files?e=1&dl=0
#####then joins parcel sales information with:
####hmda dataparcel lstby year, block imperviousness by year, and vegetative ring of parcel by year


# In[8]:


# land_use_categories = {
#     11: "water",
#     21: "opendev",
#     22: "lowdev",
#     23: "meddev",
#     24: "highdev",
#     31: "barren",
#     52: "shrub",
#     71: "shrub",
#     81: "pasture",
#     82: "pasture",
#     90: "water",
#     95: "water"
# }


# In[9]:


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
from shapely.geometry import Point
from shapely.geometry import mapping

import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from rasterio.mask import mask


from rasterio.features import shapes
import shapely.geometry


from collections import Counter

# Load Maricopa County boundary (replace with correct path if needed)
maricopa_gdf = gpd.read_file("%s/maricopa/maricopa.shp"%path)
#get zips
zips=gpd.read_file('%s/Maricopa_County_Zip_Codes/ZipCodes.shp'%path)

# Filter or ignore specific warning types
warnings.filterwarnings('ignore')


# In[10]:


# Function to process MultiPolygon and Polygon Z
def convert_to_2d_polygon(geom):

    try:
        if isinstance(geom, MultiPolygon):  # Convert MultiPolygon to Polygon
            geom = max(geom.geoms, key=lambda p: p.area)  # Keep largest polygon
        
        
        if geom.has_z:  # Convert Polygon Z to 2D
            geom = Polygon([(x, y) for x, y, _ in geom.exterior.coords])

    except:
        geom =np.nan  

    
    return geom


# ### Begin Processing for DiD regression of heat on development

# In[11]:


# #### read in residential completion data
# rescom0 = pd.read_csv('%s/Rescomps_Maricopa_2011-2019(Sheet1).csv' % path, encoding="ISO-8859-1")


# ####drop completions of 1 unit for apartments and townhomes
# rescom=rescom0[~((rescom0['UTYPE']=='TH') & (rescom0['UNITS']<2)) & ~((rescom0['UTYPE']=='AP') & (rescom0['UNITS']<3)) ].copy()

# rescom_gpd=gpd.GeoDataFrame(rescom,geometry=gpd.points_from_xy(rescom['Long'], rescom['Lat']),crs="EPSG:4326")
# rescom_gpd.to_file('%s/rescoms.shp'%path)


# In[31]:


#### Now get the test set, sfh that existed in 2009 and before
master=pd.read_csv('%s/all_sf_homes.csv'%path)
master['sf']=np.where(((master['puc']>=100) & (master['puc']<=190)) ,1,0)
master['apartment']=np.where((((master['puc']>=350) & (master['puc']<379))|((master['puc']>=390) & (master['puc']<=398)) |((master['puc']>=320) & (master['puc']<=348))) ,1,0)
master['townhome_condo']=np.where(((master['puc']>=8510) & (master['puc']<8590)) | ((master['puc']>=710) & (master['puc']<=796)),1,0)

#####clean housing attribute variables 
master=master[~(master['roof'].isin(['99','10','12','','11','`']))]
master=master[(master['roof']!='') & ~(master['roof'].isnull())]
master['roof']=master['roof'].astype(int)
####clean to get patio and garage type
master['garage_type']=master['garage'].astype(str).apply(lambda x: x[0]).astype(int)
master['patio_type']=master['patio'].astype(str).apply(lambda x: x[0]).astype(int)

####limit data and clear results from erroneous data entries
master['sale_year']=master['sale_year'].astype(int)
master['sale_month']=master['sale_month'].astype(int)
master['age']=master['age'].astype(int)
master['sq_ft']=master['sq_ft'].astype(int)

master['pool']=np.where(master['pool'].isnull(),0,master['pool'])
master['pool']=np.where(master['pool'].str.strip()=='',0,master['pool'])
# master['pool']=master['pool'].astype(str).astype(int)


#####clean house characteristic variables
master['heating_dum']=np.where(master['heating'].str.strip()=='Y',1,0)
master['stories_dum']=np.where(master['stories'].str.strip()=='M',1,0)


####homes that existed prior and during 2009
pre=master[(master['age']<=2009) & (master['age']>1850) ].groupby('APN',as_index=False).agg({'age':'mean','sq_ft':'mean','bathroom':'mean','patio':'mean','sf':'mean','townhome_condo':'mean','apartment':'mean'})

####homes built after 2009
# post=master[(master['age']>2009) & (master['age']<2019) ].groupby('APN',as_index=False).agg({'age':'mean','sq_ft':'mean','sf':'mean','townhome_condo':'mean','apartment':'mean','bathroom':'mean','patio':'mean'})

# post=pd.read_csv('%s/Rescomps_Maricopa_2011-2019(Sheet1).csv'%path,encoding="ISO-8859-1")


# In[32]:


#### import residential completions data

post1=pd.read_csv('%s/Rescomps_Maricopa_2011-2019(Sheet1).csv'%path,encoding="ISO-8859-1")
post2=pd.read_excel('%s/Rescomps_Maricopa_2000-2009 (1).xlsx'%path)
post=pd.concat([post1,post2],ignore_index=True)
post=post[post['COMYEAR']>2009].reset_index(drop=True)


# In[33]:





# In[34]:





# In[35]:


post=post.rename(columns={'PARCEL':'APN'})


# ### OK now get geometry for each home built during or after 2009

# In[37]:


rescom=post.copy()
rescom_geo_post=gpd.GeoDataFrame(geometry=[],crs="EPSG:26912")
####import parcels
parcels=gpd.read_file(f"{path}/parcels_by_year/Parcels_-_Maricopa_County%2C_Arizona_({2019}).shp")
parcels=parcels.to_crs(epsg=26912)
# rescom_tmp=rescom[rescom['sale_year']==year]
rescom_tmp_post=rescom.merge(parcels[['APN','geometry','AREA']],on=['APN'],how='inner')
rescom_geo_post=pd.concat([rescom_geo_post,rescom_tmp_post],ignore_index=True)


# ### OK now get geometry for each home built before 2009

# In[38]:


rescom=pre.copy()

rescom_geo_pre=gpd.GeoDataFrame(geometry=[],crs="EPSG:26912")
####import parcels
parcels=gpd.read_file(f"{path}/parcels_by_year/Parcels_-_Maricopa_County%2C_Arizona_({2019}).shp")
parcels=parcels.to_crs(epsg=26912)
# rescom_tmp=rescom[rescom['sale_year']==year]
rescom_tmp_pre=rescom.merge(parcels[['APN','geometry','AREA']],on=['APN'],how='inner')
rescom_geo_pre=pd.concat([rescom_geo_pre,rescom_tmp_pre],ignore_index=True)


# ### For each parcel developed between 2011 and 2018, get what its original land use was in 2009 and new land use in 2018

# In[40]:


# Initialize an empty GeoDataFrame
parcels_with_lc_change = gpd.GeoDataFrame(geometry=[], crs="EPSG:26912")

# Define raster file paths
land_use_raster0 = f"{path}/clipped_land_cover_maricopa_{2009}_full_county.tif"
land_use_raster1 = f"{path}/clipped_land_cover_maricopa_{2018}_full_county.tif"

rescom_tmp=rescom_geo_post.copy()

# Open the land use change raster
with rasterio.open(land_use_raster0) as src0, rasterio.open(land_use_raster1) as src1:
    raster_crs = src0.crs  # Get the CRS of the raster

    rescom_tmp = rescom_tmp.to_crs(raster_crs)

    # Define land use categories
    unique_land_use_classes = [11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95]
    
    for land_use in unique_land_use_classes:
        rescom_tmp[f"pct_{land_use}_0"] = 0.0
        rescom_tmp[f"pct_{land_use}_1"] = 0.0

    # Extract land use values for each parcel
    for idx, parcel in rescom_tmp.iterrows():
        
        try:

            # # Convert to a dictionary format for Rasterio
            parcel_geom_2d = [mapping(parcel.geometry)]

            # Mask the raster for the parcel
            masked_image0, _ = mask(src0, parcel_geom_2d, crop=True, indexes=1, filled=True, all_touched=True)
            masked_image1, _ = mask(src1, parcel_geom_2d, crop=True, indexes=1, filled=True, all_touched=True)

            # Convert to 1D arrays and remove NoData values
            land_use_values0 = masked_image0.flatten()
            land_use_values1 = masked_image1.flatten()

            # Remove NoData values (-9999, 255, or other values in raster profile)
            no_data_value = src1.nodata
            
            land_use_values0 = land_use_values0[land_use_values0 != no_data_value]
            land_use_values1 = land_use_values1[land_use_values1 != no_data_value]

            # Compute land cover change only if valid values exist
            if land_use_values0.size > 0 and land_use_values1.size > 0:
                
                counts0 = Counter(land_use_values0)
                counts1 = Counter(land_use_values1)

                total_pixels0 = sum(counts0.values())
                total_pixels1 = sum(counts1.values())

                # Assign percentages
                for land_use0, count0 in counts0.items():
                    column_name = f"pct_{land_use0}_0"
                    if column_name in rescom_tmp.columns:
                        rescom_tmp.at[idx, column_name] = count0 / total_pixels0 * 100
                        # / total_pixels0 * 100

                for land_use1, count1 in counts1.items():
                    column_name = f"pct_{land_use1}_1"
                    if column_name in rescom_tmp.columns:
                        rescom_tmp.at[idx, column_name] = count1 / total_pixels1 * 100
                        


        except Exception as e:
            print(f"Error processing parcel {idx}: {e}")
            continue

# Convert back to EPSG:26912
rescom_tmp = rescom_tmp.to_crs("EPSG:26912")

# Append results
parcels_with_lc_change = pd.concat([parcels_with_lc_change, rescom_tmp], ignore_index=True)

# Save the final dataset
parcels_with_lc_change.to_file(f"{path}/rescom_with_land_cover_change_2009_2018.shp")


# ### For each pixel in developed parcel file, label change as "infill" (dev to dev) or "sprawl" (farm/nat/water to dev)

# In[43]:


# List of unique land use classes
unique_land_use_classes = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]

# Create column names for 2000 and 2018
land_use_cols_2000 = [f"pct_{lu}_0" for lu in unique_land_use_classes]
land_use_cols_2018 = [f"pct_{lu}_1" for lu in unique_land_use_classes]

# Define Natural and Developed land cover categories
nat = { 81, 82}  # Natural classes
dev = { 21, 22, 23, 24}  # Developed classes

# Create column names for 2000 and 2018
dev_cols_2009 = [f"pct_{lu}_0" for lu in dev]  # Land cover 2000
dev_cols_2018 = [f"pct_{lu}_1" for lu in dev]  # Land cover 2018

# Create column names for 2000 and 2018
nat_cols_2009 = [f"pct_{lu}_0" for lu in nat]  # Land cover 2000
nat_cols_2018 = [f"pct_{lu}_1" for lu in nat]  # Land cover 2018

###sum across natural and developed uses

parcels_with_lc_change['nat_uses_2009'] = parcels_with_lc_change[nat_cols_2009].sum(axis=1)
parcels_with_lc_change['dev_uses_2009'] = parcels_with_lc_change[dev_cols_2009].sum(axis=1)

parcels_with_lc_change['nat_uses_2018'] = parcels_with_lc_change[nat_cols_2018].sum(axis=1)
parcels_with_lc_change['dev_uses_2018'] = parcels_with_lc_change[dev_cols_2018].sum(axis=1)

# Find majority land use in 2000
parcels_with_lc_change["majority_land_use_2009"] = parcels_with_lc_change[['nat_uses_2009','dev_uses_2009']].idxmax(axis=1).str.extract(r"(^.{3})")

# Find majority land use in 2018
parcels_with_lc_change["majority_land_use_2018"] = parcels_with_lc_change[['nat_uses_2018','dev_uses_2018']].idxmax(axis=1).str.extract(r"(^.{3})")
parcels_with_lc_change["land_switch"]=parcels_with_lc_change["majority_land_use_2009"].astype(str)+"_"+parcels_with_lc_change["majority_land_use_2018"].astype(str)


# In[44]:


parcels_with_lc_change[parcels_with_lc_change['land_switch']=='nat_dev'].to_file('%s/nat_dev.shp'%path)


# In[45]:


parcels_with_lc_change[parcels_with_lc_change['land_switch']=='dev_dev'].to_file('%s/dev_dev.shp'%path)


# In[46]:


parcels_with_lc_change[parcels_with_lc_change['land_switch']=='nat_nat'].to_file('%s/nat_nat.shp'%path)


# In[47]:


parcels_with_lc_change["land_switch"].value_counts()


# In[48]:


# #### now draw 1km buffer around each pre-existing home, and merge characteristics of residential development that took place in that 1km buffer
parcels_with_lc_change['area_dev_parcel']=parcels_with_lc_change['geometry'].area
# # Create a 1km buffer around the centroid of each geometry in rescom_geo_pre
rescom_geo_pre["buffer"] = rescom_geo_pre.geometry.centroid.buffer(1000)

# # Spatial join: Find parcels in parcels_with_lc_change that intersect with any buffer
joined = gpd.sjoin(parcels_with_lc_change, rescom_geo_pre.set_geometry("buffer"), predicate="intersects")
joined['area_pre_parcel']=joined['geometry_right'].area
# # Count the occurrences of each "land_switch" category within each buffer
joined_gr = joined.groupby(['index_right', "APN_right",'land_switch'],as_index=False).agg({'area_dev_parcel':sum,'sq_ft':sum,'age':max,'sf':max,'townhome_condo':max,'apartment':max,'area_pre_parcel':max})
del joined



# In[53]:


len(rescom_geo_pre)


# In[54]:


# Define batch size (adjust as needed)
batch_size = 50_000  # Process in 50k chunks

# Initialize an empty list to store grouped results
grouped_results = []

# Loop through chunks of rescom_geo_pre
for i in range(0, len(rescom_geo_pre), batch_size):
    print(f"Processing batch {i} to {i+batch_size}")

    # Get current chunk
    rescom_chunk = rescom_geo_pre.iloc[i:i+batch_size].copy()

    # Create 2km buffer (excluding 1km buffer)
    rescom_chunk["buffer_2km"] = rescom_chunk.geometry.centroid.buffer(2000).difference(
        rescom_chunk.geometry.centroid.buffer(1000)
    )

    # Spatial join: Find parcels in parcels_with_lc_change that intersect with any buffer
    joined_2km = gpd.sjoin(parcels_with_lc_change, rescom_chunk.set_geometry("buffer_2km"), predicate="intersects")

    # Compute area per parcel
    joined_2km["area_pre_parcel"] = joined_2km["geometry_right"].area

    # Group and aggregate immediately
    joined_gr_2km = (
        joined_2km.groupby([ "APN_right", 'land_switch'], as_index=False)
        .agg({'area_dev_parcel': 'sum'})
    )

    # Append result to list
    grouped_results.append(joined_gr_2km)

    # Delete joined dataset to free memory
    del joined_2km, joined_gr_2km, rescom_chunk

# Concatenate all grouped results into a single DataFrame
final_grouped_2km = pd.concat(grouped_results, ignore_index=True)


# In[55]:


###now join 1km buffer dev with 2km buffer dev
final_grouped_2km=final_grouped_2km.rename(columns={'area_dev_parcel':'area_dev_parcel_2km','land_switch':'land_switch_2km'})
joined_master=joined_gr.merge(final_grouped_2km,left_on=['APN_right','land_switch'],right_on=['APN_right','land_switch_2km'],how='inner')


###now pivot

joined_master_1km=joined_master[['APN_right','land_switch','area_dev_parcel']]
joined_master_pvt_1km = joined_master_1km.pivot_table(
    index='APN_right', 
    columns='land_switch', 
    values='area_dev_parcel', 
    aggfunc='sum'
).reset_index()

joined_master_pvt_1km=joined_master_pvt_1km.fillna(0)


joined_master_2km=joined_master[['APN_right','land_switch_2km','area_dev_parcel_2km']]

joined_master_pvt_2km = joined_master_2km.pivot_table(
    index='APN_right', 
    columns='land_switch_2km', 
    values='area_dev_parcel_2km', 
    aggfunc='sum'
).reset_index()


joined_master_pvt_2km.columns = [
    f"{j}_2km"  for j in joined_master_pvt_2km.columns
]

joined_master_pvt_2km=joined_master_pvt_2km.fillna(0)


####merge

joined_master_pvt=joined_master_pvt_1km.merge(joined_master_pvt_2km,left_on='APN_right', right_on='APN_right_2km')


# In[56]:


air_temp=gpd.GeoDataFrame(geometry=[],crs="EPSG:4326")

for year in [2009,2018]:
    print(year)
    
    data_year=gpd.GeoDataFrame(geometry=[],crs="EPSG:4326")
    
    for month in [6,7,8]:
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


# In[58]:


air_temp['t2_mean']=(air_temp['t2_max_BC']+air_temp['t2_min_BC'])/2


# In[59]:


air_temp.groupby('year',as_index=False).agg({'t2_mean':'mean'})


# In[60]:


### now merge canopy temps with parcel 1km neighborhoods for 2009

tmp_2009=rescom_geo_pre.copy()
tmp_2009['geometry_new_buffer']=tmp_2009.geometry.centroid.buffer(1000)
del tmp_2009['geometry']
tmp_2009=tmp_2009.rename(columns={'geometry_new_buffer':'geometry'})
tmp_2009.set_geometry('geometry',inplace=True)
air_2009=air_temp[air_temp['year']==2009].copy().reset_index()
tmp_2009=tmp_2009.to_crs("EPSG:4326")
join_2009=gpd.sjoin(air_2009[['t2_mean','geometry']],tmp_2009,predicate='within',how='inner')
air_temp_gr_2009=join_2009.groupby(['APN'],as_index=False).agg({'t2_mean':'mean'})
air_temp_gr_2009=air_temp_gr_2009.rename(columns={'t2_mean':'t2_mean_2009'})
    


###now get temp in 2km buffer

tmp_2009_b=rescom_geo_pre.copy()
tmp_2009_b['geometry_buffer_2km']=tmp_2009_b.geometry.centroid.buffer(2000).difference(tmp_2009_b.geometry.centroid.buffer(1000))
del tmp_2009_b['geometry']
tmp_2009_b=tmp_2009_b.rename(columns={'geometry_buffer_2km':'geometry'})
tmp_2009_b.set_geometry('geometry',inplace=True)
air_2009=air_temp[air_temp['year']==2009].copy().reset_index()
tmp_2009_b=tmp_2009_b.to_crs("EPSG:4326")
join_2009_b=gpd.sjoin(air_2009[['t2_mean','geometry']],tmp_2009_b,predicate='within',how='inner')
air_temp_gr_2009_b=join_2009_b.groupby(['APN'],as_index=False).agg({'t2_mean':'mean'})
air_temp_gr_2009_b=air_temp_gr_2009_b.rename(columns={'t2_mean':'t2_mean_2009_b'})
    


# In[61]:


### now merge canopy temps with parcel 1km neighborhoods for 2018

tmp_2018=rescom_geo_pre.copy()
tmp_2018['geometry_new_buffer']=tmp_2018.geometry.centroid.buffer(1000)
del tmp_2018['geometry']
tmp_2018=tmp_2018.rename(columns={'geometry_new_buffer':'geometry'})
tmp_2018.set_geometry('geometry',inplace=True)
air_2018=air_temp[air_temp['year']==2018].copy().reset_index()
tmp_2018=tmp_2018.to_crs("EPSG:4326")
join_2018=gpd.sjoin(air_2018[['t2_mean','geometry']],tmp_2018,predicate='within',how='inner')
air_temp_gr_2018=join_2018.groupby(['APN'],as_index=False).agg({'t2_mean':'mean'})
air_temp_gr_2018=air_temp_gr_2018.rename(columns={'t2_mean':'t2_mean_2018'})
    


###now get temp in 2km buffer

tmp_2018_b=rescom_geo_pre.copy()
tmp_2018_b['geometry_buffer_2km']=tmp_2018_b.geometry.centroid.buffer(2000).difference(tmp_2018_b.geometry.centroid.buffer(1000))
del tmp_2018_b['geometry']
tmp_2018_b=tmp_2018_b.rename(columns={'geometry_buffer_2km':'geometry'})
tmp_2018_b.set_geometry('geometry',inplace=True)
air_2018=air_temp[air_temp['year']==2018].copy().reset_index()
tmp_2018_b=tmp_2018_b.to_crs("EPSG:4326")
join_2018_b=gpd.sjoin(air_2018[['t2_mean','geometry']],tmp_2018_b,predicate='within',how='inner')
air_temp_gr_2018_b=join_2018_b.groupby(['APN'],as_index=False).agg({'t2_mean':'mean'})
air_temp_gr_2018_b=air_temp_gr_2018_b.rename(columns={'t2_mean':'t2_mean_2018_b'})
    


# In[62]:


master_final_infill=rescom_geo_pre.merge(
    air_temp_gr_2009,on='APN',how='inner').merge(
    air_temp_gr_2018,on='APN',how='inner').merge(
    air_temp_gr_2009_b,on='APN',how='inner').merge(
    air_temp_gr_2018_b,on='APN',how='inner').merge(
    joined_master_pvt,left_on='APN',right_on='APN_right')



# In[63]:


master_final_infill['change_in_temp']=master_final_infill['t2_mean_2018']-master_final_infill['t2_mean_2009']
master_final_infill['change_in_temp_b']=master_final_infill['t2_mean_2018_b']-master_final_infill['t2_mean_2009_b']


# In[64]:


master_final_infill['change_in_temp_b'].describe()


# In[75]:


# del master_final_infill['geometry']
# del master_final_infill['buffer']
# master_final_infill.to_stata('%s/infill_reg_2_17.dta'%path)


# ### Get commercial build data

# In[125]:


commercial_raw=pd.read_csv('%s/2017_CommercialMaster_All.csv'%path,header=None,delimiter="|")
parcels=gpd.read_file(f"{path}/parcels_by_year/Parcels_-_Maricopa_County%2C_Arizona_({2019}).shp")


# In[126]:


comm=commercial_raw[(commercial_raw[17]>2009)& (commercial_raw[17]<2019)].copy()
comm['APN']=comm[2].str.strip()
comm['build_year']=comm[17]
comm=comm[['APN','build_year']]

comm=comm.merge(parcels[['APN','geometry']],on='APN')
comm=gpd.GeoDataFrame(comm,geometry='geometry',crs=parcels.crs)


# #### get commercial land transitions between 2009 and 2018

# In[128]:


# Initialize an empty GeoDataFrame
comm_with_lc_change = gpd.GeoDataFrame(geometry=[], crs="EPSG:26912")

# Define raster file paths
land_use_raster0 = f"{path}/clipped_land_cover_maricopa_{2009}_full_county.tif"
land_use_raster1 = f"{path}/clipped_land_cover_maricopa_{2018}_full_county.tif"

rescom_tmp=comm.copy()

# Open the land use change raster
with rasterio.open(land_use_raster0) as src0, rasterio.open(land_use_raster1) as src1:
    raster_crs = src0.crs  # Get the CRS of the raster

    rescom_tmp = rescom_tmp.to_crs(raster_crs)

    # Define land use categories
    
    unique_land_use_classes = [11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95]
    
    for land_use in unique_land_use_classes:
        rescom_tmp[f"pct_{land_use}_0"] = 0.0
        rescom_tmp[f"pct_{land_use}_1"] = 0.0

    # Extract land use values for each parcel
    for idx, parcel in rescom_tmp.iterrows():
        
        try:

            # # Convert to a dictionary format for Rasterio
            parcel_geom_2d = [mapping(parcel.geometry)]

            # Mask the raster for the parcel
            masked_image0, _ = mask(src0, parcel_geom_2d, crop=True, indexes=1, filled=True, all_touched=True)
            masked_image1, _ = mask(src1, parcel_geom_2d, crop=True, indexes=1, filled=True, all_touched=True)

            # Convert to 1D arrays and remove NoData values
            land_use_values0 = masked_image0.flatten()
            land_use_values1 = masked_image1.flatten()

            # Remove NoData values (-9999, 255, or other values in raster profile)
            no_data_value = src1.nodata
            
            land_use_values0 = land_use_values0[land_use_values0 != no_data_value]
            land_use_values1 = land_use_values1[land_use_values1 != no_data_value]

            # Compute land cover change only if valid values exist
            if land_use_values0.size > 0 and land_use_values1.size > 0:
                
                counts0 = Counter(land_use_values0)
                counts1 = Counter(land_use_values1)

                total_pixels0 = sum(counts0.values())
                total_pixels1 = sum(counts1.values())

                # Assign percentages
                for land_use0, count0 in counts0.items():
                    column_name = f"pct_{land_use0}_0"
                    if column_name in rescom_tmp.columns:
                        rescom_tmp.at[idx, column_name] = count0 / total_pixels0 * 100
                        # / total_pixels0 * 100

                for land_use1, count1 in counts1.items():
                    column_name = f"pct_{land_use1}_1"
                    if column_name in rescom_tmp.columns:
                        rescom_tmp.at[idx, column_name] = count1 / total_pixels1 * 100
                        


        except Exception as e:
            print(f"Error processing parcel {idx}: {e}")
            continue

# Convert back to EPSG:26912
rescom_tmp = rescom_tmp.to_crs("EPSG:26912")

# Append results
comm_with_lc_change = pd.concat([comm_with_lc_change, rescom_tmp], ignore_index=True)

# Save the final dataset
comm_with_lc_change.to_file(f"{path}/comm_with_land_cover_change_2009_2018.shp")


# ### For commercial completions characterize whether development was infill or sprawl

# In[130]:


# List of unique land use classes
unique_land_use_classes = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]

# Define Natural and Developed land cover categories
nat = {81, 82}  # Natural classes
dev = { 21,22, 23, 24}  # Developed classes

# Create column names for 2000 and 2018
dev_cols_2009 = [f"pct_{lu}_0" for lu in dev]  # Land cover 2000
dev_cols_2018 = [f"pct_{lu}_1" for lu in dev]  # Land cover 2018

# Create column names for 2000 and 2018
nat_cols_2009 = [f"pct_{lu}_0" for lu in nat]  # Land cover 2000
nat_cols_2018 = [f"pct_{lu}_1" for lu in nat]  # Land cover 2018

###sum across natural and developed uses

comm_with_lc_change['nat_uses_2009'] = comm_with_lc_change[nat_cols_2009].sum(axis=1)
comm_with_lc_change['dev_uses_2009'] = comm_with_lc_change[dev_cols_2009].sum(axis=1)

comm_with_lc_change['nat_uses_2018'] = comm_with_lc_change[nat_cols_2018].sum(axis=1)
comm_with_lc_change['dev_uses_2018'] = comm_with_lc_change[dev_cols_2018].sum(axis=1)

# Find majority land use in 2009
comm_with_lc_change["majority_land_use_2009"] = comm_with_lc_change[['nat_uses_2009','dev_uses_2009']].idxmax(axis=1).str.extract(r"(^.{3})")

# Find majority land use in 2018
comm_with_lc_change["majority_land_use_2018"] = comm_with_lc_change[['nat_uses_2018','dev_uses_2018']].idxmax(axis=1).str.extract(r"(^.{3})")

comm_with_lc_change["land_switch"]=comm_with_lc_change["majority_land_use_2009"].astype(str)+"_"+comm_with_lc_change["majority_land_use_2018"].astype(str)


# In[131]:


comm_with_lc_change=comm_with_lc_change[comm_with_lc_change['geometry'].area >0].reset_index(drop=True)


# In[132]:


# #### now draw 1km buffer around each pre-existing home, and merge characteristics of residential development that took place in that 1km buffer
comm_with_lc_change['area_dev_parcel_comm']=comm_with_lc_change['geometry'].area
# # Create a 1km buffer around the centroid of each geometry in rescom_geo_pre
rescom_geo_pre["buffer"] = rescom_geo_pre.geometry.centroid.buffer(1000)

# # Spatial join: Find parcels in parcels_with_lc_change that intersect with any buffer
joined = gpd.sjoin(comm_with_lc_change, rescom_geo_pre.set_geometry("buffer"), predicate="intersects")
joined['area_pre_parcel']=joined['geometry_right'].area
# # Count the occurrences of each "land_switch" category within each buffer
joined_gr_comm = joined.groupby(['index_right', "APN_right",'land_switch'],as_index=False).agg({'area_dev_parcel_comm':sum})
del joined



# In[133]:


# Define batch size (adjust as needed)
batch_size = 50_000  # Process in 50k chunks

# Initialize an empty list to store grouped results
grouped_results = []

# Loop through chunks of rescom_geo_pre
for i in range(0, len(rescom_geo_pre), batch_size):
    print(f"Processing batch {i} to {i+batch_size}")

    # Get current chunk
    rescom_chunk = rescom_geo_pre.iloc[i:i+batch_size].copy()

    # Create 2km buffer (excluding 1km buffer)
    rescom_chunk["buffer_2km"] = rescom_chunk.geometry.centroid.buffer(2000).difference(
        rescom_chunk.geometry.centroid.buffer(1000)
    )

    # Spatial join: Find parcels in parcels_with_lc_change that intersect with any buffer
    joined_2km = gpd.sjoin(comm_with_lc_change, rescom_chunk.set_geometry("buffer_2km"), predicate="intersects")

    # Compute area per parcel
    joined_2km["area_pre_parcel"] = joined_2km["geometry_right"].area

    # Group and aggregate immediately
    joined_gr_2km = (
        joined_2km.groupby([ "APN_right", 'land_switch'], as_index=False)
        .agg({'area_dev_parcel_comm': 'sum'})
    )

    # Append result to list
    grouped_results.append(joined_gr_2km)

    # Delete joined dataset to free memory
    del joined_2km, joined_gr_2km, rescom_chunk

# Concatenate all grouped results into a single DataFrame
final_grouped_2km_comm = pd.concat(grouped_results, ignore_index=True)


# In[135]:


final_grouped_2km_comm[final_grouped_2km_comm['area_dev_parcel_comm']==0]


# In[141]:


###now join 1km buffer dev with 2km buffer dev
final_grouped_2km_comm=final_grouped_2km_comm.rename(columns={'area_dev_parcel_comm':'area_dev_parcel_comm_2km','land_switch':'land_switch_comm_2km'})

joined_gr_comm=joined_gr_comm.rename(columns={'land_switch':'land_switch_comm'})


joined_master=joined_gr_comm.merge(final_grouped_2km_comm,left_on=['APN_right','land_switch_comm'],right_on=['APN_right','land_switch_comm_2km'],how='inner')


###now pivot

joined_master_1km=joined_master[['APN_right','land_switch_comm','area_dev_parcel_comm']]
joined_master_pvt_1km = joined_master_1km.pivot_table(
    index='APN_right', 
    columns='land_switch_comm', 
    values='area_dev_parcel_comm', 
    aggfunc='sum'
).reset_index()

joined_master_pvt_1km=joined_master_pvt_1km.fillna(0)

joined_master_pvt_1km.columns = [
    f"{j}_comm"  for j in joined_master_pvt_1km.columns
]

joined_master_2km=joined_master[['APN_right','land_switch_comm_2km','area_dev_parcel_comm_2km']]

joined_master_pvt_2km = joined_master_2km.pivot_table(
    index='APN_right', 
    columns='land_switch_comm_2km', 
    values='area_dev_parcel_comm_2km', 
    aggfunc='sum'
).reset_index()


joined_master_pvt_2km.columns = [
    f"{j}_comm_2km"  for j in joined_master_pvt_2km.columns
]

joined_master_pvt_2km=joined_master_pvt_2km.fillna(0)


####merge

joined_master_pvt=joined_master_pvt_1km.merge(joined_master_pvt_2km,left_on='APN_right_comm', right_on='APN_right_comm_2km')


# #### Merge to res buffer data

# In[148]:


joined_master_pvt


# In[145]:


master_final_infill_comm=master_final_infill.merge(joined_master_pvt,left_on='APN',right_on='APN_right_comm',how='left').fillna(0)


# In[149]:


#### join in tract income and pop growth data
parcel_block=pd.read_csv('/Users/hannahkamen/Downloads/parcel_block_intersection.csv')
income=pd.read_csv('%s/nhgis0009_csv/nhgis0009_ts_nominal_tract.csv'%path)
income=income[income['COUNTY']=='Maricopa County'][['TRACTA','B79AA115','B79AA195']].copy()
income=income.rename(columns={'TRACTA':'TRACTCE10','B79AA115':'income_11','B79AA195':'income_19'})
pop=pd.read_csv('%s/nhgis0007_csv/nhgis0007_ts_nominal_tract.csv'%path)
pop=pop[pop['COUNTY']=='Maricopa County'][['TRACTA','AV0AA115','AV0AA195']].copy()
pop=pop.rename(columns={'TRACTA':'TRACTCE10','AV0AA115':'pop_11','AV0AA195':'pop_19'})

###merge

dem_data=parcel_block[['APN','TRACTCE10']].merge(income,on='TRACTCE10',how='inner').merge(pop,on='TRACTCE10',how='inner')


# In[150]:


master_final_infill_demo=master_final_infill_comm.merge(dem_data,on='APN').drop_duplicates(subset=['APN'])


# In[151]:


####get distance to city center

# Define the fixed point (Phoenix City Center) as a Point geometry
phx_cc = Point(-112.073795, 33.445592)  # (longitude, latitude)

# Convert the Phoenix City Center point to a GeoDataFrame
phx_gdf = gpd.GeoDataFrame({'geometry': [phx_cc]}, crs="EPSG:4326")
phx_gdf = phx_gdf.to_crs(epsg=26912)

# Extract reprojected Phoenix City Center coordinates
phx_cc_projected = phx_gdf.geometry.iloc[0]

# Compute distance from each parcel to the Phoenix City Center
master_final_infill_demo['distance_to_phx_cc'] = master_final_infill_demo.geometry.distance(phx_cc_projected)



# In[152]:


master_final_infill_demo['income_change']=master_final_infill_demo['income_19']-master_final_infill_demo['income_11']
master_final_infill_demo['pop_change']=master_final_infill_demo['pop_19']-master_final_infill_demo['pop_11']


# In[153]:


#### import land use profiles in 2009
prexist_with_lc_2009_1km=pd.read_csv(f"{path}/preexist_with_land_cover_pct_1km_2009.csv")
prexist_with_lc_2009_1km= prexist_with_lc_2009_1km[['APN','pct_21_2009','pct_22_2009','pct_23_2009','pct_24_2009','pct_52_2009','pct_81_2009','pct_82_2009']]


prexist_with_lc_2009_2km=pd.read_csv(f"{path}/preexist_with_land_cover_pct_2km_2009.csv")
prexist_with_lc_2009_2km= prexist_with_lc_2009_2km[['APN','pct_21_2009','pct_22_2009','pct_23_2009','pct_24_2009','pct_52_2009','pct_81_2009','pct_82_2009']]
prexist_with_lc_2009_2km=prexist_with_lc_2009_2km.rename(columns={'pct_21_2009':'pct_21_2009_2km',
                                                                 'pct_22_2009':'pct_22_2009_2km',
                                                                 'pct_23_2009':'pct_23_2009_2km',
                                                                 'pct_24_2009':'pct_24_2009_2km',
                                                                 'pct_81_2009':'pct_81_2009_2km',
                                                                 'pct_82_2009':'pct_82_2009_2km',
                                                                 'pct_52_2009':'pct_52_2009_2km',})


# In[154]:


####merge with existing dataframe
master_final_infill_demo_lc=master_final_infill_demo.merge(prexist_with_lc_2009_1km,on='APN').merge(prexist_with_lc_2009_2km,on='APN')


# In[155]:


####now merge on the size of farm parcelss in buffers
median_ag_area_final=pd.read_csv('%s/median_ag_area_1km_2km.csv'%path)

master_final_infill_demo_lc_agpcl=master_final_infill_demo_lc.merge(median_ag_area_final,on='APN')


# In[159]:


master_final_infill_demo_lc_agpcl=master_final_infill_demo_lc_agpcl.fillna(0)


# In[162]:


# del master_final_infill_demo_lc_agpcl['geometry']
# del master_final_infill_demo_lc_agpcl['buffer']
# del master_final_infill_demo_lc_agpcl['APN_right_comm']
del master_final_infill_demo_lc_agpcl['APN_right_comm_2km']
master_final_infill_demo_lc_agpcl.to_stata('%s/infill_reg_2_18_w_lc_ag_pcl_size.dta'%path)


# In[156]:


master_final_infill_demo_lc_agpcl


# In[7]:


master_final_infill_comm_geo[master_final_infill_comm_geo['dev_dev']==0]


# In[95]:


list(master_final_infill_demo_lc_agpcl)


# In[299]:


master_final_infill_comm_geo['distance_to_phx_cc'].describe()


# In[282]:





# In[ ]:


### get distance to city center


# In[ ]:


phx_cc=(33.445592, -112.073795)


# In[262]:


pop[pop['TRACTA']==814500]


# In[254]:


parcel_block


# In[ ]:




