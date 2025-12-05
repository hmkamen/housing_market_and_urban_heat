#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
from collections import Counter


path='/Users/hannahkamen/Downloads'
import warnings

# Filter or ignore specific warning types
warnings.filterwarnings('ignore')

from shapely.geometry import box, shape, mapping  # Import shape here
from rasterio.features import geometry_mask


# In[15]:


#### load parcel shapefile and parcelcsv to limit to sold parcels
##parcel shapefile
# tracts = gpd.read_file("%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp" % path)
# tracts=tracts[['TRACTCE10','geometry']].copy()
# tracts=tracts.to_crs('EPSG:26912')

tracts=gpd.read_file('%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp'%path)
tracts=tracts[['TRACTCE10','geometry']].copy()
tracts=tracts.to_crs('EPSG:26912')


# In[16]:


# Initialize an empty GeoDataFrame
# urban_rural_fringe = gpd.GeoDataFrame(geometry=[], crs="EPSG:26912")
tract_lc=pd.DataFrame()
for year in range(2009,2019):
    print(year)
    # Define raster file paths
    land_use_raster0 = f"{path}/clipped_land_cover_maricopa_{year}_full_county.tif"
    
    
    with rasterio.open(land_use_raster0) as src0:
        raster_crs = src0.crs  # Get the CRS of the raster
    
        tracts_tmp= tracts.to_crs(raster_crs)
        tracts_tmp['sale_year']=year

    
        # Define land use categories
        unique_land_use_classes = [21,22,23,24,52,81,82]
        
        for land_use in unique_land_use_classes:
            tracts_tmp[f"area_{land_use}"] = 0.0
    
        # Extract land use values for each parcel
        for idx, tract in tracts_tmp.iterrows():
            
            try:
    
                # # Convert to a dictionary format for Rasterio
                tract_geom_2d = [mapping(tract.geometry)]
    
                # Mask the raster for the parcel
                masked_image0, _ = mask(src0, tract_geom_2d, crop=True, indexes=1, filled=True, all_touched=True)
    
                # Convert to 1D arrays and remove NoData values
                land_use_values0 = masked_image0.flatten()
    
                # Remove NoData values (-9999, 255, or other values in raster profile)
                no_data_value = src0.nodata
                
                land_use_values0 = land_use_values0[land_use_values0 != no_data_value]
    
                # Compute land cover change only if valid values exist
                if land_use_values0.size > 0 :
                    
                    counts0 = Counter(land_use_values0)
                    total_pixels0 = sum(counts0.values())
    
                    # Assign percentages
                    for land_use0, count0 in counts0.items():
                        column_name = f"area_{land_use0}"
                        if column_name in tracts_tmp.columns:
                            tracts_tmp.at[idx, column_name] = count0/ total_pixels0 * 100
    
            except Exception as e:
                print(f"Error processing parcel {idx}: {e}")
                continue
    tract_lc=pd.concat([tract_lc,tracts_tmp],ignore_index=True)


# In[17]:


tract_lc


# In[18]:


del tract_lc['geometry']
tract_lc.to_csv("%s/tract_lc.csv"%path)


# In[290]:


tracts_lm=tracts[tracts['sum']>5]
tracts = gpd.read_file("%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp" % path)
tracts=tracts[tracts['TRACTCE10'].isin(tracts_lm['TRACTCE10'].unique())]
tracts.to_file('%s/rural_urban_fringe.shp'%path)


# In[291]:


tracts


# In[273]:


tracts.to_file("%s/tl_2010_04013_tract10/natural_land_cover_pct_2009.shp" % path)


# In[ ]:


tracts.to_file("%s/tl_2010_04013_tract10/natural_land_cover_pct_2018.shp" % path)


# In[ ]:


tracts = gpd.read_file("%s/tl_2010_04013_tract10/natural_land_cover_pct_2018.shp" % path)


# In[228]:


tracts['sum']=tracts['area_81']+tracts['area_82']+tracts['area_11']+tracts['area_31']+tracts['area_41']+tracts['area_42']+tracts['area_43']+tracts['area_52']+tracts['area_71']+tracts['area_90']+tracts['area_95']
# tracts['area_81']+tracts['area_82']+


# In[230]:


len(tracts)


# In[154]:


len(tracts['TRACTCE10'].unique())


# In[30]:


373224/571000


# In[ ]:




