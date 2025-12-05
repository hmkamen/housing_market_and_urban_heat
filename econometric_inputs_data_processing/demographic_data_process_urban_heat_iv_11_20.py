#!/usr/bin/env python
# coding: utf-8

# In[100]:


### this processes  all demographic data used in urban heat iv paper


# In[101]:


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

# In[102]:


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

#parcel block and tract intersection
parcel_tract=pd.read_csv('/Users/hannahkamen/Downloads/parcel_block_intersection.csv')
parcel_tract=parcel_tract.drop_duplicates(['APN'])
parcel_tract=parcel_tract[['APN','TRACTCE10']]


# ### Merge tract intersection with sold homes in sample

# In[103]:


####merge parcels to data
master_tract=sold_parcels.merge(parcel_tract,on='APN',how='inner').merge(tracts)


# ## Demographic data

# ### Income

# In[104]:


### INCOME
income=pd.read_csv('%s/nhgis0010_csv/nhgis0010_ts_nominal_tract.csv'%path)
income=income[income['COUNTY']=='Maricopa County'].copy()
income=income.rename(columns={'TRACTA':'TRACTCE10',
                              'B79AA105':'income_09',
                              'B79AA115':'income_10',
                              'B79AA125':'income_11',
                              'B79AA135':'income_12',
                              'B79AA145':'income_13',
                              'B79AA155':'income_14',
                              'B79AA165':'income_15',
                              'B79AA175':'income_16',
                              'B79AA185':'income_17',
                              'B79AA195':'income_18'
                              })
income=income[['TRACTCE10']+ [x for x in income.columns if "income_" in x]]
#### pivot incomes to get time varying values
income_melt=income.melt(id_vars='TRACTCE10',value_vars=[x for x in income.columns if "income"in x])
income_melt['sale_year']=income_melt['variable'].apply(lambda x: int(x[-2:])+2000)
del income_melt['variable']
income_melt=income_melt.rename(columns={'value':'median_income'})

### adjust income to 2025 dollars

# Map of inflation multipliers by sale_year
multipliers = {
    2009: 1.51,
    2010: 1.48,
    2011: 1.43,
    2012: 1.40,
    2013: 1.38,
    2014: 1.36,
    2015: 1.36,
    2016: 1.34,
    2017: 1.31,
    2018: 1.28,
}

# 1. Adjust income to 2025 dollars (median_income_2025)
income_melt["median_income_2025"] = (
    income_melt["median_income"]
    * income_melt["sale_year"].map(multipliers)
)

# 2. Squared term of the adjusted income
income_melt["median_income_2025_sq"] = income_melt["median_income_2025"] ** 2


# ### Age

# In[105]:


age=pd.read_csv('%s/nhgis0012_csv/nhgis0012_ts_nominal_tract.csv'%path)
age=age[age['COUNTY']=='Maricopa County'].copy()
age=age.rename(columns={'TRACTA':'TRACTCE10',
                              'D13AA105':'age_09',
                              'D13AA115':'age_10',
                              'D13AA125':'age_11',
                              'D13AA135':'age_12',
                              'D13AA145':'age_13',
                              'D13AA155':'age_14',
                              'D13AA165':'age_15',
                              'D13AA175':'age_16',
                              'D13AA185':'age_17',
                              'D13AA195':'age_18'
                              })
age=age[['TRACTCE10']+ [x for x in age.columns if "age_" in x]]

#### pivot age to get time varying values
age_melt=age.melt(id_vars='TRACTCE10',value_vars=[x for x in age.columns if "age" in x])
age_melt['sale_year']=age_melt['variable'].apply(lambda x: int(x[-2:])+2000)
del age_melt['variable']
age_melt=age_melt.rename(columns={'value':'median_age'})


# ### Population

# In[106]:


### population
pop=pd.read_csv('%s/nhgis0013_csv/nhgis0013_ts_nominal_tract.csv'%path)
pop=pop[pop['COUNTY']=='Maricopa County'].copy()
pop=pop.rename(columns={'TRACTA':'TRACTCE10',
                              'AV0AA115':'pop_09',
                              'AV0AA125':'pop_10',
                              'AV0AA135':'pop_11',
                              'AV0AA145':'pop_12',
                              'AV0AA155':'pop_13',
                              'AV0AA165':'pop_14',
                              'AV0AA175':'pop_15',
                              'AV0AA185':'pop_16',
                              'AV0AA195':'pop_17',
                              'AV0AA205':'pop_18'
                              })
pop=pop[['TRACTCE10']+ [x for x in pop.columns if "pop_" in x]]

#### pivot pop to get time varying values
pop_melt=pop.melt(id_vars='TRACTCE10',value_vars=[x for x in pop.columns if "pop" in x])
pop_melt['sale_year']=pop_melt['variable'].apply(lambda x: int(x[-2:])+2000)
del pop_melt['variable']
pop_melt=pop_melt.rename(columns={'value':'median_pop'})


# ### Households

# In[107]:


hh=pd.read_csv('%s/nhgis0016_csv/nhgis0016_ts_nominal_tract.csv'%path)
hh=hh[hh['COUNTY']=='Maricopa County'].copy()
hh=hh.rename(columns={'TRACTA':'TRACTCE10',
                              'AR5AA115':'hh_09',
                              'AR5AA125':'hh_10',
                              'AR5AA135':'hh_11',
                              'AR5AA145':'hh_12',
                              'AR5AA155':'hh_13',
                              'AR5AA165':'hh_14',
                              'AR5AA175':'hh_15',
                              'AR5AA185':'hh_16',
                              'AR5AA195':'hh_17',
                              'AR5AA205':'hh_18'
                              })
hh=hh[['TRACTCE10']+ [x for x in hh.columns if "hh_" in x]]

#### pivot hh to get time varying values
hh_melt=hh.melt(id_vars='TRACTCE10',value_vars=[x for x in hh.columns if "hh" in x])
hh_melt['sale_year']=hh_melt['variable'].apply(lambda x: int(x[-2:])+2000)
del hh_melt['variable']
hh_melt=hh_melt.rename(columns={'value':'median_hh'})


# ### Households with children

# In[108]:


#### get kids by tract density
child=pd.read_csv('%s/nhgis0014_csv/nhgis0014_ts_nominal_tract.csv'%path)
child=child[child['COUNTY']=='Maricopa County'].copy()
child=child.rename(columns={'TRACTA':'TRACTCE10',
                              'CQ9AA115':'child_09',
                              'CQ9AA125':'child_10',
                              'CQ9AA135':'child_11',
                              'CQ9AA145':'child_12',
                              'CQ9AA155':'child_13',
                              'CQ9AA165':'child_14',
                              'CQ9AA175':'child_15',
                              'CQ9AA185':'child_16',
                              'CQ9AA195':'child_17',
                              'CQ9AA205':'child_18'
                              })
child=child[['TRACTCE10']+ [x for x in child.columns if "child_" in x]]

#### pivot child to get time varying values
child_melt=child.melt(id_vars='TRACTCE10',value_vars=[x for x in child.columns if "child" in x])
child_melt['sale_year']=child_melt['variable'].apply(lambda x: int(x[-2:])+2000)
del child_melt['variable']
child_melt=child_melt.rename(columns={'value':'median_child'})


# ### Merge demographic data with parcel sales data

# In[109]:


master_demo=master_tract.merge(age_melt, on=["TRACTCE10",'sale_year']
                               ).merge(income_melt,on=["TRACTCE10",'sale_year']
                               ).merge(pop_melt,on=["TRACTCE10",'sale_year']
                               ).merge(hh_melt,on=["TRACTCE10",'sale_year']
                               ).merge(child_melt,on=["TRACTCE10",'sale_year'])



# In[110]:


### Calculate density values
master_demo['child_per_household']=master_demo['median_child']/master_demo['median_hh']
master_demo['pop_density']=master_demo['median_pop']/master_demo['tract_area']


# ### Make quantile groups

# In[114]:


### make quantile groups
master_demo['income_q'] = (
    master_demo.groupby('sale_year')['median_income']
      .transform(lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop') + 1)
)

master_demo['density_q'] = (
    master_demo.groupby('sale_year')['pop_density']
      .transform(lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop') + 1)
)

master_demo['child_q'] = (
    master_demo.groupby('sale_year')['median_child']
      .transform(lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop') + 1)
)

master_demo['age_q'] = (
    master_demo.groupby('sale_year')['median_age']
      .transform(lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop') + 1)
)


# ### Export APN, TRACT, YEAR level data

# In[113]:


master_demo['child_q'].describe()


# In[115]:


master_demo.to_csv("%s/urban_iv_paper_input_data_11_20/demographic_data_11_20.csv"%path)


# In[ ]:




