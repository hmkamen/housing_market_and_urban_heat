#!/usr/bin/env python
# coding: utf-8

# In[219]:


### this processes  all demographic data used in urban heat iv paper


# In[220]:


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


# ### Import home sales data

# In[221]:


home_sales=pd.read_csv('%s/urban_iv_paper_input_data_11_20/sold_homes_sample_11_20.csv'%path)


# ### Import temp and instr file

# In[222]:


# temp=pd.read_csv("%s/urban_iv_paper_input_data_11_20/temp_and_instr_data_11_20.csv"%path)
# temp=pd.read_csv("%s/urban_iv_paper_input_data_11_20/temp_and_instr_data_11_22_5000m.csv"%path)
# temp=pd.read_csv("%s/urban_iv_paper_input_data_11_20/temp_and_instr_data_incl_500_5500_11_22.csv"%path)
temp=pd.read_csv("%s/urban_iv_paper_input_data_11_20/temp_and_instr_data_incl_500_5500_11_30.csv"%path)
del temp['TRACTCE10']


# ### Import demographics file

# In[223]:


demos=pd.read_csv("%s/urban_iv_paper_input_data_11_20/demographic_data_11_20.csv"%path)
del demos['Unnamed: 0']
del demos['TRACTCE10']
del demos['tract_area']


# ### Import amenities file

# In[224]:


amenities=pd.read_csv("%s/urban_iv_paper_input_data_11_20/amenities_data_11_20.csv"%path)
del amenities['Unnamed: 0.1']
del amenities['Unnamed: 0']
del amenities['Unnamed: 0_y']
del amenities['geometry']
del amenities['index_right']


# ### Merge

# In[225]:


master=home_sales.merge(temp,on=['APN','sale_year'],how='inner').merge(demos,on=['APN','sale_year'],how='inner'
                 ).merge(amenities,on=['APN','sale_year'],how='inner')


# ### Create a few more vars

# In[226]:


master['block_year_id']= master['BLOCKID10']+master['sale_year']
master['tract_year_id']= master['TRACTCE10']+master['sale_year']


# ### Limit and Export

# In[ ]:





# In[227]:


master_lm=master.drop_duplicates(subset=['APN','sale_year'])

# 1. 95th percentile of area_soil
as_pct = master_lm["area_soil"].quantile(0.95)

# 2. 95th percentile of area_farm
af_pct = master_lm["area_farm"].quantile(0.95)

# 3. 90th percentile of distance_to_phx_cc
dcc_pct = master_lm["distance_to_phx_cc"].quantile(0.90)

# 4. Create far_from_cc indicator
master_lm["far_from_cc"] = (master_lm["distance_to_phx_cc"] >= dcc_pct).astype(int)

# 5. Drop rows where area_soil >= as_pct OR area_farm >= af_pct
mask = (master_lm["area_soil"] < as_pct) & (master_lm["area_farm"] < af_pct)
master_lm = master_lm[mask].copy()
# master_lm.to_stata("%s/urban_iv_paper_input_data_11_20/urban_heat_iv_11_20_500_5500_11_24.dta"%path)


# In[230]:


master_lm.to_stata("%s/urban_iv_paper_input_data_11_20/urban_heat_iv_11_20_500_5500_11_30.dta"%path)


# In[228]:


master['area_soil'].quantile(.90)


# In[229]:


master_lm['ann_sale_price_25'].describe()


# In[231]:


master_lm['ann_sale_price_25'].describe()


# In[190]:


len(master.drop_duplicates(subset=['APN','sale_year']))


# In[ ]:




