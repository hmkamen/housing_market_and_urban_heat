#!/usr/bin/env python
# coding: utf-8

# In[42]:


####this file cleans parcel sales information from the file system at:
##### https://www.dropbox.com/sh/0e8wltu2kb9s23y/AAAtlwnfP4bB3pY-Fj80YSE8a/Archived_Maricopa_Parcel_Files?e=1&dl=0
#####then joins parcel sales information with:
####hmda dataparcel lst by year, block imperviousness by year, and vegetative ring of parcel by year


# In[43]:


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
# Load Maricopa County boundary (replace with correct path if needed)
maricopa_gdf = gpd.read_file("%s/maricopa/maricopa.shp"%path)
#get zips
zips=gpd.read_file('%s/Maricopa_County_Zip_Codes/ZipCodes.shp'%path)
pumas=gpd.read_file('%s/tl_2021_04_puma10/tl_2021_04_puma10.shp'%path)
tracts=gpd.read_file('%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp'%path)
subdivisions=gpd.read_file("%s/Subdivisions/Subdivisions.shp" % path)

# Filter or ignore specific warning types
warnings.filterwarnings('ignore')


# ### Helper functions for data extraction

# In[44]:


def extract_year(lst):

    if len(lst) == 9:
        return lst[7][-4:] 
    if len(lst)==10:
        return lst[8][-4:] 
    else:
        return 0
    
def extract_month(lst):
    if len(lst) == 9:
        return lst[7][-6:][:2] 
    if len(lst) == 10:
        return lst[8][-6:][:2]
    else:
        return 0
    
def extract_price(lst):
    if len(lst) == 9:
        return lst[7][:-6]     
    if len(lst) == 10:
        return lst[8][:-6]     
    else:
        return 0
    
############################################   
    
def extract_sqft(lst):
    try:
        if len(lst) == 9:
            return lst[6][:-7]     
        if len(lst) == 10:
            return lst[6][:-7]     
        else:
            return 0
    except:
        
        print(lst)
    
def extract_pool(lst):
#     if len(lst) == 9:
#         return lst[7][:-6]     
    if len(lst) == 10:
        return lst[7]    
    else:
        return 0
    
def extract_constryr(lst):
    if len(lst) == 9:
        return lst[6][-7:][:4]     
    if len(lst) == 10:
        return lst[6][-7:][:4]     
    else:
        return 0
    
###############################################################    
def extract_class(lst):
    if len(lst[1])>6:
        return lst[1][6:]
    if len(lst[1])==1:
        return lst[1]
    else:
        return np.nan

def extract_stories(lst):  
    return lst[2][0:1]    

def extract_wall(lst):
    return lst[2][1:2]      

def extract_roof(lst):  
    return lst[3]    
    
def extract_heating(lst): 
    return lst[4][0:1]    

def extract_cooling(lst):     
    return lst[4][1:2]    

def extract_bathroom(lst): 
    return lst[5][:-3]

def extract_patio(lst):   
    return lst[5][-3:]
    
def extract_garage(lst):    
    return lst[6][-3:]  

def extract_puc(lst):    
    if len(lst) == 9:
        return lst[8]
    if len(lst)==10:
        return lst[9]
    else:
        return np.nan


# ### Import Parcel sales data

# In[45]:


###combine 2018
m2018=pd.DataFrame()
for f in Path('/Users/hannahkamen/Downloads/2018').iterdir():
    try:
        df=pd.read_csv('%s'%f, header=None, delimiter='\t')
        df=df.rename(columns={0:''})
        df['APN']=df[''].str.strip().apply(lambda x: x.split(" ")[0].strip())
        df['parsed']=df[''].apply(lambda x: x.split())
        df['len']=df['parsed'].apply(lambda x: len(x))
        df=df[df['len']>=9]

        df['sale_year']=df['parsed'].apply(extract_year).astype(int)
        df=df[((df['sale_year']>1990) &(df['sale_year']<2020))]    
        df['sale_year']=df['sale_year'].astype(int)

        df['sale_price'] = df['parsed'].apply(extract_price)
        df=df[df['sale_price']!='']
        df['sale_price']=df['sale_price'].astype(int)

        df['sale_month']=df['parsed'].apply(extract_month).astype(int).astype(str)

        df['sq_ft']=df['parsed'].apply(extract_sqft)
        df=df[df['sq_ft']!='']
        df['sq_ft']=df['sq_ft'].astype(int)
        df['pool']=df['parsed'].apply(extract_pool).astype(int)
        df['age']=df['parsed'].apply(extract_constryr).astype(int)
        df=df[(df['pool']!=df['sale_year'])]

        df['class']=df['parsed'].apply(extract_class)
        df['stories']=df['parsed'].apply(extract_stories)
        df['wall']=df['parsed'].apply(extract_wall)
        df['roof']=df['parsed'].apply(extract_roof).str.replace(" ","")        
        df['heating']=df['parsed'].apply(extract_heating)
        df['cooling']=df['parsed'].apply(extract_cooling)
        df['bathroom']=df['parsed'].apply(extract_bathroom).astype(str).str.replace(" ","")
        df['patio']=df['parsed'].apply(extract_patio)
        df['garage']=df['parsed'].apply(extract_garage)
        df['puc']=df['parsed'].apply(extract_puc)
        df['year']=2018

        m2018=pd.concat([m2018,df], ignore_index=True)
    except:        
        print(f)
        
m2018=m2018[(m2018['sale_year']>=1995) & (m2018['sale_year']<2020)].reset_index()


# In[46]:


master0=pd.DataFrame()
for year in [2011,2012,2013,2014,2015,2016,2017,2019,2020]:
    
    if year== 2019:
        
        print(year)
        df=pd.read_csv('/Users/hannahkamen/Downloads/2019.csv')
        df['APN']=df['APN'].str.strip()
        df['sale_price']=df['PRICE'].astype(float)
        df=df[~(df['SALEDATE'].isnull()) ]
        df['sale_month']=df['SALEDATE'].astype(int).astype(str).apply(lambda x: x.replace(x[-4:],''))
        df['sale_year']=df['SALEDATE'].astype(int).astype(str).apply(lambda x: x[-4:])
        df['age']=df['CONSTYR'].astype(int)
        df['sq_ft']=df['SQFT']
        df['pool']=df['POOLAREA']
        df['heating']=df['HEATING']
        df['cooling']=df['COOLING']
        df['bathroom']=df['BATHFIX'].astype(str).str.replace(" ",'')
        df['patio']=df['PATIO']
        df['garage']=df['GARAGE']
        df['stories']=df['STORIES']
        df['wall']=df['WALL']
        df['roof']=df['ROOF'].astype(str).str.replace(" ","")
        df['class']=df['CLASS'].astype(int)
        df['year']=year
        df['puc']=df['PUC']
        master0=pd.concat([master0,df],ignore_index=True)
    
    if year == 2020:
        print(year)
        df=pd.read_csv('/Users/hannahkamen/Downloads/2020.csv', header=None, delimiter='\t')
        df=df.rename(columns={0:''})
        df['APN']=df[''].str.strip().apply(lambda x: x.split(" ")[0].strip())
        df['parsed']=df[''].apply(lambda x: x.split())
        df['len']=df['parsed'].apply(lambda x: len(x))
        df=df[df['len']>=9]

        df['sale_year']=df['parsed'].apply(extract_year).astype(int)
        df=df[((df['sale_year']>1990) &(df['sale_year']<2020))]    
        df['sale_year']=df['sale_year'].astype(int)

        df['sale_price'] = df['parsed'].apply(extract_price)
        df=df[df['sale_price']!='']
        df['sale_price']=df['sale_price'].astype(int)

        df['sale_month']=df['parsed'].apply(extract_month).astype(int).astype(str)

        df['sq_ft']=df['parsed'].apply(extract_sqft)
        df=df[df['sq_ft']!='']
        df['sq_ft']=df['sq_ft'].astype(int)
        df['pool']=df['parsed'].apply(extract_pool).astype(int)
        df['age']=df['parsed'].apply(extract_constryr).astype(int)
        df=df[(df['pool']!=df['sale_year'])]
        df['year']=year
        
        df['class']=df['parsed'].apply(extract_class)
        df['stories']=df['parsed'].apply(extract_stories)
        df['wall']=df['parsed'].apply(extract_wall)
        df['roof']=df['parsed'].apply(extract_roof).str.replace(" ","")        
        df['heating']=df['parsed'].apply(extract_heating)
        df['cooling']=df['parsed'].apply(extract_cooling)
        df['bathroom']=df['parsed'].apply(extract_bathroom).astype(str).str.replace(" ","")
        df['patio']=df['parsed'].apply(extract_patio)
        df['garage']=df['parsed'].apply(extract_garage)
        df['puc']=df['parsed'].apply(extract_puc)
        master0=pd.concat([master0,df],ignore_index=True)

      
    if year in [2011,2012,2013,2014,2015,2016,2017]:
        
        print(year)
        df=pd.read_csv('/Users/hannahkamen/Downloads/%s.csv' %year,header=None,delimiter="|")
        df[15]=df[15].str.strip()
        df[16]=df[16].str.strip()
        df=df[~((df[16].isnull())| (df[16]==''))]
        df=df[~((df[15].isnull())| (df[15]==''))]
        df['APN']=df[0].str.strip()
        df['sale_price']=df[15].astype(int)
        df['sale_year']=df[16].apply(lambda x: x[-4:])
        df['sale_month']=df[16].astype(int).astype(str).apply(lambda x: x.replace(x[-4:],''))


        df['class']=df[3].astype(int)
        df['age']=df[12].astype(int)
        df['sq_ft']=df[11]
        df['pool']=df[14]
        df['heating']=df[7]
        df['cooling']=df[8]
        df['bathroom']=df[9].astype(str).str.replace(" ","")
        df['patio']=df[10]
        df['garage']=df[13]

        df['stories']=df[4]
        df['wall']=df[5]
        df['roof']=df[6].astype(str).str.replace(" ","")
        df['puc']=df[23]

        df=df[(df['pool']!=df['sale_year'])]
        df['year']=year
        df=df[df['bathroom'].astype(str)!=""]


        master0=pd.concat([master0,df],ignore_index=True).drop_duplicates(subset=['APN','sale_year','sale_price'])

master0=pd.concat([master0,m2018],ignore_index=True).drop_duplicates(subset=['APN','sale_year','sale_price'])


# In[47]:


#####clean sale year, class and property use code variables
master0['sale_year']=master0['sale_year'].astype(int)
master0['class']=master0['class'].astype(int)
master0['puc']=master0['puc'].astype(int)
#####limit to residential owner occupied and sale years berween 1990 and 2020
# master0=master0[master0['class'].isin([3,4])& (master0['sale_year']<2020) & (master0['sale_year']>1990)]
master0=master0[ (master0['sale_year']<2020) & (master0['sale_year']>1990)]
#####limit to single family homes, townhomesand apartment property use codes
master0=master0[((master0['puc']>=100) & (master0['puc']<190)) |((master0['puc']>=310) & (master0['puc']<=398))| ((master0['puc']>=710) & (master0['puc']<=796))| ((master0['puc']>=8510) & (master0['puc']<=8728))]

master0['sf']=np.where(((master0['puc']>=100) & (master0['puc']<=190)) ,1,0)
master0['apartment']=np.where((((master0['puc']>=350) & (master0['puc']<379))|((master0['puc']>=390) & (master0['puc']<=398)) |((master0['puc']>=320) & (master0['puc']<=348))) ,1,0)
master0['townhome_condo']=np.where(((master0['puc']>=8510) & (master0['puc']<8590)) | ((master0['puc']>=710) & (master0['puc']<=796)),1,0)


# In[48]:


master=master0.copy()
del master0


# In[49]:


#####clean housing attribute variables 
master['cooling']=master['cooling'].astype(str).str.strip()
master=master[~(master['roof'].isin(['99','10','12','','11','`']))]
master=master[master['cooling']!='']
master['cooling']=master['cooling'].astype(int)
# master=master[master['roof']!='']
master['wall']=master['wall'].astype(int)
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
master['pool']=master['pool'].astype(int)


#####clean house characteristic variables
master['heating_dum']=np.where(master['heating'].str.strip()=='Y',1,0)
master['stories_dum']=np.where(master['stories'].str.strip()=='M',1,0)

####sanity checks
master=master[(master['year']>=master['sale_year'])& (master['year']>=master['age'])]
master=master[(master['sale_year']>=2009)&(master['sale_year']<2020)]
master=master[(master['sale_month']<=12)]

######create home age at date of sale variable
master['home_age']=master['sale_year']-master['age']

######limit to home-level variables of interest
master=master[['APN','sale_price','sale_year','sale_month','sf','apartment','townhome_condo','age','sq_ft','pool','bathroom','garage_type','patio_type','heating_dum','stories_dum','wall','roof','cooling','home_age']].drop_duplicates()



# In[50]:


#########limit based on erroneous home characterisitics
master=master[master['bathroom']!='']
master['bathroom']=master['bathroom'].astype(float)
master=master[(master['bathroom']<=15)]
master=master[(master['sq_ft']>500)&(master['sq_ft']<5000)]
master=master[master['pool']<=900]
master=master[master['wall']!=99]
master=master[~(master['sale_price'].isnull())]
master=master[(master['age']!=master['sale_year']) & ~(master['age']>master['sale_year'])]


#####filter out sales that occured within a year from previous sale
from datetime import datetime
#### Get rid of homes that sold twice but within a year of eachother
def create_datetime(row):
    return datetime(row['sale_year'], row['sale_month'], 1)

# Apply the function to create a new datetime column
master['sale_date'] = master.apply(create_datetime, axis=1)
# del master['index']
# del master['level_0']
master=master.sort_values(by=['APN','sale_date']).reset_index()
master['time_between_sales']=master.groupby("APN")['sale_date'].diff()


#####limit to homes that sold at least a year apart
master=master[(master['time_between_sales'].isna()) |(master['time_between_sales'].dt.days>365)]


# ### Get Rid of APNs for which home improvement took place

# In[51]:


#####ind apns for which livable square footage, pool area, and garage or patio or bathrooms changed
##sqft
home_sqft_lookup=master.groupby(['APN','sq_ft'],as_index=False).agg({'sale_price':max})
duplicate_sqft=home_sqft_lookup.groupby(['APN'],as_index=False).count()
duplicate_sqft_homes=duplicate_sqft[duplicate_sqft['sq_ft']>1]['APN'].unique()

##pool
home_pool_lookup=master.groupby(['APN','pool'],as_index=False).agg({'sale_price':max})
duplicate_pool=home_pool_lookup.groupby(['APN'],as_index=False).count()
duplicate_pool_homes=duplicate_pool[duplicate_pool['pool']>1]['APN'].unique()

###garage
home_garage_lookup=master.groupby(['APN','garage_type'],as_index=False).agg({'sale_price':max})
duplicate_garage=home_garage_lookup.groupby(['APN'],as_index=False).count()
duplicate_garage_homes=duplicate_garage[duplicate_garage['garage_type']>1]['APN'].unique()


###patio
home_patio_lookup=master.groupby(['APN','patio_type'],as_index=False).agg({'sale_price':max})
duplicate_patio=home_patio_lookup.groupby(['APN'],as_index=False).count()
duplicate_patio_homes=duplicate_patio[duplicate_patio['patio_type']>1]['APN'].unique()

###bathrooms
home_bathroom_lookup=master.groupby(['APN','bathroom'],as_index=False).agg({'sale_price':max})
duplicate_bathroom=home_bathroom_lookup.groupby(['APN'],as_index=False).count()
duplicate_bathroom_homes=duplicate_bathroom[duplicate_bathroom['bathroom']>1]['APN'].unique()


###roofs
home_roof_lookup=master.groupby(['APN','roof'],as_index=False).agg({'sale_price':max})
duplicate_roof=home_roof_lookup.groupby(['APN'],as_index=False).count()
duplicate_roof_homes=duplicate_roof[duplicate_roof['roof']>1]['APN'].unique()

####find parcels with duplicate ages (tear down properties) and remove

home_age_lookup=master.groupby(['APN','age'],as_index=False).agg({'sale_price':max})
duplicate_ages=home_age_lookup.groupby(['APN'],as_index=False).count()
double_age_homes=duplicate_ages[duplicate_ages['age']>1]['APN'].unique()

####drop parcels for which a new home was built on in the sample period
master=master[~(master['APN'].isin(duplicate_sqft_homes))]
master=master[~(master['APN'].isin(duplicate_pool_homes))]
master=master[~(master['APN'].isin(duplicate_garage_homes))]
master=master[~(master['APN'].isin(duplicate_patio_homes))]
master=master[~(master['APN'].isin(duplicate_bathroom_homes))]
master=master[~(master['APN'].isin(duplicate_roof_homes))]
master=master[~(master['APN'].isin(double_age_homes))]


# ### Drop Duplicate APNs

# In[52]:


#### Drop Duplicates
master_lm=master.drop_duplicates(subset=['APN','sale_year'])


# ### Clean home prices 

# In[53]:


#transform home prices to 2025 dollars
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
master_lm["sale_price_25"] = (
    master_lm["sale_price"]
    * master_lm["sale_year"].map(multipliers)
)

# annualize sale price (multiply by 0.11), following Poterba (1991)
master_lm['ann_sale_price_25']=master_lm["sale_price_25"]*.11

########take natural log of 2010 annual sale price
master_lm['lnp_ann_25']=np.log(master_lm['ann_sale_price_25'])

#####create quantiles
master_lm['sale_quantile'] = pd.qcut(master_lm['lnp_ann_25'],100, 
                               labels = False)
####drop top 1% and bottom 1% of sale prices
master_99=master_lm[(master_lm['sale_quantile']>0)&(master_lm['sale_quantile']<99)]

#####create quantiles
master_99['sale_decile_25'] = pd.qcut(master_99['ann_sale_price_25'],20, 
                               labels = False)


# ### Make some extra variables for regressions

# In[54]:


# --- sale_quarter ---
# default = 1
master_99["sale_quarter"] = 1

master_99.loc[master_99["sale_month"].isin([4, 5, 6]),  "sale_quarter"] = 2
master_99.loc[master_99["sale_month"].isin([7, 8, 9]),  "sale_quarter"] = 3
master_99.loc[master_99["sale_month"].isin([10, 11, 12]), "sale_quarter"] = 4


# --- seasonal dummies ---

master_99["summer_sale"] = 0
master_99.loc[master_99["sale_month"].isin([7, 8, 9]), "summer_sale"] = 1

master_99["fall_sale"] = 0
master_99.loc[master_99["sale_month"].isin([10, 11, 12]), "fall_sale"] = 1

master_99["winter_sale"] = 0
master_99.loc[master_99["sale_month"].isin([1, 2, 3]), "winter_sale"] = 1

master_99["spring_sale"] = 0
master_99.loc[master_99["sale_month"].isin([4, 5, 6]), "spring_sale"] = 1


# --- amenity dummies ---

# pool > 10
master_99["has_pool"] = 0
master_99.loc[master_99["pool"] > 10, "has_pool"] = 1

# patio_type < 9
master_99["has_patio"] = 0
master_99.loc[master_99["patio_type"] < 9, "has_patio"] = 1

# garage_type < 9
master_99["has_garage"] = 0
master_99.loc[master_99["garage_type"] < 9, "has_garage"] = 1

# --- roof type dummies ---
# wood_roof: roof == 0
master_99["wood_roof"] = 0
master_99.loc[master_99["roof"] == 0, "wood_roof"] = 1

# shingle_roof: roof == 1 or 2
master_99["shingle_roof"] = 0
master_99.loc[master_99["roof"].isin([1, 2]), "shingle_roof"] = 1

# tile_roof: roof == 8 or 4 or 6
master_99["tile_roof"] = 0
master_99.loc[master_99["roof"].isin([8, 4, 6]), "tile_roof"] = 1

# hot_roof: roof == 1 or 7
master_99["hot_roof"] = 0
master_99.loc[master_99["roof"].isin([1, 7]), "hot_roof"] = 1


# --- old home dummy ---
master_99["old_home"] = 0
master_99.loc[master_99["home_age"] >= 65, "old_home"] = 1


# ### Export final sold homes sample

# In[56]:


master_99.to_csv('%s/urban_iv_paper_input_data_11_20/sold_homes_sample_11_20.csv'%path)


# In[ ]:




