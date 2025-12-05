#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import geopandas as gpd
import scipy.stats
import warnings
import mapclassify as mc
import statsmodels.api as sm
from urllib.request import urlopen
import json
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from shapely.geometry import MultiPolygon, Polygon
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from rasterio.sample import sample_gen
from collections import Counter
from rasterstats import zonal_stats
from shapely.geometry import Point

# Ignore warnings
warnings.filterwarnings('ignore')

##define path
path='/Users/hannahkamen/Downloads'
# Load Maricopa County boundary (replace with correct path if needed)
maricopa_gdf = gpd.read_file("%s/maricopa/maricopa.shp"%path)
#get zips
zips=gpd.read_file('%s/Maricopa_County_Zip_Codes/ZipCodes.shp'%path)
pumas=gpd.read_file('%s/tl_2021_04_puma10/tl_2021_04_puma10.shp'%path)
tracts=gpd.read_file('%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp'%path)
subdivisions=gpd.read_file("%s/Subdivisions/Subdivisions.shp" % path)


# ### load econometric sample

# In[3]:


### load econometric sample
sample=pd.read_stata("/Users/hannahkamen/Downloads/urban_iv_paper_input_data_11_20/urban_heat_iv_11_20_500_5500_11_22.dta")


# ### Import tract, block group, parcel and cool corridor shapefiles


# Load Maricopa County tract shapefile
tracts = gpd.read_file("%s/tl_2010_04013_tract10/tl_2010_04013_tract10.shp" % path)
##a circle with a buffer of 1500m is 3,140,000
tracts=tracts.to_crs('EPSG:26912')
tracts['area']=tracts.geometry.area
tracts['TRACTCE10']=tracts['TRACTCE10'].astype(int)

# load maricopa county block group shape file
block_groups = gpd.read_file(f"{path}/tl_2010_04013_bg10/tl_2010_04013_bg10.shp")

# TIGER 2010 BG shapefiles usually have 12-digit GEOID10 already
block_groups["bg_geoid"] = (
    block_groups["GEOID10"]
    .astype(str)
    .str.strip()
    .str[-12:]     # also force 12-char from the right, just to be safe
)

### parcels shapefile
parcels=gpd.read_file("%s/parcels_by_year/Parcels_-_Maricopa_County%%2C_Arizona_(2019).shp"%path)
# keep only parcels that appear in sample
apn_sample = sample["APN"].unique()
parcels_samp = parcels[parcels["APN"].isin(apn_sample)].copy()

cc=gpd.read_file("/Users/hannahkamen/Downloads/phoenix_cool_corridors.shp")
cc=cc.to_crs('EPSG:26912')


len(cc)


# ### Load maricopa county income by block group file and merge to block group shapefile


input_path = "/Users/hannahkamen/Downloads/nhgis0023_csv/nhgis0023_ds233_20175_blck_grp.csv"

use_cols = ["COUNTYA", "GEOID", "AH1PE001"]
df = pd.read_csv(
    input_path,
    usecols=use_cols,
    dtype={"COUNTYA": str, "GEOID": str}
)

# Filter to Maricopa (013)
df["COUNTYA"] = df["COUNTYA"].str.zfill(3)
df_013 = df[df["COUNTYA"] == "013"].copy()

# Clean join key: take LAST 12 characters from GEOID
df_013["bg_geoid"] = df_013["GEOID"].astype(str).str.strip().str[-12:]

# Make sure income is numeric
df_013["medhhinc_2017adj"] = pd.to_numeric(df_013["AH1PE001"], errors="coerce")

# Merge income onto block_groups
block_groups = block_groups.merge(
    df_013[["bg_geoid", "medhhinc_2017adj"]],
    on="bg_geoid",
    how="left"
)

# merge income onto tract shapefile, calculate weighted mean median tract income 


# ### Merge block group income onto tract and calculate area weighted mean median tract income


# ------------------------------------------------------------
# 1. Extract tract ID from NHGIS GEOID (for reference)
#    Example GEOID: "15000US040130101011"
#    Last 12 chars:  "040130101011"
#    Positions 5:11 of those 12 chars = 6-digit tract code ("010101")
# ------------------------------------------------------------
bg12 = df_013["GEOID"].astype(str).str.strip().str[-12:]
df_013["tract_code6"] = bg12.str[5:11]          # 6-digit tract code
df_013["tract_int"]   = df_013["tract_code6"].astype(int)

# ------------------------------------------------------------
# 2. Attach NHGIS block-group income to block_groups shapefile
# ------------------------------------------------------------

# Ensure a clean block-group ID on both
df_013["bg_geoid"] = df_013["GEOID"].astype(str).str.strip().str[-12:]
block_groups["bg_geoid"] = (
    block_groups["GEOID10"].astype(str).str.strip().str[-12:]
)

# Only merge if the income column isn't already present
if "medhhinc_2017adj" not in block_groups.columns:
    block_groups = block_groups.merge(
        df_013[["bg_geoid", "medhhinc_2017adj"]],
        on="bg_geoid",
        how="left"
    )

# ------------------------------------------------------------
# 3. Reproject block_groups to match tracts CRS and compute BG area
# ------------------------------------------------------------
block_groups = block_groups.to_crs(tracts.crs)   # tracts already in EPSG:26912
block_groups["bg_area"] = block_groups.geometry.area

# Make sure tract codes match type with tracts
block_groups["TRACTCE10"] = block_groups["TRACTCE10"].astype(int)

# ------------------------------------------------------------
# 4. Bring tract area onto block_groups and compute area shares
# ------------------------------------------------------------

# tract area is already in tracts['area']
tract_area_df = (
    tracts[["TRACTCE10", "area"]]
    .rename(columns={"area": "tract_area"})
)

block_groups = block_groups.merge(
    tract_area_df,
    on="TRACTCE10",
    how="left"
)

# Drop any rows without income or tract_area (should be rare)
bg_valid = block_groups.dropna(subset=["medhhinc_2017adj", "tract_area"]).copy()

# Share of each tract’s area accounted for by each block group
bg_valid["area_share_in_tract"] = bg_valid["bg_area"] / bg_valid["tract_area"]

# ------------------------------------------------------------
# 5. Area-weighted median income at tract level
# ------------------------------------------------------------

# For each tract: sum( median_income_bg * area_share_in_tract )
tract_income_aw = (
    bg_valid
    .groupby("TRACTCE10")
    .apply(lambda d: (d["medhhinc_2017adj"] * d["area_share_in_tract"]).sum())
    .reset_index(name="medhhinc_aw")
)


# ### Import and Process existing tree coverage data

# In[8]:


# --- 0. Path
raster_path = f"{path}/land_cover_5m_2015_masked.tif"

# open original raster
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

    # --- 1. Reproject vectors to raster CRS ---
    if parcels_samp.crs != raster_crs:
        parcels_samp = parcels_samp.to_crs(raster_crs)
    if tracts.crs != raster_crs:
        tracts = tracts.to_crs(raster_crs)
    if cc.crs != raster_crs:
        cc = cc.to_crs(raster_crs)

    # --- 2. Build 1500m buffer around cool corridors and clip raster ---
    # (assumes raster CRS is in meters; if not, you should reproject first)
    corridors_buffer = cc.unary_union.buffer(1500)

    # Clip raster to corridors_buffer extent (+1500m)
    out_image, out_transform = mask(src, [corridors_buffer], crop=True)
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

# Save clipped raster to disk (much smaller than full city raster)
clipped_raster_path = f"{path}/land_cover_1m_2015_coolcorridors_1500mclip.tif"
with rasterio.open(clipped_raster_path, "w", **out_meta) as dst:
    dst.write(out_image)

# --- 3. Helper to compute % tree cover within a geometry ---
def pct_tree_cover_area(geom, raster, tree_value=4):
    """
    Computes percent area in 'geom' covered by tree pixels (value tree_value)
    using a clipped 1m raster. Assumes raster CRS is in meters.
    """
    if geom.is_empty:
        return 0.0

    out, _ = mask(raster, [geom], crop=True)
    arr = out[0]

    nodata = raster.nodata
    if nodata is not None:
        valid = arr != nodata
    else:
        # if nodata is not defined, treat all cells as potentially valid
        valid = np.ones_like(arr, dtype=bool)

    # tree pixels within valid region
    tree_mask = (arr == tree_value) & valid

    # each pixel area (e.g., 1m x 1m)
    pixel_area = abs(raster.res[0] * raster.res[1])
    tree_area = tree_mask.sum() * pixel_area

    geom_area = geom.area
    if geom_area == 0:
        return 0.0
    return tree_area / geom_area

# --- 4. Open clipped raster and compute metrics ---
with rasterio.open(clipped_raster_path) as lc_src:

    # 4a. Tract-level % tree cover (2015) for tracts intersecting the buffered corridors
    tracts_cc = tracts[tracts.intersects(corridors_buffer)].copy()
    tracts_cc["pct_tree_2015"] = tracts_cc.geometry.apply(
        lambda g: pct_tree_cover_area(g, lc_src, tree_value=4)
    )

    # 4b. Parcel-level % tree in 0–500m and 500–1500m rings
    parcels_cc = parcels_samp[parcels_samp.intersects(corridors_buffer)].copy()

    pct_tree_0_500 = []
    pct_tree_500_1500 = []

    for geom in parcels_cc.geometry:
        buf_500 = geom.buffer(500)       # 0–500m buffer
        buf_1500 = geom.buffer(1500)     # 0–1500m buffer
        ring_500_1500 = buf_1500.difference(buf_500)  # 500–1500m ring

        pct_tree_0_500.append(
            pct_tree_cover_area(buf_500, lc_src, tree_value=4)
        )
        pct_tree_500_1500.append(
            pct_tree_cover_area(ring_500_1500, lc_src, tree_value=4)
        )

    parcels_cc["pct_tree_0_500m_2015"] = pct_tree_0_500
    parcels_cc["pct_tree_500_1500m_2015"] = pct_tree_500_1500

# --- 5. Save outputs ---
# tracts_cc.to_file(f"{path}/tracts_with_pct_tree_2015_coolcorridors.shp")
# parcels_cc.to_file(f"{path}/parcels_samp_pct_tree_buffers_2015_coolcorridors.shp")

tracts_cc[['TRACTCE10','pct_tree_2015']].to_csv(f"{path}/tracts_with_pct_tree_2015_coolcorridors.csv", index=False)
parcels_cc[['APN','pct_tree_0_500m_2015','pct_tree_500_1500m_2015']].to_csv(f"{path}/parcels_samp_pct_tree_buffers_2015_coolcorridors.csv", index=False)

tracts_tree_cover=tracts_cc[['TRACTCE10','pct_tree_2015']]
parcels_tree_cover=parcels_cc[['APN','pct_tree_0_500m_2015','pct_tree_500_1500m_2015']]


# ### Begin tract analysis

# In[9]:


# pick a working CRS in meters
target_crs = "EPSG:32612"  # UTM zone 12N (covers Phoenix)

cc      = cc.to_crs(target_crs)
parcels = parcels.to_crs(target_crs)
tracts  = tracts.to_crs(target_crs)


# In[10]:


# Make sure APN types are compatible
sample["APN"]  = sample["APN"].astype(str)
parcels["APN"] = parcels["APN"].astype(str)

# Merge: one row per sale, carrying parcel geometry
sample_gdf = sample.merge(
    parcels[["APN", "geometry"]],
    on="APN",
    how="left",
)

sample_gdf=sample_gdf.drop_duplicates(subset=['APN'])

# Turn into GeoDataFrame
sample_gdf = gpd.GeoDataFrame(sample_gdf, geometry="geometry", crs=parcels.crs)

##merge tracts

# keep only tract ID + geometry (adjust ID field name to your file)
tract_id_col = "TRACTCE10"  # or whatever your tract identifier is

tracts_sub = tracts.merge(tract_income_aw,on=tract_id_col)[[tract_id_col, "geometry","medhhinc_aw"]]

# spatial join: point-in-polygon (or polygon-in-polygon) sale → tract
del sample_gdf['level_0']
sample_gdf = gpd.sjoin(sample_gdf, tracts_sub, how="left", predicate="intersects")
# this adds a 'index_right' column; we don't need it
sample_gdf = sample_gdf.drop(columns=["index_right"])



# ### Define treel cooling analysis assumption params

# In[12]:


# constants
TREES_PER_MILE = 200
M_PER_MILE     = 1609.34
A_TREE         = 30.0   # m^2 canopy per mature tree (assumption)
BETA           = 0.14   # °C per 1% canopy (Middel / Cool Urban Spaces)


# ensure tract id col name matchesshapefile
tract_id_col = "TRACTCE10"  # adjust if needed
tracts[tract_id_col] = tracts[tract_id_col].astype(str)

# intersect tracts with corridors
tracts_for_intersect = tracts[[tract_id_col, "geometry"]]

cc_in_tracts = gpd.overlay(
    tracts_for_intersect,
    cc[["geometry"]],
    how="intersection"
)

# length of corridor segments inside each tract (meters)
cc_in_tracts["seg_len_m"] = cc_in_tracts.geometry.length

len_by_tract = (
    cc_in_tracts
    .groupby(tract_id_col, as_index=False)["seg_len_m"]
    .sum()
)

# total corridor length in miles and tree counts per tract
len_by_tract["seg_len_miles"] = len_by_tract["seg_len_m"] / M_PER_MILE
len_by_tract["trees_corridor"] = len_by_tract["seg_len_miles"] * TREES_PER_MILE

# tract area (m^2)
tract_area = (
    tracts
    .set_index(tract_id_col)["geometry"]
    .area
    .rename("tract_area_m2")
    .reset_index()
)

# merge area
tract_metrics = len_by_tract.merge(tract_area, on=tract_id_col, how="right")

# if a tract has no corridor, seg_len_m will be NaN -> treat as 0
tract_metrics["seg_len_m"] = tract_metrics["seg_len_m"].fillna(0)
tract_metrics["seg_len_miles"] = tract_metrics["seg_len_miles"].fillna(0)
tract_metrics["trees_corridor"] = tract_metrics["trees_corridor"].fillna(0)

## join with existing tree cover
tract_metrics['TRACTCE10']=tract_metrics['TRACTCE10'].astype(int)
tract_metrics=tract_metrics.merge(tracts_tree_cover,on="TRACTCE10")

# canopy area and canopy percentage change due to corridors
tract_metrics["canopy_area_m2"] = tract_metrics["trees_corridor"] * A_TREE
tract_metrics["delta_canopy_pct_tract"] = 100*(((tract_metrics["canopy_area_m2"] / tract_metrics["tract_area_m2"]) + tract_metrics["pct_tree_2015"])-tract_metrics["pct_tree_2015"])

# ΔT at tract level (°C)
tract_metrics["delta_T_tract"] = BETA * tract_metrics["delta_canopy_pct_tract"]

# ------------------------------------------------------------
# attach area-weighted tract income back to tracts GeoDataFrame
# ------------------------------------------------------------
tract_metrics = tract_metrics.merge(tract_income_aw, on="TRACTCE10", how="left")


# ensure tract id col name matchesshapefile
tract_id_col = "TRACTCE10"  # adjust if needed
tracts[tract_id_col] = tracts[tract_id_col].astype(str)

# intersect tracts with corridors
tracts_for_intersect = tracts[[tract_id_col, "geometry"]]

cc_in_tracts = gpd.overlay(
    tracts_for_intersect,
    cc[["geometry"]],
    how="intersection"
)

# length of corridor segments inside each tract (meters)
cc_in_tracts["seg_len_m"] = cc_in_tracts.geometry.length

len_by_tract = (
    cc_in_tracts
    .groupby(tract_id_col, as_index=False)["seg_len_m"]
    .sum()
)

# total corridor length in miles and tree counts per tract
len_by_tract["seg_len_miles"] = len_by_tract["seg_len_m"] / M_PER_MILE
len_by_tract["trees_corridor"] = len_by_tract["seg_len_miles"] * TREES_PER_MILE

# tract area (m^2)
tract_area = (
    tracts
    .set_index(tract_id_col)["geometry"]
    .area
    .rename("tract_area_m2")
    .reset_index()
)

# merge area
tract_metrics = len_by_tract.merge(tract_area, on=tract_id_col, how="right")

# if a tract has no corridor, seg_len_m will be NaN -> treat as 0
tract_metrics["seg_len_m"] = tract_metrics["seg_len_m"].fillna(0)
tract_metrics["seg_len_miles"] = tract_metrics["seg_len_miles"].fillna(0)
tract_metrics["trees_corridor"] = tract_metrics["trees_corridor"].fillna(0)

## join with existing tree cover
tract_metrics['TRACTCE10']=tract_metrics['TRACTCE10'].astype(int)
tract_metrics=tract_metrics.merge(tracts_tree_cover,on="TRACTCE10")

# canopy area and canopy percentage change due to corridors
tract_metrics["canopy_area_m2"] = tract_metrics["trees_corridor"] * A_TREE
tract_metrics["delta_canopy_pct_tract"] = 100*(((tract_metrics["canopy_area_m2"] / tract_metrics["tract_area_m2"]) + tract_metrics["pct_tree_2015"])-tract_metrics["pct_tree_2015"])

# ΔT at tract level (°C)
tract_metrics["delta_T_tract"] = BETA * tract_metrics["delta_canopy_pct_tract"]

# ------------------------------------------------------------
# attach area-weighted tract income back to tracts GeoDataFrame
# ------------------------------------------------------------
tract_metrics = tract_metrics.merge(tract_income_aw, on="TRACTCE10", how="left")



### BUILD RING GEODATAFRAMES AROUND EACH PARCEL

# 1) Make sure cc is in a metric CRS
target_crs = "EPSG:32612"   # or whatever you're using
cc = cc.to_crs(target_crs)
parcels_samp = parcels_samp.to_crs(target_crs)

# 2) Build rings *after* parcels are projected
parcels_samp["home_pt"] = parcels_samp.geometry.representative_point()
home_pts = gpd.GeoDataFrame(
    parcels_samp[["APN"]],
    geometry=parcels_samp["home_pt"],
    crs=parcels_samp.crs
)

# buffers
buf500  = home_pts.geometry.buffer(500)   # 0–500 m
buf1500 = home_pts.geometry.buffer(1500)  # 0–1500 m

# build rings as polygons
ring0_500_geom   = buf500
ring500_1500_geom = buf1500.difference(buf500)
# ring500_1500_geom = buf1500

# GeoDataFrames of rings
ring0_500 = gpd.GeoDataFrame(
    {"APN": parcels_samp["APN"]},
    geometry=ring0_500_geom,
    crs=parcels_samp.crs,
)

ring500_1500 = gpd.GeoDataFrame(
    {"APN": parcels_samp["APN"]},
    geometry=ring500_1500_geom,
    crs=parcels_samp.crs,
)

# area of rings (m^2)
ring0_500["area_m2"]   = ring0_500.geometry.area
ring500_1500["area_m2"] = ring500_1500.geometry.area

area_crs = "EPSG:26912"

bg_proj    = block_groups.to_crs(area_crs)
ring0_proj = ring0_500.to_crs(area_crs)
ring1_proj = ring500_1500.to_crs(area_crs)

def add_area_weighted_income(rings_gdf_proj, bg_gdf_proj, ring_name):
    inter = gpd.overlay(
        rings_gdf_proj[["APN", "geometry"]],
        bg_gdf_proj[["bg_geoid", "medhhinc_2017adj", "geometry"]],
        how="intersection"
    )

    inter = inter.dropna(subset=["medhhinc_2017adj"])
    if inter.empty:
        colname = f"median_weighted_income_{ring_name}"
        return pd.DataFrame({"APN": [], colname: []})

    inter["area_int"] = inter.geometry.area
    inter["inc_x_area"] = inter["medhhinc_2017adj"] * inter["area_int"]

    agg = (
        inter
        .groupby("APN", as_index=False)
        .agg(
            inc_area_sum=("inc_x_area", "sum"),
            area_sum=("area_int", "sum")
        )
    )

    colname = f"median_weighted_income_{ring_name}"
    agg[colname] = np.where(
        agg["area_sum"] > 0,
        agg["inc_area_sum"] / agg["area_sum"],
        np.nan
    )
    return agg[["APN", colname]]

income0 = add_area_weighted_income(ring0_proj, bg_proj, ring_name="0_500")
income1 = add_area_weighted_income(ring1_proj, bg_proj, ring_name="500_1500")

ring0_500 = ring0_500.merge(income0, on="APN", how="left")
ring500_1500 = ring500_1500.merge(income1, on="APN", how="left")

print(ring0_500["median_weighted_income_0_500"].describe())
print(ring500_1500["median_weighted_income_500_1500"].describe())


# 0–500 m ring ∩ corridors
cc_ring0 = gpd.overlay(
    ring0_500.to_crs(cc.crs)[["APN", "geometry"]],
    cc[["geometry"]],
    how="intersection"
)
cc_ring0["seg_len_m"] = cc_ring0.geometry.length

len0_by_parcel = (
    cc_ring0
    .groupby("APN", as_index=False)["seg_len_m"]
    .sum()
    .rename(columns={"seg_len_m": "len_0_500_m"})
)

# 500–1500 m ring ∩ corridors
cc_ring1 = gpd.overlay(
    ring500_1500.to_crs(cc.crs)[["APN", "geometry"]],
    cc[["geometry"]],
    how="intersection"
)
cc_ring1["seg_len_m"] = cc_ring1.geometry.length

len1_by_parcel = (
    cc_ring1
    .groupby("APN", as_index=False)["seg_len_m"]
    .sum()
    .rename(columns={"seg_len_m": "len_500_1500_m"})
)


# base table with ring areas
parcel_ring_metrics = (
    ring0_500[["APN", "area_m2","median_weighted_income_0_500"]]
    .rename(columns={"area_m2": "area_0_500_m2"})
    .merge(
        ring500_1500[["APN", "area_m2","median_weighted_income_500_1500"]].rename(columns={"area_m2": "area_500_1500_m2"}),
        on="APN",
        how="left",
    )
)

# merge corridor lengths (might be missing -> 0)
parcel_ring_metrics = parcel_ring_metrics.merge(len0_by_parcel, on="APN", how="left")
parcel_ring_metrics = parcel_ring_metrics.merge(len1_by_parcel, on="APN", how="left")

parcel_ring_metrics["len_0_500_m"]   = parcel_ring_metrics["len_0_500_m"].fillna(0)
parcel_ring_metrics["len_500_1500_m"] = parcel_ring_metrics["len_500_1500_m"].fillna(0)

# convert to miles
parcel_ring_metrics["len_0_500_miles"]   = parcel_ring_metrics["len_0_500_m"] / M_PER_MILE
parcel_ring_metrics["len_500_1500_miles"] = parcel_ring_metrics["len_500_1500_m"] / M_PER_MILE

# trees per ring (baseline canopy along corridor = 0)
parcel_ring_metrics["trees_0_500"]   = parcel_ring_metrics["len_0_500_miles"]   * TREES_PER_MILE
parcel_ring_metrics["trees_500_1500"] = parcel_ring_metrics["len_500_1500_miles"] * TREES_PER_MILE

# canopy area per ring due to Cool Corridors
parcel_ring_metrics["canopy_area_0_500_m2"]   = parcel_ring_metrics["trees_0_500"]   * A_TREE
parcel_ring_metrics["canopy_area_500_1500_m2"] = parcel_ring_metrics["trees_500_1500"] * A_TREE

# join with existing tree cover prior to cool corridor

parcel_ring_metrics=parcel_ring_metrics.merge(parcels_tree_cover,on='APN')

# Δcanopy% per ring
parcel_ring_metrics["delta_canopy_pct_0_500"] = 100*(((parcel_ring_metrics["canopy_area_0_500_m2"] / parcel_ring_metrics["area_0_500_m2"]) + 
                                                      parcel_ring_metrics["pct_tree_0_500m_2015"])-parcel_ring_metrics["pct_tree_0_500m_2015"])

parcel_ring_metrics["delta_canopy_pct_500_1500"]= 100*(((parcel_ring_metrics["canopy_area_500_1500_m2"] / parcel_ring_metrics["area_500_1500_m2"]) + 
                                                       parcel_ring_metrics["pct_tree_500_1500m_2015"])-parcel_ring_metrics["pct_tree_500_1500m_2015"])


# ΔT per ring (°C)
parcel_ring_metrics["delta_T_0_500"] = BETA * parcel_ring_metrics["delta_canopy_pct_0_500"]
parcel_ring_metrics["delta_T_500_1500"] = BETA * parcel_ring_metrics["delta_canopy_pct_500_1500"]


# ensure matching types
sample["TRACTCE10"] = sample["TRACTCE10"].astype(str)

# keep only relevant columns from metrics tables
tract_metrics_small = tract_metrics[[tract_id_col, "delta_T_tract","medhhinc_aw"]]

# merge parcel ring metrics -> sample by APN
sample_enriched = sample.merge(
    parcel_ring_metrics[
        ["APN", "delta_T_0_500", "delta_T_500_1500","median_weighted_income_0_500","median_weighted_income_500_1500"]
    ],
    on="APN",
    how="left"
)

sample_enriched['TRACTCE10']=sample_enriched['TRACTCE10'].astype(int)
# merge tract-level ΔT -> sample by TRACTCE10
sample_enriched = sample_enriched.merge(
    tract_metrics_small,
    left_on="TRACTCE10",
    right_on=tract_id_col,
    how="left",
)



# ### Calculate cooling benefits to tracts from cool corridors

# In[15]:


# --------------------------------------------------
# 1. Recompute benefits cleanly & drop duplicate APNs
# --------------------------------------------------
# Drop duplicate APNs (one observation per home)
sample_enriched = sample_enriched.drop_duplicates(subset=["APN"]).copy()

# Hedonic coefficients (per °C)
beta_tract      = 0.1067   # tract-level model
beta_0_500      = 0.0537   # spatial lag (0–500 m)
beta_500_1500   = 0.1567   # spatial lag (500–1500 m)

# ---------- canopy growth + discounting parameters ----------

canopy_start     = 8.0   # m² at program start (year 1)
canopy_increment = 3.0    # m² additional canopy per year
T_HORIZON_YEARS  = 5

# vector of years 1..5
years = np.arange(1, T_HORIZON_YEARS + 1)

# canopy per tree in each year: 10, 13, 16, 19, 22 m²
canopy_per_year = canopy_start + canopy_increment * (years - 1)

# public discount rate (real) – 3% per year (OMB/EPA-style)
discount_rate = 0.03
discount_factors = 1.0 / (1.0 + discount_rate) ** years

# this scalar multiplies the one-year "mature" benefit to get 5-year PV
# we assume delta_T is linear in canopy area, so scale by canopy_t / A_TREE_BASE
pv_multiplier = np.sum(discount_factors * (canopy_per_year / A_TREE_BASE))

print("Canopy per year (m²):", canopy_per_year)
print("Discount factors:", discount_factors)
print("PV multiplier applied to one-year benefits:", pv_multiplier)

# --------------------------------------------------
# 2. One-year (full-canopy) benefits at tract + spatial-lag scales
# --------------------------------------------------

# tract-based one-year benefit (your existing static calc)
sample_enriched["benefit_simple_1yr"] = (
    beta_tract  * sample_enriched["sale_price_25"]*.11 * sample_enriched["delta_T_tract"]
)

# spatial-lag one-year benefits for inner and outer rings
sample_enriched["benefit_0_500_1yr"] = (
    beta_0_500  * sample_enriched["sale_price_25"]*.11  * sample_enriched["delta_T_0_500"]
)

sample_enriched["benefit_500_1500_1yr"] = (
    beta_500_1500  * sample_enriched["sale_price_25"] *.11 * sample_enriched["delta_T_500_1500"]
)

sample_enriched["benefit_spatial_lag_1yr"] = (
    sample_enriched["benefit_0_500_1yr"] + sample_enriched["benefit_500_1500_1yr"]
)

# --------------------------------------------------
# 3. Convert 1-year benefits to 5-year discounted PV
# --------------------------------------------------

# PV over 5 years for each home, at tract scale
sample_enriched["benefit_simple_PV5"] = (
    sample_enriched["benefit_simple_1yr"] * pv_multiplier
)

# PV over 5 years for each home, at spatial-lag scale
sample_enriched["benefit_0_500_PV5"] = (
    sample_enriched["benefit_0_500_1yr"] * pv_multiplier
)
sample_enriched["benefit_500_1500_PV5"] = (
    sample_enriched["benefit_500_1500_1yr"] * pv_multiplier
)
sample_enriched["benefit_spatial_lag_PV5"] = (
    sample_enriched["benefit_0_500_PV5"] + sample_enriched["benefit_500_1500_PV5"]
)

# --------------------------------------------------
# 4. Summary stats: totals, per-household PV, comparison
# --------------------------------------------------

# Total PV benefits
slb_PV = sample_enriched["benefit_spatial_lag_PV5"].sum()
tb_PV  = sample_enriched["benefit_simple_PV5"].sum()

pct_diff_PV = (slb_PV - tb_PV) / tb_PV * 100.0
# print("Percent difference in PV valuation (spatial-lag vs tract):", pct_diff_PV)

# Households benefiting in each band (based on any positive PV benefit)
num_500 = (sample_enriched["benefit_0_500_PV5"] > 0).sum()
num_1500 = (sample_enriched["benefit_500_1500_PV5"] > 0).sum()
num_sl = num_500 + num_1500

b_per_hh_sl      = slb_PV / num_sl if num_sl > 0 else np.nan
b_per_hh_sl_500  = sample_enriched["benefit_0_500_PV5"].sum() / num_500 if num_500 > 0 else np.nan
b_per_hh_sl_1500 = sample_enriched["benefit_500_1500_PV5"].sum() / num_1500 if num_1500 > 0 else np.nan

# tract-level households benefiting
num_tract = (sample_enriched["benefit_simple_PV5"] > 0).sum()
b_per_hh_t = tb_PV / num_tract if num_tract > 0 else np.nan

print("Total 5-year benefits: Spatial lag level", slb_PV )
print("Total 5-year benefits: Tract level", tb_PV )

print("Number hh benefiting spatial lag (any band):", num_sl)
print("Number hh benefiting 0–500 m:", num_500)
print("Number hh benefiting 500–1500 m:", num_1500)
print("PV benefits per hh, spatial lag (all bands):", b_per_hh_sl)
print("PV benefits per hh, 0–500 m:", b_per_hh_sl_500)
print("PV benefits per hh, 500–1500 m:", b_per_hh_sl_1500)

print("Number hh benefiting tract:", num_tract)
print("PV benefits per hh, tract:", b_per_hh_t)



# In[16]:


# --------------------------------------------------
# 5. Build LaTeX table from computed quantities
# --------------------------------------------------

# Totals for each spatial band
total_0_500_PV     = sample_enriched["benefit_0_500_PV5"].sum()
total_500_1500_PV  = sample_enriched["benefit_500_1500_PV5"].sum()

# --------- helper formatting ---------
tb_PV_fmt            = f"{tb_PV:,.0f}"
slb_PV_fmt           = f"{slb_PV:,.0f}"
total_0_500_PV_fmt   = f"{total_0_500_PV:,.0f}"
total_500_1500_PV_fmt= f"{total_500_1500_PV:,.0f}"

num_tract_fmt        = f"{int(num_tract):,d}"
num_500_fmt          = f"{int(num_500):,d}"
num_1500_fmt         = f"{int(num_1500):,d}"
num_sl_fmt           = f"{int(num_sl):,d}"

b_per_hh_t_fmt       = f"{b_per_hh_t:,.0f}"
b_per_hh_sl_fmt      = f"{b_per_hh_sl:,.0f}"
b_per_hh_sl_500_fmt  = f"{b_per_hh_sl_500:,.0f}"
b_per_hh_sl_1500_fmt = f"{b_per_hh_sl_1500:,.0f}"

pct_diff_PV_fmt      = f"{pct_diff_PV:,.1f}"
pv_multiplier_fmt    = f"{pv_multiplier:,.2f}"

# --------- LaTeX table string ---------
latex_table = f"""
\\begin{{table}}[htbp]
    \\centering
    \\caption{{Discounted five-year benefits of the cool corridor program}}
    \\label{{tab:pv_benefits_spatial_vs_tract}}
    \\begin{{tabular}}{{lrrrr}}
        \\hline\\hline
        & Tract-level model 
        & 0--500 m band 
        & 500--1500 m band 
        & Spatial-lag (0--1500 m)\\\\
        & (1) & (2) & (3) & (4)\\\\
        \\hline
        \\noalign{{\\vskip 0.5ex}}
        \\multicolumn{{5}}{{l}}{{\\textit{{Panel A: Total discounted benefits (PV over 5 years)}}}}\\\\[0.5ex]
        Total PV benefits (2010 \\$) 
            & {tb_PV_fmt} 
            & {total_0_500_PV_fmt} 
            & {total_500_1500_PV_fmt} 
            & {slb_PV_fmt} \\\\
        Benefiting households 
            & {num_tract_fmt} 
            & {num_500_fmt} 
            & {num_1500_fmt} 
            & {num_sl_fmt} \\\\
        PV benefit per benefiting household (2010 \\$) 
            & {b_per_hh_t_fmt} 
            & {b_per_hh_sl_500_fmt} 
            & {b_per_hh_sl_1500_fmt} 
            & {b_per_hh_sl_fmt} \\\\[0.5ex]
        \\multicolumn{{5}}{{l}}{{\\textit{{Panel B: Relative valuation}}}}\\\\[0.5ex]
        Spatial-lag vs tract: total PV (\\%) 
            & \\multicolumn{{4}}{{c}}{{{pct_diff_PV_fmt}\\%}} \\\\[0.5ex]
        \\multicolumn{{5}}{{l}}{{\\textit{{Panel C: Growth and discounting assumptions}}}}\\\\[0.5ex]
        Initial canopy area per tree (m$^2$) 
            & \\multicolumn{{4}}{{c}}{{{canopy_start:.1f}}} \\\\
        Annual increase in canopy per tree (m$^2$/year) 
            & \\multicolumn{{4}}{{c}}{{{canopy_increment:.1f}}} \\\\
        Real discount rate 
            & \\multicolumn{{4}}{{c}}{{3\\% per year}} \\\\
        PV multiplier applied to one-year benefits 
            & \\multicolumn{{4}}{{c}}{{{pv_multiplier_fmt}}} \\\\
        \\hline\\hline
    \\end{{tabular}}
\\end{{table}}
"""

print(latex_table)

# Optionally write directly to a .tex file
with open("pv_benefits_table.tex", "w") as f:
    f.write(latex_table)


# ## Process spatial ring data

# ### create spatial ring geodataframes

# In[29]:





# ### Calculate area weighted average of income of block groups overlapping 500m and 500-100m rings

# In[30]:





# ### get segment length of cool corridor overlap by 500m and 500-100m ring

# In[31]:





# ### Calculate canopy and temperature change by 500m and 500-1500m ring

# In[32]:





# ### Merge tract and spatial ring data back onto original parcel sample dataframe

# In[33]:





# ### Calculate total value of cool corridors program according to each measurement strategy

# In[11]:


import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Recompute benefits cleanly & drop duplicate APNs
# --------------------------------------------------
# Drop duplicate APNs (one observation per home)
sample_enriched_lm = sample_enriched.drop_duplicates(subset=["APN"]).copy()

beta_tract      = 0.1067    # tract-level model
beta_0_500      = 0.0537  # spatial lag (0–500m)
beta_500_1500   = 0.1567  # spatial lag (500–1500m)

# tract-based benefit (already in your code)
sample_enriched["benefit_simple"] = (
    beta_tract  * sample_enriched["sale_price_25"] * sample_enriched["delta_T_tract"]
)

# decompose spatial-lag benefit into inner and outer components
sample_enriched["benefit_0_500"] = (
    beta_0_500  * sample_enriched["sale_price_25"]  * sample_enriched["delta_T_0_500"]
)

sample_enriched["benefit_500_1500"] = (
    beta_500_1500  * sample_enriched["sale_price_25"]  * sample_enriched["delta_T_500_1500"]
)

# total spatial-lag benefit is the sum
sample_enriched["benefit_spatial_lag"] = (
    sample_enriched["benefit_0_500"] + sample_enriched["benefit_500_1500"]
)


##### print percent difference in valuation between simple tract calc and spatial lag calc
### total benefits
slb=sample_enriched["benefit_spatial_lag"].sum()
tb=sample_enriched["benefit_simple"].sum()


print("Percent difference in valuation between simple tract calc and spatial lag calc:"+ str(((sample_enriched["benefit_spatial_lag"].sum()-sample_enriched["benefit_simple"].sum())/sample_enriched["benefit_simple"].sum())*100))

### get number of households benefiting from each calc
num_500 = len(sample_enriched[sample_enriched["benefit_0_500"]>0])
num_1500 = len(sample_enriched[sample_enriched["benefit_500_1500"]>0])

num_sl=num_500+ num_1500

b_per_hh_sl=slb/num_sl
b_per_hh_sl_500=sample_enriched["benefit_0_500"].sum()/num_500
b_per_hh_sl_1500=sample_enriched["benefit_500_1500"].sum()/num_1500

### get average benefit per person
num_tract = len(sample_enriched[sample_enriched["benefit_simple"]>0])
b_per_hh_t=tb/num_tract

print("Numebr hh benefiting spatial lag:" + str(num_sl))
print("Numebr hh benefiting spatial lag 500m:" + str(num_500))
print("Numebr hh benefiting spatial lag 1500m:" + str(num_1500))
print("Benefits per hh spatial lag:" + str(b_per_hh_sl))
print("Benefits per hh spatial lag 500m:" + str(b_per_hh_sl_500))
print("Benefits per hh spatial lag 1500m:" + str(b_per_hh_sl_1500))

print("Number hh benefiting tract:" + str(num_tract))
print("Benefits per hh tract" + str(b_per_hh_t))



# In[41]:





# In[40]:


sample_enriched["benefit_simple"].sum()


# In[46]:


import numpy as np
import matplotlib.pyplot as plt

# Make sure benefit_shares is sorted in the order you want on the x-axis
benefit_shares = benefit_shares.copy()

income_groups = benefit_shares.index.astype(str)
x = np.arange(len(income_groups))  # positions on x-axis
width = 0.4                        # width of each bar

fig, ax = plt.subplots(figsize=(8, 5))

# Tract-based benefits
ax.bar(
    x - width/2,
    benefit_shares["pct_of_tract_benefits"],
    width,
    label="Tract-based"
)

# Spatial-lag-based benefits
ax.bar(
    x + width/2,
    benefit_shares["pct_of_spatiallag_benefits"],
    width,
    label="Spatial lag"
)

ax.set_xlabel("Income group")
ax.set_ylabel("Percent of total benefits")
ax.set_title("Distribution of cooling benefits by income group")
ax.set_xticks(x)
ax.set_xticklabels(income_groups, rotation=45, ha="right")
ax.legend()

fig.tight_layout()
plt.show()


# In[368]:


sample_enriched_lm[sample_enriched_lm['delta_T_0_500']>0]['median_weighted_income_0_500'].describe()


# In[370]:


sample_enriched_lm[sample_enriched_lm['delta_T_0_500']>0]['median_weighted_income_500_1500'].describe()


# In[369]:


sample_enriched_lm[sample_enriched_lm['benefit_simple']>0]['median_income'].describe()


# In[303]:


sample_enriched_lm["benefit_spatial_lag"].sum()


# In[305]:


sample_enriched_lm["benefit_simple"].sum()


# In[50]:


parcel_ring_metrics['len_500_1500_m'].describe()


# In[46]:


parcel_ring_metrics['delta_T_500_1500'].describe()


# In[ ]:


# --------------------------------------------------
# 2. Ensure income variables are numeric
# --------------------------------------------------

for col in [
    "medhhinc_aw",                   # tract-level income
    "median_weighted_income_0_500",    # inner ring income (0–500m)
    "median_weighted_income_500_1500"  # outer ring income (500–1500m)
]:
    sample_enriched_lm[col] = pd.to_numeric(sample_enriched_lm[col], errors="coerce")

# --------------------------------------------------
# 3. Define income bins and labels
# --------------------------------------------------

# ABDOE002:    Less than $10,000
# ABDOE003:    $10,000 to $14,999
# ABDOE004:    $15,000 to $19,999

# ABDOE005:    $20,000 to $24,999
# ABDOE006:    $25,000 to $29,999
# ABDOE007:    $30,000 to $34,999
# ABDOE008:    $35,000 to $39,999
# ABDOE009:    $40,000 to $44,999
# ABDOE010:    $45,000 to $49,999
# ABDOE011:    $50,000 to $59,999
# ABDOE012:    $60,000 to $74,999
# ABDOE013:    $75,000 to $99,999
# ABDOE014:    $100,000 to $124,999
# ABDOE015:    $125,000 to $149,999
# ABDOE016:    $150,000 to $199,999
# ABDOE017:    $200,000 or more

bins = [0, 20000, 30000, 40000, 50000, 75000, 100000,125000, np.inf]

labels = [
    "0–20k",
    "20–30k",    
    "30–40k",   
    "40–50k",
    "50–75k",
    "75–100k",
    "100k–125k",
    "125k+"
]

# Tract-level income groups (for benefit_simple)
sample_enriched_lm["tract_inc_group"] = pd.cut(
    sample_enriched_lm["medhhinc_aw"],
    bins=bins,
    labels=labels,
    right=False
)

# Inner-ring income groups (for benefit_0_500)
sample_enriched_lm["ring0_inc_group"] = pd.cut(
    sample_enriched_lm["median_weighted_income_0_500"],
    bins=bins,
    labels=labels,
    right=False
)

# Outer-ring income groups (for benefit_500_1500)
sample_enriched_lm["ring1_inc_group"] = pd.cut(
    sample_enriched_lm["median_weighted_income_500_1500"],
    bins=bins,
    labels=labels,
    right=False
)

# --------------------------------------------------
# 4. Tract-based benefits: share by tract median income
# --------------------------------------------------

tract_benefit_by_group = (
    sample_enriched_lm
    .dropna(subset=["tract_inc_group"])
    .groupby("tract_inc_group")["benefit_simple"]
    .sum()
    .reindex(labels)
    .fillna(0)
)

total_tract_benefit = tract_benefit_by_group.sum()
tract_share = 100 * tract_benefit_by_group / total_tract_benefit

# --------------------------------------------------
# 5. Spatial-lag benefits: share by *ring-level* income
# --------------------------------------------------

# Inner-ring piece (0–500m), grouped by 0–500m ring income
spatial0_by_group = (
    sample_enriched_lm
    .dropna(subset=["ring0_inc_group"])
    .groupby("ring0_inc_group")["benefit_0_500"]
    .sum()
    .reindex(labels)
    .fillna(0)
)

# Outer-ring piece (500–1500m), grouped by 500–1500m ring income
spatial1_by_group = (
    sample_enriched_lm
    .dropna(subset=["ring1_inc_group"])
    .groupby("ring1_inc_group")["benefit_500_1500"]
    .sum()
    .reindex(labels)
    .fillna(0)
)

# Total spatial-lag benefit by income group =
#   inner contribution + outer contribution for that group
spatial_total_by_group = spatial0_by_group + spatial1_by_group
total_spatial_benefit = spatial_total_by_group.sum()
spatial_share = 100 * spatial_total_by_group / total_spatial_benefit

# --------------------------------------------------
# 6. Put results in one table
# --------------------------------------------------

benefit_shares = pd.DataFrame({
    "pct_of_tract_benefits": tract_share,
    "pct_of_spatiallag_benefits": spatial_share
})

print(benefit_shares)

