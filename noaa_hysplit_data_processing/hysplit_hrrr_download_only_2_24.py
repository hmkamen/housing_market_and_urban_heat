#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess
from datetime import datetime, timedelta
path='/Users/hannahkamen/Downloads'


# In[2]:


import pandas as pd
import numpy as np
import scipy.stats
import warnings
from urllib.request import urlopen
import json
from pathlib import Path
import math
from shapely.ops import nearest_points
from tqdm import tqdm 
import rasterio
import os
import subprocess
from pyproj import CRS
from ftplib import FTP
path='/Users/hannahkamen/Downloads'


# In[3]:


# import os
# from ftplib import FTP

# # NOAA READY FTP server
# FTP_SERVER = "arlftp.arlhq.noaa.gov"
# FTP_DIR = "/pub/archives/hrrr.v1/"

# # Local directory to save files
# OUTPUT_DIR = "/Users/hannahkamen/hysplit/working/hrrr"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Define years, months, and sample days
# years = range(2015, 2016)
# months = [ "07" ]
# sample_days = ["01"]
# sample_hours = ['18']

# # Max download size per file (3.49 GB in bytes)
# MAX_FILE_BYTES = 3_480_000_000  # 3.49 GB

# ftp = FTP(FTP_SERVER)
# ftp.login()
# ftp.cwd(FTP_DIR)

# # Function to get the size of a remote file
# def get_remote_file_size(filename):
#     try:
#         return ftp.size(filename)  # Returns file size in bytes
#     except:
#         return None  # If file size is not available

# # Function to download a file but **halt if 3.49 GB is reached**
# def download_file(filename):
#                     # Connect to FTP

#     local_path = os.path.join(OUTPUT_DIR, filename)
    
#     # Skip if file already exists
#     if os.path.exists(local_path):
#         print(f"Skipping {filename} (already downloaded)")
#         return
    
#     # Get the remote file size
#     file_size = get_remote_file_size(filename)
#     if file_size is None:
#         print(f"Skipping {filename} (size unknown)")
#         return
    
#     try:
#         with open(local_path, "wb") as f:
#             def handle_binary_data(chunk):
#                 """Writes data to file but stops when 3.49 GB is reached."""
#                 nonlocal downloaded_bytes
#                 if downloaded_bytes >= MAX_FILE_BYTES:
#                     print(f"Reached 3.49 GB limit for {filename}, stopping this file.")
#                     raise StopIteration  # Stop downloading this file
                
#                 f.write(chunk)
#                 downloaded_bytes += len(chunk)

#             downloaded_bytes = 0  # Track bytes for this file
#             ftp.retrbinary(f"RETR {filename}", handle_binary_data)
        
#         print(f"Downloaded: {filename} ({downloaded_bytes / 1e9:.2f} GB)")

        
#     except StopIteration:
#         print(f"Stopped download of {filename} at 3.49 GB, moving to next file.")
#     except Exception as e:
#         print(f"Failed to download {filename}: {e}")

    

# # Iterate over years, months, and days
# for year in years:
#     for month in months:
#         for day in sample_days:
#             for hour in sample_hours:
                
               
#                 #download file
#                 filename = f"hysplit.{year}{month}{day}.{hour}z.hrrra"
#                 download_file(filename)  # Attempt to download the file
                

# # Close FTP connection
# ftp.quit()

# print("Download complete!")


# In[16]:


import os
import time
from ftplib import FTP
from tqdm import tqdm

# NOAA READY FTP server
FTP_SERVER = "arlftp.arlhq.noaa.gov"
FTP_DIR = "/pub/archives/hrrr.v1/"

# Local directory to save files
OUTPUT_DIR = "/Users/hannahkamen/hysplit/working/hrrr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define years, months, sample days, and hours
years = range(2015, 2016)
months = ["07"]
sample_days = ["22"]
sample_hours = ["00"]

# Max download size per file (3.49 GB in bytes)
MAX_FILE_BYTES = 2_200_000_000  # 3.49 GB

# Create the list of all filenames to download
all_files = []
for year in years:
    for month in months:
        for day in sample_days:
            for hour in sample_hours:
                filename = f"hysplit.{year}{month}{day}.{hour}z.hrrra"
                all_files.append(filename)

# Download a single file
def download_file(filename):
    local_path = os.path.join(OUTPUT_DIR, filename)

    # Skip if already downloaded
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return f"✅ Skipping {filename} (already downloaded)"

    try:
        ftp = FTP(FTP_SERVER)
        ftp.login()
        ftp.cwd(FTP_DIR)

        try:
            file_size = ftp.size(filename)
        except:
            ftp.quit()
            return f"⚠️ Skipping {filename} (size unknown or file not found)"

        with open(local_path, "wb") as f:
            downloaded_bytes = 0

            def handle_binary_data(chunk):
                nonlocal downloaded_bytes
                if downloaded_bytes >= MAX_FILE_BYTES:
                    raise StopIteration
                f.write(chunk)
                downloaded_bytes += len(chunk)

            ftp.retrbinary(f"RETR {filename}", handle_binary_data)

        ftp.quit()
        return f"✅ Downloaded: {filename} ({downloaded_bytes / 1e9:.2f} GB)"

    except StopIteration:
        return f"⏭️ Stopped download of {filename} at 3.00 GB"
    except Exception as e:
        return f"❌ Failed to download {filename}: {e}"

# Loop with progress bar
print("🚀 Starting downloads...")
for filename in tqdm(all_files, desc="Downloading HRRR files"):
    msg = download_file(filename)
    tqdm.write(msg)
    time.sleep(2)  # Short delay between downloads

print("🎉 Download process complete!")


# In[ ]:




