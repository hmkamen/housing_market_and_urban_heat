# Housing Market and Urban Heat

This repository contains code, analysis notebooks, and supporting materials for my dissertation chapted entitled The Price of Cool: How Urban Heat Shapes Neighborhood Value"

# Absract: 
Urban heat is a growing concern in U.S. cities, however it remains unclear how households value the benefits of neighborhood cooling (e.g., shade trees, cool pavement, etc.) at increasing distances from their home. Measuring these benefits is challenging because temperature is correlated with many other neighborhood differences, including resident income and development patterns. I use local wind patterns to identify temperature variation that is unrelated to other neighborhood attributes, allowing me to estimate how much households value cooling at different distances. Using home sales data from Maricopa County, AZ, I find that households are willing to pay $676 annually for a 1°C decrease in summer air temperatures in their immediate surroundings, and $245 annually for equivalent cooling in their broader neighborhoods. I apply these estimates to Phoenix’s Cool Corridors Program and find that analyses ignoring these broader effects understate the program’s total value by 48%. These findings provide novel evidence that neighborhood cooling is a local public good: private adoption underprovides cooling, and conventional evaluations can understate the gains from targeted public investments. 

## Project Overview

- I use a novel airflow instrument to causally identify how micro-scale temperature variation within metropolitan Phoenix capitalizes into housing prices.
- Uses high-resolution temperature data, parcel-level housing transactions, and zoning information.
- Spatial lab model defines how heat capitalizes into home value over space.

## Repository Structure

- `noaa_hysplit_data_processing/` – data and code used to download high resolutoin meteorlogical HRRR data from noaa API, and for processing HRRR data to calculate airflow trajectory frequency over each home in maricopa county
- `econometric_inputs_data_processing/` – data and code for processing airflow instrument used in IV, demographic data, amaneities data, elevation data, and home sales transaction data
- `heatplots_and_charts/` – code used to create heatplots in final paper
- `cool_corridors/` – files needed to reconstruct the benefits calculation of Phoenix's cool corridors program
- `infill_dev_and_heat_analysis/` – auxilarry analysis testing the impact of infill versus sprawl development on urban heat changes

## Contact

Questions or comments: please reach out via email at hkamen@mines.edu
