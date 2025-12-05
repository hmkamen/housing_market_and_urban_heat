# Housing Market and Urban Heat

This repository contains code, analysis notebooks, and supporting materials for my dissertation chapted entitled The Price of Cool: How Urban Heat Shapes Neighborhood Value"

# Absract: 
US cities are increasingly investing in urban heat mitigation. Determining the benefits of these projects is challenging because urban heat conditions are tightly linked to neighborhood attributes other than temperature. I develop a novel instrument for summer air temperatures that draws on urban heat advection—a well documented meteorological phenomenon through which surface air flow transfers hot and cool air within and between neighborhoods. By combining high-resolution temperature data with modeled surface wind trajectories, I am able to isolate exogenous variation in household heat exposure and identify the causal effect of cooling on home values in Maricopa County, AZ. Because my empirical technique does not rely on granular fixed effects to establish households' marginal willingness to pay for urban heat mitigation, I am able to isolate the effect of microclimate changes at varying spatial scales. I find that households are willing to pay \$124 monthly for a 1°C decrease in summer air temperatures in their immediate surroundings, and \$45 monthly for equivalent cooling in their larger, surrounding neighborhood. I use these estimates to calculate the benefits of Phoenix's Cool Corridor program and find that relying on conventional tract-level estimates of MWTP for microclimate improvement leads to a 22\% underestimation of the benefits of public investment in urban cooling. These findings imply that urban cooling behaves like a semi-public good: policies that incentivize private adaptation of cooling measures do not facilitate internalization of positive spatial externalities from neighborhood cooling; targeted public investments in shared cooling infrastructure may therefore be more efficient, but it is critical that cities account for the full scale at which households benefit these reductions.

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

Questions or comments: please reach out via GitHub (`@hmkamen`).
<img width="468" height="657" alt="image" src="https://github.com/user-attachments/assets/aa54f7db-4f0b-4068-9803-f17bdeb12f3f" />

