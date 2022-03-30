# Workflow for generating the SLIIDERS-ECON dataset

This directory contains notebooks to generate the **SLIIDERS-ECON** dataset. The final output for future projections is a Zarr store containing socioeconomic variables binned by coastal segment, elevation slice, and Shared Socioeconomic Pathway.

The steps to produce the final output are as follows.
1. Go to the directory `country_level_ypk` and follow the instructions in the `README.md` in that directory. The workflow in `country_level_ypk` cleans (and when necessary, imputes) various country-level socioeconomic variables.
2. Run `download-sliiders-econ-input-data.ipynb` to download additional necessary datasets, including World Bank Intercomparison Project 2017 and construction cost index by Lincke and Hinkel (2021, *Earth's Future*).
3. Manually isolate the 10 km-spaced coastline points included in the CoDEC dataset from the 50 km-spaced points, and save these as `gtsm_stations_eur_tothin.shp` in the defined in `settings.py` as `DIR_CIAM_SHAPEFILES`.
4. Use `create-coastline-segments.ipynb` to thin and augment the CoDEC points to get a uniformly distributed set of coastline segments that include all areas with exposure.
5. Obtain [CoastalDEM v1.1](https://go.climatecentral.org/coastaldem/). Save `.tif` files directly in the directory defined in `settings.py` as `DIR_COASTALDEM`.
6. Adjust datum of CoastalDEM, using a global Mean Dynamic Ocean Topography dataset to convert CoastalDEM to the Mean Sea Surface (MSS 2000) datum.
7. Go to the `exposure` directory and follow the instructions in the `README.md` in that directory. The workflow in `exposure` generates current-day global exposure data by coastal segment, elevation, and other variables.
8. Use `create-SLIIDERS-ECON.ipynb` to combine disparate data sources to generate the final output.
