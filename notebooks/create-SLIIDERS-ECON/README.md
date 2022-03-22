# Workflow for generating the SLIIDERS-ECON dataset

This directory contains notebooks to generate the **SLIIDERS-ECON** dataset. The final output for future projections is a Zarr store containing socioeconomic variables binned by coastal segment, elevation slice, and Shared Socioeconomic Pathway.

The steps to produce the final output are as follows.
1. Go to the directory `country_level_ypk` and follow the instructions in the `README.md` in that directory. The workflow in `country_level_ypk` cleans (and when necessary, imputes) various country-level socioeconomic variables.
2. Go to the directory `exposure` and follow the instructions in the `README.md` in that directory. The workflow in `exposure` generates current-day global exposure data by coastal segment and elevation.
3. Come back to `create-SLIIDERS-ECON` directory, and follow the instructions below:
    a. Use `download-sliiders-econ-input-data.ipynb` to download additional necessary datasets, including World Bank Intercomparison Project 2017 and construction cost index by Lincke and Hinkel (2021, *Earth's Future*).
    b. Obtain CoastalDEM v1.1. Due to this dataset being a proprietary one, a download script is not provided.
    c. Adjust datum of CoastalDEM. That is, use a global dataset of Mean Dynamic Ocean Topography to convert CoastalDEM to a Mean Sea Surface (MSS 2000) datum.
    d. Manually isolate the 10 km-spaced coastline points included in the CoDEC dataset from the 50 km-spaced points, and save these as `gtsm_stations_eur_tothin.shp` in the directory `DIR_CIAM_SHAPEFILES`.
    e. Use `create-coastline-segments.ipynb` to thin and augment the CoDEC points to get a uniformly distributed set of coastline segments that include all areas with exposure.
    f. Use `create-SLIIDERS-ECON.ipynb` to combine disparate data sources to generate the final output.