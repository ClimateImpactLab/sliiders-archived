TODO:
1. Create automated download notebook

This directory contains notebooks to generate the SLIIDERS-ECON dataset.

The final output for future projections is a Zarr store containing socioeconomic variables binned by coastal segment, elevation slice, and Shared Socioeconomic Pathway.

The steps to produce this output are as follows:

First, manually download a variety of datasets:
1. Download a [dataset](https://github.com/daniellincke/DIVA_paper_migration/blob/master/data/csv/country_input.csv) that contains a country-level Construction Cost Index from [Lincke, 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EF001965?campaign=woletoc). Save to `PATH_EXPOSURE_LINCKE`.
2. Download construction cost and GDP data from the [World Bank Intercomparison Project 2017](https://databank.worldbank.org/source/icp-2017). Specifically, you will download the "1501200:CONSTRUCTION" series for all countries. Save this to `PATH_EXPOSURE_LINCKE`

Next, run the automated data download scripts:
1. `download-sliiders-econ-input-data.ipynb`


1. Obtain CoastalDEM v1.1. This is a proprietary dataset and thus a download script is not provided
2. ADJUST DATUM OF COASTALDEM: Use a global dataset of Mean Dynamic Ocean Topography to convert CoastalDEM to a Mean Sea Surface (MSS 2000) datum.
3. Manually isolate the 10 km-spaced coastline points included in the CoDEC dataset from the 50 km-spaced, and save these as `gtsm_stations_eur_tothin.shp`.
4. `create-coastline-segments.ipynb`: Thin and augment the CoDEC points to get a uniformly distributed set of coastline segments that include all areas with exposure.
5. `create-SLIIDERS-ECON.ipynb`: Combine disparate data sources to generate the final output.