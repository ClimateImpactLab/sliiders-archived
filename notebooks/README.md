# Instructions

This directory contains sub-directories to produce final SLIIDERS outputs. The order of execution is as follows:

1. `country_level_ypk`: Workflow to clean and impute necessary country-level socioeconomic variables
2. `exposure`: workflow to generate current-day global exposure data by coastal segment and elevation
3. `create-SLIIDERS-SLR`: Workflow to generate **SLIIDERS-SLR**, a dataset of gridded local sea-level Monte Carlo samples for each RCP scenario, year (decadal), and site ID (defined by LocalizeSL).
4. `create-SLIIDERS-ECON`: Workflow to generate **SLIIDERS-ECON**, a dataset containing socioeconomic variables by coastal segment, elevation, Shared Socioeconomic Pathway scenario. Note that this workflow uses the SLIIDERS-SLR dataset to find nearest grid cells to match to coastal segments.
