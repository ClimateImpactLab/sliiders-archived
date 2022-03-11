# Sea Level Impacts Input Dataset by Elevation, Region, and Scenario (SLIIDERS)

This repository hosts the code used to create the [SLIIDERS-ECON](https://doi.org/10.5281/zenodo.6010452) and [SLIIDERS-SLR](https://doi.org/10.5281/zenodo.6012027) datasets. The SLIIDERS datasets contain current and forecasted physical and socioeconomic metrics from 2000-2100 - organized by coastal segment, elevation slice, and scenario - for use as inputs to global coastal climate impacts research.

**SLIIDERS-ECON** contains socioeconomic variables, varying horizontally and vertically over space. **SLIIDERS-SLR** contains Monte Carlo projections of Local Sea Level Rise under different emissions and ice sheet dynamics assumptions, based on the outputs of [LocalizeSL](https://github.com/bobkopp/LocalizeSL). Coastal segments in SLIIDERS-ECON can be matched to gridded LSLR projections in SLIIDERS-SLR via the `SLR_site` key.

## Installation
Most users will want to just use the datasets directly, accessible at the DOIs linked above. If you wish to recreate and/or modify the datasets, which we encourage, you will need to run the Jupyter notebooks in this repository. A collection of helper functions, organized into a Python package, is necessary to run the notebooks and can be found within the `sliiders` directory. A simple pip install will install this package

```bash
pip install -e sliiders
```

In addition, you will need to have [Dask Gateway](gateway.dask.org) installed and configured to execute the parallel, Dask-backed workflows contained in this repo. Advanced users can use other Dask Cluster backends (including simply running [Dask Distributed](distributed.dask.org) locally), but doing so will require modifying the cluster setup portion of notebooks that employ dask.

A Conda environment file better specifying a full environment needed to execute all of the workflows in this repo is in development and will be posted when complete.

## Filepaths and other settings
All filepaths and settings for the notebooks can be found within `settings.py`. Before moving onto executing different parts of this repository, please adjust these settings to match your directory structure and desired values. Most values will not need to be updated unless you change a particular dataset. However, at minimum you should:

1. Update the `DIR_DATA` filepath within this file to point to the root directory within which all of the data consumed and generated by this workflow will live.
2. Update `DASK_IMAGE` to point to a Docker Image that you will use for Dask workers (advanced users not using Dask Gateway may not need this parameter).

## Package Structure
* `sliiders`: Contains `.py` files with essential settings and functions for the SLIIDERS workflow
  - `settings.py`: Contains essential settings, including various parameters and data storage directories
  - `gcs.py`: Contains functions related to the use of Google Cloud Storage (GCS). Users running workflows locally or on a different cloud provider are encouraged to contribute similar modules for other contexts.
  - `io.py`: Contains various I/O-related functions
  - `spatial.py`: Contains functions for executing spatial and geographic operations including those related to shapefiles, grid-cell level operations, and more.
  - `dask.py`: Contains utility functions for working with dask clusters
  - `country_level_ypk.py`: contains functions for cleaning and working with country-level socioeconomic data, especially for the workflow in `notebooks/country_level_ypk`
* `notebooks`: contains the workflows to create SLIIDERS-ECON and SLIIDERS-SLR.

## Generating SLIIDERS-ECON and SLIIDERS-SLR

To generate **SLIIDERS-ECON** and **SLIIDERS-SLR**, please follow the directions in `notebooks/README.md` and other readme files in subdirectories under `notebooks` to learn about how to execute the workflows.
