First, run all notebooks in `nearest_regions`.

Then, run the notebooks in this directory in order:

0. fill_missing_litpop_with_geg.ipynb
Fill missing regions in LitPop with data from Geg15.

1. vectorize-wetlands.ipynb
Transform wetlands rasters (GLOBCOVER and Global Mangrove Watch) into single
shapefile.

2. get_positive_elev_tiles.ipynb
Assign global 1-degree tiles to groups for tile processing notebook (step 4)

3. create_dem_mss.ipynb
Create elevation grid relative to MSS

4. generate_exposure_tiles.ipynb
Assign population, asset value, elevation, segments, protected regions, and
administrative regions to global 1-degree tiles.

5. combine_exposure_tiles.ipynb
Combine 1-degree tiles into the following datasets:
- Exposure with elevation (coastal exposure)
- Exposure without elevation (all exposure)
- Areas by elevation