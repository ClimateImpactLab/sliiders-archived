First, run all notebooks in `nearest_regions`.

Then, run the notebooks in this directory in order:

1. `create-coastline-segments`: Create segments from CoDEC points.
2. `create-segment-regions`: Divide the world up into Voronoi polygons for each segmentXregion.
3. `fill_missing_litpop_with_geg`: Fill missing regions in LitPop with data from Geg15.
4. `vectorize-wetlands`: Transform wetlands rasters (GLOBCOVER and Global Mangrove Watch) into single shapefile.
5. `get_positive_elev_tiles`: Assign global 1-degree tiles to groups for tile processing notebook (step 4)
6. `create_dem_mss`: Create elevation grid relative to MSS
7. `generate_exposure_tiles`: Assign population, asset value, elevation, segments, protected regions, and administrative regions to global 1-degree tiles.
8. `combine_exposure_tiles`: Combine 1-degree tiles into the following datasets:
    * Exposure with elevation (coastal exposure)
    * Exposure without elevation (all exposure)
    * Areas by elevation