Run the notebooks in this directory in order:

1. `create-coastline-segments`: Create segments from CoDEC points.
2. `create-segment-regions`: Divide the world up into Voronoi polygons for each segmentXregion.
3. `fill_missing_litpop_with_geg`: Fill missing regions in LitPop with data from GEG-15.
4. `vectorize-wetlands`: Transform wetlands rasters (GLOBCOVER and Global Mangrove Watch) into single shapefile.
5. `get_positive_elev_tiles`: Assign global 1-degree tiles to groups for tile processing notebook
6. `generate_datum_conversion_grid`: converts (interpolates) MDT data to match with geoid grid and combines geoid and MDT datasets
7. `create_dem_mss`: Create elevation grid relative to MSS
8. `generate_exposure_tiles`: Assign population, asset value, elevation, segments, protected regions, and administrative regions to global 1-degree tiles.
9. `combine_exposure_tiles`: Combine 1-degree tiles into the following datasets:
    * Exposure with elevation (coastal exposure)
    * Exposure without elevation (all exposure)
    * Areas by elevation