Generate globally comprehensive map of regions delineating the combination of
closest GADM adm1 regions and CIAM segments.

Run the notebooks in order, as follows:

0a. Filter adm0 and adm1
0b. Generate closest-gadm-adm1 voronoi, dissolve to get closest-gadm-adm0 voronoi

1a. Assign ISOs to CIAM points
1b. Generate ISO-level point-voronoi from CIAM points
1c. Get CIAM coastline by country
1d. Get coast-seg-by-CIAM point

2. Overlap coastline vor with adm1 vor to get spatially comprehensive seg_adm1.
Save overlap shapes and segment shapes.