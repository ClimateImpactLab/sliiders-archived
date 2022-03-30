import random
import warnings
import zipfile
from collections import defaultdict
from operator import itemgetter
from typing import Any, Sequence, Union

import geopandas as gpd
import matplotlib._color_data as mcd
import networkx as nx
import numpy as np
import pandas as pd
import pygeos
import regionmask
import rioxarray
import shapely as shp
import xarray as xr
from dask_gateway import Gateway
from IPython.display import display
from numba import jit
from scipy.spatial import SphericalVoronoi, cKDTree
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import linemerge, unary_union
from tqdm.notebook import tqdm

from . import settings as sset
from .io import load_adm0_shpfiles

assert sset.MARGIN_DIST < sset.DENSIFY_TOLERANCE
assert 10 ** (-sset.ROUND_INPUT_POINTS) < sset.MARGIN_DIST

SPHERICAL_VORONOI_THRESHOLD = (
    1e-7  # `threshold` parameter of SphericalVoronoi() (not sure it can go any lower)
)
LAT_TO_M = 111131.745
EARTH_RADIUS = 6371.009


def iso_poly_box_getter(iso, shp_df):
    """Get `box`es or rectangular areas of coordinates that contains each Polygon
    belonging to the shapefile of the country specified by the ISO code.

    Parameters
    ----------
    iso : str
        ISO code of the country that we are interested in
    shp_df : geopandas DataFrame
        with the indices being the iso codes and with a column called `geometry`
        containing the shapefile of the relevant countries

    Returns
    -------
    list of tuples (of length four)
        containing the smallest and largest x and y coordinates (longitudes and
        latitudes)

    """
    shp = shp_df.loc[iso, "geometry"]
    if type(shp) == MultiPolygon:
        shps = shp.geoms
    else:
        shps = [shp]

    poly_bounds = []
    for poly in shps:
        xx = np.array(poly.exterior.coords.xy[0])
        yy = np.array(poly.exterior.coords.xy[1])
        xmin, xmax = np.floor(xx.min()), np.ceil(xx.max())
        ymin, ymax = np.floor(yy.min()), np.ceil(yy.max())
        poly_bounds.append((xmin, xmax, ymin, ymax))

    return list(set(poly_bounds))


def get_iso_geometry(iso=""):
    """Find the index in df2 of the nearest point to each element in df1

    Parameters
    ----------
    iso : str or list of str
        three-letter code, or list of three-letter codes, referencing a
        geographic region in the natural earth shapefiles

    Returns
    -------
    :py:class:`shapely.geometry` or list of :py:class:`shapely.geometry`
    """
    input_is_list = isinstance(iso, (list, np.ndarray))

    if input_is_list:
        isos = list(iso)
    else:
        isos = [iso]

    shp_dict = load_adm0_shpfiles(
        ["countries", "map_units", "map_subunits", "disputed_areas"]
    )
    country_shps = shp_dict["countries"]
    map_unit_shps = shp_dict["map_units"]
    map_subunit_shps = shp_dict["map_subunits"]
    disputed_area_shps = shp_dict["disputed_areas"]

    for i in range(len(isos)):
        # cw between iso codes --
        if isos[i] == "ALA":
            isos[i] = "ALD"
        if isos[i] == "ESH":
            isos[i] = "SAH"
        if isos[i] == "PSE":
            isos[i] = "PSX"
        if isos[i] == "SJM":
            isos[i] = "NSV"
        if isos[i] == "SSD":
            isos[i] = "SDS"
        if isos[i] == "XKX":
            isos[i] = "KOS"
        if isos[i] == "BES":
            isos[i] = "NLY"

    geos = []
    for iso in isos:
        if iso == "SAH":
            geo = disputed_area_shps[
                disputed_area_shps["NAME_EN"] == "Western Sahara"
            ].geometry.unary_union
        else:
            # retrieve shape file
            try:
                geo = country_shps[
                    country_shps["ADM0_A3"] == iso.upper()
                ].geometry.iloc[0]
                if iso == "MAR":
                    geo = geo.difference(
                        disputed_area_shps[
                            disputed_area_shps["NAME_EN"] == "Western Sahara"
                        ].geometry.unary_union
                    )
            except IndexError:
                try:
                    geo = map_unit_shps[
                        map_unit_shps["GU_A3"] == iso.upper()
                    ].geometry.iloc[0]
                except IndexError:
                    geo = map_subunit_shps[
                        map_subunit_shps["SU_A3"] == iso.upper()
                    ].geometry.iloc[0]
        geos.append(geo)

    if input_is_list:
        return geos
    return geos[0]


def filter_spatial_warnings():
    """Suppress warnings defined in `sliiders.settings.SPATIAL_WARNINGS_TO_IGNORE`"""
    for msg in sset.SPATIAL_WARNINGS_TO_IGNORE:
        warnings.filterwarnings("ignore", message=f".*{msg}*")


def add_rand_color(gdf, col=None):
    """Get a list of random colors corresponding to either each row or each ID
    (as defined by `col`) of a GeoDataFrame. Used in `sliiders` for diagnostic
    visualizations, not functionality.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to assign colors to

    col: str
        Column name in `gdf` to use as unique ID to assign colors to

    Returns
    -------
    colors : list-like, or pandas.Series
        A list of random colors corresponding to each row, or to values
        defined in gdf[`col`]
    """
    if col is None:
        colors = random.choices(list(mcd.XKCD_COLORS.keys()), k=gdf.shape[0])
    else:
        unique_vals = gdf[col].unique()
        color_dict = {
            v: random.choice(list(mcd.XKCD_COLORS.keys())) for v in unique_vals
        }
        return gdf[col].apply(lambda v: color_dict[v])
    return colors


def get_points_on_lines(geom, distance, starting_length=0.0):
    """Return evenly spaced points on a LineString or
    MultiLineString object.

    Parameters
    ----------
    geom : :py:class:`shapely.geometry.MultiLineString` or
        :py:class:`shapely.geometry.LineString`
    distance : float
        Interval desired between points along LineString(s).
    starting_length : float
        How far in from one end of the LineString you would like
        to put your first point.

    Returns
    -------
    coast : :py:class:`shapely.geometry.MultiPoint` object
        Contains all of the points on your line.
    """

    if geom.geom_type == "LineString":
        short_length = geom.length - starting_length
        num_vert = int(short_length / distance) + 1

        # if no points should be on this linestring, return
        # empty list
        if short_length <= 0:
            return [], -short_length

        # else return list of coordinates
        remaining_length = geom.length - ((num_vert - 1) * distance + starting_length)
        return (
            shp.geometry.MultiPoint(
                [
                    geom.interpolate(n * distance + starting_length, normalized=False)
                    for n in range(num_vert)
                ]
            ),
            remaining_length,
        )
    elif geom.geom_type == "MultiLineString":
        this_length = starting_length
        parts = []
        for part in geom:
            res, this_length = get_points_on_lines(part, distance, this_length)
            parts += res
        return shp.geometry.MultiPoint(parts), this_length
    else:
        raise ValueError("unhandled geometry %s", (geom.geom_type,))


def grab_lines(g):
    """Get a LineString or MultiLineString representing all the lines in a
    geometry.

    Parameters
    ----------
    g : shapely.Geometry
        Any Geometry in Shapely

    Returns
    -------
    shapely.LineString or shapely.MultiLineString
        A shapely.Geometry object representing all LineStrings in `g`.
    """
    if isinstance(g, Point):
        return LineString()
    if isinstance(g, LineString):
        return g

    return linemerge(
        [
            component
            for component in g.geoms
            if isinstance(component, LineString)
            or isinstance(component, MultiLineString)
        ]
    )


def grab_polygons(g):
    """Get a Polygon or MultiPolygon representing all the polygons in a
    geometry.

    Parameters
    ----------
    g : shapely.Geometry
        Any Geometry in Shapely

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        A shapely.Geometry object representing all Polygons in `g`.

    """
    if isinstance(g, Point):
        return Polygon()
    if isinstance(g, Polygon):
        return g
    if isinstance(g, MultiPolygon):
        return g
    return unary_union(
        [
            component
            for component in g.geoms
            if isinstance(component, Polygon) or isinstance(component, MultiPolygon)
        ]
    )


def strip_line_interiors_poly(g):
    """Remove tiny interior Polygons from a Polygon.

    Parameters
    ----------
    g : shapely.Polygon
        A Shapely Polygon

    Returns
    -------
    shapely.Polygon
        A Shapely Polygon equivalent to `g`, removing any interior Polygons
        smaller than or equal to `sliiders.spatial.SMALLEST_INTERIOR_RING`,
        measured in "square degrees".
    """
    return Polygon(
        g.exterior,
        [i for i in g.interiors if Polygon(i).area > sset.SMALLEST_INTERIOR_RING],
    )


def strip_line_interiors(g):
    """Remove tiny interior Polygons from a Geometry.

    Parameters
    ----------
    g : shapely.Geometry
        A Shapely Geometry. Must be either an object containing Polygons, i.e.
        shapely.Polygon or shapely.MultiPolygonn or shapely.GeometryCollection

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        A collection of Shapely Polygons equivalent to the set of Polygons
        contained in `g`, removing any interior Polygons smaller than or equal
        to `sliiders.spatial.SMALLEST_INTERIOR_RING`, measured in
        "square degrees".
    """
    if isinstance(g, Polygon):
        return strip_line_interiors_poly(g)
    if isinstance(g, MultiPolygon):
        return unary_union(
            [
                strip_line_interiors_poly(component)
                for component in g.geoms
                if isinstance(component, Polygon)
            ]
        )

    # Recursively call this function for each Polygon or Multipolygon contained
    # in the geometry
    if isinstance(g, GeometryCollection):
        return unary_union(
            [
                strip_line_interiors(grab_polygons(g2))
                for g2 in g.geoms
                if (isinstance(g2, Polygon) or isinstance(g2, MultiPolygon))
            ]
        )

    raise ValueError(
        "Geometry must be of type `Polygon`, `MultiPolygon`, or `GeometryCollection`."
    )


def fill_in_gaps(gser):
    """Fill in the spatial gaps of a GeoSeries within the latitude-longitude
    coordinate system. Approximates a "nearest shape" from the original
    geometries by iteratively expanding the original shapes in degree-space.
    Not ideal for precise nearest-shape-matching, but useful in cases where
    gaps are small and/or insignificant but may lead to computational
    difficulties.

    Parameters
    ----------
    gser : :py:class:`geopandas.GeoSeries`
        A GeoSeries intended to include globally comprehensive shapes

    Returns
    -------
    out : :py:class:`geopandas.GeoSeries`
        A GeoSeries covering the globe, with initially empty spaces filled in by a
        nearby shape.
    """
    uu = gser.unary_union
    current_coverage = box(-180, -90, 180, 90).difference(uu)
    if isinstance(current_coverage, Polygon):
        current_coverage = MultiPolygon([current_coverage])

    assert all([g.type == "Polygon" for g in current_coverage.geoms])

    intersects_missing_mask = gser.intersects(current_coverage)
    intersects_missing = gser[intersects_missing_mask].copy().to_frame(name="geometry")

    for buffer_size in tqdm([0.01, 0.01, 0.01, 0.03, 0.05, 0.1, 0.1, 0.1]):
        with warnings.catch_warnings():
            filter_spatial_warnings()
            intersects_missing["buffer"] = intersects_missing["geometry"].buffer(
                buffer_size
            )

        new_buffers = []
        for i in intersects_missing.index:
            new_buffer = (
                intersects_missing.loc[i, "buffer"]
                .intersection(current_coverage)
                .buffer(0)
            )
            new_buffers.append(new_buffer)
            current_coverage = current_coverage.difference(new_buffer)

        with warnings.catch_warnings():
            filter_spatial_warnings()
            intersects_missing["new_buffer"] = gpd.GeoSeries(
                new_buffers, index=intersects_missing.index, crs=intersects_missing.crs
            ).buffer(0.00001)
            use_new_buffer_mask = intersects_missing["new_buffer"].geometry.area > 0
        intersects_missing.loc[
            use_new_buffer_mask, "geometry"
        ] = intersects_missing.loc[use_new_buffer_mask, "geometry"].union(
            intersects_missing.loc[use_new_buffer_mask, "new_buffer"]
        )

    assert current_coverage.area == 0

    out = gser[~intersects_missing_mask].copy()

    out = pd.concat(
        [out, intersects_missing.geometry],
    ).rename(out.name)

    assert intersects_missing.is_valid.all()

    return out


def get_polys_in_box(all_polys, lx, ly, ux, uy):
    """Get the subset of shapes in `all_polys` that overlap with the box defined
    by `lx`, `ly`, `ux`, `uy`.

    Parameters
    ----------
    all_polys : pygeos.Geometry
        Array of pygeos Polygons

    lx : float
        Left (western) bound of box

    ly : float
        Lower (southern) bound of box

    ux : float
        Right (eastern) bound of box

    uy : float
        Upper (northern) bound of box

    Returns
    -------
    vertical_slab : pygeos.Geometry
        List of the pygeos polygons from `all_polys` overlapped with (cut by)
        the box.

    slab_polys : np.array
        List of indices from `all_polys` corresponding to the polygons in
        `vertical_slab`.
    """

    vertical_slab = pygeos.clip_by_rect(all_polys, lx, ly, ux, uy)

    poly_found_mask = ~pygeos.is_empty(vertical_slab)
    slab_polys = np.where(poly_found_mask)

    vertical_slab = vertical_slab[poly_found_mask]

    # invalid shapes may occur from Polygons being cut into what should be MultiPolygons
    not_valid = ~pygeos.is_valid(vertical_slab)
    vertical_slab[not_valid] = pygeos.make_valid(vertical_slab[not_valid])

    vertical_slab_shapely = pygeos.to_shapely(vertical_slab)
    vertical_slab_shapely = [strip_line_interiors(p) for p in vertical_slab_shapely]
    vertical_slab = pygeos.from_shapely(vertical_slab_shapely)

    return vertical_slab, slab_polys


def grid_gdf(
    orig_gdf,
    box_size=sset.DEFAULT_BOX_SIZE,
    show_bar=True,
):
    """Divide a GeoDataFrame into a grid, returning the gridded shape-parts and
    the "empty" areas, each nested within a `box_size`-degree-width square.
    This reduces the sizes and rectangular boundaries of geometries, easing
    many computational processes, especially those that depend on a spatial
    index.

    Note: This may be deprecated in a future version if something like this
    becomes available: https://github.com/pygeos/pygeos/pull/256

    Parameters
    ----------
    orig_gdf : :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoSeries`
        GeoDataFrame/GeoSeries to be divided into a grid

    orig_geo_col : str
        Column in `orig_gdf` containing geometry to be divided

    box_size : float
        Width and height of boxes to divide geometries into

    show_bar : bool
        Show progress bar

    Returns
    -------
    gridded_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing `orig_gdf` geometries divided into grid cells.

    all_oc : pygeos.Geometry
        List of pygeos Polygons corresponding to the "ocean" shapes in each
        grid cell. Ocean shapes are defined as areas not covered by any
        geometry in `orig_gdf`.
    """

    if isinstance(orig_gdf, gpd.GeoSeries):
        orig_gdf = orig_gdf.to_frame(name="geometry")
    orig_geos = pygeos.from_shapely(orig_gdf.geometry)

    llon, llat, ulon, ulat = orig_gdf.total_bounds

    boxes = []
    ixs = []
    all_oc = []
    iterator = np.arange(llon - 1, ulon + 1, box_size)
    if show_bar:
        iterator = tqdm(iterator)
    for lx in iterator:
        ux = lx + box_size
        vertical_slab, slab_polys = get_polys_in_box(orig_geos, lx, llat, ux, ulat)
        for ly in np.arange(llat - 1, ulat + 1, box_size):
            uy = ly + box_size
            res = pygeos.clip_by_rect(vertical_slab, lx, ly, ux, uy)
            polygon_found_mask = ~pygeos.is_empty(res)
            res = res[polygon_found_mask]
            # invalid shapes may occur from Polygons being cut into what should be
            # MultiPolygons
            not_valid = ~pygeos.is_valid(res)
            res[not_valid] = pygeos.make_valid(res[not_valid])
            ix = np.take(slab_polys, np.where(polygon_found_mask))
            if res.shape[0] > 0:
                boxes.append(res)
                ixs.append(ix)

            if res.shape[0] > 0:

                this_uu = pygeos.union_all(res)

                this_oc = pygeos.difference(
                    pygeos.from_shapely(box(lx, ly, ux, uy)), this_uu
                )

                oc_parts = pygeos.get_parts(this_oc)
                all_oc += list(oc_parts)

            else:
                this_oc = pygeos.from_shapely(box(lx, ly, ux, uy))
                all_oc.append(this_oc)

    geom_ix = np.concatenate(ixs, axis=1).flatten()
    geom = np.concatenate(boxes).flatten()

    gridded_gdf = orig_gdf.drop(columns="geometry").iloc[geom_ix]
    gridded_gdf["geometry"] = geom

    all_oc = np.array(all_oc)
    all_oc = all_oc[~pygeos.is_empty(all_oc)]

    return gridded_gdf, all_oc


def divide_pts_into_categories(
    pts,
    pt_gadm_ids,
    all_oc,
    tolerance=sset.DENSIFY_TOLERANCE,
    at_blank_tolerance=sset.MARGIN_DIST,
):
    """From a set of points and IDs, divide points into "coastal-coastal" and
    "coastal-border" categories.

    "Coastal" indicates proximity to the "coast", i.e. the edges of the union
    of all original polygons, defined by `all_oc`. "Border" indicates
    non-proximity to the coast. Proximity to the coast is calculated as being
    within `at_blank_tolerance` of `all_oc`.

    "Coastal-border" points are defined as all coastal points that are within
    `tolerance` of "border" points (points that are not near the coast).

    "Coastal-coastal" points are defined as the remaining "coastal" points
    (not near a border).

    The motivation for this function is to simplify the point set used to
    generate Voronoi regions from a set of polygons. Precision matters a lot
    in parts of shapes that are near borders with other regions, and less so
    in coastal areas that are distant from the nearest non-same region. Points
    that are neither coastal, nor near a border, can be ignored, as they do not
    define the edges of a region. That is, they are entirely interior to a
    region's boundaries, so will not figure in the calculation of all areas
    nearest to that region.

    Parameters
    ----------
    pts : np.ndarray
        2D array with dimensions 2xN, representing N longitude-latitude
        coordinates.

    pt_gadm_ids : np.ndarray
        1D array representing N IDs corresponding to `pts`.

    all_oc : pygeos.Geometry
        List of pygeos Polygons corresponding to the "ocean" shapes in each
        grid cell. Ocean shapes should be defined as areas not covered by any
        geometry in the set of shapes represented here by their component
        points.

    tolerance : float
        Maximum distance from one point to a point with a different ID for the
        first point to be considered a "border" point.

    at_blank_tolerance : float
        Maximum distance from `all_oc` for a point to be considered "coastal".

    Returns
    -------
    coastal_coastal_pts : np.ndarray
        2D array representing coastal points that are not near borders.
    coastal_border_pts : np.ndarray
        2D array representing coastal points that are near borders.
    coastal_coastal_gadm : np.ndarray
        1D array representing IDs corresponding to `coastal_coastal_pts`.
    coastal_border_gadm : np.ndarray
        1D array representing IDs corresponding to `coastal_border_pts`.
    """
    at_blank_tolerance = at_blank_tolerance + (at_blank_tolerance / 10)
    tolerance = tolerance + (tolerance / 10)

    tree = cKDTree(pygeos.get_coordinates(all_oc))

    batch_size = int(1e6)
    starts = np.arange(0, pts.shape[0], batch_size)
    ends = starts + batch_size
    ends[-1] = pts.shape[0]

    pts_at_blank = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        pts_subset = pts[start:end]
        pts_tree = cKDTree(pts_subset)
        pts_at_blank_subset = pts_tree.query_ball_tree(tree, r=at_blank_tolerance)
        pts_at_blank_subset = np.array(
            [True if r else False for r in pts_at_blank_subset]
        )
        pts_at_blank.append(pts_at_blank_subset)

    pts_at_blank = np.concatenate(pts_at_blank)

    coastal_pts = pts[pts_at_blank]
    coastal_pt_gadm_ids = pt_gadm_ids[pts_at_blank]

    border_pts = pts[~pts_at_blank]

    tree = cKDTree(border_pts)

    batch_size = int(1e6)
    starts = np.arange(0, coastal_pts.shape[0], batch_size)
    ends = starts + batch_size
    ends[-1] = coastal_pts.shape[0]

    pts_at_border = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        pts_subset = coastal_pts[start:end]
        pts_tree = cKDTree(pts_subset)
        pts_at_border_subset = pts_tree.query_ball_tree(tree, r=tolerance)
        pts_at_border_subset = np.array(
            [True if r else False for r in pts_at_border_subset]
        )
        pts_at_border.append(pts_at_border_subset)

    pts_at_border = np.concatenate(pts_at_border)

    coastal_coastal_pts = coastal_pts[~pts_at_border].copy()
    coastal_coastal_gadm = coastal_pt_gadm_ids[~pts_at_border].copy()

    coastal_border_pts = coastal_pts[pts_at_border].copy()
    coastal_border_gadm = coastal_pt_gadm_ids[pts_at_border].copy()

    return (
        coastal_coastal_pts,
        coastal_border_pts,
        coastal_coastal_gadm,
        coastal_border_gadm,
    )


def simplify_nonborder(
    coastal_coastal_pts,
    coastal_border_pts,
    coastal_coastal_gadm,
    coastal_border_gadm,
    tolerance=sset.MARGIN_DIST,
):
    """Simplify coastal Voronoi generator points that are not near the border
    of another administrative region.

    Parameters
    ----------
    coastal_coastal_pts : np.ndarray
        2D array of longitude-latitude coordinates representing
        "coastal-coastal" points (see documentation in
        `divide_pts_into_categories()`.)

    coastal_border_pts : np.ndarray
        2D array of longitude-latitude coordinates representing
        "coastal-border" points (see documentation in
        `divide_pts_into_categories()`.)

    coastal_coastal_gadm : np.ndarray
        1D array of region IDs corresponding to `coastal_coastal_pts`.

    coastal_border_gadm : np.ndarray
        1D array of region IDs corresponding to `coastal_border_pts`.

    tolerance : float
        Precision in degree-distance below which we tolerate imprecision for
        all points.

    Returns
    -------
    non_border : np.ndarray
        2D array of points that are not close to the border

    non_border_gadm : np.ndarray
        1D array of region IDs corresponding to `non_border`.

    now_border : np.ndarray
        2D array of points that are close to the border

    now_border_gadm : np.ndarray
        1D array of region IDs corresponding to `now_border`.
    """
    border_tree = cKDTree(coastal_border_pts)

    d, i = border_tree.query(coastal_coastal_pts, distance_upper_bound=1)

    already_simplified = np.zeros_like(coastal_coastal_pts[:, 0], dtype="bool")
    non_border = []
    non_border_gadm = []

    for UPPER_BOUND in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        if UPPER_BOUND <= tolerance:
            break

        simplify = ~(d < UPPER_BOUND)
        this_level_nonborder = coastal_coastal_pts[simplify & (~already_simplified)]
        this_level_nonborder_gadm = coastal_coastal_gadm[
            simplify & (~already_simplified)
        ]

        already_simplified[simplify] = True

        # For points >= UPPER_BOUND away from the border, round to nearest
        # UPPER_BOUND/10
        this_level_nonborder = np.round(
            this_level_nonborder, int(-np.log10(UPPER_BOUND) + 1)
        )
        this_level_nonborder, this_level_nonborder_ix = np.unique(
            this_level_nonborder, axis=0, return_index=True
        )
        this_level_nonborder_gadm = this_level_nonborder_gadm[this_level_nonborder_ix]

        non_border.append(this_level_nonborder)
        non_border_gadm.append(this_level_nonborder_gadm)

    non_border = np.concatenate(non_border)
    non_border_gadm = np.concatenate(non_border_gadm)

    now_border = coastal_coastal_pts[~already_simplified]
    now_border_gadm = coastal_coastal_gadm[~already_simplified]

    return non_border, non_border_gadm, now_border, now_border_gadm


def explode_gdf_to_pts(geo_array, id_array):
    """Transform an array of shapes into an array of coordinate pairs, keeping
    the IDs of shapes aligned with the coordinates.

    Parameters
    ----------
    geo_array : pygeos.Geometry
        Array of pygeos geometries

    id_array : np.ndarray
        List of IDs corresponding to shapes in `geo_array`

    Returns
    -------
    pts : np.ndarray
        2D array of longitude-latitude pairs representing all points, rounded
        to `sliiders.settings.ROUND_INPUT_POINTS` precision, represented in the
        geometries of `geo_array`.

    pt_ids : np.ndarray
        1D array of IDs corresponding to `pts`.
    """
    counts = np.array([pygeos.count_coordinates(poly) for poly in geo_array])

    pt_ids = np.repeat(id_array, counts)

    pts = pygeos.get_coordinates(geo_array)

    pts, pts_ix = np.unique(
        np.round(pts, sset.ROUND_INPUT_POINTS), axis=0, return_index=True
    )
    pt_ids = pt_ids[pts_ix]

    return pts, pt_ids


def polys_to_vor_pts(regions, all_oc, tolerance=sset.DENSIFY_TOLERANCE):
    """Create a set of Voronoi region generator points from a set of shapes.

    Parameters
    ----------
    regions : geopandas.GeoDataFrame
        GeoDataFrame defining region boundaries, with `UID` unique ID field

    all_oc : pygeos.Geometry
        List of pygeos Polygons corresponding to the "ocean" shapes in each
        grid cell. Ocean shapes should be defined as areas not covered by any
        geometry in the set of `regions`.

    tolerance : float
        Desired precision of geometries in `regions`

    Returns
    -------
    :py:class:`geopandas.GeoSeries`
        Resulting points derived from `regions` to use as Voronoi generators
    """
    densified = pygeos.segmentize(pygeos.from_shapely(regions["geometry"]), tolerance)

    pts, pt_gadm_ids = explode_gdf_to_pts(densified, regions.index.values)

    all_oc_densified = pygeos.segmentize(all_oc, sset.MARGIN_DIST)

    (
        coastal_coastal_pts,
        coastal_border_pts,
        coastal_coastal_gadm,
        coastal_border_gadm,
    ) = divide_pts_into_categories(pts, pt_gadm_ids, all_oc_densified, tolerance)

    non_border, non_border_gadm, now_border, now_border_gadm = simplify_nonborder(
        coastal_coastal_pts,
        coastal_border_pts,
        coastal_coastal_gadm,
        coastal_border_gadm,
        tolerance=sset.MARGIN_DIST,
    )

    vor_pts = np.concatenate([non_border, now_border, coastal_border_pts])
    vor_gadm = np.concatenate([non_border_gadm, now_border_gadm, coastal_border_gadm])

    return remove_duplicate_points(
        gpd.GeoSeries.from_xy(
            x=vor_pts[:, 0],
            y=vor_pts[:, 1],
            index=pd.Index(vor_gadm, name=regions.index.name),
            crs=regions.crs,
        )
    )


def get_hemisphere_shape(hemisphere):
    """Define Shapely boxes for each hemisphere and the globe.

    Parameters
    ----------
    hemisphere : str
        Options are "west", "east", and "both".

    Returns
    -------
    shapely.Polygon
        A single box corresponding to the requested hemisphere(s).
    """
    if hemisphere == "west":
        return box(-180, -90, 0, 90)
    elif hemisphere == "east":
        return box(0, -90, 180, 90)
    elif hemisphere == "both":
        return box(-180, -90, 180, 90)
    else:
        raise ValueError


def make_valid_shapely(g):
    """Wrapper to call `make_valid` on a list of Shapely geometries.
    Should be deprecated upon release of Shapely 2.0.

    Parameters
    ----------
    g : list-like
        List of Shapely geometries or geopandas.GeoSeries

    Returns
    -------
    list
        List of Shapely geometries, after calling `pygeos.make_valid()` on all.
    """
    return pygeos.to_shapely(pygeos.make_valid(pygeos.from_shapely(g)))


def clip_geoseries_by_rect(gs, rect):
    """Wrapper to mask a geopandas.GeoSeries by a Shapely rectangle.
    Should be deprecated upon release of Shapely 2.0.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Any Geopandas GeoSeries

    rect : shapely.Polygon
        A Shapely rectangle

    Returns
    -------
    geopandas.GeoSeries
        A GeoSeries equivalent to `gs`, bound by `rect`.
    """
    try:
        return gpd.GeoSeries(pygeos.clip_by_rect(pygeos.from_shapely(gs), *rect.bounds))
    except Exception:  # weird issue with CYM, clip_by_rect doesn't work
        return gs.apply(lambda g: g.intersection(rect))


def diff_geoseries(gs1, gs2):
    """Wrapper to get the spatial difference between two GeoSeries.

    Parameters
    ----------
    gs1 : geopandas.GeoSeries
        Any GeoSeries

    gs2 : geopandas.GeoSeries
        Any GeoSeries

    Returns
    -------
    geopandas.GeoSeries
        The spatial difference between `gs1` and `gs2` (i.e. `gs1` - `gs2`)
    """
    return gpd.GeoSeries(
        pygeos.difference(pygeos.from_shapely(gs1), pygeos.from_shapely(gs2))
    )


@jit(nopython=True, parallel=False)
def lon_lat_to_xyz(lons, lats):
    """Transformation from longitude-latitude to an x-y-z cube
    with centroid [0, 0, 0]. Resulting points are on the unit sphere.

    Parameters
    ----------
    lons : np.ndarray
        1D array of longitudes

    lats : np.ndarray
        1D array of latitudes

    Returns
    -------
    np.ndarray
        2D array representing x-y-z coordinates equivalent to inputs
    """
    lat_radians, lon_radians = np.radians(lats), np.radians(lons)
    sin_lat, cos_lat = np.sin(lat_radians), np.cos(lat_radians)
    sin_lon, cos_lon = np.sin(lon_radians), np.cos(lon_radians)
    x = cos_lat * cos_lon
    y = cos_lat * sin_lon
    z = sin_lat
    return np.stack((x, y, z), axis=1)


@jit(nopython=True, parallel=False)
def xyz_to_lon_lat(xyz):
    """Transformation from x-y-z cube with centroid [0, 0, 0] to
    longitude-latitude.

    Parameters
    ----------
    xyz : np.ndarray
        2D array representing x-y-z coordinates on the unit sphere

    Returns
    -------
    np.ndarray
        2D array representing longitude-latitude coordinates equivalent to
        inputs
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lats = np.degrees(np.arcsin(z).flatten())
    lons = np.degrees(np.arctan2(y, x).flatten())
    # ensure consistency with points exactly on meridian
    lons = np.where(lons == -180, 180, lons)

    return np.stack((lons, lats), axis=1)


def combine_reg_group(reg_group):
    """Combine tesselated triplets on a sphere to get the points defining region
    boundaries.
    """
    pairs = defaultdict(list)

    for reg in reg_group:
        for v in range(len(reg)):
            p1 = reg[v]
            p2 = reg[(v + 1) % len(reg)]
            pairs[p1].append(p2)
            pairs[p2].append(p1)

    edge_pairs = {k: v for k, v in pairs.items() if len(v) != 6}
    edge_pairs = {
        k: [item for item in v if item in edge_pairs.keys()]
        for k, v in edge_pairs.items()
    }

    G = nx.Graph()

    G.add_nodes_from(list(edge_pairs.keys()))

    for k in edge_pairs:
        for item in edge_pairs[k]:
            G.add_edge(k, item)
            G.add_edge(item, k)

    cycles = nx.cycle_basis(G)

    return cycles


def get_reg_group(loc_reg_lists, loc, regions):
    reg_group = itemgetter(*loc_reg_lists[loc])(regions)
    if isinstance(reg_group, tuple):
        return list(reg_group)

    return [reg_group]


def fix_ring_topology(reg_group_polys, reg_group_loc_ids):

    group_polys = pygeos.from_shapely(reg_group_polys)

    tree = pygeos.STRtree(group_polys)

    contains, contained = tree.query_bulk(group_polys, "contains_properly")

    # Check that there are no rings inside rings. If there are, this function
    # And `get_groups_of_regions()` may need to be re-worked
    assert set(contains) & set(contained) == set([])

    for container_ix in np.unique(contains):

        reg_group_polys[container_ix] = pygeos.to_shapely(
            pygeos.make_valid(
                pygeos.polygons(
                    pygeos.get_exterior_ring(group_polys[container_ix]),
                    holes=pygeos.get_exterior_ring(
                        group_polys[contained[contains == container_ix]]
                    ),
                )
            )
        )

    reg_group_polys = [
        p for (i, p) in enumerate(reg_group_polys) if i not in np.unique(contained)
    ]
    reg_group_loc_ids = [
        l for (i, l) in enumerate(reg_group_loc_ids) if i not in np.unique(contained)
    ]

    return reg_group_polys, reg_group_loc_ids


@jit(nopython=True)
def numba_geometric_slerp(start, end, t):
    """Optimized version of scipy.spatial.geometric_slerp

    Adapted from:
    https://github.com/scipy/scipy/blob/master/scipy/spatial/_geometric_slerp.py

    Parameters
    ----------
    start : np.ndarray
        Single n-dimensional input coordinate in a 1-D array-like
        object. `n` must be greater than 1.

    end : np.ndarray
        Single n-dimensional input coordinate in a 1-D array-like
        object. `n` must be greater than 1.

    t : np.ndarray
        A float or 1D array-like of doubles representing interpolation
        parameters, with values required in the inclusive interval
        between 0 and 1. A common approach is to generate the array
        with ``np.linspace(0, 1, n_pts)`` for linearly spaced points.
        Ascending, descending, and scrambled orders are permitted.

    Returns
    -------
    np.ndarray
        An array of doubles containing the interpolated
        spherical path and including start and
        end when 0 and 1 t are used. The
        interpolated values should correspond to the
        same sort order provided in the t array. The result
        may be 1-dimensional if ``t`` is a float.
    """
    # create an orthogonal basis using QR decomposition
    basis = np.vstack((start, end))
    Q, R = np.linalg.qr(basis.T)
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q.T * np.reshape(signs.T, (2, 1))
    R = R.T * np.reshape(signs.T, (2, 1))

    # calculate the angle between `start` and `end`
    c = np.dot(start, end)
    s = np.linalg.det(R)
    omega = np.arctan2(s, c)

    # interpolate
    start, end = Q
    s = np.sin(t * omega)
    c = np.cos(t * omega)
    return start * np.reshape(c, (c.shape[0], 1)) + end * np.reshape(s, (s.shape[0], 1))


@jit(nopython=True, parallel=False)
def clip_to_sphere(poly_points):
    """Ensure 3D points do not reach outside of unit cube.
    As designed this should only correct for tiny differences that would make
    x-y-z to lon-lat conversion impossible.

    Parameters
    ----------
    poly_points : np.ndarray
        3D array of points (that should be) on unit sphere

    Returns
    -------
    poly_points : np.ndarray
        3D array of points (that should be) on unit sphere, clipped wherever
        they exceed the bounds of the unit cube.
    """
    poly_points = np.minimum(poly_points, 1)
    poly_points = np.maximum(poly_points, -1)
    return poly_points


def get_polygon_covering_pole(poly_points_lon_lat, nsign):
    """Convert a polygon defined by its edges into a polygon representing
    its latitude-longitude space comprehensively.

    Coordinates that cover poles may define polygon boundaries but not
    their relationship to a pole explicitly. For example, consider a polygon
    represented by these coordinates:

    [[0, 60], [120, 60], [240, 60], [0, 60]]

    This may represent the region of the earth above the 60-degree latitude
    line, or it may represent the region of the earth below that line. This
    function ensures an explicit definition on a projected coordinate system.

    Parameters
    ----------
    poly_points_lon_lat : np.ndarray
        2D array of coordinates (longitude, latitude) representing a polygon
        that covers a pole.

    nsign : int
        Integer representing positive (nsign == 1: north pole) or negative
        (nsign == -1: south pole) sign of latitude of the pole to be covered.

    Returns
    -------
    p : shapely.Polygon
        Polygon defined by `poly_points_lon_lat`, transformed to cover the pole
        indicated by `nsign` in latitude-longitude space.
    """
    diff = poly_points_lon_lat[1:] - poly_points_lon_lat[:-1]
    turnpoints = np.flip(np.where(np.abs(diff[:, 0]) > 180)[0])

    for turnpoint in turnpoints:
        esign = 1 if poly_points_lon_lat[turnpoint][0] > 0 else -1

        start, end = poly_points_lon_lat[turnpoint], poly_points_lon_lat[
            turnpoint + 1
        ] + np.array([360 * esign, 0])

        refpoint = 180 * esign
        opppoint = 180 * -esign

        xdiff = end[0] - start[0]
        ydiff = end[1] - start[1]

        xpart = (refpoint - start[0]) / xdiff if xdiff > 0 else 0.5

        newpt1 = [refpoint, start[1] + ydiff * xpart]
        newpt2 = [refpoint, 90 * nsign]
        newpt3 = [opppoint, 90 * nsign]
        newpt4 = [opppoint, start[1] + ydiff * xpart]

        insert_pts = np.array([newpt1, newpt2, newpt3, newpt4])

        poly_points_lon_lat = np.insert(
            poly_points_lon_lat, turnpoint + 1, insert_pts, axis=0
        )

    p = Polygon(poly_points_lon_lat)
    return p


@jit(nopython=True, parallel=False)
def ensure_validity(poly_points_lon_lat):
    """Resolve duplicate points and some floating point issues in polygons
    derived from `numba_process_points()`.

    Parameters
    ----------
    poly_points_lon_lat : np.ndarray
        2D array of longitude-latitude coordinates

    Returns
    -------
    np.ndarray
        A version of `poly_points_lon_lat` with duplicates removed and some
        floating point issues resolved.
    """
    same_as_next = np.zeros((poly_points_lon_lat.shape[0]), dtype=np.uint8)
    same_as_next = same_as_next > 1
    same_as_next[:-1] = (
        np.sum(poly_points_lon_lat[:-1] == poly_points_lon_lat[1:], axis=1) == 2
    )
    poly_points_lon_lat = poly_points_lon_lat[~same_as_next]
    out = np.empty_like(poly_points_lon_lat)
    return np.round(poly_points_lon_lat, 9, out)


@jit(nopython=True)
def numba_divide_polys_by_meridians(poly_points_lon_lat):
    """Transform polygons defined by vertices that wrap around the globe,
    into those same polygons represented as 2D shapes.

    Parameters
    ----------
    poly_points_lon_lat : np.ndarray
        2D array of longitude-latitude coordinates

    Returns
    -------
    list of np.ndarray
        List of 2D array of longitude-latitude coordinates, representing all
        2D polygons formed by `poly_points_lon_lat` when represented in
        projected space that does not wrap around the globe
    """

    diff = poly_points_lon_lat[1:] - poly_points_lon_lat[:-1]
    turnpoints = np.flip(np.where(np.abs(diff[:, 0]) > 180)[0])
    if turnpoints.shape[0] == 0:
        return [poly_points_lon_lat]
    else:
        for turnpoint in turnpoints:
            esign = 1 if poly_points_lon_lat[turnpoint][0] > 0 else -1

            start, end = poly_points_lon_lat[turnpoint], poly_points_lon_lat[
                turnpoint + 1
            ] + np.array([360 * esign, 0])

            refpoint = 180 * esign
            opppoint = 180 * -esign

            xdiff = end[0] - start[0]
            ydiff = end[1] - start[1]

            xpart = (refpoint - start[0]) / xdiff if xdiff > 0 else 0.5

            newpt1 = [refpoint, start[1] + ydiff * xpart]
            newpt4 = [opppoint, start[1] + ydiff * xpart]

            insert_pts = np.array([newpt1, newpt4])

            poly_points_lon_lat = np.concatenate(
                (
                    poly_points_lon_lat[: turnpoint + 1],
                    insert_pts,
                    poly_points_lon_lat[turnpoint + 1 :],
                ),
                axis=0,
            )

        diff = poly_points_lon_lat[1:] - poly_points_lon_lat[:-1]

        turnpoint_switches_off1 = np.zeros((diff[:, 0].shape[0]), dtype=np.int8)

        turnpoint_switches_off1[np.where(diff[:, 0] < -240)[0]] = 1
        turnpoint_switches_off1[np.where(diff[:, 0] > 240)[0]] = -1

        turnpoint_switches = np.zeros(
            (poly_points_lon_lat[:, 0].shape[0]), dtype=np.int8
        )

        turnpoint_switches[1:] = turnpoint_switches_off1

        turnpoints = np.where(turnpoint_switches)[0]

        shapeset = np.cumsum(turnpoint_switches)

        return [poly_points_lon_lat[shapeset == sh] for sh in np.unique(shapeset)]


@jit(nopython=True, parallel=False)
def interpolate_vertices_on_sphere(vertices):
    """Interpolate points in x-y-z space on a sphere.

    Parameters
    ----------
    vertices : np.ndarray
        2D array of x-y-z coordinates on the unit sphere

    Returns
    -------
    np.ndarray
        2D array of x-y-z coordinates on the unit sphere, interpolated with
        at least one point for every distance of length `precision`.
    """
    n = len(vertices)

    poly_interp_x = []
    poly_interp_y = []
    poly_interp_z = []
    ct = 0
    for i in range(n):
        precision = 1e-3
        start = vertices[i]
        end_ix = (i + 1) % n
        end = vertices[end_ix]
        dist = np.linalg.norm(start - end)
        n_pts = max(int(dist / precision), 2)
        t_vals = np.linspace(0, 1, n_pts)
        if i != n - 1:
            t_vals = t_vals[:-1]

        result = numba_geometric_slerp(start, end, t_vals)
        for x in result[:, 0]:
            poly_interp_x.append(x)
        for y in result[:, 1]:
            poly_interp_y.append(y)
        for z in result[:, 2]:
            poly_interp_z.append(z)

        ct += result.shape[0]

    return np.stack(
        (
            np.array(poly_interp_x)[:ct],
            np.array(poly_interp_y)[:ct],
            np.array(poly_interp_z)[:ct],
        ),
        axis=1,
    )


@jit(nopython=True)
def numba_process_points(vertices):
    """Densify x-y-z spherical vertices and convert to lon-lat space.

    Parameters
    ----------
    vertices : np.ndarray
        2D array of x-y-z coordinates on the unit sphere

    Returns
    -------
    poly_points_lon_lat : np.ndarray
        2D array of longitude-latitude coordinates, representing densified
        version of `vertices`.
    """
    poly_points = interpolate_vertices_on_sphere(vertices)
    poly_points = clip_to_sphere(poly_points)
    poly_points_lon_lat = xyz_to_lon_lat(poly_points)
    return poly_points_lon_lat


def get_groups_of_regions(
    loc_reg_lists, loc, sv, includes_southpole, includes_northpole, combine_by_id=True
):
    """Get Voronoi output shapes for generator points that are part of
    the same original Polygon.

    Parameters
    ----------
    loc_reg_lists : dict from object to list of int
        Mapping from the `UID` of the Voronoi-generating Polygon, to the list
        of all indices in `pts_df` with that `UID`.

    loc : object
        Some key in `loc_reg_lists`

    sv : scipy.spatial.SphericalVoronoi
        SphericalVoronoi object based on input points

    includes_southpole : bool
        Whether any of the Voronoi regions in `sv.regions` covers the south
        pole.

    includes_northpole : bool
        Whether any of the Voronoi regions in `sv.regions` covers the north
        pole.

    combine_by_id : bool
        Whether to combine all Voronoi regions with the same `UID`, or to keep
        them separate.

    Returns
    -------
    reg_group : list of int, or list of list of int
        List of indices in `sv.vertices` composing the nodes of the Voronoi
        polygon corresponding to `UID` == `loc`. If there are multiple polygons
        formed from the combination of Voronoi shapes (e.g. if two islands of
        a region are separated by an island with another `UID`), returns a list
        of these lists of indices.

    """

    reg_group = get_reg_group(loc_reg_lists, loc, sv.regions)
    if not combine_by_id:
        return reg_group

    if (not includes_southpole) and (not includes_northpole):
        # Optimization to combine points from the same region into one large shape
        # WARNING: Robust to interior rings, with `fix_ring_topology`, but not to
        # rings within those rings. This is ok as long as the related assertion
        # in `fix_ring_topology()` passes.
        candidate = combine_reg_group(reg_group)
        if len(candidate) == 1:
            reg_group = candidate

    return reg_group


def get_polys_from_cycles(
    loc_reg_lists,
    reg_cycles,
    sv,
    loc,
    includes_southpole,
    includes_northpole,
    ix_min,
    ix_max,
):
    """Transform Voronoi regions defined by `sv` on a sphere into the polygons
    they define in longitude-latitude space.

    Parameters
    ----------
    loc_reg_lists : dict from object to list of int
        Mapping from the `UID` of the Voronoi-generating Polygon, to the list
        of all indices in `pts_df` with that `UID`.

    reg_cycles : list of int, or list of list of int
        List of indices in `sv.vertices` composing the nodes of the Voronoi
        region corresponding to some `UID`. If there are multiple polygons
        formed from the combination of Voronoi shapes (e.g. if two islands of
        a region are separated by an island with another `UID`), this is a list
        of these lists of indices.

    sv : scipy.spatial.SphericalVoronoi
        SphericalVoronoi object based on input points

    loc : object
        Some key in `loc_reg_lists`

    includes_southpole : bool
        Whether any of the Voronoi regions in `sv.regions` covers the south
        pole.

    includes_northpole : bool
        Whether any of the Voronoi regions in `sv.regions` covers the north
        pole.

    ix_min : list of int
        Indices of most southerly origin points

    ix_max : list of int
        Indices of most northerly origin points

    Returns
    -------
    reg_group_polys : list of shapely.Polygon
        Polygons representing Voronoi outputs in longitude-latitude space

    reg_group_loc_ids : list of int
        `UID`s of `reg_group_polys`

    """
    reg_group_polys = []
    reg_group_loc_ids = []
    for i, reg in enumerate(reg_cycles):

        poly_points_lon_lat = numba_process_points(sv.vertices[reg])

        if (includes_southpole or includes_northpole) and (
            loc_reg_lists[loc][i] in (set(ix_max) | set(ix_min))
        ):
            nsign = 1 if loc_reg_lists[loc][i] in ix_max else -1
            poly_points_lon_lat = ensure_validity(poly_points_lon_lat)
            p = get_polygon_covering_pole(poly_points_lon_lat, nsign)
            reg_polys = [p]
        else:
            reg_polys = numba_divide_polys_by_meridians(
                ensure_validity(poly_points_lon_lat)
            )
            reg_polys = list(pygeos.to_shapely([pygeos.polygons(p) for p in reg_polys]))

        reg_group_polys += reg_polys
        reg_group_loc_ids += [loc for i in range(len(reg_polys))]

    return reg_group_polys, reg_group_loc_ids


def get_spherical_voronoi_gser(pts, show_bar=True):
    """From a list of points associated with IDs (which must be specified by
    ``pts_df.index``), calculate the region of a globe closest to each ID-set, and
    return a GeoSeries representing those "nearest" Polygons/MultiPolygons.

    Parameters
    ----------
    pts_df : :py:class:`geopandas.GeoSeries`
        GeoSeries of Points to be used as Voronoi generators.

    show_bar : bool
        Show progress bar

    Returns
    -------
    :py:class:`geopandas.GeoSeries` : GeoSeries representing Voronoi regions for each
        input row.
    """
    # Get indices of polar Voronoi regions
    lats = pts.y.values
    ymax = lats.max()
    ymin = lats.min()

    ix_max = np.where(lats == ymax)[0]
    ix_min = np.where(lats == ymin)[0]

    xyz_candidates = lon_lat_to_xyz(pts.x.values, lats)

    sv = SphericalVoronoi(
        xyz_candidates, radius=1, threshold=SPHERICAL_VORONOI_THRESHOLD
    )
    sv.sort_vertices_of_regions()

    polys = []
    loc_ids = []

    loc_reg_lists = (
        pts.rename_axis(index="UID")
        .reset_index(drop=False)
        .reset_index(drop=False)
        .groupby("UID")["index"]
        .unique()
        .to_dict()
    )

    iterator = tqdm(loc_reg_lists) if show_bar else loc_reg_lists
    for loc in iterator:
        includes_southpole = bool(set(ix_min) & set(loc_reg_lists[loc]))
        includes_northpole = bool(set(ix_max) & set(loc_reg_lists[loc]))

        reg_cycles = get_groups_of_regions(
            loc_reg_lists,
            loc,
            sv,
            includes_southpole,
            includes_northpole,
            combine_by_id=True,
        )

        reg_group_polys, reg_group_loc_ids = get_polys_from_cycles(
            loc_reg_lists,
            reg_cycles,
            sv,
            loc,
            includes_southpole,
            includes_northpole,
            ix_min,
            ix_max,
        )

        reg_group_polys, reg_group_loc_ids = fix_ring_topology(
            reg_group_polys, reg_group_loc_ids
        )
        polys += reg_group_polys
        loc_ids += reg_group_loc_ids

    # This should resolve some areas where regions are basically slivers, and
    # the geometric slerp is too long to capture the correct topology of the
    # region so that two lines of the same region cross along their planar coordinates.
    # Based on testing and our use case, these are rare and small enough to ignore,
    # and correcting for this with smaller slerp sections too computationally
    # intensive, but improvements on this would be welcome.
    polys = make_valid_shapely(polys)

    return (
        gpd.GeoDataFrame({pts.index.name: loc_ids}, geometry=polys, crs="EPSG:4326")
        .dissolve(pts.index.name)
        .geometry
    )


def append_extra_pts(sites):
    """Define three extra points at the pole farthest from any point in `sites`
    These can be necessary for compataibility with `SphericalVoronoi` when the
    number of original sites is less than four.

    Parameters
    ----------
    sites : :py:class:`geopandas.GeoSeries`
        GeoSeries of Points.

    Returns
    -------
    :py:class:`geopandas.GeoSeries`
        Same as input, but with extra points near one of the poles included.
    """
    y = sites.geometry.y
    nsign = -1 if np.abs(y.max()) > np.abs(y.min()) else 1

    out = sites.iloc[[0, 0, 0]].copy()
    out.index = pd.Index(
        ["placeholder1", "placeholder2", "placeholder3"], name=sites.index.name
    )

    out["geometry"] = gpd.GeoSeries.from_xy(
        x=[0, 0, 180],
        y=[90 * nsign, 89 * nsign, 89 * nsign],
        index=out.index,
        crs=sites.crs,
    )

    return pd.concat([sites, out])


def get_voronoi_from_sites(sites):
    """Get the Voronoi diagram corresponding to the points defined by `sites`.

    Parameters
    ----------
    sites : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of sites from which to generate a Voronoi diagram. Must include
        index, Point `geometry` field.

    Returns
    -------
    vor_gdf : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame where the geometry represents Voronoi regions for each site in
        ``sites``.
    """
    sites = remove_duplicate_points(sites)
    if sites.index.nunique() == 1:
        out = sites.copy()
        out["geometry"] = box(-180, -90, 180, 90)
    else:
        if sites.shape[0] <= 3:
            sites = append_extra_pts(sites)
        vor_gser = get_spherical_voronoi_gser(sites.geometry, show_bar=False)
        out = gpd.GeoDataFrame(
            sites.drop(columns="geometry", errors="ignore").join(vor_gser),
            crs=sites.crs,
        )
    return out


def get_stations_by_iso_voronoi(stations):
    """From the GeoDataFrame of GTSM stations with assigned ISO values,
    calculate a globally comprehensive set of shapes for each ISO mapping to the
    closest station that has that ISO.

    Parameters
    ----------
    stations : pandas.DataFrame
        A DataFrame with fields `ISO`, `lon`, and `lat`

    Returns
    -------
    out : geopandas.GeoDataFrame
        A GeoDataFrame with fields `station_id` and `geometry`, indexed by `ISO`
        `geometry` represents the region of the globe corresponding to the area
        closer to station `station_id` than any other station in that `ISO`

    """

    # Make sure none of the stations with too few points to calculate SphericalVoronoi
    # are anywhere near the poles, so we can introduce the poles as extra points
    iso_count = (
        stations.groupby("ISO")[["ISO"]].count().rename(columns={"ISO": "count"})
    )
    stations = stations.join(iso_count, on="ISO")

    lats = stations.geometry.y[stations["count"] <= 3]
    assert (lats.max() < 60) and (lats.min() > -60)

    # Iterate through each country, add each Voronoi gdf to `vors`
    all_isos = stations["ISO"].unique()
    all_isos.sort()

    vors = []
    for iso in all_isos:
        print(iso, end=" ")
        iso_stations = stations[stations["ISO"] == iso].copy()
        vors.append(get_voronoi_from_sites(iso_stations))

    # Combine all Voronoi diagrams into one GeoDataFrame (results overlap)
    vor_gdf = pd.concat(vors).drop(
        ["placeholder1", "placeholder2", "placeholder3"], errors="ignore"
    )

    # Check that ISOs match
    assert set(vor_gdf.index[vor_gdf["ISO"].isnull()].unique()) - set(
        ["placeholder1", "placeholder2", "placeholder3"]
    ) == set([])

    # Clean up
    vor_gdf = vor_gdf[vor_gdf["ISO"].notnull()].copy()
    vor_gdf["geometry"] = vor_gdf["geometry"].apply(grab_polygons)

    return vor_gdf[["ISO", "geometry"]]


def remove_duplicate_points(pts, threshold=SPHERICAL_VORONOI_THRESHOLD):
    """Remove points in DataFrame that are too close to each other to be
    recognized as different in the `SphericalVoronoi` algorithm.

    Parameters
    ----------
    pts : :py:class:`geopandas.GeoSeries`
        GeoSeries of Points

    Returns
    -------
    geopandas.DataFrame or pandas.DataFrame
        DataFrame of `pts_df` points, with duplicates removed (i.e. leave one
        of each set of duplicates).
    """

    xyz_candidates = lon_lat_to_xyz(pts.geometry.x.values, pts.geometry.y.values)

    res = cKDTree(xyz_candidates).query_pairs(threshold)

    first_point = np.array([p[0] for p in res])
    mask = np.ones(xyz_candidates.shape[0], dtype="bool")

    if len(first_point) > 0:
        mask[first_point] = False

    return pts[mask]


def remove_already_attributed_land_from_vor(
    vor_shapes,
    all_gridded,
    vor_ix,
    existing,
    vor_uid,
    gridded_uid,
    show_bar=True,
    crs=None,
):
    """Mask Voronoi regions with the pre-existing regions, so that the result
    includes only the parts of the Voronoi regions that are not already
    assigned to the pre-existing regions.

    Parameters
    ----------
    vor_shapes : array of pygeos.Geometry
        Shapes of globally comprehensive Voronoi regions

    all_gridded : array of pygeos.Geometry
        Shapes of original regions

    vor_ix : np.ndarray
        1D array of indices of `vor_shapes` intersecting with `all_gridded`

    existing : np.ndarray
        1D array of indices of Polygons in `all_gridded` intersecting with
        `vor_shapes`

    vor_uid : np.ndarray
        1D array of unique IDs corresponding with `vor_ix`

    gridded_uid : np.ndarray
        1D array of unique IDs corresponding with `existing`

    Returns
    -------
    geopandas.GeoSeries
        A GeoSeries based on `vor_shapes` that excludes the areas defined in
        `all_gridded`.
    """

    calculated = []

    iterator = range(len(vor_shapes))
    if show_bar:
        iterator = tqdm(iterator)
    for ix in iterator:
        overlapping_ix = list(existing[(vor_ix == ix) & (gridded_uid != vor_uid)])
        if len(overlapping_ix) > 0:
            overlapping_land = itemgetter(*overlapping_ix)(all_gridded)
            uu = pygeos.union_all(overlapping_land)
            remaining = pygeos.difference(vor_shapes[ix], uu)
        else:
            remaining = vor_shapes[ix]
        calculated.append(remaining)

    return gpd.GeoSeries(calculated, crs=crs)


def get_voronoi_regions(full_regions):
    """Computes a globally comprehensive set of shapes corresponding to the
    nearest regions in each place from the set of `full_regions`.

    Parameters
    ----------
    full_regions : :py:class:`geopandas.GeoDataFrame`
        Contains regions for which you want to create Voronoi shapes

    Returns
    -------
    out : :py:class:`geopandas.GeoDataFrame`
        Same as input but with the geometry defined as the Voronoi shapes.
    """

    out_cols = [c for c in full_regions.columns if c != "geometry"]

    # avoiding GeoDataFrame.explode until geopandas v0.10.3 b/c of
    # https://github.com/geopandas/geopandas/issues/2271
    # region_polys = full_regions.explode(index_parts=False)
    region_polys = full_regions.drop(columns="geometry").join(
        full_regions.geometry.explode(index_parts=False)
    )

    # This has been tested with XYZ coordinates so cannot guarantee performance on more
    # complex shapefiles
    assert (
        pygeos.count_coordinates(pygeos.from_shapely(region_polys.values))
        < sset.MAX_VORONOI_COMPLEXITY
    )

    gridded_gdf, all_oc = grid_gdf(region_polys)

    pts = polys_to_vor_pts(region_polys, all_oc)

    vor_gdf = get_spherical_voronoi_gser(pts).to_frame()

    vor_shapes = pygeos.from_shapely(vor_gdf["geometry"])
    all_gridded = pygeos.from_shapely(gridded_gdf["geometry"])

    tree = pygeos.STRtree(all_gridded)

    vor_ix, existing = tree.query_bulk(vor_shapes, "intersects")

    gridded_uid = np.take(gridded_gdf.index.values, existing)
    vor_uid = np.take(vor_gdf.index.values, vor_ix)

    vor_gdf["calculated"] = remove_already_attributed_land_from_vor(
        vor_shapes,
        all_gridded,
        vor_ix,
        existing,
        vor_uid,
        gridded_uid,
        crs=full_regions.crs,
    ).values

    full_regions = full_regions.join(vor_gdf.drop(columns="geometry"), how="left")

    full_regions["calculated"] = full_regions["calculated"].fillna(Polygon())

    full_regions["combined"] = full_regions["geometry"].union(
        full_regions["calculated"]
    )

    out = full_regions[full_regions.index.notnull()].combined.rename("geometry")

    out = out.apply(grab_polygons)
    out = out.apply(strip_line_interiors)

    out = fill_in_gaps(out)
    return gpd.GeoDataFrame(full_regions[out_cols].join(out, how="right"))


def get_points_along_segments(segments, tolerance=sset.DENSIFY_TOLERANCE):
    """Get a set of points along line segments. Calls `pygeos.segmentize()`
    to interpolate between endpoints of each line segment.

    Parameters
    ----------
    segments : :py:class:`geopandas.GeoDataFrame`
        Geometry column represents segments (as (Multi)LineStrings or
        (Multi)Polygons).

    Returns
    -------
    :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of resulting endpoints and interpolated points, with same
        non-geometry columns as ``segments``.
    """

    segments = segments[~segments.geometry.type.isnull()].copy()

    # avoiding GeoDataFrame.explode until geopandas v0.10.3 b/c of
    # https://github.com/geopandas/geopandas/issues/2271
    # segments = segments.explode(index_parts=False)
    segments = segments.drop(columns="geometry").join(
        segments.geometry.explode(index_parts=False)
    )

    segments = segments[~segments["geometry"].is_empty].copy()

    segments["geometry"] = pygeos.segmentize(
        pygeos.from_shapely(segments["geometry"]), tolerance
    )

    pts, pts_ix = pygeos.get_coordinates(
        pygeos.from_shapely(segments["geometry"]), return_index=True
    )

    return gpd.GeoDataFrame(
        segments.drop(columns="geometry").iloc[pts_ix],
        geometry=gpd.points_from_xy(pts[:, 0], pts[:, 1]),
        crs=segments.crs,
    )


def constrain_lons(arr, lon_mask):
    if lon_mask is False:
        return arr
    out = arr.copy()
    out = np.where((out > 180) & lon_mask, -360 + out, out)
    out = np.where((out <= -180) & lon_mask, 360 + out, out)
    return out


def grid_val_to_ix(
    vals: Any,
    cell_size: Union[int, Sequence],
    map_nans: Union[int, Sequence] = None,
    lon_mask: Union[bool, Sequence] = False,
) -> Any:
    """Converts grid cell lon/lat/elevation values to i/j/k values. The function is
    indifferent to order, of dimensions, but the order returned matches the order of the
    inputs, which in turn must match the order of ``cell_size``. The origin of the grid
    is the grid cell that has West, South, and bottom edges at (0,0,0) in
    (lon, lat, elev) space, and we map everything to (-180,180] longitude.

    Parameters
    ----------
    vals : array-like
        The values in lon, lat, or elevation-space. The dimensions of this array should
        be n_vals X n_dims (where dims is either 1, 2, or 3 depending on which of
        lat/lon/elev are in the array).
    cell_size : int or Sequence
        The size of a cells along the dimensions included in ``vals``. If int, applies
        to all columns of ``vals``. If Sequence, must be same length as the number of
        columns of ``vals``.
    map_nans : int or Sequence, optional
        If not None, map this value in the input array to ``np.nan`` in the output
        array. If int, applies to all columns of ``vals``. If Sequence, must be the same
        length as ``vals``, with each element applied to the corresponding column of
        ``vals``.
    lon_mask : bool or array-like, optional
        Specify an mask for values to constrain to (-180, 180] space. If value is a
        bool, apply mask to all (True) or none of (False) the input ``vals``. If value
        is array-like, must be broadcastable to the shape of ``vals`` and castable to
        bool.

    Returns
    -------
    out : array-like
        An integer dtype object of the same type as vals defining the index of each grid
        cell in ``vals``.

    Raises
    ------
    ValueError
        If `vals` contains null values but `map_nans` is None.

    Example
    -------
    >>> import numpy as np
    >>> lons = [-180.5, np.nan]
    >>> lats = [-45, 0]
    >>> elevs = [-5, 3.2]
    >>> inputs = np.stack((lons, lats, elevs)).T
    >>> grid_val_to_ix(
    ...     inputs,
    ...     cell_size=(.25, .25, .1),
    ...     map_nans=-9999,
    ...     lon_mask=np.array([1, 0, 0])
    ... ) # doctest: +NORMALIZE_WHITESPACE
    array([[    718,  -180,   -50],
           [-9999,     0,    32]], dtype=int32)
    """

    # handle nans
    nan_mask = np.isnan(vals)
    is_nans = nan_mask.sum()

    out = vals.copy()

    if is_nans != 0:
        if map_nans is None:
            raise ValueError(
                "NaNs not allowed in `vals`, unless `map_nans` is specified."
            )
        else:
            # convert to 0s to avoid warning in later type conversion
            out = np.where(nan_mask, 0, out)

    out = constrain_lons(out, lon_mask)

    # convert to index
    out = np.floor(out / cell_size).astype(np.int32)

    # fix nans to our chosen no data int value
    if is_nans:
        out = np.where(nan_mask, map_nans, out)

    return out


def grid_ix_to_val(
    vals: Any,
    cell_size: Union[int, Sequence],
    map_nans: Union[int, Sequence] = None,
    lon_mask: Union[bool, Sequence] = False,
) -> Any:
    """Converts grid cell i/j/k values to lon/lat/elevation values. The function is
    indifferent to order, of dimensions, but the order returned matches the order of the
    inputs, which in turn must match the order of ``cell_size``. The origin of the grid
    is the grid cell that has West, South, and bottom edges at (0,0,0) in
    (lon, lat, elev) space, and we map everything to (-180,180] longitude.

    Parameters
    ----------
    vals : array-like
        The values in i, j, or k-space. The dimensions of this array should be
        n_vals X n_dims (where dims is either 1, 2, or 3 depending on which of i/j/k are
        in the array).
    cell_size : Sequence
        The size of a cells along the dimensions included in ``vals``. If int, applies
        to all columns of ``vals``. If Sequence, must be same length as the number of
        columns of ``vals``.
    map_nans : int or Sequence, optional
        If not None, map this value in the input array to ``np.nan`` in the output
        array. If int, applies to all columns of ``vals``. If Sequence, must be the same
        length as ``vals``, with each element applied to the corresponding column of
        ``vals``.
    lon_mask : bool or array-like, optional
        Specify an mask for values to constrain to (-180, 180] space. If value is a
        bool, apply mask to all (True) or none of (False) the input ``vals``. If value
        is array-like, must be broadcastable to the shape of ``vals`` and castable to
        bool.

    Returns
    -------
    out : array-like
        A float dtype object of the same type as vals defining the lat/lon/elev of each
        grid cell in ``vals``.

    Raises
    ------
    AssertionError
        If `vals` is not an integer object

    Example
    -------
    >>> i = [750, 100]
    >>> j = [-3, 2]
    >>> k = [-14, 34]
    >>> inputs = np.stack((i, j, k)).T
    >>> grid_ix_to_val(
    ... inputs,
    ... cell_size=(.25, .25, .1),
    ... map_nans=-14,
    ... lon_mask=np.array([1, 0, 0])
    ... ) # doctest: +NORMALIZE_WHITESPACE
    array([[-172.375, -0.625, nan],
           [  25.125,  0.625,  3.45 ]])
    """

    assert np.issubdtype(vals.dtype, np.integer)

    out = cell_size * (vals + 0.5)
    out = constrain_lons(out, lon_mask)

    # apply nans
    if map_nans is not None:
        valid = vals != map_nans
        out = np.where(valid, out, np.nan)

    return out


def great_circle_dist(
    ax,
    ay,
    bx,
    by,
):
    """Calculate pair-wise Great Circle Distance (in km) between two sets of points.

    Note: ``ax``, ``ay``, ``bx``, ``by`` must be either:
        a. 1-D, with the same length, in which case the distances are element-wise and
           a 1-D array is returned, or
        b. Broadcastable to a common shape, in which case a distance matrix is returned.

    Parameters
    ----------
    ax, bx : array-like
        Longitudes of the two point sets
    ay, by : array-like
        Latitudes of the two point sets

    Returns
    -------
    array-like
        The distance vector (if inputs are 1-D) or distance matrix (if inputs are
        multidimensional and broadcastable to the same shape).

    Examples
    --------
    We can calculate element-wise distances

    >>> lon1 = [0, 90]
    >>> lat1 = [-45, 0]
    >>> lon2 = [10, 100]
    >>> lat2 = [-45, 10]

    >>> great_circle_dist(lon1, lat1, lon2, lat2)
    array([ 785.76833086, 1568.52277257])

    We can also create a distance matrix w/ 2-D inputs

    >>> lon1 = np.array(lon1)[:,np.newaxis]
    >>> lat1 = np.array(lat1)[:,np.newaxis]
    >>> lon2 = np.array(lon2)[np.newaxis,:]
    >>> lat2 = np.array(lat2)[np.newaxis,:]

    >>> great_circle_dist(lon1, lat1, lon2, lat2)
    array([[  785.76833086, 11576.03341028],
           [ 9223.29614889,  1568.52277257]])
    """
    radius = 6371.009  # earth radius
    lat1, lon1, lat2, lon2 = map(np.radians, (ay, ax, by, bx))

    # broadcast so it's easier to work with einstein summation below
    if all(map(lambda x: isinstance(x, xr.DataArray), (lat1, lon1, lat2, lon2))):
        broadcaster = xr.broadcast
    else:
        broadcaster = np.broadcast_arrays
    lat1, lon1, lat2, lon2 = broadcaster(lat1, lon1, lat2, lon2)

    dlat = 0.5 * (lat2 - lat1)
    dlon = 0.5 * (lon2 - lon1)

    # haversine formula:
    hav = np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
    return 2 * np.arcsin(np.sqrt(hav)) * radius


def get_great_circle_nearest_index(df1, df2, x1="lon", y1="lat", x2="lon", y2="lat"):
    """
    Finds the index in df2 of the nearest point to each element in df1

    Parameters
    ----------
    df1 : pandas.DataFrame
        Points that will be assigned great circle nearest neighbors from df2
    df2 : pandas.DataFrame
        Location of points to within which to select data
    x1 : str
        Name of x column in df1
    y1 : str
        Name of y column in df1
    x2 : str
        Name of x column in df2
    y2 : str
        Name of y column in df2

    Returns
    -------
    nearest_indices : pandas.Series
        :py:class:`pandas.Series` of indices in df2 for the nearest entries to
        each row in df1, indexed by df1's index.
    """

    dists = great_circle_dist(
        df1[[x1]].values, df1[[y1]].values, df2[x2].values, df2[y2].values
    )

    nearest_indices = pd.Series(df2.index.values[dists.argmin(axis=1)], index=df1.index)

    return nearest_indices


def coastlen_poly(
    i,
    coastlines_shp_path,
    seg_adm_voronoi_parquet_path,
    seg_var="seg_adm",
    filesystem=None,
):
    lensum = 0

    # Import coastlines, CIAM seg and ADM1 voronoi polygons
    clines = gpd.read_file(coastlines_shp_path)
    poly = gpd.read_parquet(
        seg_adm_voronoi_parquet_path,
        filesystem=filesystem,
        columns=["geometry"],
        filters=[(seg_var, "=", i)],
    )

    assert len(poly) == 1

    # Intersect polygon with coastlines
    if not clines.intersects(poly.iloc[0].loc["geometry"]).any():
        return 0
    lines_int = gpd.overlay(clines, poly, how="intersection", keep_geom_type=True)
    if len(lines_int) > 0:
        for idx0 in range(len(lines_int)):

            def line_dist(line, npts=50):
                dist = 0
                pts = get_points_on_lines(line, line.length / npts)[0]
                for p in range(1, len(pts.geoms)):
                    dist += great_circle_dist(
                        pts.geoms[p - 1].x,
                        pts.geoms[p - 1].y,
                        pts.geoms[p].x,
                        pts.geoms[p].y,
                    )
                return dist

            line = lines_int.iloc[idx0]

            if line.geometry.type == "MultiLineString":
                lines = line.geometry.explode()
                for idx1 in range(len(lines)):
                    line = lines.iloc[idx1]
                    lensum += line_dist(line)
            else:
                lensum += line_dist(line.geometry)

    return lensum


def simplify_coastlines(coastlines):
    """Read in coastlines and break them up into their component (2-point)
    line segments

    Parameters
    ----------
    coastlines : :py:class:`geopandas.GeoSeries`
        GeoSeries containing a set of global coastline `LINESTRING`s.

    Returns
    -------
    :py:class:`geopandas.GeoSeries` :
        GeoSeries containing broken-up coastlines with their original
        associated index.

    """

    coords, linestring_ix = pygeos.get_coordinates(
        pygeos.from_shapely(coastlines.values), return_index=True
    )

    start, end = coords[:-1], coords[1:]

    tiny_segs = pygeos.linestrings(
        np.stack((start[:, 0], end[:, 0]), axis=1),
        np.stack((start[:, 1], end[:, 1]), axis=1),
    )

    tiny_segs = tiny_segs[linestring_ix[:-1] == linestring_ix[1:]]

    linestring_ix = linestring_ix[:-1][linestring_ix[:-1] == linestring_ix[1:]]

    return gpd.GeoSeries(
        tiny_segs, crs=coastlines.crs, index=coastlines.iloc[linestring_ix].index
    )


def join_coastlines_to_isos(coastlines, regions_voronoi):
    """Get country-level coastlines by calculating intersection between
    coastlines and countries.

    Parameters
    ----------
    coastlines: :py:class:`geopandas.GeoSeries`
        GeoSeries representing simplified global coastlines, i.e. outputs of
        ``simplify_coastlines``.

    regions_voronoi: :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame including an `ISO` field and a `geometry` field. Should be globally
        comprehensive, with a one-to-one mapping from coordinates to ISO values.

    Returns
    -------
    joined : geopandas.GeoDataFrame
        A GeoDataFrame with fields `region_geo`, `ISO`, and `geometry`, where `geometry`
        represents the (entire) original linestring corresponding that overlaps with the
        `region_geo` defined by ``regions_voronoi``.
    """
    regions = regions_voronoi.to_crs(coastlines.crs)

    # Use regions as a proxy for countries. It's faster because the regions are more
    # narrowly located than the countries in the STRtree, but could instead subdivide
    # countries
    tree = pygeos.STRtree(pygeos.from_shapely(regions["geometry"]))

    coastal_ix, region_ix = tree.query_bulk(
        pygeos.from_shapely(coastlines), "intersects"
    )

    coastal_geo = coastlines.iloc[coastal_ix]
    regions_out = regions.iloc[region_ix]

    joined = gpd.GeoDataFrame(
        {
            "region_geo": regions_out.geometry.values,
            "ISO": regions_out.ISO.values,
        },
        geometry=coastal_geo.values,
        crs=coastal_geo.crs,
        index=coastal_geo.index,
    )

    return joined


def get_coastlines_by_iso(coastlines, regions_voronoi, plot=True):
    """Get country-level coastlines by calculating intersection between
    coastlines and countries.

    Parameters
    ----------
    coastlines : :py:class:`geopandas.GeoSeries`
        GeoSeries containing a set of global coastline `LINESTRING`s.

    regions_voronoi: :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame including an `ISO` field and a `geometry` field. Should be globally
        comprehensive, with a one-to-one mapping from coordinates to ISO values.

    plot : bool
        True to see resulting coastlines by country, False to suppress plotting

    Returns
    -------
    :py:class:`geopandas.GeoSeries`
        Indexed by country, contains coastlines for each country.
    """

    # Get coastal components (line segments)
    coastlines = simplify_coastlines(coastlines)

    # Get all matches between coastal components and regions
    coastlines = join_coastlines_to_isos(coastlines, regions_voronoi)

    # Clip matched coastal components to the regions they are matched with
    coastlines["geometry"] = coastlines["geometry"].intersection(
        coastlines["region_geo"]
    )
    coastlines = coastlines.drop(columns=["region_geo"])
    coastlines = coastlines[~coastlines["geometry"].is_empty]

    # Merge LineStrings where possible
    coastlines["geometry"] = coastlines["geometry"].apply(grab_lines)

    out = coastlines.dissolve("ISO").geometry

    # Check output
    if plot:
        coastlines.reset_index(drop=False).plot(
            color=add_rand_color(coastlines, col="ISO"), figsize=(20, 20)
        )

    return out


def get_coastal_segments_by_ciam_site(site_vor, coastlines, plot=True):
    """Generate coastal segments corresponding to each CoDEC site.

    Parameters
    ----------
    site_vor : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame with fields `ISO` and `geometry`, indexed by `station_id`,
        where `geometry` represents the region of the globe corresponding
        to the area closer to station `station_id` than any other station
        in that `ISO`. (i.e. the output of ``get_stations_by_iso_voronoi``)

    coastlines : :py:class:`geopandas.GeoSeries`
        Contains coastlines by country (i.e. the output of ``get_coastlines_by_iso``)

    Returns
    -------
    coastal_segs : :py:class:`geopandas.GeoDataFrame`
        Contains `ISO` and `geometry`, where `geometry` represents the coastline within
        some ISO that is closer to the associated `station_id` than any other site
        within that ISO.
    """

    # Join coastlines to CIAM site Voronoi
    site_vor = site_vor.join(coastlines.rename("coastline"), on="ISO", how="left")

    assert site_vor["ISO"].isnull().sum() == 0

    # Clip coastal segments within point-based Voronoi shapes
    site_vor["segment"] = site_vor["coastline"].intersection(site_vor["geometry"])

    coastal_segs = site_vor.drop(columns=["geometry", "coastline"]).rename(
        columns={"segment": "geometry"}
    )

    # Merge LineStrings where possible
    coastal_segs["geometry"] = coastal_segs["geometry"].apply(grab_lines)

    # Check output
    if plot:
        coastal_segs.plot(color=add_rand_color(coastal_segs, "ISO"), figsize=(20, 20))

    return coastal_segs


def dist_matrix(
    ax: Any, ay: Any, bx: Any, by: Any, radius: float = EARTH_RADIUS
) -> Any:
    """Get the distance matrix (in km) between two sets of points defined by lat/lon.

    Parameters
    ----------
    ax, bx : 1-d array-like
        Longitudes of the two point sets
    ay, by : 1-d array-like
        Latitudes of the two point sets

    Returns
    -------
    :class:`numpy.ndarray`
        The distance distance matrix between the two point sets.

    Example
    -------
    >>> lon1 = np.array([0, 90, 270])
    >>> lat1 = np.array([-45, 0, -60])
    >>> lon2 = np.array([10, 100])
    >>> lat2 = np.array([-45, 10])

    >>> dist_matrix(lon1, lat1, lon2, lat2)
    array([[  785.76833086, 11576.03341028],
           [ 9223.29614889,  1568.52277257],
           [ 6289.84215841, 14393.39737057]])
    """

    # broadcast manually
    ax1 = ax[:, np.newaxis].repeat(bx.shape[0], axis=1)
    ay1 = ay[:, np.newaxis].repeat(by.shape[0], axis=1)
    bx1 = bx[np.newaxis, :].repeat(ax.shape[0], axis=0)
    by1 = by[np.newaxis, :].repeat(ay.shape[0], axis=0)

    # get dist
    return great_circle_dist(ax1, ay1, bx1, by1)


def create_overlay_voronois(
    regions, seg_centroids, coastlines, overlay_name, plot=False
):
    """Create two Voronoi objects necessary for assigning values to coastal segments in
    SLIIDERS.

    Parameters
    ----------
    regions : :py:class:`geopandas.GeoDataFrame`
        Contains the Polygon/MultiPolygons of each region that you wish to run analyses
        on separately. Columns are ``ISO`` and ``geometry``. Each region must be mapped
        to a country (``ISO``).
    seg_centroids : :py:class:`pandas.DataFrame`
        Contains ``lon`` and ``lat`` columns, specifying the location of coastal segment
        centroids.
    coastlines : :py:class:`geopandas.GeoSeries`
        Contains LineStrings representing global coastlines. The index is not important.
    overlay_name : str
        What you would like the variable representing each combination of segment and
        region to be called
    plot : bool
        Whether to produce some diagnostic plots during calculation. Only valuable if
        running in an interactive setting.

    Returns
    -------
    all_overlays : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame representing Voronoi shapes, as administrative Voronoi
        regions intersected with segment-based Voronoi regions.

    ciam_polys : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame representing segment-based Voronoi regions.
    """
    # Generate global Voronoi shapes for regions
    print("Generating global Voronoi shapes for regions...")
    reg_vor = get_voronoi_regions(regions)
    reg_vor["ISO"] = regions.ISO
    adm0 = reg_vor.dissolve("ISO")

    # Assign ISO to seg centroids based on country Voronoi
    print("Assigning countries to segment centroids...")
    stations = (
        seg_centroids.rename("geometry")
        .to_frame()
        .sjoin(adm0, how="left", predicate="within")
        .rename(columns={"index_right": adm0.index.name})
    )

    # Generate ISO-level point-voronoi from CIAM points
    print("Generating within-country Voronoi shapes for segment centroids...")
    vor_gdf = get_stations_by_iso_voronoi(stations)

    # Get coastline by country
    print("Generating country-level coastlines...")
    coastlines_by_iso = get_coastlines_by_iso(coastlines, reg_vor, plot=plot)

    # Get coast-seg-by-CIAM point
    print("Assigning segments to each centroid point...")
    coastal_segs = get_coastal_segments_by_ciam_site(
        vor_gdf, coastlines_by_iso, plot=plot
    )

    # Overlap coastline vor with region vor to get spatially comprehensive seg_reg.
    print("Creating segment X region Voronoi shapes...")
    return generate_voronoi_from_segments(
        coastal_segs,
        reg_vor,
        overlay_name,
    )


def get_country_level_voronoi_gdf(all_pts_df):
    """Get Voronoi diagram within a country based on a set of points derived
    from that country's coastal segments.

    Parameters
    ----------
    all_pts_df : :py:class:`geopandas.GeoDataFrame`
        Voronoi-generator points within a country, containing `ISO` and `geometry`
        columns.

    Returns
    -------
    vor_gdf : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame representing Voronoi regions for each input point
    """

    all_isos = all_pts_df["ISO"].unique()
    all_isos.sort()

    vors = []

    for iso in all_isos:
        print(iso, end=" ")
        station_pts = all_pts_df[all_pts_df["ISO"] == iso].copy()
        vors.append(get_voronoi_from_sites(station_pts))

    vor_gdf = pd.concat(vors).drop(
        ["placeholder1", "placeholder2", "placeholder3"], errors="ignore"
    )

    # Assign ISO to point-region shapes
    assert vor_gdf["ISO"].isnull().sum() == 0

    # Aggregate to site level
    return vor_gdf.dissolve(vor_gdf.index.name)


def generate_voronoi_from_segments(segments, region_gdf, overlay_name):
    """Get global Voronoi diagram based on a set of coastal segments and
    administrative regions.

    Parameters
    ----------
    segments : :py:class:`geopandas.GeoDataFrame`
        Coastal segments, including `ISO` column.

    region_gdf : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame representing administrative Voronoi regions

    overlay_name : str
        Name of the field in the returned GeoDataFrame representing the
        intersections between administrative Voronoi regions and segment-based
        Voronoi regions.

    Returns
    -------
    all_overlays : geopandas.GeoDataFrame
        GeoDataFrame representing Voronoi shapes, as administrative Voronoi
        regions intersected with segment-based Voronoi regions.

    ciam_polys : geopandas.GeoDataFrame
        GeoDataFrame representing segment-based Voronoi regions.
    """

    all_pts_df = get_points_along_segments(segments)

    vor_gdf = get_country_level_voronoi_gdf(all_pts_df)

    # Calculate Voronoi diagram of all coastal segments, independent of ISO
    all_stations_vor = get_voronoi_from_sites(all_pts_df.drop(columns="ISO")).dissolve(
        all_pts_df.index.name
    )

    # Join ISO-level Voronoi diagrams with country shapes to get final seg-region
    # polygons

    coastal_isos = vor_gdf["ISO"].unique()
    coastal_isos.sort()
    landlocked_isos = sorted(list(set(region_gdf["ISO"].unique()) - set(coastal_isos)))

    coastal_overlays = []

    for iso in tqdm(coastal_isos):
        print(iso, end=" ")

        ciam_iso = vor_gdf[vor_gdf["ISO"] == iso].copy()

        region_iso = region_gdf[region_gdf["ISO"] == iso].copy()

        coastal_overlays.append(
            gpd.overlay(
                ciam_iso.reset_index(),
                region_iso.reset_index().drop(columns=["ISO"]),
                keep_geom_type=True,
            )
        )

    coastal_overlays = pd.concat(coastal_overlays, ignore_index=True)

    landlocked_overlays = []
    for iso in tqdm(landlocked_isos):
        print(iso, end=" ")

        region_iso = region_gdf[region_gdf["ISO"] == iso].copy()

        landlocked_overlays.append(
            gpd.overlay(
                all_stations_vor.reset_index(),
                region_iso.reset_index(),
                keep_geom_type=True,
            )
        )

    if len(landlocked_overlays):
        landlocked_overlays = pd.concat(landlocked_overlays, ignore_index=True)
        all_overlays = pd.concat(
            [landlocked_overlays, coastal_overlays], ignore_index=True
        )
    else:
        all_overlays = coastal_overlays

    assert all_overlays.is_valid.all()

    all_overlays["geometry"] = fill_in_gaps(all_overlays.geometry)

    all_overlays[overlay_name] = (
        "seg_"
        + all_overlays[segments.index.name].str.split("_").str[-1]
        + f"_{region_gdf.index.name}_"
        + all_overlays[region_gdf.index.name].astype(str)
    )

    return all_overlays


def get_degree_box(row):
    """
    Get a 1-degree box containing a centroid
    defined by row["lon"] and row["lat"]

    Parameters
    ----------
    row : dict
        A dictionary including values for "lon" and "lat" indicating the center
        of the 1-degree box

    Returns
    -------
    shapely.Polygon
        A Shapely box representing the spatial extent of the 1-degree tile
    """
    return box(
        row["lon"] - 0.5,
        row["lat"] - 0.5,
        row["lon"] + 0.5,
        row["lat"] + 0.5,
    )


def get_tile_names(df, lon_col, lat_col):
    """Get tile names in the format used by CoastalDEM.
    Defined by the southeastern point's 2-digit degree-distance
    north (N) or south (S) of the equator, and then its 3-digit
    distance east (E) or west (W) of the prime meridian.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with latitude and longitude

    lon_col : str
        Name of field representing longitude in `df`

    lat_col : str
        Name of field representing latitude in `df`

    Returns
    -------
    np.ndarray
        Array of strings. Tile names defined by latitude and longitude.
    """
    tlon = np.floor(df[lon_col]).astype(int)
    tlat = np.floor(df[lat_col]).astype(int)

    NS = np.where(tlat >= 0, "N", "S")
    EW = np.where(tlon >= 0, "E", "W")

    return (
        NS
        + np.abs(tlat).astype(int).astype(str).str.zfill(2)
        + EW
        + np.abs(tlon).astype(int).astype(str).str.zfill(3)
    )


def get_all_exp_tiles(exp_path, filter_field=None):
    """
    Get the list of tiles included in an exposure dataset.

    Parameters
    ----------
    exp_path : pathlib.Path
        Path to exposure Parquet. Must be indexed by `x_ix` and `y_ix`

    filter_field : str
        Field to filter on. Filter is set to filter out values that are not
        greater than 0.

    Returns
    -------
    np.ndarray
        1D array of unique 1-degree tile names

    """

    pq_filter = None
    if filter_field is not None:
        pq_filter = [[(filter_field, ">", 0)]]

    exp = pd.read_parquet(exp_path, columns=["x_ix", "y_ix"], filters=pq_filter)

    out = grid_ix_to_val(
        np.stack((exp["x_ix"], exp["y_ix"])).T, cell_size=sset.LITPOP_GRID_WIDTH
    )

    exp = exp.drop(columns=["x_ix", "y_ix"])
    exp["lon"] = np.floor(out[:, 0]).astype(int)
    exp["lat"] = np.floor(out[:, 1]).astype(int)

    lonlats = exp.drop_duplicates(["lon", "lat"]).reset_index(drop=True)

    lonlats["londir"] = np.where(lonlats["lon"] < 0, "W", "E")
    lonlats["latdir"] = np.where(lonlats["lat"] < 0, "S", "N")

    lonlats["lonnum"] = np.abs(lonlats["lon"]).astype(int).astype(str).str.zfill(3)
    lonlats["latnum"] = np.abs(lonlats["lat"]).astype(int).astype(str).str.zfill(2)

    lonlats["tile_name"] = (
        lonlats["latdir"] + lonlats["latnum"] + lonlats["londir"] + lonlats["lonnum"]
    )

    return lonlats["tile_name"].values


def get_bbox(tile_name):
    """
    Return bounding box from tile name in the string format "VXXHYYY"
    representing the southwestern corner of a 1-degree tile, where "V" is "N"
    (north) or "S" (south), "H" is "E" (east) or "W" (west), "XX" is a
    two-digit zero-padded number indicating the number of degrees north or
    south from 0,0, and "YYY" is a three-digit zero-padded number indicating
    the number of degrees east or west from 0,0.

    Parameters
    ----------
    tile_name : str
        Tile name in the format described above

    Returns
    -------
    shapely.Polygon
        A box representing the spatial coverage of the tile
    """
    lat_term, lon_term = tile_name[:3], tile_name[3:]

    lat_direction, lat_value = lat_term[0], int(lat_term[1:])
    lon_direction, lon_value = lon_term[0], int(lon_term[1:])

    lat_sign = 1 if lat_direction == "N" else -1
    lon_sign = 1 if lon_direction == "E" else -1

    llat = lat_sign * lat_value
    llon = lon_sign * lon_value

    ulat = llat + 1
    ulon = llon + 1

    return box(llon, llat, ulon, ulat)


def get_partial_covering_matches(elev_tile, bbox, gdf, id_name=None):
    """
    Get shapes in `gdf` that overlap with `bbox`, as flattened array corresponding
    to the indices of `elev_tile`

    Parameters
    ----------
    elev_tile : xarray.DataArray
        Elevation raster tile

    bbox : shapely.Polygon
        Bounding box of `elev_tile`

    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries to match to `elev_tile` coordinates

    id_name : str
        Field of `gdf` to use as IDs in returned `region_matches`

    Returns
    -------
    region_matches : np.ndarray
        1D array of matches between `elev_tile` and `gdf`. Represented as a
        flattened array along coordinates of `elev_tile`. If `id_name` is None,
        returns flag indicating there is some match. If `id_name` is defined,
        returns ID of the match, or -1 if there's no match.
    """
    gdf = gdf[gdf["geometry"].intersects(bbox)].copy()

    gdf["geometry"] = gdf["geometry"].intersection(bbox)
    gdf = gdf[gdf["geometry"].area > 0].copy()

    if len(gdf) == 0:
        res = np.zeros(elev_tile.size, dtype=int)
        if id_name is None:
            return res
        return res - 1

    gdf = gdf.reset_index(drop=True)

    regions = regionmask.from_geopandas(gdf, names=id_name, name="regions")

    mask = regions.mask(elev_tile.x.values, elev_tile.y.values)

    if id_name is None:
        mask = ~np.isnan(mask)

    mask_df = mask.astype(bool if id_name is None else int)

    mask_df = mask_df.to_pandas().stack().reset_index().rename(columns={0: "region_id"})

    if id_name is None:
        return mask_df["region_id"].to_numpy()

    mask_df.loc[mask_df["region_id"] < 0, "region_id"] = len(regions.names)

    region_matches = np.take(
        np.array(regions.names + [-1]), np.array(mask_df["region_id"])
    )

    return region_matches


def get_vor_matches(elev_tile, bbox, regions_df, id_name, out_name, assert_filled=True):
    """For each pixel of `elev_tile`, assign the corresponding shape in
    `regions_df`.

    Parameters
    ----------
    elev_tile : xarray.DataArray
        Elevation raster tile

    bbox : shapely.Polygon
        Bounding box of `elev_tile`

    regions_df : geopandas.GeoDataFrame
        GeoDataFrame containing geometries to match to `elev_tile` coordinates

    id_name : str
        Field of `regions_df` to use as IDs in returned array

    out_name : str
        Name to use for `id_name` field in output

    assert_filled : bool
        Whether to assert that regions are defined over the entire `elev_tile`

    Returns
    -------
    pandas.Series
        Array of matches between `elev_tile` and `regions_df`, defined by the field
        `id_name` in `regions_df`.

    """
    regions = regionmask.from_geopandas(regions_df, names=id_name, name=out_name)

    mask = regions.mask(elev_tile.x.values, elev_tile.y.values, wrap_lon=False)

    # if there are pixels without shapes, buffer shapes
    assert mask.isnull().sum().item() == 0
    mask_df = (
        mask.astype(int)
        .to_pandas()
        .stack()
        .reset_index()
        .rename(columns={0: "region_ix"})
    )

    if assert_filled:
        assert (mask_df["region_ix"] < 0).sum() == 0
    else:
        mask_df.loc[mask_df["region_ix"] < 0, "region_ix"] = len(regions.names)

    mask_df[out_name] = np.take(
        np.array(regions.names + [""]), np.array(mask_df["region_ix"])
    )

    return mask_df[out_name]


def get_empty_exp_grid(elev_tile, grid_width):
    """Initialize DataFrame template representing an exposure tile covering
    the same space as `elev_tile`.

    Parameters
    ----------
    elev_tile : xarray.DataArray
        Elevation raster tile

    grid_width : float
        Width of grid cells indexing exposure, in degrees

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with fields `lat`, `lon`, `x_ix`, and `y_ix`, providing
        template to fill in exposure fields like asset value and population
    """

    mg = np.meshgrid(elev_tile.x.values, elev_tile.y.values)

    df = pd.DataFrame({"lat": mg[1].flatten(), "lon": mg[0].flatten()})

    df["x_ix"] = grid_val_to_ix(df["lon"], grid_width)
    df["y_ix"] = grid_val_to_ix(df["lat"], grid_width)

    out_types = {
        "lat": np.float32,
        "lon": np.float32,
        "x_ix": np.int16,
        "y_ix": np.int16,
    }

    df = df.astype({k: v for k, v in out_types.items() if k in df.columns})

    return df


def get_cell_size_km(elev_tile, bbox):
    """Get approximate size of a grid cell in `elev_tile`. Assumes the median
    latitude extends over the entire cell, so that all grid cells are equal in
    size. Could be improved by evaluating size at each pixel's latitude in
    `elev_tile`.

    Parameters
    ----------
    elev_tile : xarray.DataArray
        Elevation raster tile

    bbox : shapely.Polygon
        Bounding box of `elev_tile`

    Returns
    -------
    cell_size_km : float
        Approximate size in km2 of each pixel in `elev_tile`

    """

    # grid cell area is determined by latitude
    tile_size_km = np.cos(np.deg2rad(bbox.centroid.y)) * (LAT_TO_M / 1000) ** 2
    cell_size_km = tile_size_km / elev_tile.size

    return cell_size_km


def get_closest_valid_exp_tiles(
    missing_exp_tiles, valid_exp_tiles, max_batch_comparisons=int(2e7)
):
    """Get the closest valid exposure tiles to the invalid tiles with exposure
    values that need to be re-assigned.

    Parameters
    ----------
    missing_exp_tiles : pandas.DataFrame
        Tiles with attributed exposure (asset value or population) that do not
        overlap any land areas in the Digital Elevation Model.

    valid_exp_tiles : pandas.DataFrame
        Tiles that overlap land areas in the Digital Elevation Model. (i.e.,
        candidates for re-assignment of the missing exposure)

    max_batch_comparisons : int
        Maximum number of simultaneous comparisons to make using
        `dist_matrix()`. High number of comparisons can reduce computation time
        but increase memory footprint.

    Returns
    -------
    pandas.DataFrame
        DataFrame mapping `missing_exp_tiles` to their closest
        `valid_exp_tiles`. Original indices are `x_ix` and `y_ix`, and the
        valid indices are labelled `valid_x_ix` and `valid_y_ix`.
    """

    if len(valid_exp_tiles) == 0:
        return None

    src_locs = missing_exp_tiles[["lon", "lat"]].to_numpy()
    dst_locs = valid_exp_tiles[["lon", "lat"]].to_numpy()

    total_comparisons = len(src_locs) * len(dst_locs)

    closest_ix = np.zeros(len(src_locs), dtype=int) - 1

    num_batches = int(total_comparisons / max_batch_comparisons) + 1
    batch_size_src = int(len(src_locs) / num_batches)

    for batch in range(num_batches):
        batch_start = batch * batch_size_src
        batch_end = min((batch + 1) * batch_size_src, len(src_locs))
        closest_ix_batch = dist_matrix(
            src_locs[batch_start:batch_end, 0],
            src_locs[batch_start:batch_end, 1],
            dst_locs[:, 0],
            dst_locs[:, 1],
        ).argmin(axis=1)

        closest_ix[batch_start:batch_end] = closest_ix_batch

    missing_exp_tiles["closest_ix"] = closest_ix

    missing_exp_tiles["valid_x_ix"] = np.take(
        valid_exp_tiles["x_ix"].to_numpy(),
        missing_exp_tiles["closest_ix"].to_numpy(),
    )
    missing_exp_tiles["valid_y_ix"] = np.take(
        valid_exp_tiles["y_ix"].to_numpy(),
        missing_exp_tiles["closest_ix"].to_numpy(),
    )

    return missing_exp_tiles[["x_ix", "y_ix", "valid_x_ix", "valid_y_ix"]]


def get_granular_grid(bbox, grid_width=3601, cap=sset.ELEV_CAP):
    """Generate a dummy grid on a 1-degree tile using the same format and level
    of granularity as the elevation tiles used in `sliiders` (i.e. 1 arcsec).

    Parameters
    ----------
    bbox : shapely.Polygon
        Bounding box of the 1-degree tile

    grid_width : int
        Number of pixels to use as width and height in the area defined by
        `bbox`.

    cap : int
        An arbitrary integer higher than any elevations saved in SLIIDERS-ECON
        outputs. Allows compatibility with functions that process elevation
        tiles.

    Returns
    -------
    granular_grid : xarray.DataArray
        Grid with a dummy elevation variable set at `cap`, in the same format
        as `sliiders` 1-degree elevation tiles.

    """

    size = 1 / grid_width

    llon, llat, ulon, ulat = bbox.bounds

    lons_small = np.arange(llon + (size / 2), ulon, size)
    lats_small = np.flip(np.arange(llat + (size / 2), ulat, size))

    xx, yy = [i.flatten() for i in np.meshgrid(lons_small, lats_small)]

    granular_grid = pd.DataFrame(
        {"y": yy, "x": xx, "v": cap * np.ones(grid_width**2)}
    ).set_index(["y", "x"])
    granular_grid = granular_grid.to_xarray().v

    return granular_grid


def process_landscan(
    landscan_zip,
    dir_landscan_raw,
    dir_landscan_int,
    landscan_year,
    save_to_file=True,
    NWORKERS=20,
):
    """Convert raw LandScan Zip-file into format suitable for global `sliiders`
    grid.

    Parameters
    ----------
    landscan_zip : pathlib.Path
        Path to raw Zip-file downloaded from LandScan

    dir_landscan_raw : pathlib.Path
        Path to unzipped LandScan directory containing raw files

    dir_landscan_int : pathlib.Path
        Path to directory in which to store output of this function

    landscan_year : str
        Year (i.e. version) of LandScan dataset, e.g. "2019"

    save_to_file : bool
        Whether to save the output in a file. If True, two files will be saved
        within `dir_landscan_int`, one with `x_ix`, `y_ix`, and `population`,
        and another with the additional fields `x` and `y` to represent
        longitude and latitude.

    NWORKERS : int
        Number of Dask workers with which to run this function.

    Returns
    -------
    pop_df : pandas.DataFrame
        DataFrame of global population indexed by `x_ix`, `y_ix` coordinates
    """
    warnings.filterwarnings("ignore", message=".*Reshaping is producing a large chunk*")

    ### Unzipping
    path_landscan = dir_landscan_raw / f"lspop{landscan_year}" / "hdr.adf"

    if not path_landscan.exists():
        with zipfile.ZipFile(landscan_zip, "r") as zip_ref:
            zip_ref.extractall(dir_landscan_raw)

    ### Organizing TIF to parquet

    image_name = sset.DASK_IMAGE
    gateway = Gateway()
    cluster = gateway.new_cluster(worker_image=image_name, profile="micro")
    client = cluster.get_client()
    cluster.scale(NWORKERS)
    display(cluster)

    #### Open raw population raster

    pop_ds = rioxarray.open_rasterio(path_landscan, chunks={"x": 2700, "y": 10440})
    pop_ds = pop_ds.squeeze().drop("band")

    # Replace null values with 0's
    pop_ds = pop_ds.where(pop_ds >= 0, 0)
    pop_ds = pop_ds.persist()

    #### Transform to dataframe

    pop_da = pop_ds.to_dataset(name="population")
    pop_ddf = pop_da.to_dask_dataframe()
    pop_ddf = pop_ddf.drop(columns=["spatial_ref"])
    pop_ddf = pop_ddf.persist()
    pop_ddf = pop_ddf[pop_ddf["population"] > 0].persist()

    # Bring to local
    pop_df = pop_ddf.compute()

    #### Convert coordinates to indices

    pop_df["x_ix"] = grid_val_to_ix(pop_df["x"].to_numpy(), sset.LANDSCAN_GRID_WIDTH)

    pop_df["y_ix"] = grid_val_to_ix(pop_df["y"].to_numpy(), sset.LANDSCAN_GRID_WIDTH)

    #### Drop unnecessary columns

    pop_with_xy = pop_df.copy()
    pop_df = pop_df.drop(columns=["x", "y"]).reset_index(drop=True)
    pop_with_xy = pop_with_xy.reset_index(drop=True)

    #### Save and shut down workers
    if save_to_file:
        dir_landscan_int.mkdir(exist_ok=True)
        pop_df.to_parquet(dir_landscan_int / "population.parquet", index=False)
        pop_with_xy.to_parquet(
            dir_landscan_int / "population_with_xy.parquet", index=False
        )

    cluster.scale(0)
    client.close()
    cluster.close()
    cluster.shutdown()

    return pop_df
