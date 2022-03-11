import random
import warnings
from collections import defaultdict
from operator import itemgetter
from typing import Any, Sequence, Union

import geopandas as gpd
import matplotlib._color_data as mcd
import networkx as nx
import numpy as np
import pandas as pd
import pygeos
import shapely as shp
import xarray as xr
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
    """
    Find the index in df2 of the nearest point to each element in df1

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
    for msg in sset.SPATIAL_WARNINGS_TO_IGNORE:
        warnings.filterwarnings("ignore", message=f".*{msg}*")


def add_rand_color(gdf, col=None):
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
    """
    Get a LineString or MultiLineString representing all the lines in a geometry
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
    """
    Get a Polygon or MultiPolygon representing all the polygons in a geometry
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
    """
    Remove tiny interiors from a polygon
    """
    return Polygon(
        g.exterior,
        [i for i in g.interiors if Polygon(i).area > sset.SMALLEST_INTERIOR_RING],
    )


def strip_line_interiors(g):
    """
    Remove tiny interiors from a geometry
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

    # Recursively call this function for each Polygon or Multipolygon contained in the
    # geometry
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


def fill_in_gaps(gdf):
    """Fill in small spaces between shapes to produce a globally comprehensive
    GeoDataFrame."""
    uu = gdf.unary_union
    missing = box(-180, -90, 180, 90).difference(uu)

    assert all([g.type == "Polygon" for g in missing.geoms])

    intersects_missing_mask = gdf["geometry"].intersects(missing)
    intersects_missing = gdf[intersects_missing_mask].copy()

    current_coverage = missing

    for buffer_size in tqdm([0.01, 0.01, 0.01, 0.03, 0.05, 0.1, 0.1, 0.1]):
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

        intersects_missing["new_buffer"] = gpd.GeoSeries(
            new_buffers, index=intersects_missing.index
        ).buffer(0.00001)
        use_new_buffer_mask = intersects_missing["new_buffer"].geometry.area > 0
        intersects_missing.loc[
            use_new_buffer_mask, "geometry"
        ] = intersects_missing.loc[use_new_buffer_mask, "geometry"].union(
            intersects_missing.loc[use_new_buffer_mask, "new_buffer"]
        )

    assert current_coverage.area == 0

    gdf = gdf[~intersects_missing_mask].copy()

    gdf = pd.concat(
        [gdf, intersects_missing.drop(columns=["buffer", "new_buffer"])],
        ignore_index=True,
    )

    assert intersects_missing.is_valid.all()

    return gdf


def get_polys_in_slab(all_polys, lx, ly, ux, uy):
    """
    Get the subset of shapes in `all_polys` that overlap with the box defined
    by `lx`, `ly`, `ux`, `uy`
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
    orig_geo_col="geometry",
    orig_id_col="UID",
    box_size=sset.DEFAULT_BOX_SIZE,
    show_bar=True,
):
    """
    Divide a GeoDataFrame into a grid, returning the gridded shape-parts and the
    "empty" areas, each nested within a `box_size`-degree-width square

    Note: This may be deprecated in a future version if something like this
    becomes available: https://github.com/pygeos/pygeos/pull/256
    """
    orig_geos = pygeos.from_shapely(orig_gdf[orig_geo_col])

    llon, llat, ulon, ulat = orig_gdf.total_bounds

    boxes = []
    ixs = []
    all_oc = []
    iterator = np.arange(llon - 1, ulon + 1, box_size)
    if show_bar:
        iterator = tqdm(iterator)
    for lx in iterator:
        ux = lx + box_size
        vertical_slab, slab_polys = get_polys_in_slab(orig_geos, lx, llat, ux, ulat)
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

    gridded_gdf = gpd.GeoDataFrame(
        {"orig_ix": geom_ix}, geometry=pygeos.to_shapely(geom)
    )
    gridded_gdf["UID"] = np.take(
        orig_gdf[orig_id_col].to_numpy(), gridded_gdf["orig_ix"].to_numpy()
    )

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
    """From a set of points and IDs, divide points into categories according to their
    proximity to the "coast" (i.e. the edges of the union of all original polygons),
    and their proximity to points with other IDs. Proximity to the coast is calculated
    from being within `at_blank_tolerance`, and proximity to points with other IDs is
    within `tolerance`.
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
    total_tolerance=1,
):
    """
    Simplify coastal Voronoi generators that are not near the border of
    another administrative region.
    """
    border_tree = cKDTree(coastal_border_pts)

    d, i = border_tree.query(coastal_coastal_pts, distance_upper_bound=1)

    already_simplified = np.zeros_like(coastal_coastal_pts[:, 0], dtype="bool")
    non_border = []
    non_border_gadm = []

    for UPPER_BOUND in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        if UPPER_BOUND <= tolerance:
            break
        if UPPER_BOUND >= total_tolerance:
            continue

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
    """
    Transform an array of shapes into an array of coordinate pairs, keeping the
    IDs of shapes aligned with the coordinates.
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
    """
    Create a set of Voronoi region generator points from a set of shapes.
    """
    densified = pygeos.segmentize(pygeos.from_shapely(regions["geometry"]), tolerance)

    pts, pt_gadm_ids = explode_gdf_to_pts(densified, regions["UID"].to_numpy())

    all_oc_densified = pygeos.segmentize(all_oc, sset.MARGIN_DIST)

    (
        coastal_coastal_pts,
        coastal_border_pts,
        coastal_coastal_gadm,
        coastal_border_gadm,
    ) = divide_pts_into_categories(
        pts, pt_gadm_ids, all_oc_densified, sset.DENSIFY_TOLERANCE
    )

    non_border, non_border_gadm, now_border, now_border_gadm = simplify_nonborder(
        coastal_coastal_pts,
        coastal_border_pts,
        coastal_coastal_gadm,
        coastal_border_gadm,
        tolerance=sset.MARGIN_DIST,
        # total_tolerance=total_tolerance,
    )

    vor_pts = np.concatenate([non_border, now_border, coastal_border_pts])
    vor_gadm = np.concatenate([non_border_gadm, now_border_gadm, coastal_border_gadm])

    pts_df = pd.DataFrame({"x": vor_pts[:, 0], "y": vor_pts[:, 1], "UID": vor_gadm})

    pts_df = remove_duplicate_points(pts_df)

    return pts_df


def get_hemisphere_shape(hemisphere):
    """
    Define Shapely boxes for each hemisphere and the globe.
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
    """
    Helper function to make use of `pygeos.make_valid()` directly on Shapely objects.
    Can likely be deprecated in Shapely 2.0
    """
    return pygeos.to_shapely(pygeos.make_valid(pygeos.from_shapely(g)))


def clip_geoseries_by_rect(gs, rect):
    """
    Helper function to make use of `pygeos.clip_by_rect()` directly on
    geopandas.GeoSeries. Can likely be deprecated in Shapely 2.0
    """
    try:
        return gpd.GeoSeries(
            pygeos.to_shapely(
                pygeos.clip_by_rect(pygeos.from_shapely(gs), *rect.bounds)
            )
        )
    except Exception:  # weird issue with CYM, clip_by_rect doesn't work
        return gs.apply(lambda g: g.intersection(rect))


def diff_geoseries(gs1, gs2):
    """
    Helper function to make use of `pygeos.difference()` directly on
    geopandas.GeoSeries. Can likely be deprecated in Shapely 2.0
    """
    return gpd.GeoSeries(
        pygeos.to_shapely(
            pygeos.difference(pygeos.from_shapely(gs1), pygeos.from_shapely(gs2))
        )
    )


@jit(nopython=True, parallel=False)
def lon_lat_to_xyz(lons, lats):
    """
    Optimized transformation from longitude / latitude to x / y / z cube
    with centroid [0, 0, 0]
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
    """
    Optimized transformation from x / y / z cube with centroid [0, 0, 0]
    to longitude / latitude
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lats = np.arcsin(z)
    lons = np.arctan2(y, x)
    return np.stack((np.degrees(lons.flatten()), np.degrees(lats.flatten())), axis=1)


def combine_reg_group(reg_group):
    """
    Combine tesselated triplets on a sphere to get the points defining region
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


def adjust_vor_shapes_to_projected_space(polys_gdf):
    polys_gdf = polys_gdf[(~polys_gdf.is_empty) & (polys_gdf.area.notnull())].copy()

    polys_gdf.loc[(~polys_gdf.is_valid), "geometry"] = polys_gdf.loc[
        (~polys_gdf.is_valid), "geometry"
    ].apply(make_valid_shapely)

    polys_gdf = polys_gdf.dissolve(by="UID")
    polys_gdf = polys_gdf.reset_index(drop=False)

    polys_gdf["geometry"] = polys_gdf["geometry"].apply(grab_polygons)
    polys_gdf["geometry"] = polys_gdf["geometry"].apply(strip_line_interiors)

    polys_gdf["west"] = clip_geoseries_by_rect(
        polys_gdf["geometry"], get_hemisphere_shape("west")
    )
    polys_gdf["east"] = clip_geoseries_by_rect(
        polys_gdf["geometry"], get_hemisphere_shape("east")
    )
    polys_gdf["off"] = diff_geoseries(
        polys_gdf["geometry"], get_hemisphere_shape("both")
    )

    east_polys = (
        polys_gdf[~polys_gdf["east"].is_empty]
        .drop(columns=["west", "off", "geometry"])
        .rename(columns={"east": "geometry"})
    )
    west_polys = (
        polys_gdf[~polys_gdf["west"].is_empty]
        .drop(columns=["east", "off", "geometry"])
        .rename(columns={"west": "geometry"})
    )
    off_polys = (
        polys_gdf[~polys_gdf["off"].is_empty]
        .drop(columns=["west", "east", "geometry"])
        .rename(columns={"off": "geometry"})
    )

    off_polys["geometry"] = off_polys["geometry"].translate(xoff=360)

    full_gdf = pd.concat([east_polys, off_polys, west_polys], ignore_index=True)

    full_gdf = full_gdf.dissolve(by="UID").reset_index(drop=False)
    full_gdf["geometry"] = full_gdf["geometry"].apply(grab_polygons)

    return full_gdf


@jit(nopython=True)
def numba_geometric_slerp(start, end, t):
    """
    Adapted from:
    https://github.com/scipy/scipy/blob/master/scipy/spatial/_geometric_slerp.py
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
    poly_points = np.minimum(poly_points, 1)
    poly_points = np.maximum(poly_points, -1)
    return poly_points


@jit(nopython=True, parallel=False)
def geoize_shapes(vertices):
    poly_interp_points = []
    n = len(vertices)

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
        poly_interp_points.append(result)

    return poly_interp_points


def get_polygon_covering_pole(poly_points_lon_lat, nsign):
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
    """Not robust, just resolves duplicate points and floating point issues"""
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
    """
    Insert interpolated points in x-y-z space
    on a sphere. Use geometric slerp to interpolate,
    with at least one point for every distance of length `precision`.
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
    poly_points = interpolate_vertices_on_sphere(vertices)
    poly_points = clip_to_sphere(poly_points)
    poly_points_lon_lat = xyz_to_lon_lat(poly_points)
    return poly_points_lon_lat


def get_groups_of_regions(
    loc_reg_lists, loc, sv, includes_southpole, includes_northpole, combine_by_id=True
):
    reg_group = get_reg_group(loc_reg_lists, loc, sv.regions)
    if not combine_by_id:
        return reg_group

    if (not includes_southpole) and (not includes_northpole):
        # Optimization to combine points from the same region into one large shape
        # WARNING: Robust to interior rings, with `fix_ring_topology`, but not to
        # rings within those rings
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


def get_spherical_voronoi_gdf(pts_df, show_bar=True):
    """
    From a list of points associated with IDs (which must be specified by 'UID'),
    calculate the region of a globe closest to each ID-set, and return a
    GeoDataFrame representing those "nearest" Polygons/MultiPolygons. Improvements
    needed: Ensure entire surface area of globe is covered exactly once.
    Attempts have been made with `fix_ring_topology()`.
    """

    # Get indices of polar Voronoi regions
    ymax = pts_df["y"].max()
    ymin = pts_df["y"].min()

    ix_max = np.where(pts_df["y"] == ymax)[0]
    ix_min = np.where(pts_df["y"] == ymin)[0]

    xyz_candidates = lon_lat_to_xyz(pts_df["x"].to_numpy(), pts_df["y"].to_numpy())

    sv = SphericalVoronoi(
        xyz_candidates, radius=1, threshold=SPHERICAL_VORONOI_THRESHOLD
    )
    sv.sort_vertices_of_regions()

    polys = []
    loc_ids = []

    loc_reg_lists = (
        pts_df.reset_index(drop=True)
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

    polys_gdf = gpd.GeoDataFrame({"UID": loc_ids}, geometry=polys, crs="EPSG:4326")

    # This should resolve some areas where regions are basically slivers, and
    # the geometric slerp is too long to capture the correct topology of the
    # region so that two lines of the same region cross along their planar coordinates.
    # Based on testing and our use case, these are rare and small enough to ignore,
    # and correcting for this with smaller slerp sections too computationally
    # intensive, but improvements on this would be welcome.
    polys_gdf["geometry"] = make_valid_shapely(polys_gdf["geometry"])

    polys_gdf = polys_gdf.dissolve("UID").reset_index(drop=False)

    return polys_gdf


def remove_duplicate_points(pts_df):

    xyz_candidates = lon_lat_to_xyz(pts_df["x"].to_numpy(), pts_df["y"].to_numpy())

    res = cKDTree(xyz_candidates).query_pairs(SPHERICAL_VORONOI_THRESHOLD * 1.0)

    first_point = np.array([p[0] for p in res])
    mask = np.ones(xyz_candidates.shape[0], dtype="bool")

    if len(first_point) > 0:
        mask[first_point] = False

    pts_df = pts_df[mask].copy().reset_index(drop=True)

    return pts_df


def remove_already_attributed_land_from_vor(
    existing, vor_shapes, vor_ix, gridded_uid, vor_uid, all_gridded, show_bar=True
):
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

    return gpd.GeoSeries(pygeos.to_shapely(calculated))


def split_into_rings(regions):
    regions["interiors"] = regions["geometry"].interiors

    region_interiors = regions.loc[regions["interiors"].str.len() > 0].copy()
    region_interiors = region_interiors.drop(columns=["geometry"])
    region_interiors["geometry"] = region_interiors["interiors"].apply(
        lambda lrs: MultiPolygon([Polygon(lr) for lr in lrs])
    )
    region_interiors = region_interiors.explode().reset_index(drop=True)

    regions = pd.concat([regions, region_interiors], ignore_index=True).drop(
        columns=["interiors"]
    )

    regions["geometry"] = regions["geometry"].exterior

    return regions


def get_segs_from_regions(all_coords):

    all_seg_pts = []
    all_seg_indices = []
    all_polys = []

    prev_high_index = -1
    for coords, poly_id in zip(all_coords, np.arange(all_coords.shape[0])):
        seg_pts = np.repeat(coords, 2, axis=0)[1:-1]
        seg_indices = np.repeat(np.arange(coords.shape[0] - 1), 2)
        all_polys.append(np.repeat(poly_id, seg_indices[-1] + 1))

        seg_indices = seg_indices + prev_high_index + 1
        prev_high_index = seg_indices[-1]

        all_seg_pts.append(seg_pts)
        all_seg_indices.append(seg_indices)

    all_seg_pts = np.concatenate(all_seg_pts)
    all_seg_indices = np.concatenate(all_seg_indices)
    all_polys = np.concatenate(all_polys)

    all_segs = pygeos.linestrings(all_seg_pts, indices=all_seg_indices)

    return all_segs, all_polys


####################
####################
####################
####################
####################
####################

LAT_TO_M = 111131.745
EARTH_RADIUS = 6371.009


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
    coastlines_shp_path=sset.PATH_CIAM_COASTLINES,
    seg_adm_voronoi_parquet_path=sset.PATH_CIAM_ADM1_VORONOI_INTERSECTIONS,
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
    lines_int = gpd.overlay(clines, poly, how="intersection")
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
                lines = line.explode().geometry
                for idx1 in range(len(lines)):
                    line = lines.iloc[idx1]
                    lensum += line_dist(line)
            else:
                lensum += line_dist(line.geometry)

    return lensum


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
