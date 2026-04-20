"""
Transit connectivity analysis.

For each worker in LODES, check if they can actually reach their
workplace by bus: walk 0.5mi to a stop, ride with up to 2 transfers,
walk 0.5mi to work. If not, they're "stranded".
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from collections import defaultdict

WALK_RADIUS_MI = 0.5
WALK_RADIUS_DEG = WALK_RADIUS_MI / 69

TRANSFER_RADIUS_MI = 0.1   # ~500ft, close enough to transfer between routes
TRANSFER_RADIUS_DEG = TRANSFER_RADIUS_MI / 69


def build_bg_centroids(geometry):
    geo = geometry.to_crs("EPSG:4326").copy()
    geo["lat"] = geo.geometry.centroid.y
    geo["lon"] = geo.geometry.centroid.x
    return geo[["GEOID", "lat", "lon"]]


def map_bgs_to_routes(bg_centroids, stop_locs, stop_route_map):
    """For each block group, find which transit routes are within walking distance."""
    stop_tree = cKDTree(stop_locs[["stop_lat", "stop_lon"]].values)
    stop_ids = stop_locs["stop_id"].values

    bg_routes = {}
    for _, row in bg_centroids.iterrows():
        nearby_idx = stop_tree.query_ball_point(
            [row["lat"], row["lon"]], WALK_RADIUS_DEG
        )
        routes = set()
        for idx in nearby_idx:
            routes.update(stop_route_map.get(stop_ids[idx], set()))
        if routes:
            bg_routes[row["GEOID"]] = routes

    route_bgs = defaultdict(set)
    for bg_id, routes in bg_routes.items():
        for r in routes:
            route_bgs[r].add(bg_id)

    return bg_routes, dict(route_bgs)


def find_route_transfers(stop_route_map, stop_locs):
    """Figure out which routes can transfer to each other (stops within ~500ft)."""
    route_coords = defaultdict(list)
    for stop_id, routes in stop_route_map.items():
        loc = stop_locs[stop_locs["stop_id"] == stop_id]
        if loc.empty:
            continue
        coord = (loc.iloc[0]["stop_lat"], loc.iloc[0]["stop_lon"])
        for r in routes:
            route_coords[r].append(coord)

    route_names = list(route_coords.keys())
    transfers = defaultdict(set)

    for i, r1 in enumerate(route_names):
        tree1 = cKDTree(np.array(route_coords[r1]))
        for j, r2 in enumerate(route_names):
            if i >= j:
                continue
            for pt in route_coords[r2]:
                dist, _ = tree1.query(pt, k=1)
                if dist < TRANSFER_RADIUS_DEG:
                    transfers[r1].add(r2)
                    transfers[r2].add(r1)
                    break

    return dict(transfers)


def build_reachability(bg_routes, route_bgs, transfers):
    """For each BG with transit, find all other BGs reachable with up to 2 transfers."""
    reachability = {}
    for bg_id, routes in bg_routes.items():
        # expand to 1 transfer
        routes_1t = set(routes)
        for r in routes:
            routes_1t.update(transfers.get(r, set()))

        # expand to 2 transfers
        routes_2t = set(routes_1t)
        for r in routes_1t:
            routes_2t.update(transfers.get(r, set()))

        # collect all BGs served by any reachable route
        reachable = set()
        for r in routes_2t:
            reachable.update(route_bgs.get(r, set()))
        reachable.discard(bg_id)
        reachability[bg_id] = reachable

    return reachability


def build_transit_graph(geometry, stop_locs, stop_route_map):
    """Full pipeline: geometry + GTFS -> reachability graph."""
    bg_centroids = build_bg_centroids(geometry)
    bg_routes, route_bgs = map_bgs_to_routes(bg_centroids, stop_locs, stop_route_map)
    transfers = find_route_transfers(stop_route_map, stop_locs)
    reachability = build_reachability(bg_routes, route_bgs, transfers)

    n_transfer_pairs = sum(len(v) for v in transfers.values()) // 2
    stats = {
        "total_bgs": len(bg_centroids),
        "bgs_with_transit": len(bg_routes),
        "bgs_without_transit": len(bg_centroids) - len(bg_routes),
        "routes": len(route_bgs),
        "transfer_pairs": n_transfer_pairs,
    }

    return reachability, bg_centroids, stats


def check_commutes(lodes_df, reachability, county_fips):
    """Check each home->work commute pair: can transit get them there?
    Only checks internal commutes (both ends in the county)."""
    county_od = lodes_df[
        (lodes_df["home_bg"].str[:5] == county_fips) &
        (lodes_df["work_bg"].str[:5] == county_fips)
    ].copy()

    county_od["connected"] = [
        work in reachability.get(home, set())
        for home, work in zip(county_od["home_bg"], county_od["work_bg"])
    ]

    return county_od


def summarize_stranded(checked_df):
    """Break down stranded workers by where they live and where they work."""
    stranded = checked_df[~checked_df["connected"]].copy()
    stranded["home_tract"] = stranded["home_bg"].str[:11]
    stranded["work_tract"] = stranded["work_bg"].str[:11]

    total_workers = checked_df["total_workers"].sum()
    connected_workers = checked_df[checked_df["connected"]]["total_workers"].sum()
    stranded_workers = stranded["total_workers"].sum()

    home_summary = stranded.groupby("home_tract").agg(
        stranded=("total_workers", "sum"),
        low_wage=("low_wage", "sum"),
        workplaces=("work_bg", "nunique"),
    ).reset_index()

    tract_totals = checked_df.copy()
    tract_totals["home_tract"] = tract_totals["home_bg"].str[:11]
    tract_totals = tract_totals.groupby("home_tract")["total_workers"].sum().to_dict()
    home_summary["total"] = home_summary["home_tract"].map(tract_totals)
    home_summary["pct_stranded"] = home_summary["stranded"] / home_summary["total"]

    work_summary = stranded.groupby("work_tract").agg(
        stranded_inbound=("total_workers", "sum"),
        low_wage=("low_wage", "sum"),
        from_tracts=("home_tract", "nunique"),
    ).reset_index()

    totals = {
        "total_commuters": int(total_workers),
        "connected": int(connected_workers),
        "stranded": int(stranded_workers),
        "pct_connected": connected_workers / max(total_workers, 1),
        "pct_stranded": stranded_workers / max(total_workers, 1),
    }

    return home_summary, work_summary, stranded, totals
