"""
Route recommendation engine.

Given stranded workers (from connectivity analysis), find the highest-impact
corridors for new transit routes. Groups feeder tracts by compass direction
from each job center so each proposed route is a plausible linear corridor.
"""

import pandas as pd
import numpy as np
from math import atan2, degrees


def find_top_job_centers(work_summary, top_n=10):
    """Rank job centers by number of stranded workers needing to reach them."""
    return work_summary.nlargest(top_n, "stranded_inbound").reset_index(drop=True)


def get_tract_centroids(geometry):
    """Compute centroids per tract from block group geometry."""
    geo = geometry.to_crs("EPSG:4326").copy()
    geo["tract"] = geo["GEOID"].str[:11]
    geo["lat"] = geo.geometry.centroid.y
    geo["lon"] = geo.geometry.centroid.x
    tract_cents = geo.groupby("tract").agg(
        lat=("lat", "mean"), lon=("lon", "mean")
    ).to_dict("index")
    return tract_cents


def _distance_mi(lat1, lon1, lat2, lon2):
    """Rough distance in miles from lat/lon."""
    return ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5 * 69


def _bearing(lat1, lon1, lat2, lon2):
    """Compass bearing in degrees from point 1 to point 2 (0=N, 90=E, etc.)"""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    angle = degrees(atan2(dlon, dlat))  # 0=N, 90=E
    return angle % 360


DIRECTION_NAMES = {
    0: "N", 1: "NE", 2: "E", 3: "SE", 4: "S", 5: "SW", 6: "W", 7: "NW"
}


def _direction_bucket(bearing):
    """Convert bearing to one of 8 compass directions."""
    bucket = int((bearing + 22.5) / 45) % 8
    return bucket


def _direction_label(bucket):
    return DIRECTION_NAMES[bucket]


def find_feeder_tracts(stranded_df, job_center_tract, tract_centroids,
                       min_workers=10, max_dist_mi=20):
    # min_workers=10: below this the tract contributes negligible demand
    # max_dist_mi=20: typical max length for an urban bus route (APTA data
    # shows avg US bus route is 10-15 mi; 20 gives some headroom)
    """
    For a given job center, find residential tracts sending stranded workers,
    with distance and direction from the job center.
    """
    feeders = (stranded_df[stranded_df["work_tract"] == job_center_tract]
               .groupby("home_tract")
               .agg(workers=("total_workers", "sum"),
                    low_wage=("low_wage", "sum"))
               .reset_index())

    feeders = feeders[feeders["workers"] >= min_workers]

    jc = tract_centroids.get(job_center_tract)
    if not jc:
        return feeders.head(0)

    dists, bearings, lats, lons = [], [], [], []
    for _, row in feeders.iterrows():
        tc = tract_centroids.get(row["home_tract"])
        if tc:
            dists.append(_distance_mi(tc["lat"], tc["lon"], jc["lat"], jc["lon"]))
            bearings.append(_bearing(jc["lat"], jc["lon"], tc["lat"], tc["lon"]))
            lats.append(tc["lat"])
            lons.append(tc["lon"])
        else:
            dists.append(None)
            bearings.append(None)
            lats.append(None)
            lons.append(None)

    feeders["dist_mi"] = dists
    feeders["bearing"] = bearings
    feeders["lat"] = lats
    feeders["lon"] = lons

    feeders = feeders.dropna(subset=["dist_mi"])
    feeders = feeders[feeders["dist_mi"] <= max_dist_mi]

    # Assign compass direction bucket
    feeders["direction"] = feeders["bearing"].apply(_direction_bucket)
    feeders["direction_name"] = feeders["direction"].apply(_direction_label)

    return feeders.sort_values("workers", ascending=False)


def cluster_feeders_by_direction(feeders, max_stops=8):
    """
    Group feeder tracts by compass direction from the job center.
    Each group = a potential linear route corridor.

    max_stops=8: typical urban bus routes have 3-8 major stops per
    directional segment. Keeps routes realistic and visualization clean.
    """
    if feeders.empty:
        return []

    corridors = []
    for direction, group in feeders.groupby("direction"):
        # Take the top stops by worker count, then order by distance for the route path
        top_stops = group.nlargest(max_stops, "workers").sort_values("dist_mi")

        corridors.append({
            "direction": int(direction),
            "direction_name": _direction_label(direction),
            "workers": int(top_stops["workers"].sum()),
            "low_wage": int(top_stops["low_wage"].sum()),
            "num_tracts": len(top_stops),
            "avg_dist_mi": round(top_stops["dist_mi"].mean(), 1),
            "max_dist_mi": round(top_stops["dist_mi"].max(), 1),
            "tracts": [{
                "tract": row["home_tract"],
                "workers": int(row["workers"]),
                "low_wage": int(row["low_wage"]),
                "dist_mi": round(row["dist_mi"], 1),
                "lat": row["lat"],
                "lon": row["lon"],
            } for _, row in top_stops.iterrows()],
        })

    corridors.sort(key=lambda c: c["workers"], reverse=True)
    return corridors


def recommend_routes(stranded_df, work_summary, geometry,
                     top_centers=3, corridors_per_center=3, min_corridor_workers=50):
    # min_corridor_workers=50: below ~50 daily riders a bus route is not
    # cost-effective (TCRP standards suggest ~10-15 boardings/service hour,
    # which translates to roughly 120-180/day for a 12-hr route)
    """
    For each of the top job centers, find the best 3 directional corridors.
    Each corridor includes all feeder tracts in that direction — no cap on stops.
    """
    tract_centroids = get_tract_centroids(geometry)
    job_centers = find_top_job_centers(work_summary, top_n=top_centers)

    all_routes = []
    for _, jc in job_centers.iterrows():
        jc_tract = jc["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})

        feeders = find_feeder_tracts(stranded_df, jc_tract, tract_centroids)
        corridors = cluster_feeders_by_direction(feeders)

        # Take top 3 corridors for this job center
        kept = 0
        for corridor in corridors:
            if corridor["workers"] < min_corridor_workers:
                continue
            if kept >= corridors_per_center:
                break

            all_routes.append({
                "job_center": jc_tract,
                "jc_stranded_total": int(jc["stranded_inbound"]),
                "jc_lat": jc_loc.get("lat"),
                "jc_lon": jc_loc.get("lon"),
                "direction": corridor["direction_name"],
                "route_workers": corridor["workers"],
                "route_low_wage": corridor["low_wage"],
                "num_feeders": corridor["num_tracts"],
                "avg_dist_mi": corridor["avg_dist_mi"],
                "max_dist_mi": corridor["max_dist_mi"],
                "feeders": corridor["tracts"],
            })
            kept += 1

    return all_routes
