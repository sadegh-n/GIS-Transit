"""
Route recommendation engine.

Given stranded workers (from connectivity analysis), find the highest-impact
corridors for new transit routes. Groups feeder tracts by compass direction
from each job center so each proposed route is a plausible linear corridor.
"""

import pandas as pd
import numpy as np
from math import atan2, degrees, radians, sin, cos


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
                       min_workers=10, min_dist_mi=1.0, max_dist_mi=20):
    # min_workers=10: below this the tract contributes negligible demand
    # min_dist_mi=1.0: closer than this, workers can walk
    # max_dist_mi=20: typical max length for an urban bus route
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
    feeders = feeders[(feeders["dist_mi"] >= min_dist_mi) &
                      (feeders["dist_mi"] <= max_dist_mi)]

    # Assign compass direction bucket
    feeders["direction"] = feeders["bearing"].apply(_direction_bucket)
    feeders["direction_name"] = feeders["direction"].apply(_direction_label)

    return feeders.sort_values("workers", ascending=False)


def _lateral_offset_mi(jc_lat, jc_lon, pt_lat, pt_lon, corridor_bearing):
    """How far off the corridor center-line a point is, in miles."""
    # Vector from JC to point
    dlat = pt_lat - jc_lat
    dlon = pt_lon - jc_lon
    # Corridor direction as unit vector
    b = radians(corridor_bearing)
    along = dlat * cos(b) + dlon * sin(b)  # projection along corridor
    perp = -dlat * sin(b) + dlon * cos(b)  # perpendicular offset
    return abs(perp) * 69  # convert degrees to miles


def cluster_feeders_by_direction(feeders, max_stops=8, max_lateral_mi=3.0):
    """
    Group feeder tracts by compass direction from the job center.
    Filters out tracts that deviate too far laterally from the corridor
    center-line, reducing zigzag. Orders remaining tracts by distance
    from the job center to form a plausible linear route.
    """
    if feeders.empty:
        return []

    # Need JC coordinates — infer from the bearing origin
    # All bearings are from JC, so we can get JC lat/lon from any row
    # by reversing the bearing+distance. Simpler: just use the group's
    # average bearing as the corridor direction and filter lateral outliers.

    corridors = []
    for direction, group in feeders.groupby("direction"):
        if group.empty:
            continue

        # Corridor center bearing (average of all bearings in this direction)
        avg_bearing = group["bearing"].mean()

        # For lateral filtering, we need the JC position
        # Approximate by working backwards from the nearest feeder
        nearest = group.nsmallest(1, "dist_mi").iloc[0]
        b = radians(avg_bearing)
        jc_lat_approx = nearest["lat"] - (nearest["dist_mi"] / 69) * cos(b)
        jc_lon_approx = nearest["lon"] - (nearest["dist_mi"] / 69) * sin(b)

        # Filter out lateral outliers
        group = group.copy()
        group["lateral"] = group.apply(
            lambda r: _lateral_offset_mi(
                jc_lat_approx, jc_lon_approx,
                r["lat"], r["lon"], avg_bearing
            ), axis=1
        )
        on_corridor = group[group["lateral"] <= max_lateral_mi]

        if on_corridor.empty:
            on_corridor = group  # fallback: keep all if filter is too strict

        # Take top stops by workers, then order by distance
        top_stops = on_corridor.nlargest(max_stops, "workers").sort_values("dist_mi")

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


def _compute_corridor_opp_score(corridor_tracts, sld_df):
    """
    Compute average Opportunity Score for the tracts in a corridor.
    Opp Score = R_PCTLOWWAGE * (P_WrkAge - Workers/TotPop) * (1 - transit_ratio)
    Higher = more underserved, higher priority for investment.
    """
    tract_ids = [t["tract"] for t in corridor_tracts]
    sld_tracts = sld_df[sld_df["tract"].isin(tract_ids)]
    if sld_tracts.empty:
        return 0.0

    scores = (sld_tracts["R_PCTLOWWAGE"]
              * (sld_tracts["P_WrkAge"] - sld_tracts["Workers"] / sld_tracts["TotPop"].clip(lower=1))
              * (1 - sld_tracts["transit_ratio"]))
    return float(scores.mean()) if not scores.empty else 0.0


def recommend_routes(stranded_df, work_summary, geometry, sld_df=None,
                     top_centers=3, corridors_per_center=3, min_corridor_workers=50):
    # min_corridor_workers=50: below ~50 daily riders a bus route is not
    # cost-effective (TCRP standards suggest ~10-15 boardings/service hour)
    """
    For each top job center, find directional corridors and rank them
    by a combined score of worker volume and Opportunity Score.
    """
    tract_centroids = get_tract_centroids(geometry)
    job_centers = find_top_job_centers(work_summary, top_n=top_centers)

    all_routes = []
    for _, jc in job_centers.iterrows():
        jc_tract = jc["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})

        feeders = find_feeder_tracts(stranded_df, jc_tract, tract_centroids)
        corridors = cluster_feeders_by_direction(feeders)

        kept = 0
        for corridor in corridors:
            if corridor["workers"] < min_corridor_workers:
                continue
            if kept >= corridors_per_center:
                break

            # Compute Opportunity Score for this corridor
            opp_score = 0.0
            if sld_df is not None:
                opp_score = _compute_corridor_opp_score(corridor["tracts"], sld_df)

            # Combined priority: workers * (1 + opp_score)
            # Higher opp_score = more underserved = higher priority
            priority = corridor["workers"] * (1 + max(opp_score, 0))

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
                "opp_score": round(opp_score, 4),
                "priority": round(priority, 1),
                "feeders": corridor["tracts"],
            })
            kept += 1

    # Sort by priority (workers * equity weight) instead of just workers
    all_routes.sort(key=lambda r: r["priority"], reverse=True)
    return all_routes
