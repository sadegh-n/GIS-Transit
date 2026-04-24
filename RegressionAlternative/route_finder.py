"""
Route recommendation engine for proposing new bus corridors.

The basic idea:
1. Find the job centers where the most stranded workers need to go
2. Cluster nearby residential tracts into pickup zones using DBSCAN
3. Build a directed graph where edges point inward toward the job center
4. Search the graph for paths that serve the most workers
5. Rank routes by worker count weighted by an equity score
"""

import pandas as pd
import numpy as np
from math import atan2, degrees, cos, radians
from sklearn.cluster import DBSCAN
import networkx as nx


def find_top_job_centers(work_summary, top_n=10):
    """Get the job centers (work tracts) with the most stranded inbound workers."""
    return work_summary.nlargest(top_n, "stranded_inbound").reset_index(drop=True)


def get_tract_centroids(geometry):
    """Compute the center lat/lon of each census tract (by averaging its block groups)."""
    geo = geometry.to_crs("EPSG:4326").copy()
    geo["tract"] = geo["GEOID"].str[:11]
    geo["lat"] = geo.geometry.centroid.y
    geo["lon"] = geo.geometry.centroid.x
    return geo.groupby("tract").agg(
        lat=("lat", "mean"), lon=("lon", "mean")
    ).to_dict("index")


def _distance_mi(lat1, lon1, lat2, lon2):
    """Approximate distance in miles between two lat/lon points.
    Corrects for longitude compression at higher latitudes."""
    lat_avg = radians((lat1 + lat2) / 2)
    dlat = (lat1 - lat2) * 69  # 1 degree latitude is about 69 miles
    dlon = (lon1 - lon2) * 69 * cos(lat_avg)
    return (dlat**2 + dlon**2)**0.5


def _build_feeder_df(stranded_df, jc_tract, tract_centroids,
                     min_workers=5, min_dist_mi=1.0, max_dist_mi=20):
    """Get all residential tracts that send stranded workers to a given job center.
    Filters out tracts that are too close (they can probably walk) or too far."""
    feeders = (stranded_df[stranded_df["work_tract"] == jc_tract]
               .groupby("home_tract")
               .agg(workers=("total_workers", "sum"), low_wage=("low_wage", "sum"))
               .reset_index())
    feeders = feeders[feeders["workers"] >= min_workers]

    jc = tract_centroids.get(jc_tract)
    if not jc:
        return pd.DataFrame()

    rows = []
    for _, row in feeders.iterrows():
        tc = tract_centroids.get(row["home_tract"])
        if not tc:
            continue
        dist = _distance_mi(tc["lat"], tc["lon"], jc["lat"], jc["lon"])
        if dist < min_dist_mi or dist > max_dist_mi:
            continue
        rows.append({
            "tract": row["home_tract"], "workers": row["workers"],
            "low_wage": row["low_wage"], "lat": tc["lat"], "lon": tc["lon"],
        })
    return pd.DataFrame(rows)


def _cluster_pickup_zones(df, eps_mi=1.0):
    """Use DBSCAN to group nearby feeder tracts into pickup zones.
    eps_mi=1.0 means tracts within 1 mile of each other get clustered together.
    Noise points (unclustered) are kept as their own zone if they have enough workers."""
    if df.empty:
        return []

    # convert miles to degrees for DBSCAN (1 degree lat ~ 69 miles)
    labels = DBSCAN(eps=eps_mi / 69, min_samples=2).fit_predict(
        df[["lat", "lon"]].values
    )
    df = df.copy()
    df["cluster"] = labels

    zones = []
    for c in sorted(df["cluster"].unique()):
        if c == -1:
            continue  # handle noise points separately below
        g = df[df["cluster"] == c]
        zones.append({
            "lat": g["lat"].mean(), "lon": g["lon"].mean(),
            "workers": int(g["workers"].sum()),
            "low_wage": int(g["low_wage"].sum()),
            "n_tracts": len(g),
        })

    # noise points with at least 20 workers become their own zone
    noise = df[df["cluster"] == -1]
    for _, row in noise.iterrows():
        if row["workers"] >= 20:
            zones.append({
                "lat": row["lat"], "lon": row["lon"],
                "workers": int(row["workers"]),
                "low_wage": int(row["low_wage"]),
                "n_tracts": 1,
            })
    return zones


def _build_corridor_graph(zones, jc_lat, jc_lon,
                          max_angle_diff=45, max_hop_mi=10):
    """Build a directed graph where edges point inward toward the job center.
    The bearing constraint (max_angle_diff=45 degrees) keeps routes roughly
    linear instead of zigzagging all over the place. Short hops get a slightly
    relaxed angle limit since small detours for nearby zones are fine."""
    for z in zones:
        z["dist_jc"] = _distance_mi(z["lat"], z["lon"], jc_lat, jc_lon)
        z["bearing_jc"] = degrees(
            atan2(z["lon"] - jc_lon, z["lat"] - jc_lat)
        ) % 360

    G = nx.DiGraph()
    for i in range(len(zones)):
        G.add_node(i)

    for i, a in enumerate(zones):
        for j, b in enumerate(zones):
            if i == j:
                continue
            # edges only go inward (closer to job center)
            if b["dist_jc"] >= a["dist_jc"]:
                continue

            angle_diff = abs(a["bearing_jc"] - b["bearing_jc"])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            hop = _distance_mi(a["lat"], a["lon"], b["lat"], b["lon"])

            # short hops (under 3mi) get a wider angle tolerance
            effective_max_angle = max_angle_diff
            if hop < 3.0:
                effective_max_angle = min(max_angle_diff + 20, 65)

            if angle_diff > effective_max_angle:
                continue
            if hop > max_hop_mi:
                continue

            # edge weight: prefer direct paths to high-worker zones
            direct_saved = a["dist_jc"] - b["dist_jc"]
            detour_ratio = hop / max(direct_saved, 0.5)
            weight = detour_ratio / max(b["workers"], 1)
            G.add_edge(i, j, weight=weight)

    return G


def _find_best_paths(G, zones, max_routes=3):
    """Find paths through the graph that serve the most workers.
    Starts from the outer zones and works inward toward the job center.
    Uses shortest path first for speed, then checks alternatives."""
    dists = [z["dist_jc"] for z in zones]
    median_dist = np.median(dists)
    # inner zones are close to the JC, outer zones are far away
    inner_thresh = max(median_dist * 0.4, 2.0)
    outer_thresh = max(median_dist * 0.6, 3.0)

    inner = [i for i, z in enumerate(zones) if z["dist_jc"] <= inner_thresh]
    outer = [i for i, z in enumerate(zones) if z["dist_jc"] >= outer_thresh]

    # fallback if the thresholds don't split things well
    if not inner or not outer:
        sorted_idx = sorted(range(len(zones)), key=lambda i: zones[i]["dist_jc"])
        mid = len(sorted_idx) // 2
        inner = sorted_idx[:mid]
        outer = sorted_idx[mid:]

    used = set()  # track which zones have already been assigned to a route
    routes = []

    for start in sorted(outer, key=lambda i: -zones[i]["workers"]):
        if start in used:
            continue

        best_path, best_w = None, 0
        for target in inner:
            if target in used:
                continue

            # try weighted shortest path first (fast)
            try:
                sp = nx.shortest_path(G, start, target, weight="weight")
                if not any(n in used for n in sp):
                    w = sum(zones[n]["workers"] for n in sp)
                    if w > best_w:
                        best_w = w
                        best_path = list(sp)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            # also enumerate simple paths to see if there's something better
            # cap at 50 per start/target pair so this doesn't run forever
            try:
                count = 0
                for path in nx.all_simple_paths(G, start, target, cutoff=5):
                    count += 1
                    if count > 50:
                        break
                    if any(n in used for n in path):
                        continue
                    w = sum(zones[n]["workers"] for n in path)
                    if w > best_w:
                        best_w = w
                        best_path = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        # only keep routes with at least 30 workers
        if best_path and best_w >= 30:
            pz = [zones[n] for n in best_path]
            # figure out the compass direction of this route
            cb = np.mean([z["bearing_jc"] for z in pz])
            dirs16 = [
                "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
            ]
            routes.append({
                "direction": dirs16[int((cb + 11.25) / 22.5) % 16],
                "workers": best_w,
                "low_wage": sum(z["low_wage"] for z in pz),
                "num_stops": len(best_path),
                "max_dist_mi": round(max(z["dist_jc"] for z in pz), 1),
                "avg_dist_mi": round(np.mean([z["dist_jc"] for z in pz]), 1),
                "stops": [{
                    "lat": z["lat"], "lon": z["lon"],
                    "workers": z["workers"], "low_wage": z["low_wage"],
                    "dist_mi": round(z["dist_jc"], 1), "n_tracts": z["n_tracts"],
                } for z in pz],
            })
            for n in best_path:
                used.add(n)

        if len(routes) >= max_routes:
            break

    routes.sort(key=lambda r: -r["workers"])
    return routes


def _compute_opp_score(feeder_df, sld_df):
    """Compute an opportunity score that measures how underserved an area is.
    Combines three factors: share of low-wage workers, employment gap
    (working-age pop minus actual workers), and transit gap (1 - transit ratio).
    Higher score means the area needs transit more."""
    if sld_df is None or feeder_df.empty:
        return 0.0
    tracts = feeder_df["tract"].unique()
    match = sld_df[sld_df["tract"].isin(tracts)]
    if match.empty:
        return 0.0
    scores = (match["R_PCTLOWWAGE"]
              * (match["P_WrkAge"] - match["Workers"] / match["TotPop"].clip(lower=1))
              * (1 - match["transit_ratio"]))
    return float(scores.mean())


def recommend_routes(stranded_df, work_summary, geometry, sld_df=None,
                     top_centers=15, routes_per_center=4,
                     min_route_workers=250, max_final_routes=8,
                     _tract_centroids=None):
    """Main pipeline: look at many job centers, build corridor routes for each one,
    and keep the best ones ranked by workers * equity score."""
    tract_centroids = _tract_centroids or get_tract_centroids(geometry)
    job_centers = find_top_job_centers(work_summary, top_n=top_centers)

    all_routes = []
    for _, jc_row in job_centers.iterrows():
        jc_tract = jc_row["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})
        if not jc_loc:
            continue

        feeder_df = _build_feeder_df(stranded_df, jc_tract, tract_centroids)
        if feeder_df.empty:
            continue

        zones = _cluster_pickup_zones(feeder_df)
        if len(zones) < 2:
            continue  # need at least 2 zones to form a corridor

        G = _build_corridor_graph(zones, jc_loc["lat"], jc_loc["lon"])
        corridor_routes = _find_best_paths(G, zones, max_routes=routes_per_center)

        opp_score = _compute_opp_score(feeder_df, sld_df)

        for route in corridor_routes:
            feeders = [{
                "tract": f"zone_{i}",
                "workers": s["workers"],
                "low_wage": s["low_wage"],
                "dist_mi": s["dist_mi"],
                "lat": s["lat"],
                "lon": s["lon"],
            } for i, s in enumerate(route["stops"])]

            # pass along all feeder block groups for this JC
            # (app.py will filter them down to ones near the actual road path later)
            bg_data = [{
                "tract": row["tract"],
                "workers": int(row["workers"]),
                "low_wage": int(row["low_wage"]),
                "lat": row["lat"],
                "lon": row["lon"],
            } for _, row in feeder_df.iterrows()]

            # priority = worker count boosted by the equity score
            priority = route["workers"] * (1 + max(opp_score, 0))

            all_routes.append({
                "job_center": jc_tract,
                "jc_stranded_total": int(jc_row["stranded_inbound"]),
                "jc_lat": jc_loc["lat"],
                "jc_lon": jc_loc["lon"],
                "direction": route["direction"],
                "route_workers": route["workers"],
                "route_low_wage": route["low_wage"],
                "num_feeders": route["num_stops"],
                "avg_dist_mi": route["avg_dist_mi"],
                "max_dist_mi": route["max_dist_mi"],
                "opp_score": round(opp_score, 4),
                "priority": round(priority, 1),
                "feeders": feeders,
                "bg_data": bg_data,
            })

    all_routes.sort(key=lambda r: -r["priority"])

    # rough pre-filter before road paths are drawn (the real filter happens in app.py
    # after we know the actual road geometry)
    all_routes = [r for r in all_routes if r["route_workers"] >= min_route_workers // 2]
    return all_routes[:max_final_routes * 2]
