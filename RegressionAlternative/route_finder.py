"""
Route recommendation engine for proposing new bus corridors.

Steps:
1. Find the job centers with the most stranded workers
2. Cluster nearby feeder tracts with DBSCAN
3. Build a directed graph connecting clusters inward toward each job center
4. Search for the best path (most workers) through the graph
5. Rank routes by worker count weighted by an equity/opportunity score
"""

import pandas as pd
import numpy as np
from math import atan2, degrees, cos, radians
from sklearn.cluster import DBSCAN
import networkx as nx


def find_top_job_centers(work_summary, top_n=10):
    return work_summary.nlargest(top_n, "stranded_inbound").reset_index(drop=True)


def get_tract_centroids(geometry):
    geo = geometry.to_crs("EPSG:4326").copy()
    geo["tract"] = geo["GEOID"].str[:11]
    geo["lat"] = geo.geometry.centroid.y
    geo["lon"] = geo.geometry.centroid.x
    return geo.groupby("tract").agg(
        lat=("lat", "mean"), lon=("lon", "mean")
    ).to_dict("index")


def _distance_mi(lat1, lon1, lat2, lon2):
    """Approximate distance in miles, with longitude correction."""
    lat_avg = radians((lat1 + lat2) / 2)
    dlat = (lat1 - lat2) * 69
    dlon = (lon1 - lon2) * 69 * cos(lat_avg)
    return (dlat**2 + dlon**2)**0.5


def _build_feeder_df(stranded_df, jc_tract, tract_centroids,
                     min_workers=5, min_dist_mi=1.0, max_dist_mi=20):
    """Get all residential tracts sending stranded workers to this job center."""
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
    """Use DBSCAN to group feeder tracts into pickup zones.
    Noise points with enough workers get kept as their own zone."""
    if df.empty:
        return []

    labels = DBSCAN(eps=eps_mi / 69, min_samples=2).fit_predict(
        df[["lat", "lon"]].values
    )
    df = df.copy()
    df["cluster"] = labels

    zones = []
    for c in sorted(df["cluster"].unique()):
        if c == -1:
            continue
        g = df[df["cluster"] == c]
        zones.append({
            "lat": g["lat"].mean(), "lon": g["lon"].mean(),
            "workers": int(g["workers"].sum()),
            "low_wage": int(g["low_wage"].sum()),
            "n_tracts": len(g),
        })

    # keep noise points that have a decent number of workers
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
    """Build a directed graph where edges go inward toward the job center.
    Only connects zones within a bearing constraint so routes stay linear.
    Short hops get a slightly relaxed angle limit since nearby detours are ok."""
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
            if b["dist_jc"] >= a["dist_jc"]:
                continue

            angle_diff = abs(a["bearing_jc"] - b["bearing_jc"])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            hop = _distance_mi(a["lat"], a["lon"], b["lat"], b["lon"])

            # relax the angle limit for short hops
            effective_max_angle = max_angle_diff
            if hop < 3.0:
                effective_max_angle = min(max_angle_diff + 20, 65)

            if angle_diff > effective_max_angle:
                continue
            if hop > max_hop_mi:
                continue

            # weight: prefer direct routes to high-worker zones
            direct_saved = a["dist_jc"] - b["dist_jc"]
            detour_ratio = hop / max(direct_saved, 0.5)
            weight = detour_ratio / max(b["workers"], 1)
            G.add_edge(i, j, weight=weight)

    return G


def _find_best_paths(G, zones, max_routes=3):
    """Find the paths through the graph that serve the most workers.
    Tries weighted shortest path first then enumerates alternatives."""
    dists = [z["dist_jc"] for z in zones]
    median_dist = np.median(dists)
    inner_thresh = max(median_dist * 0.4, 2.0)
    outer_thresh = max(median_dist * 0.6, 3.0)

    inner = [i for i, z in enumerate(zones) if z["dist_jc"] <= inner_thresh]
    outer = [i for i, z in enumerate(zones) if z["dist_jc"] >= outer_thresh]

    if not inner or not outer:
        sorted_idx = sorted(range(len(zones)), key=lambda i: zones[i]["dist_jc"])
        mid = len(sorted_idx) // 2
        inner = sorted_idx[:mid]
        outer = sorted_idx[mid:]

    used = set()
    routes = []

    for start in sorted(outer, key=lambda i: -zones[i]["workers"]):
        if start in used:
            continue

        best_path, best_w = None, 0
        for target in inner:
            if target in used:
                continue

            # try shortest path first (fast)
            try:
                sp = nx.shortest_path(G, start, target, weight="weight")
                if not any(n in used for n in sp):
                    w = sum(zones[n]["workers"] for n in sp)
                    if w > best_w:
                        best_w = w
                        best_path = list(sp)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            # also try all simple paths in case there's a better one
            # cap at 50 paths per start/target pair to avoid blowup
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

        if best_path and best_w >= 30:
            pz = [zones[n] for n in best_path]
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
    """Opportunity score: how underserved is this area?
    Combines low-wage share, employment gap, and transit gap."""
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
    """Main pipeline: scan many job centers, build corridors, keep the best ones."""
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
            continue

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

            # all feeder BGs for this JC (filtered to nearby ones later in app.py)
            bg_data = [{
                "tract": row["tract"],
                "workers": int(row["workers"]),
                "low_wage": int(row["low_wage"]),
                "lat": row["lat"],
                "lon": row["lon"],
            } for _, row in feeder_df.iterrows()]

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

    # rough pre-filter before road paths are drawn (real filter happens after)
    all_routes = [r for r in all_routes if r["route_workers"] >= min_route_workers // 2]
    return all_routes[:max_final_routes * 2]
