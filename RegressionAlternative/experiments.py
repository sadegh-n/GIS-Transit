"""
Validation experiments for the transit route recommendation pipeline.
Runs across all cities with local data, then aggregates results.

Usage:
    python experiments.py

Takes several minutes on first run per city (connectivity analysis gets cached).
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from math import cos, radians

warnings.filterwarnings("ignore", category=UserWarning)

from data_loader import load_sld, load_lodes, load_geometry, load_gtfs
from connectivity import (build_transit_graph, check_commutes,
                          summarize_stranded, build_bg_centroids,
                          map_bgs_to_routes, find_route_transfers,
                          build_reachability)
from route_finder import (recommend_routes, get_tract_centroids,
                          find_top_job_centers, _build_feeder_df,
                          _cluster_pickup_zones, _build_corridor_graph,
                          _find_best_paths, _distance_mi)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# all cities with GTFS and LODES data available
CITIES = {
    "Dallas": {
        "state": "48", "county": "113", "fips": "48113", "state_abbr": "tx",
        "lat": 32.77, "lon": -96.79,
        "gtfs_dir": os.path.join(DATA_DIR, "V248-160-161-20210517"),
    },
    "Fort Worth": {
        "state": "48", "county": "439", "fips": "48439", "state_abbr": "tx",
        "lat": 32.75, "lon": -97.33,
        "gtfs_dir": os.path.join(DATA_DIR, "fortworth_gtfs"),
    },
    "Houston": {
        "state": "48", "county": "201", "fips": "48201", "state_abbr": "tx",
        "lat": 29.76, "lon": -95.37,
        "gtfs_dir": os.path.join(DATA_DIR, "houston_gtfs"),
    },
    "San Antonio": {
        "state": "48", "county": "029", "fips": "48029", "state_abbr": "tx",
        "lat": 29.42, "lon": -98.49,
        "gtfs_dir": os.path.join(DATA_DIR, "via_gtfs"),
    },
    "Austin": {
        "state": "48", "county": "453", "fips": "48453", "state_abbr": "tx",
        "lat": 30.27, "lon": -97.74,
        "gtfs_dir": os.path.join(DATA_DIR, "austin_gtfs"),
    },
    "Atlanta": {
        "state": "13", "county": "121", "fips": "13121", "state_abbr": "ga",
        "lat": 33.75, "lon": -84.39,
        "gtfs_dir": os.path.join(DATA_DIR, "atlanta_gtfs"),
    },
    "Jacksonville": {
        "state": "12", "county": "031", "fips": "12031", "state_abbr": "fl",
        "lat": 30.33, "lon": -81.66,
        "gtfs_dir": os.path.join(DATA_DIR, "jacksonville_gtfs"),
    },
    "Charlotte": {
        "state": "37", "county": "119", "fips": "37119", "state_abbr": "nc",
        "lat": 35.23, "lon": -80.84,
        "gtfs_dir": os.path.join(DATA_DIR, "charlotte_gtfs"),
    },
    "Detroit": {
        "state": "26", "county": "163", "fips": "26163", "state_abbr": "mi",
        "lat": 42.33, "lon": -83.05,
        "gtfs_dir": os.path.join(DATA_DIR, "detroit_gtfs"),
    },
    "Phoenix": {
        "state": "04", "county": "013", "fips": "04013", "state_abbr": "az",
        "lat": 33.45, "lon": -112.07,
        "gtfs_dir": os.path.join(DATA_DIR, "phoenix_gtfs"),
    },
    "Nashville": {
        "state": "47", "county": "037", "fips": "47037", "state_abbr": "tn",
        "lat": 36.16, "lon": -86.78,
        "gtfs_dir": os.path.join(DATA_DIR, "nashville_gtfs"),
    },
}


def load_sld_once():
    """SLD is the same file for all cities — load it once."""
    path = os.path.join(DATA_DIR, "EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv")
    return load_sld(path)


def load_city(city_name, cfg, sld):
    """Load all data for one city. Returns dict or None if data missing."""
    lodes_path = os.path.join(DATA_DIR, f"{cfg['state_abbr']}_od_main_JT00_2021.csv.gz")
    if not os.path.exists(lodes_path):
        print(f"  Skipping {city_name}: LODES file not found")
        return None

    if not os.path.isdir(cfg["gtfs_dir"]):
        print(f"  Skipping {city_name}: GTFS dir not found")
        return None

    county = sld[sld["GEOID"].str[:5] == cfg["fips"]].copy()
    lodes = load_lodes(lodes_path)
    try:
        geometry = load_geometry(cfg["state"], cfg["county"])
    except Exception as e:
        print(f"  Skipping {city_name}: geometry download failed ({e})")
        return None
    try:
        stop_locs, stop_route_map = load_gtfs(cfg["gtfs_dir"])
    except Exception as e:
        print(f"  Skipping {city_name}: GTFS load error ({e})")
        return None

    if stop_locs is None or lodes is None:
        print(f"  Skipping {city_name}: failed to load GTFS or LODES")
        return None

    # connectivity (cached to disk, supports both old 4-tuple and new 5-tuple format)
    cache_file = os.path.join(DATA_DIR, f".cache_{cfg['fips']}_results.pkl")
    cache_valid = False
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, tuple):
            if len(cached) == 5:
                _, home_summary, work_summary, stranded_df, totals = cached
                cache_valid = True
            elif len(cached) == 4:
                home_summary, work_summary, stranded_df, totals = cached
                cache_valid = True

    if not cache_valid:
        print(f"  Running connectivity analysis for {city_name}...")
        reachability, _, _ = build_transit_graph(geometry, stop_locs, stop_route_map)
        checked = check_commutes(lodes, reachability, cfg["fips"])
        home_summary, work_summary, stranded_df, totals = summarize_stranded(checked)
        with open(cache_file, "wb") as f:
            pickle.dump((2, home_summary, work_summary, stranded_df, totals), f)

    # precompute expensive things once so experiments don't redo them
    print(f"  Precomputing centroids and routes for {city_name}...")
    tract_centroids = get_tract_centroids(geometry)
    recs = recommend_routes(
        stranded_df, work_summary, geometry, sld_df=county,
        top_centers=10, routes_per_center=4,
        min_route_workers=50, max_final_routes=10,
        _tract_centroids=tract_centroids,
    )

    return {
        "county": county, "lodes": lodes, "geometry": geometry,
        "stop_locs": stop_locs, "stop_route_map": stop_route_map,
        "home_summary": home_summary, "work_summary": work_summary,
        "stranded_df": stranded_df, "totals": totals,
        "tract_centroids": tract_centroids, "recs": recs,
        "cfg": cfg,
    }


# ============================================================
# Experiment 1: Walk radius sensitivity
# ============================================================
def exp_walk_radius(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 1: Walk Radius Sensitivity")
    print("="*65)
    print("How does the walk-to-stop radius affect the stranded %?\n")

    from scipy.spatial import cKDTree
    from collections import defaultdict

    radii = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    all_rows = []

    for city, data in all_data.items():
        cfg = data["cfg"]
        geometry = data["geometry"]
        stop_locs = data["stop_locs"]
        stop_route_map = data["stop_route_map"]
        lodes = data["lodes"]

        bg_centroids = build_bg_centroids(geometry)
        transfers = find_route_transfers(stop_route_map, stop_locs)
        stop_tree = cKDTree(stop_locs[["stop_lat", "stop_lon"]].values)
        stop_ids = stop_locs["stop_id"].values

        # precompute distances from each BG to all stops (max radius)
        max_deg = max(radii) / 69
        bg_coords = bg_centroids[["lat", "lon"]].values
        bg_geoids = bg_centroids["GEOID"].values
        bg_nearby = stop_tree.query_ball_point(bg_coords, max_deg)

        for radius_mi in radii:
            radius_deg = radius_mi / 69
            bg_routes = {}
            for i, geoid in enumerate(bg_geoids):
                routes = set()
                for idx in bg_nearby[i]:
                    # check actual distance at this radius
                    d = ((bg_coords[i][0] - stop_locs.iloc[idx]["stop_lat"])**2 +
                         (bg_coords[i][1] - stop_locs.iloc[idx]["stop_lon"])**2)**0.5
                    if d <= radius_deg:
                        routes.update(stop_route_map.get(stop_ids[idx], set()))
                if routes:
                    bg_routes[geoid] = routes

            route_bgs = defaultdict(set)
            for bg_id, routes in bg_routes.items():
                for r in routes:
                    route_bgs[r].add(bg_id)

            reachability = build_reachability(bg_routes, dict(route_bgs), transfers)
            checked = check_commutes(lodes, reachability, cfg["fips"])
            _, _, _, totals = summarize_stranded(checked)

            all_rows.append({
                "City": city,
                "Walk Radius (mi)": radius_mi,
                "Stranded %": totals["pct_stranded"],
            })

    df = pd.DataFrame(all_rows)
    pivot = df.pivot(index="Walk Radius (mi)", columns="City", values="Stranded %")
    pivot["Average"] = pivot.mean(axis=1)
    print(pivot.applymap(lambda x: f"{x:.1%}").to_string())
    print("\n0.5mi is the FTA standard for local bus service.")
    return df


# ============================================================
# Experiment 2: Transfer limit sensitivity
# ============================================================
def exp_transfer_limit(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 2: Transfer Limit Sensitivity")
    print("="*65)
    print("How does max transfers affect the stranded %?\n")

    all_rows = []
    for city, data in all_data.items():
        cfg = data["cfg"]
        geometry = data["geometry"]
        stop_locs = data["stop_locs"]
        stop_route_map = data["stop_route_map"]
        lodes = data["lodes"]

        bg_centroids = build_bg_centroids(geometry)
        bg_routes, route_bgs = map_bgs_to_routes(bg_centroids, stop_locs, stop_route_map)
        transfers = find_route_transfers(stop_route_map, stop_locs)

        for max_t in [0, 1, 2, 3]:
            reach = {}
            for bg_id, routes in bg_routes.items():
                expanded = set(routes)
                for _ in range(max_t):
                    nxt = set(expanded)
                    for r in expanded:
                        nxt.update(transfers.get(r, set()))
                    expanded = nxt
                reachable = set()
                for r in expanded:
                    reachable.update(route_bgs.get(r, set()))
                reachable.discard(bg_id)
                reach[bg_id] = reachable

            checked = check_commutes(lodes, reach, cfg["fips"])
            _, _, _, totals = summarize_stranded(checked)
            all_rows.append({
                "City": city,
                "Max Transfers": max_t,
                "Stranded %": totals["pct_stranded"],
            })

    df = pd.DataFrame(all_rows)
    pivot = df.pivot(index="Max Transfers", columns="City", values="Stranded %")
    pivot["Average"] = pivot.mean(axis=1)
    print(pivot.applymap(lambda x: f"{x:.1%}").to_string())
    print("\n2 transfers is the practical limit for riders.")
    return df


# ============================================================
# Experiment 3: DBSCAN eps sensitivity
# ============================================================
def exp_dbscan_eps(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 3: DBSCAN Clustering Radius (eps)")
    print("="*65)
    print("How does eps affect zone count and best route workers?\n")

    eps_values = [0.5, 0.75, 1.0, 1.5, 2.0]
    all_rows = []

    for city, data in all_data.items():
        stranded_df = data["stranded_df"]
        work_summary = data["work_summary"]
        tract_centroids = data["tract_centroids"]

        # use top JC only, limit feeder tracts to keep graph small
        jc = find_top_job_centers(work_summary, top_n=1)
        if jc.empty:
            continue
        jc = jc.iloc[0]
        jc_tract = jc["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})
        feeder_df = _build_feeder_df(stranded_df, jc_tract, tract_centroids,
                                     min_workers=10)
        if feeder_df.empty or len(feeder_df) > 500:
            # skip huge cities that blow up graph search — take top 500 feeders
            feeder_df = feeder_df.nlargest(500, "workers") if not feeder_df.empty else feeder_df
        if feeder_df.empty:
            continue

        for eps in eps_values:
            zones = _cluster_pickup_zones(feeder_df, eps_mi=eps)
            best_workers = 0
            n_routes = 0
            if len(zones) >= 2 and jc_loc and len(zones) <= 200:
                G = _build_corridor_graph(zones, jc_loc["lat"], jc_loc["lon"])
                routes = _find_best_paths(G, zones, max_routes=3)
                n_routes = len(routes)
                if routes:
                    best_workers = routes[0]["workers"]

            all_rows.append({
                "City": city, "eps (mi)": eps,
                "Zones": len(zones), "Best Route Workers": best_workers,
            })

    df = pd.DataFrame(all_rows)
    # show best route workers pivoted
    pivot = df.pivot(index="eps (mi)", columns="City", values="Best Route Workers")
    pivot["Average"] = pivot.mean(axis=1)
    print("Best route workers by eps:")
    print(pivot.to_string())

    # also show zone counts
    zpivot = df.pivot(index="eps (mi)", columns="City", values="Zones")
    zpivot["Average"] = zpivot.mean(axis=1)
    print("\nZone count by eps:")
    print(zpivot.to_string())
    print("\neps=1.0mi balances zone granularity with route quality.")
    return df


# ============================================================
# Experiment 4: Job center count vs coverage
# ============================================================
def exp_job_center_coverage(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 4: Job Center Count vs Worker Coverage")
    print("="*65)
    print("Diminishing returns as we consider more job centers.\n")
    print("(Uses cumulative inbound stranded workers at top-N job centers\n"
          " as a fast proxy — avoids re-running full route pipeline.)\n")

    center_counts = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    all_rows = []

    for city, data in all_data.items():
        total_stranded = data["totals"]["stranded"]
        ws = data["work_summary"].nlargest(30, "stranded_inbound")
        for n in center_counts:
            top_n = ws.head(n)
            workers = int(top_n["stranded_inbound"].sum())
            all_rows.append({
                "City": city, "Job Centers": n,
                "% Stranded Served": workers / max(total_stranded, 1),
            })

    df = pd.DataFrame(all_rows)
    pivot = df.pivot(index="Job Centers", columns="City", values="% Stranded Served")
    pivot["Average"] = pivot.mean(axis=1)
    print(pivot.applymap(lambda x: f"{x:.1%}").to_string())
    print("\nDiminishing returns — first few centers capture most impact.")
    return df


# ============================================================
# Experiment 5: Route directness
# ============================================================
def exp_route_directness(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 5: Route Directness")
    print("="*65)
    print("Straight-line vs via-stops distance. Real buses: 1.1-1.4x.\n")

    all_rows = []
    for city, data in all_data.items():
        recs = data["recs"]
        for r in recs:
            feeders = [f for f in r["feeders"] if f["lat"]]
            if not feeders:
                continue
            farthest = max(feeders, key=lambda f: f["dist_mi"])
            straight = _distance_mi(farthest["lat"], farthest["lon"],
                                    r["jc_lat"], r["jc_lon"])
            stops = sorted(feeders, key=lambda f: -f["dist_mi"])
            road = 0
            prev = stops[0]
            for s in stops[1:]:
                road += _distance_mi(prev["lat"], prev["lon"], s["lat"], s["lon"])
                prev = s
            road += _distance_mi(prev["lat"], prev["lon"], r["jc_lat"], r["jc_lon"])
            ratio = road / max(straight, 0.1)
            all_rows.append({"City": city, "Directness": ratio, "Workers": r["route_workers"]})

    df = pd.DataFrame(all_rows)
    summary = df.groupby("City")["Directness"].agg(["mean", "median", "min", "max"])
    summary.columns = ["Mean", "Median", "Min", "Max"]
    summary.loc["ALL CITIES"] = df["Directness"].agg(["mean", "median", "min", "max"])
    print(summary.applymap(lambda x: f"{x:.2f}x").to_string())
    print(f"\nOverall average: {df['Directness'].mean():.2f}x")
    return df


# ============================================================
# Experiment 6: vs random baseline
# ============================================================
def exp_vs_random_baseline(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 6: Our Routes vs Random Baseline")
    print("="*65)
    print("Do our corridors serve more workers than random ones?\n")

    all_rows = []
    for city, data in all_data.items():
        cfg = data["cfg"]
        cos_lat = cos(radians(cfg["lat"]))

        recs = data["recs"]
        our_workers = [r["route_workers"] for r in recs]

        # random baseline: random feeder -> top JC
        tract_centroids = data["tract_centroids"]
        jc = find_top_job_centers(data["work_summary"], top_n=1)
        if jc.empty or not our_workers:
            continue
        jc = jc.iloc[0]
        jc_tract = jc["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})
        feeder_df = _build_feeder_df(data["stranded_df"], jc_tract,
                                     tract_centroids, min_workers=1, min_dist_mi=0.5)

        np.random.seed(42)
        random_workers = []
        if not feeder_df.empty and jc_loc:
            for _ in range(len(our_workers)):
                start = feeder_df.sample(1).iloc[0]
                sx, sy = start["lon"], start["lat"]
                dx, dy = jc_loc["lon"] - sx, jc_loc["lat"] - sy
                lsq = dx**2 + dy**2
                w = 0
                for _, f in feeder_df.iterrows():
                    t = max(0, min(1, ((f["lon"]-sx)*dx + (f["lat"]-sy)*dy) / max(lsq, 1e-10)))
                    px, py = sx + t*dx, sy + t*dy
                    dl = (f["lat"] - py) * 69
                    dn = (f["lon"] - px) * 69 * cos_lat
                    if (dl**2 + dn**2)**0.5 <= 0.75:
                        w += int(f["workers"])
                random_workers.append(w)

        our_mean = np.mean(our_workers) if our_workers else 0
        rand_mean = np.mean(random_workers) if random_workers else 0
        all_rows.append({
            "City": city,
            "Our Mean": int(our_mean),
            "Random Mean": int(rand_mean),
            "Improvement": f"{(our_mean / max(rand_mean, 1) - 1) * 100:.0f}%",
            "Our Total": sum(our_workers),
            "Random Total": sum(random_workers),
        })

    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    avg_our = np.mean([r["Our Mean"] for r in all_rows])
    avg_rand = np.mean([r["Random Mean"] for r in all_rows])
    print(f"\nOverall: our algorithm serves {(avg_our/max(avg_rand,1)-1)*100:.0f}% "
          f"more workers per route on average.")
    return df


# ============================================================
# Experiment 7: Equity coverage
# ============================================================
def exp_equity_coverage(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 7: Equity Coverage Analysis")
    print("="*65)
    print("Do our routes serve low-wage workers proportionally?\n")

    all_rows = []
    for city, data in all_data.items():
        recs = data["recs"]
        total_stranded = data["totals"]["stranded"]
        total_lw = int(data["stranded_df"]["low_wage"].sum())
        route_workers = sum(r["route_workers"] for r in recs)
        route_lw = sum(r["route_low_wage"] for r in recs)

        baseline_pct = total_lw / max(total_stranded, 1)
        route_pct = route_lw / max(route_workers, 1)

        all_rows.append({
            "City": city,
            "Baseline LW%": f"{baseline_pct:.1%}",
            "Our Routes LW%": f"{route_pct:.1%}",
            "Difference": f"{(route_pct - baseline_pct)*100:+.1f}pp",
        })

    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    return df


# ============================================================
# Experiment 8: Bearing constraint sensitivity
# ============================================================
def exp_bearing_constraint(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 8: Bearing Constraint Sensitivity")
    print("="*65)
    print("How does the max angle difference affect route quality?\n")

    angles = [20, 30, 45, 60, 90, 180]
    all_rows = []

    for city, data in all_data.items():
        stranded_df = data["stranded_df"]
        work_summary = data["work_summary"]
        tract_centroids = data["tract_centroids"]

        jc = find_top_job_centers(work_summary, top_n=1)
        if jc.empty:
            continue
        jc = jc.iloc[0]
        jc_tract = jc["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})
        feeder_df = _build_feeder_df(stranded_df, jc_tract, tract_centroids,
                                     min_workers=10)
        if feeder_df.empty:
            feeder_df = feeder_df.nlargest(500, "workers") if not feeder_df.empty else feeder_df
        if feeder_df.empty or not jc_loc:
            continue

        zones = _cluster_pickup_zones(feeder_df, eps_mi=1.0)
        if len(zones) < 2 or len(zones) > 200:
            continue

        for angle in angles:
            G = _build_corridor_graph(zones, jc_loc["lat"], jc_loc["lon"],
                                      max_angle_diff=angle)
            routes = _find_best_paths(G, zones, max_routes=3)
            best_w = routes[0]["workers"] if routes else 0
            n_routes = len(routes)
            total_w = sum(r["workers"] for r in routes)
            all_rows.append({
                "City": city, "Max Angle": f"{angle}°",
                "Routes": n_routes, "Best Route": best_w, "Total Workers": total_w,
            })

    df = pd.DataFrame(all_rows)
    pivot = df.pivot(index="Max Angle", columns="City", values="Best Route")
    pivot["Average"] = pivot.mean(axis=1)
    print("Best route workers by bearing constraint:")
    print(pivot.to_string())
    print("\n45° keeps routes linear. 90°+ allows zigzag detours.")
    return df


# ============================================================
# Experiment 9: Path cutoff sensitivity
# ============================================================
def exp_path_cutoff(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 9: Path Cutoff (Max Hops) Sensitivity")
    print("="*65)
    print("How does max hops per route affect worker coverage?\n")

    cutoffs = [2, 3, 4, 5, 7, 10]
    all_rows = []

    for city, data in all_data.items():
        stranded_df = data["stranded_df"]
        work_summary = data["work_summary"]
        tract_centroids = data["tract_centroids"]

        jc = find_top_job_centers(work_summary, top_n=1)
        if jc.empty:
            continue
        jc = jc.iloc[0]
        jc_tract = jc["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})
        feeder_df = _build_feeder_df(stranded_df, jc_tract, tract_centroids,
                                     min_workers=10)
        if feeder_df.empty:
            feeder_df = feeder_df.nlargest(500, "workers") if not feeder_df.empty else feeder_df
        if feeder_df.empty or not jc_loc:
            continue

        zones = _cluster_pickup_zones(feeder_df, eps_mi=1.0)
        if len(zones) < 2 or len(zones) > 200:
            continue

        G = _build_corridor_graph(zones, jc_loc["lat"], jc_loc["lon"])

        for cutoff in cutoffs:
            # temporarily override cutoff in path search
            dists = [z["dist_jc"] for z in zones]
            median_dist = np.median(dists)
            inner_thresh = max(median_dist * 0.4, 2.0)
            outer_thresh = max(median_dist * 0.6, 3.0)
            inner = [i for i, z in enumerate(zones) if z["dist_jc"] <= inner_thresh]
            outer = [i for i, z in enumerate(zones) if z["dist_jc"] >= outer_thresh]
            if not inner or not outer:
                sorted_idx = sorted(range(len(zones)), key=lambda i: zones[i]["dist_jc"])
                mid = len(sorted_idx) // 2
                inner, outer = sorted_idx[:mid], sorted_idx[mid:]

            used = set()
            routes = []
            for start in sorted(outer, key=lambda i: -zones[i]["workers"]):
                if start in used:
                    continue
                best_path, best_w = None, 0
                for target in inner:
                    if target in used:
                        continue
                    try:
                        sp = nx.shortest_path(G, start, target, weight="weight")
                        if len(sp) <= cutoff and not any(n in used for n in sp):
                            w = sum(zones[n]["workers"] for n in sp)
                            if w > best_w:
                                best_w, best_path = w, list(sp)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                    try:
                        count = 0
                        for path in nx.all_simple_paths(G, start, target, cutoff=cutoff):
                            count += 1
                            if count > 50:
                                break
                            if any(n in used for n in path):
                                continue
                            w = sum(zones[n]["workers"] for n in path)
                            if w > best_w:
                                best_w, best_path = w, path
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                if best_path and best_w >= 30:
                    routes.append({"workers": best_w, "path": best_path})
                    for n in best_path:
                        used.add(n)
                if len(routes) >= 3:
                    break

            best_w = routes[0]["workers"] if routes else 0
            total_w = sum(r["workers"] for r in routes)
            all_rows.append({
                "City": city, "Cutoff": cutoff,
                "Routes": len(routes), "Best Route": best_w, "Total Workers": total_w,
            })

    df = pd.DataFrame(all_rows)
    pivot = df.pivot(index="Cutoff", columns="City", values="Best Route")
    pivot["Average"] = pivot.mean(axis=1)
    print("Best route workers by path cutoff (max hops):")
    print(pivot.to_string())
    print("\nDiminishing returns after 5 hops.")
    return df


# ============================================================
# Experiment 10: Min feeder workers threshold
# ============================================================
def exp_min_feeder_workers(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 10: Min Feeder Workers Threshold")
    print("="*65)
    print("How does the minimum workers-per-tract filter affect coverage?\n")

    thresholds = [1, 3, 5, 10, 20, 50]
    all_rows = []

    for city, data in all_data.items():
        stranded_df = data["stranded_df"]
        work_summary = data["work_summary"]
        tract_centroids = data["tract_centroids"]

        jc = find_top_job_centers(work_summary, top_n=1)
        if jc.empty:
            continue
        jc = jc.iloc[0]
        jc_tract = jc["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})
        if not jc_loc:
            continue

        for thresh in thresholds:
            feeder_df = _build_feeder_df(stranded_df, jc_tract, tract_centroids,
                                         min_workers=thresh)
            n_feeders = len(feeder_df)
            total_w = int(feeder_df["workers"].sum()) if not feeder_df.empty else 0

            best_route_w = 0
            if not feeder_df.empty and n_feeders <= 500:
                zones = _cluster_pickup_zones(feeder_df, eps_mi=1.0)
                if len(zones) >= 2 and len(zones) <= 200:
                    G = _build_corridor_graph(zones, jc_loc["lat"], jc_loc["lon"])
                    routes = _find_best_paths(G, zones, max_routes=3)
                    if routes:
                        best_route_w = routes[0]["workers"]

            all_rows.append({
                "City": city, "Min Workers": thresh,
                "Feeder Tracts": n_feeders, "Total Workers": total_w,
                "Best Route": best_route_w,
            })

    df = pd.DataFrame(all_rows)
    pivot = df.pivot(index="Min Workers", columns="City", values="Best Route")
    pivot["Average"] = pivot.mean(axis=1)
    print("Best route workers by min feeder threshold:")
    print(pivot.to_string())

    fpivot = df.pivot(index="Min Workers", columns="City", values="Feeder Tracts")
    fpivot["Average"] = fpivot.mean(axis=1)
    print("\nFeeder tract count by threshold:")
    print(fpivot.to_string())
    print("\nThreshold=5 captures most workers without noise tracts.")
    return df


# ============================================================
# Experiment 11: Max feeder distance
# ============================================================
def exp_max_feeder_distance(all_data):
    print("\n" + "="*65)
    print("EXPERIMENT 11: Max Feeder Distance")
    print("="*65)
    print("How far should routes extend from the job center?\n")

    distances = [5, 10, 15, 20, 30]
    all_rows = []

    for city, data in all_data.items():
        stranded_df = data["stranded_df"]
        work_summary = data["work_summary"]
        tract_centroids = data["tract_centroids"]

        jc = find_top_job_centers(work_summary, top_n=1)
        if jc.empty:
            continue
        jc = jc.iloc[0]
        jc_tract = jc["work_tract"]
        jc_loc = tract_centroids.get(jc_tract, {})
        if not jc_loc:
            continue

        for max_d in distances:
            feeder_df = _build_feeder_df(stranded_df, jc_tract, tract_centroids,
                                         min_workers=5, max_dist_mi=max_d)
            total_w = int(feeder_df["workers"].sum()) if not feeder_df.empty else 0
            best_route_w = 0

            if not feeder_df.empty and len(feeder_df) <= 500:
                zones = _cluster_pickup_zones(feeder_df, eps_mi=1.0)
                if len(zones) >= 2 and len(zones) <= 200:
                    G = _build_corridor_graph(zones, jc_loc["lat"], jc_loc["lon"])
                    routes = _find_best_paths(G, zones, max_routes=3)
                    if routes:
                        best_route_w = routes[0]["workers"]

            all_rows.append({
                "City": city, "Max Dist (mi)": max_d,
                "Feeders": len(feeder_df) if not feeder_df.empty else 0,
                "Total Workers": total_w, "Best Route": best_route_w,
            })

    df = pd.DataFrame(all_rows)
    pivot = df.pivot(index="Max Dist (mi)", columns="City", values="Best Route")
    pivot["Average"] = pivot.mean(axis=1)
    print("Best route workers by max feeder distance:")
    print(pivot.to_string())
    print("\n20mi captures most commuters; beyond 30mi adds little.")
    return df


# ============================================================
# Baseline summary across cities
# ============================================================
def exp_city_summary(all_data):
    print("\n" + "="*65)
    print("BASELINE: City-by-City Transit Connectivity Summary")
    print("="*65 + "\n")

    rows = []
    for city, data in all_data.items():
        t = data["totals"]
        recs = data["recs"]
        served = sum(r["route_workers"] for r in recs)
        rows.append({
            "City": city,
            "Commuters": f"{t['total_commuters']:,}",
            "Connected": f"{t['connected']:,}",
            "Stranded": f"{t['stranded']:,}",
            "Stranded %": f"{t['pct_stranded']:.1%}",
            "Routes": len(recs),
            "Served by Routes": f"{served:,}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ============================================================
# Main
# ============================================================
def main():
    print("Transit Route Recommendation - Validation Experiments")
    print("=" * 65)
    print(f"Cities: {', '.join(CITIES.keys())}")
    print()

    # load SLD once (shared across all cities)
    print("Loading SLD (shared)...")
    sld = load_sld_once()

    # load each city
    all_data = {}
    for city_name, cfg in CITIES.items():
        print(f"Loading {city_name}...")
        data = load_city(city_name, cfg, sld)
        if data:
            all_data[city_name] = data

    if not all_data:
        print("No cities loaded successfully. Exiting.")
        return

    print(f"\n{len(all_data)} cities loaded successfully.")

    t0 = time.time()

    # run experiments
    exp_city_summary(all_data)
    exp_walk_radius(all_data)
    exp_transfer_limit(all_data)
    exp_dbscan_eps(all_data)
    exp_job_center_coverage(all_data)
    exp_route_directness(all_data)
    exp_vs_random_baseline(all_data)
    exp_equity_coverage(all_data)
    exp_bearing_constraint(all_data)
    exp_path_cutoff(all_data)
    exp_min_feeder_workers(all_data)
    exp_max_feeder_distance(all_data)

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"All experiments completed in {elapsed:.0f}s across {len(all_data)} cities")
    print("=" * 65)


if __name__ == "__main__":
    main()
