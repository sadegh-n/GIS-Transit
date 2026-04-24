"""
CSE 6242 - Transit Connectivity Dashboard
Tab 1: Overview - shows a map of transit deserts
Tab 2: Proposed Routes - shows where we think new bus lines should go
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import json
import os
from math import cos, radians

import requests as _requests

from data_loader import (load_sld, load_lodes, load_geometry, load_gtfs,
                         auto_download_gtfs, fetch_route_stops_api)
from connectivity import build_transit_graph, check_commutes, summarize_stranded
from route_finder import recommend_routes

ORS_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImJjMGE5MTNjMmU4NDRmYTVhZTBjY2VjZDEwNDFhNzU4IiwiaCI6Im11cm11cjY0In0="

# bump this when connectivity logic changes so old caches get regenerated
CACHE_VERSION = 2

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(name):
    """Look for a data file in the script directory or its parent."""
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        return p
    # check parent dir in case running from a subdirectory
    parent = os.path.join(os.path.dirname(DATA_DIR), name)
    if os.path.exists(parent):
        return parent
    return p

st.set_page_config(layout="wide", page_title="Transit Connectivity Dashboard")

# Custom CSS to make the dashboard look nicer
st.markdown("""
<style>
    /* Tighter padding */
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 600;
    }
    /* Metric cards - theme aware */
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    /* Expander styling */
    .streamlit-expanderHeader { font-weight: 600; font-size: 0.95rem; }
    /* Caption */
    .stCaption { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# Each city we support, with its FIPS codes, coordinates, GTFS directory, and bounding box
CITIES = {
    # Texas cities (we already have local GTFS + LODES data for these)
    "Dallas, TX": {
        "state": "48", "county": "113", "fips": "48113", "state_abbr": "tx",
        "lat": 32.77, "lon": -96.79, "zoom": 10,
        "gtfs_dir": "V248-160-161-20210517",
        "bbox": "-97.1,32.55,-96.45,33.05",
    },
    "Fort Worth, TX": {
        "state": "48", "county": "439", "fips": "48439", "state_abbr": "tx",
        "lat": 32.75, "lon": -97.33, "zoom": 10,
        "gtfs_dir": "fortworth_gtfs",
        "bbox": "-97.65,32.5,-97.0,32.95",
    },
    "Houston, TX": {
        "state": "48", "county": "201", "fips": "48201", "state_abbr": "tx",
        "lat": 29.76, "lon": -95.37, "zoom": 10,
        "gtfs_dir": "houston_gtfs",
        "bbox": "-95.8,29.5,-95.0,30.1",
    },
    "San Antonio, TX": {
        "state": "48", "county": "029", "fips": "48029", "state_abbr": "tx",
        "lat": 29.42, "lon": -98.49, "zoom": 10,
        "gtfs_dir": "via_gtfs",
        "bbox": "-98.8,29.1,-98.1,29.8",
    },
    "Austin, TX": {
        "state": "48", "county": "453", "fips": "48453", "state_abbr": "tx",
        "lat": 30.27, "lon": -97.74, "zoom": 10,
        "gtfs_dir": "austin_gtfs",
        "bbox": "-98.1,30.0,-97.4,30.55",
    },
    # Southeast cities (auto-downloads GTFS + LODES on first load)
    "Atlanta, GA": {
        "state": "13", "county": "121", "fips": "13121", "state_abbr": "ga",
        "lat": 33.75, "lon": -84.39, "zoom": 10,
        "gtfs_dir": "atlanta_gtfs",
        "bbox": "-84.65,33.55,-84.15,33.95",
    },
    "Jacksonville, FL": {
        "state": "12", "county": "031", "fips": "12031", "state_abbr": "fl",
        "lat": 30.33, "lon": -81.66, "zoom": 10,
        "gtfs_dir": "jacksonville_gtfs",
        "bbox": "-82.0,30.1,-81.3,30.6",
    },
    "Charlotte, NC": {
        "state": "37", "county": "119", "fips": "37119", "state_abbr": "nc",
        "lat": 35.23, "lon": -80.84, "zoom": 10,
        "gtfs_dir": "charlotte_gtfs",
        "bbox": "-81.1,35.0,-80.55,35.45",
    },
    # Midwest/West cities
    "Detroit, MI": {
        "state": "26", "county": "163", "fips": "26163", "state_abbr": "mi",
        "lat": 42.33, "lon": -83.05, "zoom": 10,
        "gtfs_dir": "detroit_gtfs",
        "bbox": "-83.35,42.15,-82.8,42.55",
    },
    "Phoenix, AZ": {
        "state": "04", "county": "013", "fips": "04013", "state_abbr": "az",
        "lat": 33.45, "lon": -112.07, "zoom": 10,
        "gtfs_dir": "phoenix_gtfs",
        "bbox": "-112.4,33.2,-111.7,33.7",
    },
    "Nashville, TN": {
        "state": "47", "county": "037", "fips": "47037", "state_abbr": "tn",
        "lat": 36.16, "lon": -86.78, "zoom": 10,
        "gtfs_dir": "nashville_gtfs",
        "bbox": "-87.1,35.95,-86.5,36.4",
    },
}

# colors for drawing different routes on the map
ROUTE_COLORS = [
    [230, 50, 50], [50, 130, 230], [230, 160, 30],
    [140, 50, 200], [50, 180, 100], [230, 100, 180],
    [80, 200, 200], [200, 120, 50], [100, 100, 230],
    [180, 80, 80], [60, 160, 180], [200, 180, 60],
    [180, 60, 160], [100, 200, 80], [220, 130, 100],
    [70, 100, 180], [160, 140, 60], [200, 60, 120],
    [100, 180, 140], [180, 100, 200], [140, 160, 100],
    [80, 140, 220], [220, 80, 80], [60, 200, 160],
    [200, 100, 60], [120, 60, 180], [180, 200, 80],
    [100, 60, 140], [60, 180, 120], [220, 140, 160],
    [140, 100, 60], [80, 120, 200],
]


# Cached data loaders
# We use st.cache_data so Streamlit doesn't reload these every time the page refreshes

@st.cache_data
def cached_sld():
    """Load the EPA Smart Location Database. Uses the pre-filtered version if available (way smaller)."""
    filtered = find_file("sld_filtered.csv")
    if os.path.exists(filtered):
        return load_sld(filtered)
    return load_sld(find_file("EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"))

@st.cache_data
def cached_lodes(state_abbr):
    """Load LODES commute data for the given state."""
    filename = f"{state_abbr}_od_main_JT00_2021.csv.gz"
    return load_lodes(find_file(filename), state_abbr=state_abbr)

@st.cache_data
def cached_geometry(state, county):
    """Download and cache census block group boundaries."""
    return load_geometry(state, county)

@st.cache_data
def cached_gtfs(gtfs_dir, bbox):
    """Try to load GTFS from local files first, then download, then fall back to API."""
    for d in [gtfs_dir, os.path.join(DATA_DIR, gtfs_dir)]:
        sl, srm = load_gtfs(d)
        if sl is not None:
            return sl, srm, "local"
    result = auto_download_gtfs(bbox, gtfs_dir)
    if result:
        sl, srm = load_gtfs(gtfs_dir)
        if sl is not None:
            return sl, srm, "downloaded"
    sl, srm = fetch_route_stops_api(bbox)
    if sl is not None:
        return sl, srm, "api"
    return None, None, None

@st.cache_data
def cached_connectivity(_geometry, _stop_locs, _stop_route_map, city_fips=""):
    """Build the transit reachability graph for a city."""
    return build_transit_graph(_geometry, _stop_locs, _stop_route_map)

@st.cache_data
def cached_commute_check(_lodes, _reachability, fips):
    """Check which commutes can actually be made by transit."""
    return check_commutes(_lodes, _reachability, fips)


@st.cache_data(show_spinner=False)
def _fetch_ors_route_cached(waypoints_tuple):
    """Call ORS API to snap our proposed route waypoints to real roads."""
    if len(waypoints_tuple) < 2:
        return list(waypoints_tuple)
    try:
        r = _requests.post(
            "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
            json={"coordinates": waypoints_tuple},
            headers={"Authorization": ORS_KEY, "Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()["features"][0]["geometry"]["coordinates"]
    except Exception:
        pass
    return list(waypoints_tuple)


def get_road_route(waypoints):
    """Round waypoints to 4 decimal places and call ORS. The rounding helps with caching."""
    standardized = tuple(
        (round(float(pt[0]), 4), round(float(pt[1]), 4))
        for pt in waypoints
    )
    return _fetch_ors_route_cached(standardized)


# Map helpers

@st.cache_data
def build_geojson(_geometry, county_df, fill_gaps=True):
    """Build GeoJSON for the choropleth map. Colors each block group by transit ratio."""
    keep_cols = ["GEOID", "transit_ratio", "TotPop", "Pct_AO0"]
    for extra in ["D5AR", "D5BR", "R_PCTLOWWAGE", "employed_residents", "zero_car_pop"]:
        if extra in county_df.columns:
            keep_cols.append(extra)
    merged = _geometry.merge(county_df[keep_cols], on="GEOID", how="left").to_crs("EPSG:4326")

    has_data = merged["transit_ratio"].notna()
    vals = merged["transit_ratio"].fillna(0)
    lo = 0
    hi = merged["transit_ratio"].quantile(0.95)
    norm = ((vals - lo) / max(hi - lo, 0.001)).clip(0, 1)

    # red = worst transit, blue = best transit, gray = no data
    merged["r"] = np.where(has_data, (200 - norm * 160).astype(int), 200)
    merged["g"] = np.where(has_data, (40 + norm * 60).astype(int), 200)
    merged["b"] = np.where(has_data, (40 + norm * 180).astype(int), 200)
    merged["a"] = np.where(has_data, 170, 30)

    merged["has_data"] = has_data.map({True: "yes", False: "no"})
    merged["transit_pct"] = np.where(has_data, (merged["transit_ratio"] * 100).round(1), -1)
    merged["pop"] = merged["TotPop"].fillna(0).astype(int)
    merged["zero_car_pct"] = np.where(has_data, (merged["Pct_AO0"] * 100).round(1), -1)
    merged["transit_str"] = np.where(has_data,
        merged["transit_ratio"].apply(lambda x: f"{x*100:.1f}%"), "No data")
    merged["zerocar_str"] = np.where(has_data,
        merged["Pct_AO0"].apply(lambda x: f"{x*100:.1f}%"), "No data")

    return json.loads(merged.to_json())


# Tab 1: Overview

def tab_overview(county_df, geojson, stop_locs, totals, cfg):
    """Render the overview tab with the transit desert map and summary stats."""
    if totals and totals["total_commuters"] > 0:
        pct = totals["pct_stranded"]
        st.markdown(f"## {pct:.0%} of commuters can't reach their job by transit")
        st.caption(f"{totals['stranded']:,} out of {totals['total_commuters']:,} "
                   f"internal commuters have no viable bus path to work "
                   f"(0.5 mi walk, up to 2 transfers)")
    else:
        deserts = int((county_df["transit_ratio"] < 0.05).sum())
        st.markdown(f"## {deserts:,} neighborhoods are transit deserts")

    pop = int(county_df["TotPop"].sum())
    deserts = int((county_df["transit_ratio"] < 0.05).sum())
    desert_pop = int(county_df[county_df["transit_ratio"] < 0.05]["TotPop"].sum())
    zero_car = int(county_df[county_df["transit_ratio"] < 0.05]["zero_car_pop"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population", f"{pop:,}")
    c2.metric("Transit Deserts (<5%)", f"{deserts:,}")
    c3.metric("People in Deserts", f"{desert_pop:,}")
    c4.metric("Zero-Car in Deserts", f"{zero_car:,}")

    # green/red bar showing connected vs stranded percentages
    if totals and totals["total_commuters"] > 0:
        pct_conn = totals["pct_connected"]
        pct_strand = totals["pct_stranded"]
        st.markdown(
            f'<div style="background:#e9ecef;border-radius:6px;height:36px;margin:8px 0 4px 0;'
            f'display:flex;overflow:hidden">'
            f'<div style="background:#2ecc71;width:{pct_conn*100:.0f}%;height:100%;'
            f'display:flex;align-items:center;justify-content:center">'
            f'<span style="color:white;font-weight:700;font-size:0.85rem">'
            f'{pct_conn:.0%}</span></div>'
            f'<div style="background:#e74c3c;width:{pct_strand*100:.0f}%;height:100%;'
            f'display:flex;align-items:center;justify-content:center">'
            f'<span style="color:white;font-weight:700;font-size:0.85rem">'
            f'{pct_strand:.0%}</span></div></div>'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:1rem;font-weight:600;margin:4px 0 16px 0">'
            f'<span style="color:#2ecc71">Can reach work: {totals["connected"]:,}</span>'
            f'<span style="color:#e74c3c">Cannot reach work: {totals["stranded"]:,}</span></div>',
            unsafe_allow_html=True,
        )

    layers = [
        pdk.Layer("GeoJsonLayer", geojson,
                  pickable=True, stroked=True, filled=True,
                  get_fill_color="[properties.r, properties.g, properties.b, properties.a]",
                  get_line_color=[30, 30, 30, 40], line_width_min_pixels=0.3),
    ]
    if stop_locs is not None and not stop_locs.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            stop_locs[["stop_lat", "stop_lon"]].dropna().to_dict("records"),
            get_position=["stop_lon", "stop_lat"],
            get_fill_color=[255, 255, 255, 50], get_radius=20,
        ))

    event = st.pydeck_chart(pdk.Deck(
        layers=layers,
        map_style="road",
        initial_view_state=pdk.ViewState(
            latitude=cfg["lat"], longitude=cfg["lon"], zoom=cfg["zoom"]),
        tooltip={"text": "Transit / Driving Ratio: {transit_str}\nPopulation: {pop}\nZero-car HH: {zerocar_str}"},
    ), on_select="rerun", selection_mode="single-object",
       use_container_width=True, key="overview_map")

    # save whatever block group the user clicked on
    if event and event.selection:
        objs = event.selection.get("objects", {})
        for layer_hits in objs.values():
            if layer_hits:
                props = layer_hits[0].get("properties", layer_hits[0])
                if props:
                    st.session_state["selected_bg"] = props
                break

    # show details for the selected block group in the sidebar
    selected = st.session_state.get("selected_bg")
    if selected:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Selected Area")

        if selected.get("has_data") == "yes":
            tr = selected.get("transit_pct", 0)
            p = selected.get("pop", 0)
            zc = selected.get("zero_car_pct", 0)

            severity = "Transit desert" if tr < 5 else "Low access" if tr < 15 else "Moderate" if tr < 30 else "Good"
            st.sidebar.markdown(f"**{severity}**")
            st.sidebar.metric("Transit / Driving Ratio", f"{tr}%")
            st.sidebar.metric("Population", f"{p:,}")
            st.sidebar.metric("Zero-Car HH", f"{zc}%")

            d5ar = selected.get("D5AR")
            d5br = selected.get("D5BR")
            if d5ar is not None:
                st.sidebar.metric("Jobs by Car (45m)", f"{int(d5ar):,}")
            if d5br is not None:
                st.sidebar.metric("Jobs by Transit (45m)", f"{int(d5br):,}")

            low_wage = selected.get("R_PCTLOWWAGE")
            if low_wage is not None:
                st.sidebar.metric("Low-Wage Workers", f"{low_wage*100:.0f}%")
        else:
            st.sidebar.info("No transit data for this area")

        if st.sidebar.button("Clear selection"):
            del st.session_state["selected_bg"]
            st.rerun()


    # color legend bar at the bottom of the map
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin:4px 0 8px 0">'
        '<span style="font-size:0.8rem;color:#888">Worse transit</span>'
        '<div style="flex:1;height:14px;border-radius:3px;'
        'background:linear-gradient(to right, rgb(200,40,40), rgb(160,50,85), '
        'rgb(120,70,130), rgb(80,85,175), rgb(40,100,220))"></div>'
        '<span style="font-size:0.8rem;color:#888">Better transit</span>'
        '<span style="display:inline-block;width:20px;height:14px;'
        'background:rgb(220,220,220);border-radius:3px;margin-left:8px"></span>'
        '<span style="font-size:0.8rem;color:#888">No data</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Click a block group for details in the sidebar.")
    with st.expander("What do these values mean?"):
        st.markdown(
            "**Transit / Driving Ratio** -- Jobs reachable in 45 min by transit "
            "divided by jobs reachable by car. "
            "0.10 = transit reaches 10% of what driving can. "
            "1.0 = equal access. Above 1.0 = transit outperforms driving "
            "(common near rail in dense areas).\n\n"
            "**Zero-car HH** -- Share of households with no vehicle. "
            "These residents depend entirely on transit, walking, or rides.\n\n"
            "**Transit Desert** -- Areas where the transit/driving ratio is below 0.05 (5%)."
        )


# Tab 2: Proposed Routes

def _deg_to_mi(lon1, lat1, lon2, lat2, cos_lat):
    """Quick distance approximation in miles using lat/lon and a precomputed cos(lat)."""
    dlat = (lat1 - lat2) * 69
    dlon = (lon1 - lon2) * 69 * cos_lat
    return (dlat**2 + dlon**2)**0.5


def _precompute_route_stats(recommendations, cfg):
    """For each recommended route, snap it to roads, figure out which block groups
    are close enough to walk to the route, and place bus stops along it."""
    if not recommendations or "_road_path" in recommendations[0]:
        return  # already done

    cos_lat = cos(radians(cfg["lat"]))

    # Pass 1: get road-snapped paths and find walkable block groups for each route
    for route in recommendations:
        jc_lon, jc_lat = route["jc_lon"], route["jc_lat"]
        feeders = [f for f in route["feeders"] if f["lat"]]
        if not feeders:
            route["_road_path"] = []
            route["_candidate_bgs"] = []
            continue

        # Start from the farthest feeder and route toward the job center.
        # Intermediate feeders become waypoints only if they're close to the
        # direct line (within 0.3mi perpendicular), so the route stays straight.
        # Feeders farther off the line still get counted via the walk radius.
        sorted_by_dist = sorted(feeders, key=lambda f: -f["dist_mi"])
        start_f = sorted_by_dist[0]
        sx, sy = start_f["lon"], start_f["lat"]
        dx, dy = jc_lon - sx, jc_lat - sy
        line_len_sq = dx**2 + dy**2

        intermediates = []
        for f in sorted_by_dist[1:]:
            # project this feeder onto the line from start to job center
            t = ((f["lon"] - sx) * dx + (f["lat"] - sy) * dy) / max(line_len_sq, 1e-10)
            if t <= 0.05 or t >= 0.95:
                continue  # too close to the endpoints, skip
            proj_lon = sx + t * dx
            proj_lat = sy + t * dy
            perp_mi = _deg_to_mi(f["lon"], f["lat"], proj_lon, proj_lat, cos_lat)
            if perp_mi <= 0.5:
                intermediates.append({"lon": f["lon"], "lat": f["lat"],
                                      "t": t, "workers": f["workers"]})

        # pick the highest-worker detours, but space them at least 1.5mi apart, max 4
        intermediates.sort(key=lambda p: -p["workers"])
        picks = []
        for p in intermediates:
            too_close = any(
                _deg_to_mi(p["lon"], p["lat"], q["lon"], q["lat"], cos_lat) < 1.5
                for q in picks
            )
            if not too_close:
                picks.append(p)
            if len(picks) >= 4:
                break
        picks.sort(key=lambda p: p["t"])  # put them back in order along the line

        waypoints = [[round(sx, 6), round(sy, 6)]]
        for p in picks:
            waypoints.append([round(p["lon"], 6), round(p["lat"], 6)])
        waypoints.append([round(jc_lon, 6), round(jc_lat, 6)])
        road_path = get_road_route(waypoints)

        route["_road_path"] = road_path

        # find block groups within 0.75mi walk of the route
        # 0.75mi is the FTA standard walking distance for high-frequency/BRT service
        walk_radius = 0.75
        bg_data = route.get("bg_data", [])
        candidates = []
        if bg_data and len(road_path) > 1:
            road_arr = np.array(road_path)
            for bg in bg_data:
                dlats = (road_arr[:, 1] - bg["lat"]) * 69
                dlons = (road_arr[:, 0] - bg["lon"]) * 69 * cos_lat
                dists_mi = np.sqrt(dlats**2 + dlons**2)
                dist_route_mi = float(dists_mi.min())
                nearest_idx = int(dists_mi.argmin())
                dist_jc_mi = _deg_to_mi(bg["lon"], bg["lat"], jc_lon, jc_lat, cos_lat)
                # only count BGs that are closer to the route than to the job center
                # (otherwise they don't really need this route)
                if dist_route_mi <= walk_radius and dist_jc_mi > dist_route_mi:
                    candidates.append({
                        **bg, "_road_idx": nearest_idx, "_dist_route": dist_route_mi,
                    })
        route["_candidate_bgs"] = candidates

    # Pass 2: if a block group is near multiple routes going to the same job center,
    # only assign it to the closest route so we don't double-count workers
    from collections import defaultdict
    jc_groups = defaultdict(list)
    for ri, route in enumerate(recommendations):
        jc_groups[route["job_center"]].append(ri)

    for jc, route_indices in jc_groups.items():
        if len(route_indices) <= 1:
            continue

        bg_best_route = {}  # tract -> (route_index, dist)
        for ri in route_indices:
            for bg in recommendations[ri]["_candidate_bgs"]:
                key = bg["tract"]
                if key not in bg_best_route or bg["_dist_route"] < bg_best_route[key][1]:
                    bg_best_route[key] = (ri, bg["_dist_route"])

        for ri in route_indices:
            route = recommendations[ri]
            route["_candidate_bgs"] = [
                bg for bg in route["_candidate_bgs"]
                if bg_best_route.get(bg["tract"], (ri,))[0] == ri
            ]

    # Pass 3: finalize worker counts and place stops along each route
    for route in recommendations:
        walkable_bgs = route.pop("_candidate_bgs", [])
        road_path = route["_road_path"]

        route["_walkable_bgs"] = walkable_bgs
        route["_walkable_workers"] = sum(bg["workers"] for bg in walkable_bgs)
        route["_walkable_lowwage"] = sum(bg["low_wage"] for bg in walkable_bgs)

        # place stops where workers cluster along the route, at least 1mi apart
        placed_stops = []
        if walkable_bgs and len(road_path) > 2:
            eligible = sorted(walkable_bgs, key=lambda bg: bg["_road_idx"])

            covered = set()
            for ei, entry in enumerate(eligible):
                if id(entry) in covered:
                    continue

                anchor_pt = road_path[min(entry["_road_idx"], len(road_path) - 1)]
                group = [entry]

                # group nearby BGs together so they share one stop
                for ej in range(ei + 1, len(eligible)):
                    if id(eligible[ej]) in covered:
                        continue
                    pt = road_path[min(eligible[ej]["_road_idx"], len(road_path) - 1)]
                    d = _deg_to_mi(pt[0], pt[1], anchor_pt[0], anchor_pt[1], cos_lat)
                    if d < 0.75:
                        group.append(eligible[ej])

                med_idx = min(group[len(group) // 2]["_road_idx"], len(road_path) - 1)
                slon, slat = road_path[med_idx]

                # merge into an existing stop if it's less than 1mi away
                too_close = False
                for prev in placed_stops:
                    d = _deg_to_mi(slon, slat, prev["lon"], prev["lat"], cos_lat)
                    if d < 1.0:
                        prev["workers"] += sum(g["workers"] for g in group)
                        prev["low_wage"] += sum(g["low_wage"] for g in group)
                        too_close = True
                        break

                if not too_close:
                    placed_stops.append({
                        "lon": slon, "lat": slat,
                        "workers": sum(g["workers"] for g in group),
                        "low_wage": sum(g["low_wage"] for g in group),
                    })

                for g in group:
                    covered.add(id(g))

        route["_placed_stops"] = placed_stops


def tab_routes(recommendations, stop_locs, geojson_base, totals, cfg):
    """Render the proposed routes tab with an interactive map and route details."""
    if not recommendations:
        st.warning("No route recommendations available.")
        return

    # get road paths and accurate worker counts for each route
    _precompute_route_stats(recommendations, cfg)

    # drop routes that turned out too weak after road-snapping
    recommendations[:] = [r for r in recommendations if r["_walkable_workers"] >= 100]
    recommendations.sort(key=lambda r: -r["_walkable_workers"])
    del recommendations[8:]  # keep at most 8 routes

    total_served = sum(r["_walkable_workers"] for r in recommendations)
    st.markdown(f"## {len(recommendations)} proposed bus corridors")
    st.caption(f"Serving {total_served:,} stranded workers across "
               f"{len(set(r['job_center'] for r in recommendations))} job centers")

    # checkboxes to toggle routes on and off
    st.markdown("#### Select routes to display")
    cols = st.columns(min(len(recommendations), 4))
    visible = []
    for i, route in enumerate(recommendations):
        jc_short = f"...{route['job_center'][-4:]}"
        label = f"{route['direction']} -> {jc_short} ({route['_walkable_workers']:,})"
        if cols[i % len(cols)].checkbox(label, value=(i < 3), key=f"route_{i}"):
            visible.append(i)

    # build map layers
    layers = [
        pdk.Layer("GeoJsonLayer", geojson_base, filled=True,
                  get_fill_color="[properties.r, properties.g, properties.b, 40]",
                  get_line_color=[30, 30, 30, 20], line_width_min_pixels=0.2),
    ]

    all_markers = []
    all_stop_markers = []
    jc_markers = []
    all_lats, all_lons = [], []
    shown_jcs = set()

    for i in visible:
        route = recommendations[i]
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        jc_lat, jc_lon = route["jc_lat"], route["jc_lon"]
        if not jc_lat:
            continue

        road_path = route.get("_road_path", [])
        if road_path and len(road_path) > 1:
            rw = route["_walkable_workers"]
            rlw = route["_walkable_lowwage"]
            route_label = (
                f"{route['direction']} corridor: "
                f"{rw:,} workers, "
                f"{rlw:,} low-wage"
            )

            # draw the route line
            layers.append(pdk.Layer(
                "PathLayer",
                data=[{"path": road_path, "color": color + [220],
                       "label": route_label}],
                get_path="path", get_color="color",
                get_width=8, width_min_pixels=4, pickable=True,
            ))

            for pt in road_path[::10]:
                all_lons.append(pt[0])
                all_lats.append(pt[1])

            # proposed stop markers (black dots with white outline)
            for ps in route.get("_placed_stops", []):
                all_stop_markers.append({
                    "position": [ps["lon"], ps["lat"]],
                    "label": f"Proposed stop: {ps['workers']:,} workers within 0.5mi",
                })

        # dots for worker block groups near this route
        walkable_bgs = route.get("_walkable_bgs", [])
        if walkable_bgs:
            max_bg_w = max(bg["workers"] for bg in walkable_bgs)
            for bg in walkable_bgs:
                ratio = bg["workers"] / max(max_bg_w, 1)
                radius = int(80 + ratio * 250)
                all_markers.append({
                    "position": [bg["lon"], bg["lat"]],
                    "label": f"{bg['workers']:,} workers ({bg['tract'][-6:]})",
                    "color": color + [150], "radius": radius,
                })

        # big green dot for the job center
        if route["job_center"] not in shown_jcs:
            jc_markers.append({
                "position": [jc_lon, jc_lat],
                "label": f"Job center: {route['jc_stranded_total']:,} workers need to reach here",
                "color": [30, 200, 60], "radius": 800,
            })
            shown_jcs.add(route["job_center"])
            all_lats.append(jc_lat)
            all_lons.append(jc_lon)

    # existing transit stops shown faintly in the background
    if stop_locs is not None and not stop_locs.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            stop_locs[["stop_lat", "stop_lon"]].dropna().to_dict("records"),
            get_position=["stop_lon", "stop_lat"],
            get_fill_color=[255, 255, 255, 40], get_radius=15,
        ))

    # worker block group dots
    if all_markers:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=all_markers,
            get_position="position", get_fill_color="color",
            get_radius="radius", pickable=True,
        ))

    # proposed stop markers (white outline + black fill)
    if all_stop_markers:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=all_stop_markers,
            get_position="position",
            get_fill_color=[255, 255, 255, 220],
            get_radius=220, pickable=False,
        ))
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=all_stop_markers,
            get_position="position",
            get_fill_color=[30, 30, 30, 240],
            get_radius=140, pickable=True,
        ))

    # job center markers (largest, drawn on top)
    if jc_markers:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=jc_markers,
            get_position="position",
            get_fill_color=[255, 255, 255, 200],
            get_radius=900, pickable=False,
        ))
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=jc_markers,
            get_position="position", get_fill_color="color",
            get_radius="radius", pickable=True,
        ))

    mid_lat = np.mean(all_lats) if all_lats else cfg["lat"]
    mid_lon = np.mean(all_lons) if all_lons else cfg["lon"]

    # render the map
    route_event = st.pydeck_chart(pdk.Deck(
        layers=layers,
        map_style="road",
        initial_view_state=pdk.ViewState(
            latitude=mid_lat, longitude=mid_lon, zoom=10.5),
        tooltip={"text": "{label}"},
    ), on_select="rerun", selection_mode="single-object",
       use_container_width=True, key="routes_map")

    st.caption("Click a route or community dot for details in the sidebar. "
               "Green circles = job centers.")

    # figure out which route the user clicked on
    if route_event and route_event.selection:
        objs = route_event.selection.get("objects", {})
        for layer_hits in objs.values():
            if layer_hits:
                clicked = layer_hits[0]
                clicked_label = clicked.get("label", "")
                for i in visible:
                    r = recommendations[i]
                    if r["direction"] in clicked_label or f"...{r['job_center'][-4:]}" in clicked_label:
                        st.session_state["selected_route"] = i
                        break
                    for f in r["feeders"]:
                        if f"{f['workers']:,} workers" in clicked_label:
                            st.session_state["selected_route"] = i
                            break
                break

    # show details for the selected route in the sidebar
    sel_idx = st.session_state.get("selected_route")
    if sel_idx is not None and sel_idx < len(recommendations):
        route = recommendations[sel_idx]
        color = ROUTE_COLORS[sel_idx % len(ROUTE_COLORS)]
        hex_c = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### <span style='color:{hex_c}'>Route {sel_idx+1}</span>",
                           unsafe_allow_html=True)
        st.sidebar.markdown(
            f"**{route['direction']}** corridor to job center "
            f"...{route['job_center'][-4:]}"
        )

        st.sidebar.metric("Workers Served", f"{route['_walkable_workers']:,}")
        sc1, sc2 = st.sidebar.columns(2)
        sc1.metric("Low-Wage", f"{route['_walkable_lowwage']:,}")
        sc2.metric("Equity", f"{route.get('opp_score', 0):.3f}")
        st.sidebar.metric("Max Distance", f"{route['max_dist_mi']} mi")

        # list the stops we placed along this route
        placed = route.get("_placed_stops", [])
        if placed:
            total_stop_workers = sum(ps["workers"] for ps in placed)
            st.sidebar.markdown(f"**{len(placed)} stops along route** ({total_stop_workers:,} workers)")
            _cos = cos(radians(route["jc_lat"]))
            for j, ps in enumerate(placed):
                dist_jc = _deg_to_mi(ps["lon"], ps["lat"],
                                     route["jc_lon"], route["jc_lat"], _cos)
                st.sidebar.markdown(
                    f"<div style='display:flex;align-items:center;padding:4px 0;"
                    f"border-bottom:1px solid rgba(128,128,128,0.15)'>"
                    f"<span style='color:{hex_c};font-size:14px;margin-right:8px'>◆</span>"
                    f"<div><b>Stop {j+1}</b> — {dist_jc:.1f} mi from job center<br>"
                    f"<span style='font-size:0.85rem;color:#888'>"
                    f"{ps['workers']:,} workers within walking distance"
                    f"</span></div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.sidebar.caption("No stops computed yet -- refresh to recalculate")

        st.sidebar.markdown(
            f"<div style='display:flex;align-items:center;padding:4px 0'>"
            f"<span style='color:#32dc50;font-size:18px;margin-right:8px'>★</span>"
            f"<div><b>Job Center</b> ...{route['job_center'][-4:]}<br>"
            f"<span style='font-size:0.85rem;color:#888'>"
            f"{route['_walkable_workers']:,}"
            f" workers served by this route</span></div></div>",
            unsafe_allow_html=True,
        )

        if st.sidebar.button("Close", key="close_route_detail"):
            del st.session_state["selected_route"]
            st.rerun()

    # route comparison table below the map
    st.divider()
    st.markdown("#### Route comparison")
    summary_rows = []
    for i, r in enumerate(recommendations):
        summary_rows.append({
            "Route": i + 1,
            "Direction": r["direction"],
            "Job Center": f"...{r['job_center'][-4:]}",
            "Workers": r["_walkable_workers"],
            "Low-Wage": r["_walkable_lowwage"],
            "Communities": r["num_feeders"],
            "Max Dist (mi)": r["max_dist_mi"],
            "Equity Score": round(r.get("opp_score", 0), 3),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with st.expander("What do these values mean?"):
        st.markdown(
            "**Workers** -- Number of commuters along this corridor who currently "
            "cannot reach the job center by transit.\n\n"
            "**Low-Wage** -- Workers earning $1,250/month or less. "
            "These workers are least likely to have alternatives to transit.\n\n"
            "**Equity Score** -- Combines low-wage worker share, employment gap, "
            "and transit gap into a single priority metric. Higher score = "
            "more underserved community. Routes are ranked by workers weighted "
            "by this score, so corridors serving the most vulnerable populations "
            "rank higher even if they have fewer total workers.\n\n"
            "**Communities** -- Number of residential areas (census tracts) "
            "along the corridor that would be served by this route.\n\n"
            "**Max Dist** -- Distance from the farthest community to the job center."
        )


def clear_selection():
    """Remove any selected block group or route from session state."""
    for key in list(st.session_state.keys()):
        if key in ("selected_bg", "selected_route") or key.startswith("_recs_"):
            del st.session_state[key]


# Main

def main():
    """Entry point for the Streamlit app."""
    city = st.sidebar.selectbox("City", list(CITIES.keys()), on_change=clear_selection)
    cfg = CITIES[city]

    st.title("Where should we build new bus routes?")
    st.caption(f"{city}")

    with st.spinner("Loading data..."):
        sld = cached_sld()
    county = sld[sld["GEOID"].str[:5] == cfg["fips"]].copy()

    with st.spinner("Loading commute data..."):
        lodes = cached_lodes(cfg["state_abbr"])

    with st.spinner("Loading geometry..."):
        geometry = cached_geometry(cfg["state"], cfg["county"])

    with st.spinner("Loading transit routes..."):
        stop_locs, stop_route_map, gtfs_source = cached_gtfs(cfg["gtfs_dir"], cfg["bbox"])

    st.sidebar.markdown("---")

    # run the connectivity analysis to figure out who is stranded
    totals = None
    recommendations = []

    if stop_locs is not None and stop_route_map is not None and lodes is not None:
        # check for cached results on disk first so we don't redo expensive work
        cache_file = os.path.join(DATA_DIR, f".cache_{cfg['fips']}_results.pkl")
        cache_valid = False
        if os.path.exists(cache_file):
            import pickle
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            # handle both old format (4-tuple) and new format (5-tuple with version)
            if isinstance(cached, tuple) and len(cached) == 5 and cached[0] == CACHE_VERSION:
                _, home_summary, work_summary, stranded_df, totals = cached
                cache_valid = True

        if not cache_valid:
            with st.spinner("Checking transit connectivity (first time, may take a minute)..."):
                reachability, _, conn_stats = cached_connectivity(
                    geometry, stop_locs, stop_route_map, city_fips=cfg["fips"])
                checked = cached_commute_check(lodes, reachability, cfg["fips"])
                home_summary, work_summary, stranded_df, totals = summarize_stranded(checked)
                import pickle
                with open(cache_file, "wb") as f:
                    pickle.dump((CACHE_VERSION, home_summary, work_summary, stranded_df, totals), f)

        if totals["stranded"] > 0:
            # cache recommendations in session state so toggling checkboxes
            # doesn't re-run the whole route recommendation pipeline
            cache_key = f"_recs_{cfg['fips']}"
            if cache_key not in st.session_state:
                with st.spinner("Finding route corridors..."):
                    st.session_state[cache_key] = recommend_routes(
                        stranded_df, work_summary, geometry, sld_df=county,
                    )
            recommendations = st.session_state[cache_key]

        st.sidebar.caption(f"Stranded: {totals['stranded']:,} ({totals['pct_stranded']:.0%})")

    # build the map data and render both tabs
    geojson = build_geojson(geometry, county, fill_gaps=True)

    t1, t2 = st.tabs(["Overview", "Proposed Routes"])
    with t1:
        tab_overview(county, geojson, stop_locs, totals, cfg)
    with t2:
        tab_routes(recommendations, stop_locs, geojson, totals, cfg)


if __name__ == "__main__":
    main()
