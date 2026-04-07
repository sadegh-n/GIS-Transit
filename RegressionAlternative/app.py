"""
Transit Connectivity Dashboard — CSE 6242 Group 123

Tab 1: Overview — where is transit failing workers?
Tab 2: Routes — where should new bus routes go?
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import json
import os

import requests as _requests

from data_loader import (load_sld, load_lodes, load_geometry, load_gtfs,
                         auto_download_gtfs, fetch_route_stops_api)
from connectivity import build_transit_graph, check_commutes, summarize_stranded
from route_finder import recommend_routes

ORS_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImJjMGE5MTNjMmU4NDRmYTVhZTBjY2VjZDEwNDFhNzU4IiwiaCI6Im11cm11cjY0In0="

# Resolve file paths across directories
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ALT_DIR = os.path.join(DATA_DIR, "RegressionAlternative")

def find_file(name):
    for d in [DATA_DIR, ALT_DIR]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, name)

st.set_page_config(layout="wide", page_title="Transit Connectivity Dashboard")

CITIES = {
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
    "San Antonio, TX": {
        "state": "48", "county": "029", "fips": "48029", "state_abbr": "tx",
        "lat": 29.42, "lon": -98.49, "zoom": 10,
        "gtfs_dir": "via_gtfs",
        "bbox": "-98.8,29.1,-98.1,29.8",
    },
    "Houston, TX": {
        "state": "48", "county": "201", "fips": "48201", "state_abbr": "tx",
        "lat": 29.76, "lon": -95.37, "zoom": 10,
        "gtfs_dir": "houston_gtfs",
        "bbox": "-95.8,29.5,-95.0,30.1",
    },
    "Austin, TX": {
        "state": "48", "county": "453", "fips": "48453", "state_abbr": "tx",
        "lat": 30.27, "lon": -97.74, "zoom": 10,
        "gtfs_dir": "austin_gtfs",
        "bbox": "-98.1,30.0,-97.4,30.55",
    },
    "Toledo, OH": {
        "state": "39", "county": "095", "fips": "39095", "state_abbr": "oh",
        "lat": 41.66, "lon": -83.55, "zoom": 11,
        "gtfs_dir": "toledo_gtfs",
        "bbox": "-83.8,41.5,-83.3,41.8",
    },
    "Birmingham, AL": {
        "state": "01", "county": "073", "fips": "01073", "state_abbr": "al",
        "lat": 33.52, "lon": -86.80, "zoom": 11,
        "gtfs_dir": "birmingham_gtfs",
        "bbox": "-87.1,33.3,-86.5,33.7",
    },
    "Jacksonville, FL": {
        "state": "12", "county": "031", "fips": "12031", "state_abbr": "fl",
        "lat": 30.33, "lon": -81.66, "zoom": 10,
        "gtfs_dir": "jacksonville_gtfs",
        "bbox": "-82.0,30.1,-81.3,30.6",
    },
}

ROUTE_COLORS = [
    [230, 50, 50], [50, 130, 230], [230, 160, 30],
    [140, 50, 200], [50, 180, 100], [230, 100, 180],
    [80, 200, 200], [200, 120, 50], [100, 100, 230],
]


# ── Cached loaders ───────────────────────────────────────────────────

@st.cache_data
def cached_sld():
    return load_sld(find_file("EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"))

@st.cache_data
def cached_lodes(state_abbr):
    filename = f"{state_abbr}_od_main_JT00_2021.csv.gz"
    return load_lodes(find_file(filename), state_abbr=state_abbr)

@st.cache_data
def cached_geometry(state, county):
    return load_geometry(state, county)

@st.cache_data
def cached_gtfs(gtfs_dir, bbox):
    # Try local (both dirs), then auto-download, then API
    for d in [gtfs_dir, os.path.join(ALT_DIR, gtfs_dir)]:
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
def cached_connectivity(_geometry, _stop_locs, _stop_route_map):
    return build_transit_graph(_geometry, _stop_locs, _stop_route_map)

@st.cache_data
def cached_commute_check(_lodes, _reachability, fips):
    return check_commutes(_lodes, _reachability, fips)




@st.cache_data(show_spinner=False)
def _fetch_ors_route_cached(waypoints_tuple):
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
    except Exception as e:
        pass
        
    return list(waypoints_tuple) 

def get_road_route(waypoints):
    standardized_waypoints = tuple(
        (round(float(pt[0]), 4), round(float(pt[1]), 4)) 
        for pt in waypoints
    )
    
    return _fetch_ors_route_cached(standardized_waypoints)


# ── Map helpers ──────────────────────────────────────────────────────

@st.cache_data
def build_geojson(_geometry, county_df, fill_gaps=True):
    """Build GeoJSON with transit ratio coloring. Gray for missing data."""
    keep_cols = ["GEOID", "transit_ratio", "TotPop", "Pct_AO0"]
    for extra in ["D5AR", "D5BR", "R_PCTLOWWAGE", "employed_residents", "zero_car_pop"]:
        if extra in county_df.columns:
            keep_cols.append(extra)
    merged = _geometry.merge(county_df[keep_cols], on="GEOID", how="left").to_crs("EPSG:4326")

    has_data = merged["transit_ratio"].notna()
    vals = merged["transit_ratio"].fillna(0)
    lo = 0
    hi = merged['transit_ratio'].quantile(0.95)
    norm = ((vals - lo) / max(hi - lo, 0.001)).clip(0, 1)

    # Red (bad) → Yellow (mid) → Blue (good)
    merged["r"] = np.where(has_data, ((1 - norm) * 220).astype(int), 50)
    merged["g"] = np.where(has_data, (norm * 60 + (1 - norm) * 40).astype(int), 50)
    merged["b"] = np.where(has_data, (norm * 200).astype(int), 50)
    merged["a"] = np.where(has_data, 170, 40)

    merged["has_data"] = has_data.map({True: "yes", False: "no"})
    merged["transit_pct"] = np.where(has_data, (merged["transit_ratio"] * 100).round(1), -1)
    merged["pop"] = merged["TotPop"].fillna(0).astype(int)
    merged["zero_car_pct"] = np.where(has_data, (merged["Pct_AO0"] * 100).round(1), -1)
    # Display-friendly strings for tooltip
    merged["transit_str"] = np.where(has_data,
        merged["transit_ratio"].apply(lambda x: f"{x*100:.1f}%"), "No data")
    merged["zerocar_str"] = np.where(has_data,
        merged["Pct_AO0"].apply(lambda x: f"{x*100:.1f}%"), "No data")

    return json.loads(merged.to_json())


# ── Tab 1: Overview ──────────────────────────────────────────────────

def tab_overview(county_df, geojson, stop_locs, totals, cfg):
    # Headline stat
    if totals and totals["total_commuters"] > 0:
        pct = totals["pct_stranded"]
        st.markdown(f"## {pct:.0%} of commuters can't reach their job by transit")
        st.caption(f"{totals['stranded']:,} out of {totals['total_commuters']:,} "
                   f"internal commuters have no viable bus path to work "
                   f"(0.5 mi walk, up to 2 transfers)")
    else:
        deserts = int((county_df["transit_ratio"] < 0.05).sum())
        st.markdown(f"## {deserts:,} neighborhoods are transit deserts")

    # Metrics row
    pop = int(county_df["TotPop"].sum())
    deserts = int((county_df["transit_ratio"] < 0.05).sum())
    desert_pop = int(county_df[county_df["transit_ratio"] < 0.05]["TotPop"].sum())
    zero_car = int(county_df[county_df["transit_ratio"] < 0.05]["zero_car_pop"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population", f"{pop:,}")
    c2.metric("Transit Deserts", f"{deserts:,}")
    c3.metric("People in Deserts", f"{desert_pop:,}")
    c4.metric("Zero-Car in Deserts", f"{zero_car:,}")

    # Map
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
        tooltip={"text": "Transit access: {transit_str}\nPopulation: {pop}\nZero-car HH: {zerocar_str}"},
    ), on_select="rerun", selection_mode="single-object",
       use_container_width=True, key="overview_map")

    # Save selection to session state
    if event and event.selection:
        objs = event.selection.get("objects", {})
        for layer_hits in objs.values():
            if layer_hits:
                props = layer_hits[0].get("properties", layer_hits[0])
                if props:
                    st.session_state["selected_bg"] = props
                break

    # Show selection detail in sidebar (doesn't cause map to re-render)
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
            st.sidebar.metric("Transit Access", f"{tr}%")
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

    st.caption("Click a block group for details in the sidebar. "
               "Red = transit desert. Blue = well-served. Gray = no data.")


# ── Tab 2: Routes ────────────────────────────────────────────────────

def tab_routes(recommendations, stop_locs, geojson_base, totals, cfg):
    if not recommendations:
        st.warning("No route recommendations available.")
        return

    total_served = sum(r["route_workers"] for r in recommendations)
    st.markdown(f"## {len(recommendations)} proposed bus corridors")
    st.caption(f"Serving {total_served:,} stranded workers across "
               f"{len(set(r['job_center'] for r in recommendations))} job centers")

    # Toggle checkboxes for each route
    st.markdown("### Show routes")
    cols = st.columns(min(len(recommendations), 4))
    visible = []
    
    for i, route in enumerate(recommendations):
        label = f"{route['direction']} ({route['route_workers']:,})"
        if cols[i % len(cols)].checkbox(label, value=(i < 3), key=f"route_{i}"):
            visible.append(i)

    # Build map
    layers = [
        pdk.Layer("GeoJsonLayer", geojson_base, filled=True,
                  get_fill_color="[properties.r, properties.g, properties.b, 40]",
                  get_line_color=[30, 30, 30, 20], line_width_min_pixels=0.2),
    ]

    all_markers = []
    all_lats, all_lons = [], []
    shown_jcs = set()

    for i in visible:
        route = recommendations[i]
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        jc_lat, jc_lon = route["jc_lat"], route["jc_lon"]
        if not jc_lat:
            continue

        # Build route: farthest feeder → through cluster center → job center
        # Use 3 waypoints so the route passes near all communities
        feeders_with_coords = [f for f in route["feeders"] if f["lat"]]

        if feeders_with_coords:
            farthest = max(feeders_with_coords, key=lambda f: f["dist_mi"])

            # Midpoint = center of all feeder communities
            mid_lat = np.mean([f["lat"] for f in feeders_with_coords])
            mid_lon = np.mean([f["lon"] for f in feeders_with_coords])

            waypoints = [
                [round(farthest["lon"], 6), round(farthest["lat"], 6)],
                [round(mid_lon, 6), round(mid_lat, 6)],
                [round(jc_lon, 6), round(jc_lat, 6)],
            ]
            road_path = get_road_route(waypoints)

            opp = route.get('opp_score', 0)
            route_label = (
                f"{route['direction']} corridor: "
                f"{route['route_workers']:,} workers, "
                f"{route['route_low_wage']:,} low-wage, "
                f"{route['num_feeders']} communities, "
                f"equity score: {opp:.3f}"
            )

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

        # Job center marker (once per unique job center)
        if route["job_center"] not in shown_jcs:
            all_markers.append({
                "position": [jc_lon, jc_lat],
                "label": f"Job center: {route['jc_stranded_total']:,} workers need to reach here",
                "color": [50, 220, 80], "radius": 500,
            })
            shown_jcs.add(route["job_center"])

        # Feeder communities as dots sized by worker count
        max_workers = max((f["workers"] for f in feeders_with_coords), default=1)
        for f in feeders_with_coords:
            # Scale radius: 150 (smallest) to 500 (largest)
            ratio = f["workers"] / max(max_workers, 1)
            radius = int(150 + ratio * 350)
            all_markers.append({
                "position": [f["lon"], f["lat"]],
                "label": f"{f['workers']:,} workers, {f['dist_mi']} mi from job center",
                "color": color, "radius": radius,
            })

    if all_markers:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=all_markers,
            get_position="position", get_fill_color="color",
            get_radius="radius", pickable=True,
        ))

    if stop_locs is not None and not stop_locs.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            stop_locs[["stop_lat", "stop_lon"]].dropna().to_dict("records"),
            get_position=["stop_lon", "stop_lat"],
            get_fill_color=[255, 255, 255, 40], get_radius=15,
        ))

    mid_lat = np.mean(all_lats) if all_lats else cfg["lat"]
    mid_lon = np.mean(all_lons) if all_lons else cfg["lon"]

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        map_style="road",
        initial_view_state=pdk.ViewState(
            latitude=mid_lat, longitude=mid_lon, zoom=10.5),
        tooltip={"text": "{label}"},
    ), use_container_width=True)

    # Route detail cards
    for i in visible:
        route = recommendations[i]
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        hex_c = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        with st.expander(
            f"{route['direction']} corridor → job center "
            f"...{route['job_center'][-4:]}  |  "
            f"{route['route_workers']:,} workers  |  "
            f"{route['num_feeders']} stops  |  "
            f"up to {route['max_dist_mi']} mi",
            expanded=(len(visible) == 1),
        ):
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            rc1.metric("Workers Served", f"{route['route_workers']:,}")
            rc2.metric("Low-Wage", f"{route['route_low_wage']:,}")
            rc3.metric("Communities", route["num_feeders"])
            rc4.metric("Max Distance", f"{route['max_dist_mi']} mi")
            rc5.metric("Equity Score", f"{route.get('opp_score', 0):.3f}",
                       help="Opportunity Score: higher = more low-wage workers with poor transit")

            fd = pd.DataFrame(route["feeders"])
            if not fd.empty:
                fd["stop"] = range(len(fd), 0, -1)
                display = fd[["stop", "workers", "low_wage", "dist_mi"]].copy()
                display.columns = ["Stop #", "Workers", "Low-Wage", "Distance (mi)"]
                st.dataframe(display.reset_index(drop=True), use_container_width=True,
                             hide_index=True)

def clear_selection():
    if "selected_bg" in st.session_state:
        del st.session_state["selected_bg"]
# ── Main ─────────────────────────────────────────────────────────────

def main():
    city = st.sidebar.selectbox("City", list(CITIES.keys()), on_change=clear_selection)
    cfg = CITIES[city]

    st.title("Where should we build new bus routes?")
    st.caption(f"{city}")

    # Load data
    with st.spinner("Loading data..."):
        sld = cached_sld()
    county = sld[sld["GEOID"].str[:5] == cfg["fips"]].copy()

    with st.spinner("Loading commute data..."):
        lodes = cached_lodes(cfg["state_abbr"])

    with st.spinner("Loading geometry..."):
        geometry = cached_geometry(cfg["state"], cfg["county"])

    with st.spinner("Loading transit routes..."):
        stop_locs, stop_route_map, gtfs_source = cached_gtfs(cfg["gtfs_dir"], cfg["bbox"])

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Pop: {county['TotPop'].sum():,.0f} | "
                       f"BGs: {len(county):,}")
    if stop_locs is not None:
        st.sidebar.caption(f"Stops: {len(stop_locs):,} ({gtfs_source})")

    # Connectivity analysis
    totals = None
    recommendations = []

    if stop_locs is not None and stop_route_map is not None and lodes is not None:
        with st.spinner("Checking transit connectivity..."):
            reachability, _, conn_stats = cached_connectivity(
                geometry, stop_locs, stop_route_map)
            checked = cached_commute_check(lodes, reachability, cfg["fips"])
            home_summary, work_summary, stranded_df, totals = summarize_stranded(checked)

        if totals["stranded"] > 0:
            with st.spinner("Finding route corridors..."):
                recommendations = recommend_routes(
                    stranded_df, work_summary, geometry, sld_df=county
                )

        st.sidebar.caption(f"Stranded: {totals['stranded']:,} ({totals['pct_stranded']:.0%})")

    # Build maps
    geojson = build_geojson(geometry, county, fill_gaps=True)
    #geojson_base = build_geojson(geometry, county, fill_gaps=True)

    # Tabs
    t1, t2 = st.tabs(["Overview", "Proposed Routes"])
    with t1:
        tab_overview(county, geojson, stop_locs, totals, cfg)
    with t2:
        tab_routes(recommendations, stop_locs, geojson, totals, cfg)


if __name__ == "__main__":
    main()
