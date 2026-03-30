"""
Data loading and preprocessing for transit connectivity analysis.
Handles EPA SLD, LODES commute data, census geometry, and GTFS transit feeds.
"""

import pandas as pd
import numpy as np
import os


def load_sld(path="EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"):
    """Load and clean the EPA Smart Location Database."""
    df = pd.read_csv(path)

    # The raw GEOID columns are stored as floats (truncated), so reconstruct from components
    df["GEOID"] = (df["STATEFP"].astype(str).str.zfill(2) +
                   df["COUNTYFP"].astype(str).str.zfill(3) +
                   df["TRACTCE"].astype(str).str.zfill(6) +
                   df["BLKGRPCE"].astype(str).str.zfill(1))

    # Drop block groups with near-zero population (airports, industrial, etc.)
    df = df[df["TotPop"] > 50].copy()

    # -99999 is the sentinel for "no transit service"
    df["D4A"] = df["D4A"].replace(-99999, np.nan)
    for col in ["D4C", "D4D", "D4E"]:
        df[col] = df[col].replace(-99999, 0)
    df = df[df["D5BR"] != -99999].copy()

    # Transit ratio: what fraction of jobs reachable by car can you also reach by transit?
    df["transit_ratio"] = np.where(df["D5AR"] > 0, df["D5BR"] / df["D5AR"], 0)

    df["employed_residents"] = df["R_LowWageWk"] + df["R_MedWageWk"] + df["R_HiWageWk"]
    df["zero_car_pop"] = df["TotPop"] * df["Pct_AO0"]
    df["tract"] = df["GEOID"].str[:11]

    return df


def load_lodes(path="tx_od_main_JT00_2021.csv.gz"):
    """Load LODES origin-destination commute data, aggregate to block group level."""
    if not os.path.exists(path):
        return None

    od = pd.read_csv(path, dtype={"w_geocode": str, "h_geocode": str})

    # LODES uses 15-digit block GEOIDs; truncate to 12-digit block groups
    od["home_bg"] = od["h_geocode"].str[:12]
    od["work_bg"] = od["w_geocode"].str[:12]

    agg = od.groupby(["home_bg", "work_bg"]).agg(
        total_workers=("S000", "sum"),
        low_wage=("SE01", "sum"),
    ).reset_index()

    # Drop people who live and work in the same block group
    agg = agg[agg["home_bg"] != agg["work_bg"]].copy()
    return agg


def load_geometry(state, county):
    """Download census block group boundaries from TIGER/Line via pygris."""
    import pygris
    bg = pygris.block_groups(state=state, county=county, year=2020)
    bg["GEOID"] = bg["GEOID"].astype(str).str.zfill(12)
    return bg


def load_gtfs(gtfs_dir):
    """
    Load GTFS feed from a local directory.
    Returns (stops_df, stop_route_mapping) or (None, None) if dir doesn't exist.
    """
    if not os.path.isdir(gtfs_dir):
        return None, None

    stops = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"))
    trips = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"))
    routes = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"))

    # Build the mapping: which routes serve which stops?
    trip_routes = trips[["trip_id", "route_id"]].drop_duplicates()
    sr = (stop_times[["trip_id", "stop_id"]]
          .merge(trip_routes, on="trip_id")[["stop_id", "route_id"]]
          .drop_duplicates())
    sr = sr.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id")
    sr = sr.merge(routes[["route_id", "route_short_name"]], on="route_id")

    stop_route_map = sr.groupby("stop_id")["route_short_name"].apply(set).to_dict()
    stop_locs = sr[["stop_id", "stop_lat", "stop_lon"]].drop_duplicates("stop_id")

    return stop_locs, stop_route_map


def fetch_stops_api(bbox, api_key):
    """
    Fetch transit stop locations from the Transitland API.
    Paginates through all results for the given bounding box.
    Returns a DataFrame with stop_lat, stop_lon, stop_name.
    """
    import requests
    import time

    url = "https://transit.land/api/v2/rest/stops"
    params = {"bbox": bbox, "limit": 100, "apikey": api_key}

    all_stops = []
    for _ in range(200):
        try:
            r = requests.get(url, params=params, timeout=20)
        except Exception:
            break
        if r.status_code != 200:
            break

        data = r.json()
        page_stops = data.get("stops", [])
        all_stops.extend(page_stops)

        nxt = data.get("meta", {}).get("next")
        if not nxt or not page_stops:
            break
        url, params = nxt, {}
        time.sleep(0.2)

    if not all_stops:
        return None

    rows = [{
        "stop_lat": s["geometry"]["coordinates"][1],
        "stop_lon": s["geometry"]["coordinates"][0],
        "stop_name": s.get("stop_name", ""),
    } for s in all_stops]

    return pd.DataFrame(rows)
