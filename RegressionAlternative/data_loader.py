"""
Data loading for the transit connectivity app.
Handles the EPA Smart Location Database, LODES commute data,
census block group geometry, and GTFS transit feeds.
"""

import pandas as pd
import numpy as np
import os

TRANSITLAND_API_KEY = "Hs25CefN5AwZRUzJg4C84v1HnY9YnckH"


def load_sld(path="EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"):
    """Load the EPA Smart Location Database and compute some columns we need.
    Filters out block groups with tiny populations and bad data flags."""
    df = pd.read_csv(path)

    # build the 12-digit GEOID from the component FIPS columns
    df["GEOID"] = (df["STATEFP"].astype(str).str.zfill(2) +
                   df["COUNTYFP"].astype(str).str.zfill(3) +
                   df["TRACTCE"].astype(str).str.zfill(6) +
                   df["BLKGRPCE"].astype(str).str.zfill(1))

    # drop very small block groups since their stats are unreliable
    df = df[df["TotPop"] > 50].copy()

    # -99999 is the SLD's "no data" sentinel value
    df["D4A"] = df["D4A"].replace(-99999, np.nan)
    for col in ["D4C", "D4D", "D4E"]:
        df[col] = df[col].replace(-99999, 0)
    df = df[df["D5BR"] != -99999].copy()

    # transit_ratio = jobs reachable by transit / jobs reachable by car (both in 45 min)
    # this is the core metric for identifying transit deserts
    df["transit_ratio"] = np.where(df["D5AR"] > 0, df["D5BR"] / df["D5AR"], 0)
    df["employed_residents"] = df["R_LowWageWk"] + df["R_MedWageWk"] + df["R_HiWageWk"]
    df["zero_car_pop"] = df["TotPop"] * df["Pct_AO0"]
    df["tract"] = df["GEOID"].str[:11]

    return df


def load_lodes(path="tx_od_main_JT00_2021.csv.gz", state_abbr=None):
    """Load LODES origin-destination commute data and aggregate to block group level.
    Each row in the output is a home_bg -> work_bg pair with worker counts."""
    if not os.path.exists(path) and state_abbr:
        path = auto_download_lodes(state_abbr)
        if path is None:
            return None

    if not os.path.exists(path):
        return None

    od = pd.read_csv(path, dtype={"w_geocode": str, "h_geocode": str})
    # truncate the 15-digit block geocode to 12 digits for block group
    od["home_bg"] = od["h_geocode"].str[:12]
    od["work_bg"] = od["w_geocode"].str[:12]

    agg = od.groupby(["home_bg", "work_bg"]).agg(
        total_workers=("S000", "sum"),
        low_wage=("SE01", "sum"),  # SE01 = workers earning $1,250/month or less
    ).reset_index()

    # drop people who live and work in the same block group
    agg = agg[agg["home_bg"] != agg["work_bg"]].copy()
    return agg


def auto_download_lodes(state_abbr):
    """Download the LODES OD file from Census LEHD if we don't have it locally."""
    import requests

    filename = f"{state_abbr}_od_main_JT00_2021.csv.gz"
    if os.path.exists(filename):
        return filename

    url = f"https://lehd.ces.census.gov/data/lodes/LODES8/{state_abbr}/od/{filename}"
    try:
        r = requests.get(url, timeout=120, stream=True)
        if r.status_code != 200:
            return None
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename
    except Exception:
        return None


def load_geometry(state, county):
    """Download census block group boundaries using pygris."""
    import pygris
    bg = pygris.block_groups(state=state, county=county, year=2020)
    bg["GEOID"] = bg["GEOID"].astype(str).str.zfill(12)
    return bg


def load_gtfs(gtfs_dir):
    """Load a local GTFS feed from a directory. Returns (stop_locations, stop_route_map)
    where stop_route_map is a dict mapping stop_id to the set of route names serving it."""
    if not os.path.isdir(gtfs_dir):
        return None, None

    stops = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"), low_memory=False)
    trips = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"), low_memory=False)
    stop_times = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"), low_memory=False)
    routes = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"), low_memory=False)

    # make sure ID columns are strings so merges work across different GTFS feeds
    for df in [stops, trips, stop_times, routes]:
        for col in ["stop_id", "trip_id", "route_id"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # pick the best available route name (short name preferred, then long, then ID)
    if "route_short_name" in routes.columns and routes["route_short_name"].notna().any():
        fallback = routes["route_long_name"] if "route_long_name" in routes.columns else routes["route_id"].astype(str)
        routes["_route_name"] = routes["route_short_name"].fillna(fallback)
    elif "route_long_name" in routes.columns:
        routes["_route_name"] = routes["route_long_name"]
    else:
        routes["_route_name"] = routes["route_id"].astype(str)

    # join stop_times -> trips -> routes to figure out which routes serve which stops
    trip_routes = trips[["trip_id", "route_id"]].drop_duplicates()
    sr = (stop_times[["trip_id", "stop_id"]]
          .merge(trip_routes, on="trip_id")[["stop_id", "route_id"]]
          .drop_duplicates())
    sr = sr.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id")
    sr = sr.merge(routes[["route_id", "_route_name"]], on="route_id")

    stop_route_map = sr.groupby("stop_id")["_route_name"].apply(set).to_dict()
    stop_locs = sr[["stop_id", "stop_lat", "stop_lon"]].drop_duplicates("stop_id")

    return stop_locs, stop_route_map


def auto_download_gtfs(bbox, gtfs_dir, api_key=TRANSITLAND_API_KEY):
    """Try to download a GTFS feed for the area using the Transitland API.
    Skips intercity services like Amtrak and Greyhound since we only care about local transit."""
    import requests
    import zipfile
    import io

    if os.path.isdir(gtfs_dir) and os.path.exists(os.path.join(gtfs_dir, "stops.txt")):
        return gtfs_dir

    url = "https://transit.land/api/v2/rest/feeds"
    params = {"bbox": bbox, "limit": 20, "apikey": api_key}
    try:
        r = requests.get(url, params=params, timeout=15)
    except Exception:
        return None
    if r.status_code != 200:
        return None

    feeds = r.json().get("feeds", [])

    # filter out intercity services since they won't help local commuters
    skip = ["amtrak", "greyhound", "flixbus", "megabus", "vonlane", "redcoach",
            "tornado", "omnibus", "intercity"]
    download_url = None
    for feed in feeds:
        oid = feed.get("onestop_id", "").lower()
        if any(s in oid for s in skip):
            continue
        static = feed.get("urls", {}).get("static_current", "")
        if static:
            download_url = static
            break

    if not download_url:
        return None

    try:
        r = requests.get(download_url, timeout=60)
        if r.status_code != 200 or len(r.content) < 1000:
            return None
        os.makedirs(gtfs_dir, exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(gtfs_dir)
        return gtfs_dir
    except Exception:
        return None


def fetch_route_stops_api(bbox, api_key=TRANSITLAND_API_KEY):
    """Fallback when we can't get a GTFS file: fetch route and stop data
    directly from the Transitland REST API, one route at a time."""
    import requests
    import time
    from collections import defaultdict

    url = "https://transit.land/api/v2/rest/routes"
    params = {"bbox": bbox, "limit": 100, "apikey": api_key}

    # paginate through all routes in the bounding box
    all_routes = []
    for _ in range(20):
        try:
            r = requests.get(url, params=params, timeout=20)
        except Exception:
            break
        if r.status_code != 200:
            break
        data = r.json()
        routes = data.get("routes", [])
        all_routes.extend(routes)
        nxt = data.get("meta", {}).get("next")
        if not nxt or not routes:
            break
        url, params = nxt, {}
        time.sleep(0.2)  # be nice to the API

    # only keep local transit types (bus, rail, etc), not intercity
    skip_agencies = {"greyhound", "amtrak", "flixbus", "megabus", "vonlane",
                     "redcoach", "tornado", "omnibus"}
    local = [rt for rt in all_routes
             if rt.get("route_type") in [0, 1, 2, 3]
             and not any(s in rt.get("agency", {}).get("agency_name", "").lower()
                         for s in skip_agencies)]

    if not local:
        return None, None

    stop_route_map = defaultdict(set)
    stop_locs_dict = {}

    # fetch stops for each route individually
    for rt in local:
        rid = rt["id"]
        name = rt.get("route_short_name", str(rid))
        try:
            r2 = requests.get(
                f"https://transit.land/api/v2/rest/routes/{rid}",
                params={"apikey": api_key}, timeout=20
            )
            if r2.status_code == 200:
                rd = r2.json().get("routes", [{}])[0]
                for s in rd.get("route_stops", []):
                    stop = s.get("stop", {})
                    sid = stop.get("stop_id", str(stop.get("id", "")))
                    coords = stop.get("geometry", {}).get("coordinates", [None, None])
                    if coords[0] is not None:
                        stop_route_map[sid].add(name)
                        stop_locs_dict[sid] = {
                            "stop_id": sid,
                            "stop_lat": coords[1],
                            "stop_lon": coords[0],
                        }
        except Exception:
            pass
        time.sleep(0.3)

    if not stop_locs_dict:
        return None, None

    stop_locs = pd.DataFrame(stop_locs_dict.values())
    return stop_locs, dict(stop_route_map)
