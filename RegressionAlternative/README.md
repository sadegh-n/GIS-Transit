# Transit Connectivity Dashboard — CSE 6242 Group 123

Interactive tool that identifies where new bus routes should go, based on actual commute data. Tested across 11 U.S. cities.

## Setup

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scikit-learn scipy networkx pygris streamlit pydeck requests
```

## Data

Most data auto-downloads on first run. The following are included in the repo since they can't be auto-downloaded reliably:

| File | What it is |
|------|-----------|
| `sld_filtered.csv` | EPA Smart Location Database, pre-filtered to our 11 counties (12 MB) |
| `V248-160-161-20210517/` | Dallas DART GTFS feed |
| `charlotte_gtfs/` | Charlotte CATS GTFS feed |

Everything else auto-downloads when you first load a city:
- **LODES commute data** downloads from Census LEHD (~14-66 MB per state)
- **GTFS transit feeds** download from Transitland API
- **Census block group geometry** downloads from Census TIGER via pygris

No need to manually download anything.

## Running the Dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Select a city from the sidebar dropdown.

- First load for a new city takes 1-2 minutes (downloads data + runs connectivity analysis). After that it's cached and loads instantly.
- Texas cities (Dallas, Fort Worth, Houston, San Antonio, Austin) share the same LODES file so they're fastest after Dallas loads.
- Other cities (Atlanta, Jacksonville, Charlotte, Detroit, Phoenix, Nashville) download their state's LODES on first use.

## How to Navigate the Dashboard

### Tab 1: Overview
- Map shows block groups colored **red** (worst transit access) to **blue** (best transit access). Gray = no data.
- Top bar shows what % of commuters can vs. can't reach their job by bus.
- **Click any block group** to see its transit ratio, population, zero-car %, and jobs reachable in the sidebar.
- Legend at the bottom explains the color scale.

### Tab 2: Proposed Routes
- Each proposed bus corridor is a **colored line following real roads**.
- **Black dots** = proposed bus stops. **Colored dots** = worker clusters near the route (sized by worker count). **Green circles** = job centers.
- **Toggle routes on/off** with the checkboxes at the top.
- **Click a route line or worker dot** to see stop-by-stop detail in the sidebar (stop locations, workers served, distance from job center).
- **Comparison table** at the bottom ranks all routes by workers served, low-wage count, equity score, and distance.
- "What do these values mean?" expander explains each metric.

## How the Pipeline Works

1. **Data Loading**: Combines EPA Smart Location Database (transit metrics per block group), Census LODES (where every worker commutes from/to), and GTFS (where bus stops and routes are).

2. **Connectivity Check**: For each worker's home-to-work commute, checks if they can get there by bus — walk 0.5mi to a stop, ride with up to 2 transfers, walk 0.5mi to work. If no path exists, they're "stranded."

3. **Route Recommendation**: 
   - Finds the job centers with the most stranded workers
   - Clusters nearby residential tracts into pickup zones using DBSCAN
   - Builds a directed graph where edges only go inward toward the job center (within a 45° bearing constraint so routes stay linear)
   - Searches for the path through the graph that picks up the most workers
   - Ranks routes by workers served × equity score (prioritizes low-wage, car-limited areas)
   - Snaps routes to real roads using OpenRouteService API
   - Places bus stops where workers cluster along the route

4. **Visualization**: Two-tab Streamlit dashboard — Overview (transit desert choropleth, stranded worker stats) and Proposed Routes (toggleable corridors on real roads with stops, worker dots, job centers, comparison table).

## Tips for Presenting

1. Start on the **Overview tab** — point out the red transit deserts and the stranded worker count.
2. Switch to **Proposed Routes** — show 2-3 routes, zoom in to show they follow real roads.
3. Hover over worker dots to show counts. Click a route to show the sidebar detail.
4. Switch cities to show the tool generalizes (try Fort Worth for high stranded %, or Phoenix for a large city).
5. Key stat: our algorithm serves **167% more workers per route** than random baseline corridors.

## Running Experiments

```bash
python experiments.py
```

Runs 11 validation experiments (walk radius, transfer limits, DBSCAN eps, bearing constraint, path cutoff, min feeder workers, max feeder distance, job center coverage, route directness, random baseline comparison, equity analysis) across all cities. Takes ~10-15 minutes. Results print to stdout and save to `experiment_results.txt`.

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit interactive dashboard |
| `data_loader.py` | Loads EPA SLD, LODES, census geometry, GTFS feeds |
| `connectivity.py` | Checks if transit can connect each worker to their job |
| `route_finder.py` | Proposes new route corridors (DBSCAN + directed graph) |
| `experiments.py` | Validation experiments across multiple cities |
| `clustering.py` | K-means neighborhood clustering (standalone analysis) |
| `network.py` | Commute network analysis with community detection (standalone) |
| `results_demo.ipynb` | Notebook demo of the connectivity analysis |

## Cities

| City | State | Transit System |
|------|-------|---------------|
| Dallas | TX | DART (9,992 stops) |
| Fort Worth | TX | Trinity Metro (1,584 stops) |
| Houston | TX | METRO (8,806 stops) |
| San Antonio | TX | VIA (5,991 stops) |
| Austin | TX | Capital Metro (2,350 stops) |
| Atlanta | GA | MARTA (8,724 stops) |
| Jacksonville | FL | JTA (2,500 stops) |
| Charlotte | NC | CATS (2,946 stops) |
| Detroit | MI | DDOT/SMART (5,115 stops) |
| Phoenix | AZ | Valley Metro (8,030 stops) |
| Nashville | TN | WeGo (1,630 stops) |
