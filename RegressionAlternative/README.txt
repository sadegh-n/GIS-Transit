DESCRIPTION

Transit Connectivity Dashboard - CSE 6242 Group 123

This project analyzes transit accessibility gaps across 11 U.S. cities
and recommends where new bus routes should go. It has three components:

1. Regression Analysis: Uses the EPA Smart Location Database to model
   transit accessibility and develop an equity-weighted Opportunity Score.
   Five regression specifications were compared. Low explanatory power
   (best R2 = 0.031) showed that block-group aggregation misses individual
   commute gaps, motivating the connectivity approach.

2. Connectivity Analysis and Route Recommendation: Combines three federal
   datasets (EPA SLD, Census LODES commute data, and GTFS transit feeds)
   to check whether each worker can reach their job by bus. For stranded
   workers, proposes new bus corridors using DBSCAN clustering and a
   directed corridor graph. Routes follow real roads and are ranked by
   an equity-weighted score.

3. Interactive Visualization: A Streamlit dashboard with two tabs.
   The Overview tab shows a transit access choropleth with stranded
   worker statistics. The Proposed Routes tab displays recommended bus
   corridors on real roads with toggleable routes, proposed stop markers,
   worker cluster dots, job center markers, and a route comparison table.
   Users can click any element for details and switch between 11 cities.


INSTALLATION

Requires Python 3.9 or higher.

1. Install dependencies:
   pip install -r requirements.txt

2. The regression notebooks require the full EPA Smart Location Database:
   Download from https://www.epa.gov/smartgrowth/smart-location-mapping#SLD
   Save as EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv in the project folder.
   (The dashboard uses a pre-filtered version, sld_filtered.csv, included
   in the submission.)

3. The following data files are included in the submission:
   - sld_filtered.csv (EPA Smart Location Database, filtered to 11 counties)
   - dallas_gtfs.zip (Dallas DART GTFS feed, included because auto-download
     from Transitland sometimes returns the wrong transit agency for this city)
   - charlotte_gtfs/ (Charlotte CATS GTFS feed, included for the same reason)

4. Unzip the Dallas GTFS data before running:
   unzip dallas_gtfs.zip

5. All other data (LODES commute files, other GTFS feeds, census geometry)
   auto-downloads on first run. Internet connection required for first load.

6. API keys for OpenRouteService and Transitland are included in the code
   (free tier). If rate-limited, a new ORS key can be obtained for free at
   https://openrouteservice.org


EXECUTION

To run the interactive dashboard:
   streamlit run app.py

This opens a browser at http://localhost:8501

- Select a city from the sidebar dropdown
- First load for each city takes 1-2 minutes (downloads data and runs
  connectivity analysis). After that, results are cached and load instantly.
- Tab 1 (Overview): shows transit access map colored red (worst) to blue
  (best), with stranded worker statistics. Click any block group for details.
- Tab 2 (Proposed Routes): shows recommended bus corridors on real roads.
  Toggle routes on/off, click for stop-by-stop detail, compare routes in
  the summary table.

To run the validation experiments:
   python experiments.py

This runs sensitivity analysis across all 11 cities (walk radius, transfer
limits, DBSCAN clustering, bearing constraints, etc.) and prints results
to stdout. Takes approximately 10-15 minutes.

To run the regression analysis:
   jupyter notebook regression_models.ipynb

This trains and evaluates all five regression models on the EPA SLD data.
Requires thefull EPA SLD CSV file (see Installation step 2).


DEMO VIDEO

[INSERT YOUTUBE URL HERE]

A 1-minute unlisted YouTube video showing installation and execution
of the dashboard, including loading a city, exploring the overview map,
and viewing proposed bus routes.


FILES

Dashboard and Route Recommendation:
  app.py                  Main Streamlit dashboard and visualization
  data_loader.py          Loads EPA SLD, LODES, census geometry, GTFS data
  connectivity.py         Checks if workers can reach jobs by transit
  route_finder.py         Proposes new bus corridors (DBSCAN + directed graph)
  experiments.py          Validation experiments across 11 cities
  requirements.txt        Python package dependencies
  sld_filtered.csv        Pre-filtered EPA Smart Location Database (11 counties)
  dallas_gtfs.zip         Dallas DART GTFS feed (unzip before running)
  charlotte_gtfs/         Charlotte CATS GTFS feed

Regression Analysis:
  regression_models.ipynb         Main regression models (5 specifications, MSE/R2)
  01_regression_exploration.ipynb Initial EDA and feature exploration
  CSE_6242.ipynb                  Data cleaning and preparation
  CSE_6242_v2.ipynb               Updated data processing pipeline


DATA SOURCES

EPA Smart Location Database:
   https://www.epa.gov/smartgrowth/smart-location-mapping#SLD

Census LODES Origin-Destination (2021):
   https://lehd.ces.census.gov/data/

GTFS Transit Feeds:
   https://transit.land (Transitland API)

Census TIGER Block Group Boundaries:
   Downloaded via pygris Python library

OpenRouteService Directions API:
   https://openrouteservice.org
