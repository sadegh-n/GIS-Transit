DESCRIPTION

Transit Connectivity Dashboard - CSE 6242 Group 123

This project identifies where new bus routes should go based on actual
commute data. It has two main components:

1. Regression Analysis: Uses the EPA Smart Location Database to model
   transit accessibility and develop an equity-weighted Opportunity Score.
   Five regression specifications (full features, pre-selected, reduced,
   LASSO, Elastic Net) were compared to understand which variables predict
   underserved areas. Low explanatory power (best R2 = 0.031) motivated
   the connectivity approach.

2. Connectivity Analysis and Route Recommendation: Combines three federal
   datasets (EPA SLD, Census LODES commute data, and GTFS transit feeds)
   to check whether each worker can reach their job by bus. For stranded
   workers, proposes new bus corridors using DBSCAN clustering and a
   directed corridor graph with bearing constraints. Routes follow real
   roads and are ranked by an equity-weighted score.

The tool covers 11 U.S. cities across 5 states and was validated with
sensitivity analysis on all major parameters.


INSTALLATION

Requires Python 3.9 or higher.

1. Install dependencies:
   pip install -r requirements.txt

2. Unzip the Dallas GTFS data (if submitted as zip):
   unzip dallas_gtfs.zip

3. The regression notebooks require the full EPA Smart Location Database:
   Download from https://www.epa.gov/smartgrowth/smart-location-mapping#SLD
   Save as EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv in the project folder.
   (The dashboard uses a pre-filtered version, sld_filtered.csv, included
   in the submission.)

4. All other data (LODES commute files, other GTFS feeds, census geometry)
   auto-downloads on first run. Internet connection required for first load.

5. API keys for OpenRouteService and Transitland are included in the code
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
  (best), with stranded worker statistics
- Tab 2 (Proposed Routes): shows recommended bus corridors on real roads,
  with toggleable routes, stop markers, and worker counts

To run the validation experiments:
   python experiments.py

This runs sensitivity analysis across all 11 cities (walk radius, transfer
limits, DBSCAN clustering, bearing constraints, etc.) and prints results
to stdout. Takes approximately 10-15 minutes.

To run the regression analysis:
   jupyter notebook regression_models.ipynb

This trains and evaluates all five regression models on the EPA SLD data.
Requires the full EPA SLD CSV file (see Installation step 3).


FILES

Dashboard and Route Recommendation:
  app.py                  Main Streamlit dashboard
  data_loader.py          Loads EPA SLD, LODES, census geometry, GTFS data
  connectivity.py         Checks if workers can reach jobs by transit
  route_finder.py         Proposes new bus corridors (DBSCAN + directed graph)
  experiments.py          Validation experiments across 11 cities
  requirements.txt        Python package dependencies
  sld_filtered.csv        Pre-filtered EPA Smart Location Database (11 counties)

Regression Analysis:
  regression_models.ipynb         Main regression models (5 specifications, MSE/R2)
  01_regression_exploration.ipynb Initial EDA and feature exploration
  CSE_6242.ipynb                  Data cleaning and preparation
  CSE_6242_v2.ipynb               Updated data processing pipeline

Standalone Analysis:
  clustering.py           Neighborhood typology clustering (K-means)
  network.py              Commute network analysis (Louvain, PageRank)


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
