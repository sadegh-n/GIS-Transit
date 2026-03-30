# Transit Connectivity Analysis — CSE 6242 Group 123

## Setup

```bash
pip install pandas numpy scikit-learn scipy networkx python-louvain pygris streamlit pydeck requests matplotlib
```

## Data

The data files are too large for GitHub. Download these and place in this folder (same directory as the notebook) before running. Keep the filenames as-is:

| File | Download | Size |
|------|----------|------|
| `EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv` | [EPA Smart Location Database](https://www.epa.gov/smartgrowth/smart-location-mapping#SLD) → download the CSV | ~192MB |
| `tx_od_main_JT00_2021.csv.gz` | [LEHD LODES](https://lehd.ces.census.gov/data/) → `lodes/LODES8/tx/od/tx_od_main_JT00_2021.csv.gz` | ~63MB |

| `V248-160-161-20210517/` (folder) | [DART GTFS archive](https://transitfeeds.com/p/dart/225) → download a 2021 feed, unzip into this folder | ~15MB |

## Running

**Notebook (see results):**
```bash
jupyter notebook results_demo.ipynb
```

**Dashboard:**
```bash
streamlit run app.py
```

## Files

| File | Description |
|------|-------------|
| `results_demo.ipynb` | Standalone demo of the connectivity analysis |
| `data_loader.py` | Loads SLD, LODES, GTFS data |
| `connectivity.py` | Checks if transit can connect each worker to their job |
| `route_finder.py` | Proposes new route corridors based on stranded workers |
| `clustering.py` | K-means neighborhood clustering (auto-selects k) |
| `network.py` | Commute network analysis (Louvain, PageRank) |
| `app.py` | Streamlit interactive dashboard |
| `01_regression_exploration.ipynb` | Regression analysis (exploration) |
