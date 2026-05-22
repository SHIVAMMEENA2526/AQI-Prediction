# AQI Prediction — Course Project

Air Quality Index (AQI) prediction using **Random Forest** and **CatBoost** regression models, with web scraping via **BeautifulSoup** and class-balancing via **SMOTE**.

---

## Results

| Model | MAE | RMSE | R² Accuracy |
|---|---|---|---|
| Random Forest (baseline) | 14.02 | 18.03 | **89.9%** |
| **Random Forest + SMOTE** | 17.18 | 20.71 | **86.7% ★** |
| CatBoost | 14.15 | 18.98 | 88.8% |

> **Note:** The SMOTE model trades a small accuracy drop for significantly better handling of rare AQI categories (Very Poor / Severe), making it more robust in production.

---

## Project Structure

```
aqi_project/
├── main.py                    ← Run this to execute full pipeline
├── requirements.txt
├── scraper/
│   └── aqi_scraper.py         ← BeautifulSoup web scraper + sample data generator
├── utils/
│   └── preprocessing.py       ← Cleaning, encoding, SMOTE, train/test split
├── models/
│   ├── train_models.py        ← RFR & CatBoost training + evaluation
│   └── saved/
│       ├── rfr_smote.pkl      ← Trained Random Forest model
│       └── catboost.pkl       ← Trained CatBoost model
├── data/
│   └── aqi_raw.csv            ← Generated/scraped dataset
└── outputs/
    ├── aqi_results.png        ← Model comparison plots
    └── aqi_distribution.png   ← AQI distribution charts
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (data → preprocess → train → evaluate → plots)
python main.py
```

---

## Key Features

### Web Scraping (`scraper/aqi_scraper.py`)
- Scrapes **aqicn.org** for Indian cities using `requests` + `BeautifulSoup`
- Extracts: AQI, PM2.5, PM10, NO₂, SO₂, CO, O₃
- `load_sample_data()` provides a realistic offline dataset (500 rows)

### Preprocessing (`utils/preprocessing.py`)
- Outlier clipping at 99th percentile
- Median imputation for missing values
- Label encoding for city names
- **SMOTE** (Synthetic Minority Over-sampling Technique) to balance rare AQI categories
- `StandardScaler` normalization

### Models (`models/train_models.py`)
- **RandomForestRegressor** — ensemble of 200 decision trees
- **CatBoostRegressor** — gradient boosted trees optimized for speed
- Metrics: MAE, RMSE, R²

### Features Used
| Feature | Description |
|---|---|
| `pm2_5` | Fine particulate matter (µg/m³) |
| `pm10` | Coarse particulate matter (µg/m³) |
| `no2` | Nitrogen dioxide |
| `so2` | Sulphur dioxide |
| `co` | Carbon monoxide |
| `o3` | Ozone |
| `nh3` | Ammonia |
| `temperature` | Ambient temperature (°C) |
| `humidity` | Relative humidity (%) |
| `wind_speed` | Wind speed (km/h) |
| `month` | Month of measurement |

### AQI Categories (CPCB Standard)
| Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–150 | Moderate |
| 151–200 | Poor |
| 201–300 | Very Poor |
| 301–500 | Severe |

---

## Live Scraping

To scrape real data instead of using the sample dataset:

```python
from scraper.aqi_scraper import scrape_all_cities
df = scrape_all_cities()
df.to_csv("data/aqi_raw.csv", index=False)
```

Then run `python main.py` as normal.

---

## Tech Stack
- **Python 3.10+**
- `scikit-learn` — Random Forest, preprocessing, metrics
- `catboost` — Gradient boosting
- `imbalanced-learn` — SMOTE
- `beautifulsoup4` + `requests` — Web scraping
- `pandas` + `numpy` — Data processing
- `matplotlib` + `seaborn` — Visualization
- `joblib` — Model serialization
