"""
AQI Web Scraper using BeautifulSoup
Scrapes air quality data from aqicn.org for Indian cities.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CITIES = [
    "delhi", "mumbai", "kolkata", "chennai", "bengaluru",
    "hyderabad", "ahmedabad", "pune", "jaipur", "lucknow",
    "kanpur", "nagpur", "patna", "indore", "bhopal"
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def scrape_city_aqi(city: str) -> dict | None:
    """Scrape AQI data for a single city from aqicn.org."""
    url = f"https://aqicn.org/city/india/{city}/"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        aqi_div = soup.find("div", {"id": "aqiwgt"})
        aqi_value = None
        if aqi_div:
            aqi_span = aqi_div.find("span", {"class": "aqivalue"})
            if aqi_span:
                aqi_value = aqi_span.text.strip()

        pollutants = {}
        rows = soup.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2:
                pollutant = cols[0].text.strip().lower().replace(".", "_")
                value = cols[1].text.strip()
                if pollutant in ["pm2_5", "pm10", "o3", "no2", "so2", "co"]:
                    try:
                        pollutants[pollutant] = float(value)
                    except ValueError:
                        pollutants[pollutant] = None

        return {
            "city": city,
            "aqi": aqi_value,
            "timestamp": datetime.now().isoformat(),
            **pollutants
        }
    except Exception as e:
        logger.warning(f"Failed to scrape {city}: {e}")
        return None


def scrape_all_cities(cities: list = CITIES, delay: float = 1.5) -> pd.DataFrame:
    """Scrape AQI data for all cities and return as DataFrame."""
    results = []
    for city in cities:
        logger.info(f"Scraping {city}...")
        data = scrape_city_aqi(city)
        if data:
            results.append(data)
        time.sleep(delay)

    df = pd.DataFrame(results)
    logger.info(f"Scraped {len(df)} cities successfully.")
    return df


def load_sample_data() -> pd.DataFrame:
    """
    Return a realistic synthetic dataset mimicking scraped AQI data.
    Use this when live scraping is not needed / for offline development.
    """
    import numpy as np
    np.random.seed(42)
    n = 500

    cities = CITIES * (n // len(CITIES) + 1)
    df = pd.DataFrame({
        "city":    cities[:n],
        "pm2_5":   np.random.uniform(10, 300, n),
        "pm10":    np.random.uniform(20, 400, n),
        "no2":     np.random.uniform(5, 120, n),
        "so2":     np.random.uniform(2, 80, n),
        "co":      np.random.uniform(0.1, 10, n),
        "o3":      np.random.uniform(10, 180, n),
        "nh3":     np.random.uniform(1, 50, n),
        "temperature": np.random.uniform(10, 45, n),
        "humidity":    np.random.uniform(20, 95, n),
        "wind_speed":  np.random.uniform(0, 30, n),
        "month":       np.random.randint(1, 13, n),
    })

    # Derive AQI from pollutants (simplified linear combo + noise)
    df["aqi"] = (
        0.5  * df["pm2_5"]  +
        0.3  * df["pm10"]   +
        0.15 * df["no2"]    +
        0.05 * df["so2"]    +
        np.random.normal(0, 15, n)
    ).clip(0, 500).round(1)

    # Add AQI category
    bins   = [0, 50, 100, 150, 200, 300, 500]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    df["aqi_category"] = pd.cut(df["aqi"], bins=bins, labels=labels)

    return df


if __name__ == "__main__":
    df = load_sample_data()
    df.to_csv("data/aqi_raw.csv", index=False)
    logger.info(f"Sample data saved: {df.shape}")
    print(df.head())
