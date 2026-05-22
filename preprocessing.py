"""
Data Preprocessing for AQI Prediction
- Handles missing values, outliers, encoding, scaling
- Applies SMOTE for class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)


FEATURE_COLS = ["pm2_5", "pm10", "no2", "so2", "co", "o3",
                "nh3", "temperature", "humidity", "wind_speed", "month"]
TARGET_COL   = "aqi"
CAT_TARGET   = "aqi_category"


def load_data(path: str = "data/aqi_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded data: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fix dtypes, clip extreme outliers."""
    df = df.drop_duplicates().copy()

    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip pollutants at 99th percentile to remove extreme outliers
    for col in ["pm2_5", "pm10", "no2", "so2", "co", "o3"]:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)

    logger.info(f"After cleaning: {df.shape}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with median; drop rows still missing target."""
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[TARGET_COL])
    logger.info(f"After missing-value handling: {df.shape}")
    return df


def encode_city(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the city column if present."""
    if "city" in df.columns:
        le = LabelEncoder()
        df["city_encoded"] = le.fit_transform(df["city"].astype(str))
    return df


def get_features_target(df: pd.DataFrame):
    """Return X (features) and y (AQI) arrays."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if "city_encoded" in df.columns:
        cols.append("city_encoded")
    X = df[cols].values
    y = df[TARGET_COL].values
    return X, y, cols


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Train/test split + StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def apply_smote_for_classification(X_train, y_train_cat, random_state=42):
    """
    Apply SMOTE to balance AQI category classes.
    Classes with fewer than 2 samples are dropped before resampling.
    """
    from collections import Counter
    counts = Counter(y_train_cat)
    # Keep only classes that have at least 2 samples
    valid_classes = {c for c, n in counts.items() if n >= 2}
    mask = np.array([c in valid_classes for c in y_train_cat])
    X_f = X_train[mask]
    y_f = np.array(y_train_cat)[mask]

    min_count = min(Counter(y_f).values())
    k = max(1, min(5, min_count - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_f, y_f)
    logger.info(f"After SMOTE: {X_res.shape[0]} samples "
                f"(was {X_train.shape[0]})")
    return X_res, y_res


def aqi_to_category(aqi_series: pd.Series) -> pd.Series:
    bins   = [0, 50, 100, 150, 200, 300, 500]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    return pd.cut(aqi_series, bins=bins, labels=labels)


def full_pipeline(path: str = "data/aqi_raw.csv"):
    """End-to-end preprocessing pipeline. Returns ready-to-train arrays."""
    df = load_data(path)
    df = clean_data(df)
    df = handle_missing(df)
    df = encode_city(df)

    if CAT_TARGET not in df.columns:
        df[CAT_TARGET] = aqi_to_category(df[TARGET_COL])

    X, y, feature_names = get_features_target(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # For SMOTE, use binned categories as class labels
    y_train_cat = aqi_to_category(pd.Series(y_train)).astype(str)
    X_train_sm, y_train_cat_sm = apply_smote_for_classification(X_train, y_train_cat)

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "X_train_smote": X_train_sm,
        "y_train_cat_smote": y_train_cat_sm,
        "scaler":  scaler,
        "feature_names": feature_names,
        "df": df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = full_pipeline()
    print("Feature names:", result["feature_names"])
    print("Train size:", result["X_train"].shape)
    print("Test size:",  result["X_test"].shape)
