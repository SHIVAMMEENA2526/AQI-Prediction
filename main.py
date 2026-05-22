"""
AQI Prediction — Main Runner
Run this file to execute the full pipeline:
  1. Generate / load data
  2. Preprocess + SMOTE
  3. Train Random Forest & CatBoost
  4. Evaluate & display results
  5. Save trained models
"""

import sys, os, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "aqi_raw.csv")
OUT_DIR   = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── local imports ──────────────────────────────
sys.path.insert(0, ROOT)
from scraper.aqi_scraper   import load_sample_data
from utils.preprocessing   import full_pipeline, aqi_to_category
from models.train_models   import (
    train_random_forest, train_catboost,
    evaluate_regressor, feature_importance_df, save_model
)


def step1_generate_data():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        logger.info("Generating sample AQI dataset...")
        df = load_sample_data()
        df.to_csv(DATA_PATH, index=False)
        logger.info(f"Saved {len(df)} rows → {DATA_PATH}")
    else:
        logger.info(f"Data already exists at {DATA_PATH}")


def step2_preprocess():
    logger.info("Running preprocessing pipeline...")
    data = full_pipeline(DATA_PATH)
    logger.info(
        f"Train: {data['X_train'].shape} | Test: {data['X_test'].shape} | "
        f"SMOTE train: {data['X_train_smote'].shape}"
    )
    return data


def smote_y_to_numeric(y_cat):
    """Map category labels → approximate numeric AQI for regression."""
    midpoints = {
        "Good": 25, "Satisfactory": 75, "Moderate": 125,
        "Poor": 175, "Very Poor": 250, "Severe": 400
    }
    return np.array([midpoints.get(c, 100) for c in y_cat])


def step3_train(data):
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # ── Random Forest (baseline) ──
    rfr_base = train_random_forest(X_train, y_train)
    m_rfr_base = evaluate_regressor(rfr_base, X_test, y_test, "RFR (baseline)")

    # ── Random Forest + SMOTE ──
    X_sm = data["X_train_smote"]
    y_sm = smote_y_to_numeric(data["y_train_cat_smote"])
    rfr_smote = train_random_forest(X_sm, y_sm)
    m_rfr_smote = evaluate_regressor(rfr_smote, X_test, y_test, "RFR + SMOTE ★")

    # ── CatBoost ──
    cb = train_catboost(X_train, y_train, iterations=300)
    m_cb = evaluate_regressor(cb, X_test, y_test, "CatBoost")

    return {
        "rfr_base":   (rfr_base,  m_rfr_base),
        "rfr_smote":  (rfr_smote, m_rfr_smote),
        "catboost":   (cb,        m_cb),
        "X_test":  X_test,
        "y_test":  y_test,
        "feature_names": data["feature_names"],
    }


def step4_plots(results, data):
    fn   = results["feature_names"]
    rfr  = results["rfr_smote"][0]
    cb   = results["catboost"][0]
    X_te = results["X_test"]
    y_te = results["y_test"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AQI Prediction — Model Results", fontsize=16, fontweight="bold")

    # 1. Actual vs Predicted (RFR + SMOTE)
    y_pred_rfr = rfr.predict(X_te)
    ax = axes[0, 0]
    ax.scatter(y_te, y_pred_rfr, alpha=0.4, color="steelblue", s=20)
    lims = [min(y_te.min(), y_pred_rfr.min()), max(y_te.max(), y_pred_rfr.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual AQI"); ax.set_ylabel("Predicted AQI")
    ax.set_title("RFR+SMOTE — Actual vs Predicted")
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Residuals
    residuals = y_te - y_pred_rfr
    ax = axes[0, 1]
    ax.hist(residuals, bins=30, color="salmon", edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, linestyle="--")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("RFR+SMOTE — Residual Distribution")
    ax.grid(alpha=0.3)

    # 3. Feature Importance (RFR)
    fi = feature_importance_df(rfr, fn).head(8)
    ax = axes[1, 0]
    ax.barh(fi["feature"], fi["importance"], color="mediumseagreen")
    ax.set_xlabel("Importance"); ax.set_title("Top Features — Random Forest")
    ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")

    # 4. Model comparison bar chart
    metrics_list = [
        results["rfr_base"][1],
        results["rfr_smote"][1],
        results["catboost"][1],
    ]
    df_m = pd.DataFrame(metrics_list)
    ax = axes[1, 1]
    bars = ax.bar(df_m["model"], df_m["R2"] * 100,
                  color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("R² Accuracy (%)"); ax.set_title("Model Comparison")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, df_m["R2"] * 100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticklabels(df_m["model"], rotation=15, ha="right")

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "aqi_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {plot_path}")

    # AQI Distribution
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    df = data["df"]
    axes2[0].hist(df["aqi"], bins=40, color="cornflowerblue", edgecolor="white")
    axes2[0].set_xlabel("AQI"); axes2[0].set_ylabel("Count")
    axes2[0].set_title("AQI Distribution"); axes2[0].grid(alpha=0.3)

    cat_counts = df["aqi_category"].value_counts()
    axes2[1].bar(cat_counts.index.astype(str), cat_counts.values,
                 color=["green","yellowgreen","yellow","orange","red","purple"])
    axes2[1].set_title("AQI Category Distribution")
    axes2[1].set_xlabel("Category"); axes2[1].set_ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    dist_path = os.path.join(OUT_DIR, "aqi_distribution.png")
    fig2.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Distribution plot → {dist_path}")


def step5_save(results):
    models_dir = os.path.join(ROOT, "models", "saved")
    os.makedirs(models_dir, exist_ok=True)
    save_model(results["rfr_smote"][0], os.path.join(models_dir, "rfr_smote.pkl"))
    save_model(results["catboost"][0],  os.path.join(models_dir, "catboost.pkl"))


def print_summary(results):
    metrics = [
        results["rfr_base"][1],
        results["rfr_smote"][1],
        results["catboost"][1],
    ]
    df = pd.DataFrame(metrics)[["model", "MAE", "RMSE", "R2", "Accuracy_%"]]
    df["Accuracy_%"] = df["Accuracy_%"].round(2)
    df["R2"]         = df["R2"].round(4)
    df["MAE"]        = df["MAE"].round(2)
    df["RMSE"]       = df["RMSE"].round(2)
    print("\n" + "═"*65)
    print("  FINAL MODEL COMPARISON")
    print("═"*65)
    print(df.to_string(index=False))
    print("═"*65 + "\n")


# ── Entry Point ─────────────────────────────────
if __name__ == "__main__":
    step1_generate_data()
    data    = step2_preprocess()
    results = step3_train(data)
    results["df"] = data["df"]
    step4_plots(results, data)
    step5_save(results)
    print_summary(results)
    logger.info("✓ Pipeline complete. Check outputs/ for plots and models/saved/ for .pkl files.")
