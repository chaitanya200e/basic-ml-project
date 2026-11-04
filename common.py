"""Shared paths, constants and helpers used by the notebooks, main.py and app.py."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"              # original data, never modified
PROCESSED_DIR = DATA_DIR / "processed"  # output of each pipeline step
ARTIFACT_DIR = DATA_DIR / "artifacts"   # model, scaler, encoder, metrics
BASELINE_DIR = ARTIFACT_DIR / "baseline"

RAW_FILE = DATA_DIR / "Raw_laptop_data.csv"
INGESTED_FILE = PROCESSED_DIR / "01_ingested.csv"
PREPROCESSED_FILE = PROCESSED_DIR / "02_preprocessed.csv"
FEATURES_FILE = PROCESSED_DIR / "03_features.csv"

# final artifacts used by app.py (saved by model_tuning.ipynb)
MODEL_FILE = ARTIFACT_DIR / "model.joblib"
SCALER_FILE = ARTIFACT_DIR / "scaler.joblib"
ENCODER_FILE = ARTIFACT_DIR / "label_encoder.joblib"
METRICS_FILE = ARTIFACT_DIR / "metrics.json"

# baseline comparison (saved by model_training.ipynb)
BASELINE_PIPELINE_FILE = BASELINE_DIR / "baseline_pipeline.joblib"
BASELINE_RESULTS_FILE = BASELINE_DIR / "baseline_results.csv"

for _folder in (RAW_DIR, PROCESSED_DIR, BASELINE_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- columns
TARGET = "Price"
CATEGORICAL = ["Company", "TypeName", "OpSys", "CPU_name", "Gpu_brand"]
NUMERIC = ["Ram", "Weight", "TouchScreen", "IPS", "PPI", "HDD", "SSD"]
FEATURES = CATEGORICAL + NUMERIC

RANDOM_STATE = 42


# ---------------------------------------------------------------- data
def load_features() -> pd.DataFrame:
    """Final feature table written by data_feature_engineering.ipynb."""
    return pd.read_csv(FEATURES_FILE)


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Train/test split. The target is log(Price): prices are right-skewed."""
    X = df[FEATURES]
    y = np.log(df[TARGET])
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def get_cv() -> KFold:
    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------- modelling
def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, label-encode categorical ones.

    OrdinalEncoder is the feature-column version of LabelEncoder (LabelEncoder itself is only
    meant for the target). Unseen categories become -1 instead of crashing.
    Both are fitted inside the pipeline, so nothing from the test set leaks into them.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ]
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def evaluate(pipe, X, y_log) -> dict:
    """Metrics for a fitted pipeline. R2 is reported on the log scale and on real prices;
    MAE is in the same currency as the dataset's Price column."""
    pred_log = pipe.predict(X)
    y_true, y_pred = np.exp(y_log), np.exp(pred_log)
    return {
        "r2_log": r2_score(y_log, pred_log),
        "r2_price": r2_score(y_true, y_pred),
        "mae_price": mean_absolute_error(y_true, y_pred),
    }


# ---------------------------------------------------------------- artifacts
def save_artifacts(pipe) -> None:
    """Split a fitted pipeline into model / scaler / label encoder and save them separately."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prep = pipe.named_steps["prep"]
    joblib.dump(pipe.named_steps["model"], MODEL_FILE)
    joblib.dump(prep.named_transformers_["num"], SCALER_FILE)
    joblib.dump(prep.named_transformers_["cat"], ENCODER_FILE)


def load_artifacts() -> dict:
    return {
        "model": joblib.load(MODEL_FILE),
        "scaler": joblib.load(SCALER_FILE),
        "encoder": joblib.load(ENCODER_FILE),
    }


def transform(X: pd.DataFrame, scaler, encoder) -> np.ndarray:
    """Same column order the model was trained on: scaled numeric first, then encoded categorical."""
    return np.hstack([scaler.transform(X[NUMERIC]), encoder.transform(X[CATEGORICAL])])


# ---------------------------------------------------------------- inference
def compute_ppi(x_res: int, y_res: int, inches: float) -> float:
    return float(np.sqrt(x_res**2 + y_res**2) / inches)


def predict_price(artifacts: dict, **inputs) -> float:
    """Predict a price from one laptop described by the FEATURES columns."""
    row = pd.DataFrame([inputs])[FEATURES]
    X = transform(row, artifacts["scaler"], artifacts["encoder"])
    return float(np.exp(artifacts["model"].predict(X))[0])
