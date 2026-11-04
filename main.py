"""Run the whole pipeline: ingestion -> preprocessing -> features -> training -> tuning.

    python main.py

Each notebook is executed in order (outputs are saved back into the notebook).
The run stops at the first failure, because later notebooks depend on earlier outputs.
"""
import subprocess
import sys
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from common import (
    BASELINE_PIPELINE_FILE,
    BASELINE_RESULTS_FILE,
    get_cv,
    load_features,
    make_pipeline,
    split_data,
)

ROOT = Path(__file__).resolve().parent

NOTEBOOKS = [
    "data_ingestion .ipynb",
    "data_preprocessing.ipynb",
    "data_feature_engineering.ipynb",
    "model_tuning.ipynb",
]


def create_baseline() -> None:
    df = load_features()
    X_train, _, y_train, _ = split_data(df)
    baseline = make_pipeline(RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ))
    cv_score = cross_val_score(baseline, X_train, y_train, cv=get_cv(), scoring="r2", n_jobs=-1).mean()
    baseline.fit(X_train, y_train)
    BASELINE_PIPELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, BASELINE_PIPELINE_FILE)
    BASELINE_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_RESULTS_FILE.write_text(
        "model,cv_r2_log\nRandom Forest,"
        f"{cv_score}\n",
        encoding="utf-8",
    )


def run_notebook(nb: str) -> None:
    start = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--execute", "--to", "notebook", "--inplace", nb,
            "--ExecutePreprocessor.timeout=-1",
        ],
        cwd=ROOT,                      # notebooks import common.py and need this as cwd
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[FAILED] {nb}\n{result.stderr}")
        sys.exit(1)
    print(f"[ OK ] {nb}  ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    for notebook in NOTEBOOKS:
        if notebook == "model_tuning.ipynb":
            create_baseline()
        run_notebook(notebook)
    print('\nPipeline finished. Start the app with:  streamlit run "app (1).py"')
