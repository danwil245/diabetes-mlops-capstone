# Diabetes Prediction — MLOps Capstone

An end-to-end MLOps pipeline that predicts diabetes onset from the **Pima Indians
Diabetes** dataset (a binary classification problem). It ties together the whole
lifecycle: data versioning → multi-model training → experiment tracking & model
registry → a prediction API → containerisation → CI.

```
Dataset (DVC) ──► Train 3 models ──► MLflow (track + registry) ──► best_model.joblib
                                                                        │
                                                        FastAPI /predit  ▼
                                                     Docker image ──► CI (build + test)
```

## What it does

- **Data:** the real 768-row Pima Indians Diabetes CSV, versioned with **DVC**.
- **Training:** trains **three** classifiers, evaluates each on five metrics,
  logs everything to **MLflow**, and registers the best one in the MLflow
  **Model Registry**.
- **Serving:** a **FastAPI** service exposes `/predict` and `/health`.
- **Packaging:** a **Dockerfile** builds a self-contained image (code + model).
- **CI:** **GitHub Actions** installs, trains, runs tests, and builds the image
  on every push (build + test only — no deployment).

## The three models & their results

All three are wrapped in an sklearn `Pipeline` that first turns impossible zeros
(e.g. a Glucose or BMI of 0 — physiologically impossible, so really "missing")
into `NaN`, imputes them with the column median, and (for Logistic Regression)
standard-scales the features. Metrics are on a stratified 20% hold-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| **GradientBoosting** ⭐ | 0.766 | 0.688 | 0.611 | 0.647 | **0.831** |
| LogisticRegression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |
| RandomForest | 0.734 | 0.651 | 0.519 | 0.577 | 0.808 |

**Best model = Gradient Boosting** (selected by highest ROC-AUC). It's registered
in MLflow as `diabetes-classifier` with the alias **`production`**, and saved to
`models/best_model.joblib` for the API to serve.

### Why these choices (plain-English notes)

- **Why ROC-AUC to pick the winner?** In a medical screen the two classes are
  imbalanced (~35% diabetic) and the decision threshold matters. ROC-AUC measures
  how well the model *ranks* diabetics above non-diabetics across all thresholds,
  so it's a more honest single-number comparison than raw accuracy.
- **The five metrics:** *accuracy* = overall % correct; *precision* = of those
  flagged diabetic, how many really are; *recall* = of the true diabetics, how
  many we caught; *F1* = balance of precision & recall; *ROC-AUC* = ranking
  quality (1.0 perfect, 0.5 = coin flip).
- **Why Gradient Boosting over XGBoost?** `GradientBoostingClassifier` is pure
  scikit-learn — no extra native library — so it Dockerises cleanly. XGBoost
  would work too but adds a dependency for no real gain on a dataset this small.
- **How the API loads the model:** it loads the plain `models/best_model.joblib`
  file, *not* the MLflow registry. Registry loading needs the `mlflow.db` +
  artifact store to travel with the app; a single joblib file makes the Docker
  image fully self-contained. The registry still records provenance (which run,
  which metrics) for tracking — the two roles are complementary.

## Project layout

```
diabetes-mlops-capstone/
├── data/diabetes.csv          # tracked by DVC (see data/diabetes.csv.dvc)
├── models/                    # best_model.joblib + metrics + metadata (built)
├── src/
│   ├── utils.py               # schema + shared preprocessing (zero→NaN→impute)
│   ├── train.py               # trains 3 models, MLflow logging + registry
│   ├── predict.py             # load model + predict() helper
│   └── app.py                 # FastAPI: /predict, /health
├── tests/                     # pytest sanity checks
├── .github/workflows/ci.yml   # GitHub Actions: install → train → test → docker
├── Dockerfile
├── dvc.yaml / dvc.lock        # DVC pipeline (train stage)
├── requirements.txt
└── README.md
```

## Setup

Python 3.13. On this machine use the `py -3.13` launcher (plain `python` may
resolve to a different install).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # activate (so `python` = the venv)
pip install -r requirements.txt
```

## Run training

```powershell
python src/train.py
```

This trains all three models, prints the comparison table, logs runs to MLflow
(local `mlflow.db`), registers the best model, and writes
`models/best_model.joblib` + `models/model_metadata.json`.

### Compare the runs in the MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open <http://127.0.0.1:5000> → experiment **diabetes-classification** to compare
the three runs side by side, and see the registered model under **Models**.

## Run the API

```powershell
uvicorn src.app:app --reload
```

- Interactive docs: <http://127.0.0.1:8000/docs>
- Health: `GET http://127.0.0.1:8000/health`
- Predict:

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
# -> {"prediction":1,"label":"diabetic","probability":0.7598}
```

## Run in Docker

```powershell
docker build -t diabetes-mlops-capstone:local .
docker run -d -p 8000:8000 --name diabetes-api diabetes-mlops-capstone:local
# test it, then:
docker rm -f diabetes-api
```

The image bundles the code **and** the trained model, so the container is fully
self-contained — no training or external files needed at runtime.

## DVC (data versioning)

DVC was initialised in `--subdir` mode (this folder lives inside the parent
`devops` git repo). The dataset is tracked by its content hash:

```powershell
dvc dag        # shows: data/diabetes.csv.dvc ──► train
dvc status     # "Data and pipelines are up to date."
dvc repro      # re-runs the train stage only if inputs changed
```

Git tracks the **pointers** (`data/diabetes.csv.dvc`, `dvc.lock`) while
`data/.gitignore` and `models/.gitignore` keep the large CSV and model binaries
out of git. To share the actual data later, add a DVC remote and `dvc push`.

## Tests

```powershell
pytest tests/ -v
```

Checks that the model loads, `predict()` returns a valid shape, `/health` is ok,
`/predict` returns a valid prediction, and bad input is rejected with a 422. If
no model exists yet, the test fixture trains one first.

## CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every push/PR: checkout → set up Python 3.13
→ install deps → **train** (produces the model) → **pytest** → **docker build**
→ smoke-test `/health` in the container. It's build-and-test only — there is no
deployment step.
