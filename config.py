import os

# Databricks connection
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

# GitHub repo
GITHUB_REPO_URL = "https://github.com/birbalin25/mlflow_local.git"
GIT_PROVIDER = "gitHub"

# Databricks Repos paths
REPOS_PATH = "/Repos/birbal.das@databricks.com/mlflow_local"
REPOS_TRAIN_PATH = "/Repos/birbal.das@databricks.com/mlflow_local/train.py"

# MLflow experiment
EXPERIMENT_NAME = "/Users/birbal.das@databricks.com/tmp_fake_exp_local"

# Dataset
CSV_PATH = "/tmp/fake_classification_data.csv"
DATASET_NAME = "fake_classification_data"
NUM_ROWS = 500
NUM_FEATURES = 10

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}
