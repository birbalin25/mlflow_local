import logging
import os
import subprocess

import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from faker import Faker
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config
print("hello10")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clone_or_sync_repo_to_databricks():
    """Clone the GitHub repo into Databricks Repos, or sync it if it already exists.

    Returns the head_commit_id from the Databricks API response.
    """
    host = config.DATABRICKS_HOST.rstrip("/")
    headers = {"Authorization": f"Bearer {config.DATABRICKS_TOKEN}"}

    if not host or not config.DATABRICKS_TOKEN:
        logger.warning("DATABRICKS_HOST or DATABRICKS_TOKEN not set; skipping Repos sync")
        return ""

    # Try to create the repo
    create_payload = {
        "url": config.GITHUB_REPO_URL,
        "provider": config.GIT_PROVIDER,
        "path": config.REPOS_PATH,
    }
    try:
        resp = requests.post(f"{host}/api/2.0/repos", json=create_payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            commit = data.get("head_commit_id", "")
            logger.info("Repo cloned to Databricks Repos. Commit: %s", commit)
            return commit

        # Repo already exists — find its ID and sync
        if resp.status_code in (400, 409):
            logger.info("Repo already exists in Databricks Repos; syncing to latest.")
            list_resp = requests.get(
                f"{host}/api/2.0/repos",
                headers=headers,
                params={"path_prefix": config.REPOS_PATH},
                timeout=30,
            )
            list_resp.raise_for_status()
            repos = list_resp.json().get("repos", [])
            repo_id = None
            for repo in repos:
                if repo.get("path") == config.REPOS_PATH:
                    repo_id = repo.get("id")
                    break

            if repo_id is None:
                logger.warning("Could not find repo ID for %s", config.REPOS_PATH)
                return ""

            patch_resp = requests.patch(
                f"{host}/api/2.0/repos/{repo_id}",
                json={"branch": "main"},
                headers=headers,
                timeout=30,
            )
            patch_resp.raise_for_status()
            data = patch_resp.json()
            commit = data.get("head_commit_id", "")
            logger.info("Repo synced to latest. Commit: %s", commit)
            return commit

        resp.raise_for_status()
    except Exception:
        logger.warning("Failed to clone/sync repo to Databricks Repos", exc_info=True)

    return ""


def get_git_commit_hash():
    """Return the local git HEAD commit hash."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        logger.warning("Could not get local git commit hash", exc_info=True)
        return ""


def log_dataset_version(df, csv_path):
    """Log the dataset as an MLflow input with automatic digest-based versioning."""
    try:
        dataset = mlflow.data.from_pandas(
            df, source=csv_path, name=config.DATASET_NAME, targets="label"
        )
        mlflow.log_input(dataset, context="training")
        logger.info("Dataset logged: %s (digest: %s)", config.DATASET_NAME, dataset.digest)
    except Exception:
        logger.warning("Failed to log dataset version", exc_info=True)


def main():
    # --- Databricks Repos clone/sync ---
    databricks_commit = clone_or_sync_repo_to_databricks()
    local_commit = get_git_commit_hash()
    commit_hash = databricks_commit or local_commit
    logger.info("Using commit hash: %s", commit_hash or "(unknown)")

    # --- Experiment setup ---
    client = MlflowClient()
    if not client.get_experiment_by_name(config.EXPERIMENT_NAME):
        client.create_experiment(config.EXPERIMENT_NAME)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # --- Generate synthetic dataset ---
    fake = Faker()
    data = []
    for _ in range(config.NUM_ROWS):
        features = [
            fake.pyfloat(left_digits=2, right_digits=2, positive=True)
            for _ in range(config.NUM_FEATURES)
        ]
        label = fake.random_int(min=0, max=1)
        data.append(features + [label])

    columns = [f"feature_{i+1}" for i in range(config.NUM_FEATURES)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(config.CSV_PATH, index=False)

    # --- Training run ---
    with mlflow.start_run() as run:
        # Source tags — link to exact code version on GitHub
        github_source_url = f"{config.GITHUB_REPO_URL.replace('.git', '')}/blob/{commit_hash}/train.py" if commit_hash else config.GITHUB_REPO_URL
        mlflow.set_tag("mlflow.source.name", github_source_url)
        mlflow.set_tag("mlflow.source.type", "OTHER")
        mlflow.set_tag("mlflow.source.git.repoURL", config.GITHUB_REPO_URL)
        mlflow.set_tag("mlflow.source.git.commit", commit_hash)
        mlflow.set_tag("mlflow.databricks.gitRepoUrl", config.GITHUB_REPO_URL)
        mlflow.set_tag("mlflow.databricks.gitRepoCommit", commit_hash)
        mlflow.set_tag("mlflow.databricks.gitRepoRelativePath", "train.py")

        # Log dataset artifact and versioned input
        mlflow.log_artifact(config.CSV_PATH, artifact_path="dataset")
        log_dataset_version(df, config.CSV_PATH)

        # Train/test split
        X = df.iloc[:, :-1]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        clf = RandomForestClassifier(**config.MODEL_PARAMS)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_params(config.MODEL_PARAMS)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, "model")

        logger.info("Run %s completed. Metrics: %s", run.info.run_id, metrics)


if __name__ == "__main__":
    main()
