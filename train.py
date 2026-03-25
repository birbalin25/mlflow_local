from faker import Faker
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import inspect
import os
from mlflow.tracking import MlflowClient

experiment_name = "/Users/birbal.das@databricks.com/tmp_fake_exp_local"
client = MlflowClient()
if not client.get_experiment_by_name(experiment_name):
    client.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Generate synthetic dataset
fake = Faker()
num_rows = 500
num_features = 10

data = []
for _ in range(num_rows):
    features = [fake.pyfloat(left_digits=2, right_digits=2, positive=True) for _ in range(num_features)]
    label = fake.random_int(min=0, max=1)
    data.append(features + [label])

columns = [f"feature_{i+1}" for i in range(num_features)] + ["label"]
df = pd.DataFrame(data, columns=columns)

csv_path = "/tmp/fake_classification_data.csv"
df.to_csv(csv_path, index=False)

with mlflow.start_run() as run:
    mlflow.log_artifact(csv_path, artifact_path="dataset")
    
    X = df.iloc[:, :-1]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(clf, "model")
    
    module = inspect.getmodule(inspect.currentframe())
    if module is not None:
        source_code = inspect.getsource(module.__loader__.get_source(module.__name__)) if hasattr(module, "__loader__") else inspect.getsource(module)
    else:
        source_code = "# Unable to log Source code"
    code_path = "/tmp/source_code.py"
    with open(code_path, "w") as f:
        f.write(source_code)
    mlflow.log_artifact(code_path, artifact_path="source_code")