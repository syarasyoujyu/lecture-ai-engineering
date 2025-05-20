import os
import tempfile
import shutil
import time
import types
import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

# test_pipeline.py


from day5.演習1.pipeline import prepare_data, train_and_evaluate, log_model

# --- Fixtures ---

@pytest.fixture
def titanic_csv(tmp_path):
    # Minimal valid Titanic data for testing
    df = pd.DataFrame({
        "Pclass": [1, 3, 2, 1, 3],
        "Sex": ["male", "female", "female", "male", "male"],
        "Age": [22, 38, 26, 35, 28],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
        "Survived": [0, 1, 1, 1, 0]
    })
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "Titanic.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def patch_data_path(monkeypatch, titanic_csv):
    # 元の pd.read_csv を保存
    orig_read_csv = pd.read_csv
    def fake_read_csv(path, *args, **kwargs):
        if "Titanic.csv" in str(path):
            return orig_read_csv(titanic_csv, *args, **kwargs)
        return orig_read_csv(path, *args, **kwargs)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("pandas.read_csv", fake_read_csv)

@pytest.fixture
def prepared_data(monkeypatch, titanic_csv):
    orig_read_csv = pd.read_csv
    def fake_read_csv(path, *args, **kwargs):
        if "Titanic.csv" in str(path):
            return orig_read_csv(titanic_csv, *args, **kwargs)
        return orig_read_csv(path, *args, **kwargs)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("pandas.read_csv", fake_read_csv)
    return prepare_data()


# --- Tests ---

def test_prepare_data_shape_and_types(monkeypatch, titanic_csv):
    orig_read_csv = pd.read_csv
    def fake_read_csv(path, *args, **kwargs):
        if "Titanic.csv" in str(path):
            return orig_read_csv(titanic_csv, *args, **kwargs)
        return orig_read_csv(path, *args, **kwargs)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("pandas.read_csv", fake_read_csv)
    X_train, X_test, y_train, y_test = prepare_data()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4

def test_train_and_evaluate_output_types(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    model, accuracy, params = train_and_evaluate(X_train, X_test, y_train, y_test)
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(accuracy, float)
    assert isinstance(params, dict)
    assert 0.0 <= accuracy <= 1.0

def test_train_and_evaluate_speed(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    start = time.time()
    model, accuracy, params = train_and_evaluate(X_train, X_test, y_train, y_test)
    elapsed = time.time() - start
    assert elapsed < 2.0  # Should be fast on small data

def test_regression_accuracy(prepared_data):
    # Baseline accuracy (simulate previous run)
    BASELINE_ACCURACY = 0.5
    X_train, X_test, y_train, y_test = prepared_data
    _, accuracy, _ = train_and_evaluate(X_train, X_test, y_train, y_test)
    # Allow small fluctuation
    assert accuracy >= BASELINE_ACCURACY - 0.05

def test_log_model_runs(monkeypatch, prepared_data):
    # Mock mlflow methods to avoid actual logging
    class DummyRun:
        info = types.SimpleNamespace(run_id="dummy_run_id")
    class DummyMlflow:
        def set_experiment(self, name): pass
        def start_run(self): return self
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def log_metric(self, *a, **k): pass
        def log_params(self, *a, **k): pass
        def sklearn(self): return self
        def log_model(self, *a, **k): pass
        def active_run(self): return DummyRun()
    dummy_mlflow = DummyMlflow()
    monkeypatch.setattr("mlflow.set_experiment", dummy_mlflow.set_experiment)
    monkeypatch.setattr("mlflow.start_run", dummy_mlflow.start_run)
    monkeypatch.setattr("mlflow.log_metric", dummy_mlflow.log_metric)
    monkeypatch.setattr("mlflow.log_params", dummy_mlflow.log_params)
    monkeypatch.setattr("mlflow.sklearn.log_model", dummy_mlflow.log_model)
    monkeypatch.setattr("mlflow.active_run", dummy_mlflow.active_run)
    monkeypatch.setattr("mlflow.models.signature.infer_signature", lambda X, y: None)
    # Prepare dummy model/data
    X_train, X_test, y_train, y_test = prepared_data
    model, accuracy, params = train_and_evaluate(X_train, X_test, y_train, y_test)
    # Should not raise
    log_model(model, accuracy, params, X_train, X_test)