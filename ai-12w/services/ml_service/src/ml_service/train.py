from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import settings


@dataclass
class RunInfo:
    dataset: str
    n_samples: int
    n_features: int
    test_size: float
    random_state: int
    model: str
    created_at_epoch: float


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def train_and_save(
    out_model: Path,
    out_metrics: Path,
    out_run: Path,
    test_size: float,
    random_state: int,
) -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_seconds = time.time() - t0

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_seconds": float(train_seconds),
    }

    run = RunInfo(
        dataset="sklearn_breast_cancer",
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        test_size=float(test_size),
        random_state=int(random_state),
        model="StandardScaler + LogisticRegression",
        created_at_epoch=time.time(),
    )

    ensure_dir(out_model)
    ensure_dir(out_metrics)
    ensure_dir(out_run)

    joblib.dump(pipe, out_model)

    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out_run.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")

    print("✅ Saved:")
    print(f"  model   -> {out_model}")
    print(
        f"  metrics -> {out_metrics}  (accuracy={metrics['accuracy']:.4f}, auc={metrics['roc_auc']:.4f})"
    )
    print(f"  run     -> {out_run}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=settings.model_path)
    parser.add_argument("--metrics-path", default="artifacts/metrics.json")
    parser.add_argument("--run-path", default="artifacts/run.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_and_save(
        out_model=Path(args.model_path),
        out_metrics=Path(args.metrics_path),
        out_run=Path(args.run_path),
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()