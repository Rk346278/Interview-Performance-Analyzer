"""Model selection review — Interview Performance Analyzer."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_preprocess():
    """Match notebook preprocessing exactly."""
    df = pd.read_csv(PROJECT_DIR / "placementdata.csv")
    df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({"No": 0, "Yes": 1})
    df["PlacementTraining"] = df["PlacementTraining"].map({"No": 0, "Yes": 1})
    df["PlacementStatus"] = df["PlacementStatus"].map({"NotPlaced": 0, "Placed": 1})

    X = df.drop("PlacementStatus", axis=1)
    y = df["PlacementStatus"]
    return df, X, y


def evaluate_model(name, estimator, X_train, X_test, y_train, y_test):
    estimator.fit(X_train, y_train)
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    gap = train_acc - test_acc
    if gap > 0.05:
        fit_status = "overfitting"
    elif test_acc < 0.75 and train_acc < 0.78:
        fit_status = "underfitting"
    elif gap <= 0.05 and test_acc >= 0.75:
        fit_status = "good generalization"
    else:
        fit_status = "acceptable"

    return {
        "model": name,
        "accuracy": round(test_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "train_test_gap": round(gap, 4),
        "fit_status": fit_status,
        "confusion_matrix": cm.tolist(),
    }


def main():
    df, X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ("AdaBoost", AdaBoostClassifier(random_state=RANDOM_STATE)),
        (
            "SVM",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", random_state=RANDOM_STATE)),
                ]
            ),
        ),
        (
            "KNN",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
        ),
        ("Decision Tree (baseline)", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]

    optional = []
    try:
        from xgboost import XGBClassifier

        optional.append(
            ("XGBoost", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0))
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        optional.append(
            ("LightGBM", LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1))
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier

        optional.append(
            (
                "CatBoost",
                CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False),
            )
        )
    except ImportError:
        pass

    models.extend(optional)

    results = []
    for name, est in models:
        print(f"Training {name}...")
        results.append(evaluate_model(name, est, X_train, X_test, y_train, y_test))

    ranking = sorted(
        results,
        key=lambda r: (r["f1_score"], r["test_accuracy"], -r["train_test_gap"]),
        reverse=True,
    )

    for i, row in enumerate(ranking, 1):
        row["rank"] = i

    output = {
        "dataset": {
            "rows": len(df),
            "features": list(X.columns),
            "target": "PlacementStatus",
            "class_distribution": y.value_counts().to_dict(),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "unavailable_libraries": [
            lib
            for lib, avail in [
                ("xgboost", "XGBoost" not in [m[0] for m in optional]),
                ("lightgbm", "LightGBM" not in [m[0] for m in optional]),
                ("catboost", "CatBoost" not in [m[0] for m in optional]),
            ]
            if avail
        ],
        "results": results,
        "ranking": ranking,
    }

    out_path = PROJECT_DIR / "model_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    for row in ranking:
        print(f"\n--- {row['model']} (Rank #{row['rank']}) ---")
        print(f"  Accuracy:       {row['accuracy']:.4f}")
        print(f"  Precision:      {row['precision']:.4f}")
        print(f"  Recall:         {row['recall']:.4f}")
        print(f"  F1 Score:       {row['f1_score']:.4f}")
        print(f"  Train Accuracy: {row['train_accuracy']:.4f}")
        print(f"  Test Accuracy:  {row['test_accuracy']:.4f}")
        print(f"  Train-Test Gap: {row['train_test_gap']:.4f}")
        print(f"  Fit Status:     {row['fit_status']}")
        print(f"  Confusion Matrix: {row['confusion_matrix']}")

    print("\n" + "=" * 80)
    print("RANKING TABLE (by F1, then test accuracy, then lower overfit gap)")
    print("=" * 80)
    table = pd.DataFrame(ranking)[
        [
            "rank",
            "model",
            "test_accuracy",
            "precision",
            "recall",
            "f1_score",
            "train_accuracy",
            "train_test_gap",
            "fit_status",
        ]
    ]
    print(table.to_string(index=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
