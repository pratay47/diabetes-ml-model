"""
train_model.py
==============
Type 2 Diabetes Prediction System - Model Training & Preprocessing Pipeline

This script:
1. Loads the documented Pima Indians Diabetes Database.
2. Identifies and converts biologically invalid zero values to NaN.
3. Splits data into training and test sets using stratified sampling to prevent data leakage.
4. Encapsulates preprocessing (median imputation & scaling) and estimators in Scikit-Learn Pipelines.
5. Performs 5-fold Stratified Cross-Validation on the training set for model selection.
6. Evaluates baseline Logistic Regression, Decision Tree, and Random Forest on the held-out test set.
7. Saves the trained pipelines, test metrics, and metadata to disk for the Streamlit web application.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Constants
DATA_PATH = os.path.join("data", "diabetes.csv")
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "all_models.joblib")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "metrics.json")
RANDOM_STATE = 42

# Columns where 0 indicates missing/invalid biological readings
# Pregnancies CAN legitimately be 0, but Glucose, BloodPressure, SkinThickness, Insulin, and BMI cannot.
INVALID_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_and_preprocess_data(filepath: str = DATA_PATH):
    """
    Loads diabetes dataset and converts invalid zero values to NaN.
    Does NOT impute or scale here to prevent data leakage prior to train/test split.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure the file exists.")

    df = pd.read_csv(filepath)
    print(f"[1] Dataset loaded successfully from {filepath}.")
    print(f"    - Total records: {df.shape[0]}, Features: {df.shape[1] - 1}")
    print(f"    - Outcome distribution: 0={sum(df['Outcome'] == 0)} ({sum(df['Outcome'] == 0)/len(df):.1%}), "
          f"1={sum(df['Outcome'] == 1)} ({sum(df['Outcome'] == 1)/len(df):.1%})")

    # Replace invalid zeros with NaN in selected clinical columns
    df_clean = df.copy()
    zero_counts = {}
    for col in INVALID_ZERO_COLS:
        zero_counts[col] = int((df_clean[col] == 0).sum())
        df_clean[col] = df_clean[col].replace(0, np.nan)

    print("    - Identified invalid zeros treated as missing values:")
    for col, count in zero_counts.items():
        pct = count / len(df) * 100
        print(f"      * {col}: {count} zeros ({pct:.1f}%)")

    return df, df_clean, zero_counts


def create_pipelines():
    """
    Builds Scikit-learn Pipelines for Logistic Regression, Decision Tree, and Random Forest.
    Embedding imputation and scaling inside Pipeline guarantees zero data leakage across folds.
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            # Moderate depth and leaf constraint to prevent severe overfitting on small dataset
            ("classifier", DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=10,
                random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                random_state=RANDOM_STATE
            ))
        ])
    }
    return pipelines


def evaluate_cross_validation(pipelines, X_train, y_train):
    """
    Evaluates each candidate pipeline using 5-fold Stratified Cross-Validation on the TRAINING set.
    Model selection is performed on CV metrics to avoid test set snooping.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = {}

    print("\n[2] Performing 5-Fold Stratified Cross-Validation on Training Set...")
    print("-" * 75)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)

    best_model_name = None
    best_cv_f1 = -1.0

    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=cv_metrics, return_train_score=False)
        model_summary = {}
        for m in cv_metrics:
            mean_score = float(np.mean(scores[f"test_{m}"]))
            std_score = float(np.std(scores[f"test_{m}"]))
            model_summary[m] = {"mean": mean_score, "std": std_score}

        cv_results[name] = model_summary

        print(f"{name:<22} | "
              f"{model_summary['accuracy']['mean']:.3f}±{model_summary['accuracy']['std']:.2f} | "
              f"{model_summary['precision']['mean']:.3f}±{model_summary['precision']['std']:.2f} | "
              f"{model_summary['recall']['mean']:.3f}±{model_summary['recall']['std']:.2f} | "
              f"{model_summary['f1']['mean']:.3f}±{model_summary['f1']['std']:.2f} | "
              f"{model_summary['roc_auc']['mean']:.3f}±{model_summary['roc_auc']['std']:.2f}")

        # In medical risk screening, balanced F1 / ROC-AUC is prioritized over plain accuracy
        if model_summary["f1"]["mean"] > best_cv_f1:
            best_cv_f1 = model_summary["f1"]["mean"]
            best_model_name = name

    print("-" * 75)
    print(f"Selected best model via Training Cross-Validation: '{best_model_name}' (Mean CV F1: {best_cv_f1:.3f})")
    return cv_results, best_model_name


def evaluate_test_set(pipelines, X_train, y_train, X_test, y_test):
    """
    Fits each pipeline on full training data and evaluates generalization on the held-out test set.
    """
    test_results = {}

    print("\n[3] Evaluating Models on Held-Out Test Set (20% of data)...")
    print("-" * 75)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)

    for name, pipe in pipelines.items():
        # Fit on training data
        pipe.fit(X_train, y_train)

        # Predictions & Probabilities
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precisions, recalls, _ = precision_recall_curve(y_test, y_proba)

        test_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": precisions.tolist(), "recall": recalls.tolist()},
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        print(f"{name:<22} | {acc:<10.3f} | {prec:<10.3f} | {rec:<10.3f} | {f1:<10.3f} | {roc_auc:<10.3f}")

    print("-" * 75)
    return test_results


def extract_feature_stats(df_clean, feature_cols):
    """
    Extracts descriptive statistics for feature bounds and realistic input ranges in Streamlit.
    """
    stats = {}
    for col in feature_cols:
        series = df_clean[col].dropna()
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "std": float(series.std()),
        }
    return stats


def train_and_save():
    """
    Main training workflow execution.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    df_raw, df_clean, zero_counts = load_and_preprocess_data(DATA_PATH)
    feature_cols = [col for col in df_clean.columns if col != "Outcome"]
    X = df_clean[feature_cols]
    y = df_clean["Outcome"]

    # 2. Stratified Train-Test Split (Hold out 20% test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")

    # 3. Pipelines
    pipelines = create_pipelines()

    # 4. Cross-Validation on training set
    cv_results, best_model_name = evaluate_cross_validation(pipelines, X_train, y_train)

    # 5. Test set evaluation
    test_results = evaluate_test_set(pipelines, X_train, y_train, X_test, y_test)

    # 6. Feature stats for input ranges
    feature_stats = extract_feature_stats(df_clean, feature_cols)

    # 7. Save winning pipeline
    best_pipeline = pipelines[best_model_name]
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"\n[4] Saved best pipeline ('{best_model_name}') to '{BEST_MODEL_PATH}'.")

    # 8. Save all models bundle and metadata
    all_artifacts = {
        "models": pipelines,
        "best_model_name": best_model_name,
        "feature_names": feature_cols,
        "invalid_zero_cols": INVALID_ZERO_COLS,
        "cv_results": cv_results,
        "test_results": test_results,
        "feature_stats": feature_stats,
        "zero_counts": zero_counts,
        "sample_counts": {
            "total": len(df_raw),
            "train": len(X_train),
            "test": len(X_test),
            "outcome_0": int((y == 0).sum()),
            "outcome_1": int((y == 1).sum()),
        }
    }
    joblib.dump(all_artifacts, ALL_MODELS_PATH)
    print(f"    Saved full model bundle and evaluation data to '{ALL_MODELS_PATH}'.")

    # 9. Save JSON metrics for easy verification and basic unit tests
    simplified_metrics = {
        "best_model": best_model_name,
        "cv_results": {
            m: {metric: vals["mean"] for metric, vals in metrics.items()}
            for m, metrics in cv_results.items()
        },
        "test_results": {
            m: {
                "accuracy": vals["accuracy"],
                "precision": vals["precision"],
                "recall": vals["recall"],
                "f1": vals["f1"],
                "roc_auc": vals["roc_auc"],
                "confusion_matrix": vals["confusion_matrix"],
            }
            for m, vals in test_results.items()
        }
    }
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(simplified_metrics, f, indent=2)
    print(f"    Saved metrics summary to '{METRICS_JSON_PATH}'.")

    print("\nTraining completed successfully! Preprocessing and models are ready.")
    return all_artifacts


if __name__ == "__main__":
    train_and_save()
