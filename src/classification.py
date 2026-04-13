"""
Utilities for classification and regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.pipeline import Pipeline

CLASSIFIERS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              random_state=42, eval_metric="logloss", verbosity=0),
}

REGRESSORS = {
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ]),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                             random_state=42, verbosity=0),
}

GKF = GroupKFold(n_splits=5)

def prepare_data(df, feature_sets, target_cols, uid_col="uid"):
    """
    Clean the dataframe: fill NaN features with 0, drop rows with missing targets
    """
    all_features = list(set(f for feats in feature_sets.values() for f in feats))
    cols = all_features + target_cols + [uid_col]

    df_clean = df[cols].copy()
    df_clean[all_features] = df_clean[all_features].fillna(0)
    df_clean = df_clean.dropna()

    print(f"After cleaning: {len(df_clean)} sessions, {df_clean[uid_col].nunique()} users")
    return df_clean, df_clean[uid_col].values


def run_classification_binary(df, features, y, groups, classifiers=None):
    """
    Run binary classification with GroupKFold for all feature sets
    """
    if classifiers is None:
        classifiers = CLASSIFIERS

    results = []
    for features_name, features_cols in features.items():
        X = df[features_cols].values
        for model_name, model in classifiers.items():
            y_pred = cross_val_predict(model, X, y, cv=GKF, groups=groups)
            try:
                y_prob = cross_val_predict(model, X, y, cv=GKF, groups=groups, method="predict_proba")[:, 1]
                auc = roc_auc_score(y, y_prob)
            except Exception:
                auc = np.nan

            results.append({
                "feature_set": features_name,
                "model": model_name,
                "accuracy": accuracy_score(y, y_pred),
                "f1_weighted": f1_score(y, y_pred, average="weighted"),
                "auc_roc": auc,
                "n_features": len(features_cols),
            })
            print(f"[Binary] {features_name} + {model_name}: "f"Acc={results[-1]['accuracy']:.3f}, F1={results[-1]['f1_weighted']:.3f}, AUC={auc:.3f}")

    return pd.DataFrame(results).sort_values("auc_roc", ascending=False)


def run_classification_multiclass(df, features, y, groups, classifiers=None):
    """
    Run multi-class classification with GroupKFold for all feature sets
    """
    if classifiers is None:
        classifiers = CLASSIFIERS

    results = []
    for feat_name, feat_cols in features.items():
        X = df[feat_cols].values
        for model_name, model in classifiers.items():
            y_pred = cross_val_predict(model, X, y, cv=GKF, groups=groups)
            results.append({
                "feature_set": feat_name,
                "model": model_name,
                "accuracy": accuracy_score(y, y_pred),
                "f1_weighted": f1_score(y, y_pred, average="weighted"),
                "n_features": len(feat_cols),
            })
            print(f"[Multi] {feat_name} + {model_name}: "
                  f"Acc={results[-1]['accuracy']:.3f}, F1={results[-1]['f1_weighted']:.3f}")

    return pd.DataFrame(results).sort_values("f1_weighted", ascending=False)


def run_regression(df, features, y, groups, regressors=None):
    """
    Run regression with GroupKFold for all feature sets.
    Returns results DataFrame.
    """
    if regressors is None:
        regressors = REGRESSORS

    results = []
    for feat_name, feat_cols in features.items():
        X = df[feat_cols].values
        for model_name, model in regressors.items():
            y_pred = cross_val_predict(model, X, y, cv=GKF, groups=groups)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            results.append({
                "feature_set": feat_name,
                "model": model_name,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "n_features": len(feat_cols),
            })
            print(f"[Reg] {feat_name} + {model_name}: R2={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

    return pd.DataFrame(results).sort_values("r2", ascending=False)


def plot_best_classification_binary(df, features, y, groups, results_df, classifiers=None):
    """
    Plot confusion matrix and ROC curve for the best binary model.
    """
    if classifiers is None:
        classifiers = CLASSIFIERS

    best = results_df.iloc[0]
    X = df[features[best["feature_set"]]].values
    model = classifiers[best["model"]]

    y_pred = cross_val_predict(model, X, y, cv=GKF, groups=groups)
    y_prob = cross_val_predict(model, X, y, cv=GKF, groups=groups, method="predict_proba")[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not depressed", "Depressed"]).plot(ax=axes[0], cmap="Blues")
    axes[0].set_title(f"{best['model']} + {best['feature_set']}")

    RocCurveDisplay.from_predictions(y, y_prob, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_title(f"AUC = {best['auc_roc']:.3f}")

    plt.tight_layout()
    plt.show()

    print(classification_report(y, y_pred, target_names=["Not depressed", "Depressed"]))


def plot_best_classification_multiclass(df, features, y, groups, results_df, labels, classifiers=None):
    """
    Plot confusion matrix for the best multi-class model.
    """
    if classifiers is None:
        classifiers = CLASSIFIERS

    best = results_df.iloc[0]
    X = df[features[best["feature_set"]]].values
    model = classifiers[best["model"]]

    y_pred = cross_val_predict(model, X, y, cv=GKF, groups=groups)

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{best['model']} + {best['feature_set']}")
    plt.tight_layout()
    plt.show()

    print(classification_report(y, y_pred, target_names=labels))


def plot_best_regression(df, features, y, groups, results_df, score_name="PHQ-9", regressors=None):
    """
    Plot predicted vs actual and residuals for the best regression model.
    """
    if regressors is None:
        regressors = REGRESSORS

    best = results_df.iloc[0]
    X = df[features[best["feature_set"]]].values
    model = regressors[best["model"]]

    y_pred = cross_val_predict(model, X, y, cv=GKF, groups=groups)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y, y_pred, s=8, alpha=0.3, color="steelblue")
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2, alpha=0.7)
    axes[0].set_xlabel(f"Actual {score_name}")
    axes[0].set_ylabel(f"Predicted {score_name}")
    axes[0].set_title(f"{best['model']} + {best['feature_set']}")
    axes[0].grid(True, alpha=0.3)

    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, s=8, alpha=0.3, color="coral")
    axes[1].axhline(y=0, color="black", linewidth=1)
    axes[1].set_xlabel(f"Predicted {score_name}")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Best: R2={best['r2']:.3f}, MAE={best['mae']:.2f}")


def plot_summary(binary_df, multi_df, reg_df, feature_order, title="Model Performance"):
    """
    3-panel bar chart comparing feature sets across tasks.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    binary_df.pivot(index="model", columns="feature_set", values="auc_roc")[feature_order] \
        .plot(kind="bar", ax=axes[0], rot=15, edgecolor="white")
    axes[0].set_title("Binary — AUC-ROC")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=7)

    multi_df.pivot(index="model", columns="feature_set", values="f1_weighted")[feature_order] \
        .plot(kind="bar", ax=axes[1], rot=15, edgecolor="white")
    axes[1].set_title("Multi-class — Weighted F1")
    axes[1].set_ylabel("F1")

    reg_df.pivot(index="model", columns="feature_set", values="r2")[feature_order] \
        .plot(kind="bar", ax=axes[2], rot=15, edgecolor="white")
    axes[2].set_title("Regression — R2")
    axes[2].set_ylabel("R2")
    axes[2].axhline(y=0, color="red", linestyle="--", alpha=0.5)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(df, features, y, top_n=20, title="Feature Importance"):
    """
    Train Random Forest on all features and plot top N importances.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(df[features].values, y)

    df_imp = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(df_imp)), df_imp["importance"], color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(df_imp)))
    ax.set_yticklabels(df_imp["feature"], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
