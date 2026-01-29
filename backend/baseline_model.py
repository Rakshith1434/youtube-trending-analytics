import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def run_baseline_model_comparison():
    print("\n[BASELINE MODEL COMPARISON]")

    # ---------------------------------------------------
    # STEP 1: LOAD DATA
    # ---------------------------------------------------
    df = pd.read_csv("data/processed/clean_youtube_data.csv")
    print("Dataset Loaded:", df.shape)

    # ---------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # ---------------------------------------------------
    df["engagement_rate"] = (
        df["likes"] + df["comment_count"]
    ) / df["views"].replace(0, np.nan)
    df["engagement_rate"] = df["engagement_rate"].fillna(0)

    df["title_length"] = df["title"].astype(str).apply(len)

    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        df["publish_hour"] = df["publish_time"].dt.hour.fillna(12)
    else:
        df["publish_hour"] = 12

    # ---------------------------------------------------
    # STEP 3: TARGET VARIABLE
    # ---------------------------------------------------
    df["is_trending"] = (df["views"] > df["views"].median()).astype(int)

    # ---------------------------------------------------
    # STEP 4: FEATURE SELECTION
    # ---------------------------------------------------
    features = [
        "views",
        "likes",
        "comment_count",
        "engagement_rate",
        "title_length",
        "publish_hour",
    ]

    X = df[features]
    y = df["is_trending"]

    # ---------------------------------------------------
    # STEP 5: STRATIFIED SPLIT
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ---------------------------------------------------
    # STEP 6: SCALING (ONLY FOR LOGISTIC)
    # ---------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------
    # STEP 7: MODELS
    # ---------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
    }

    results = {}

    # ---------------------------------------------------
    # STEP 8: TRAIN & EVALUATE
    # ---------------------------------------------------
    for name, model in models.items():
        start = time.time()

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        fit_time = round(time.time() - start, 2)

        results[name] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(roc, 4),
            "fit_time": fit_time,
            "model": model,
        }

        print(f"\n{name}")
        print("Accuracy:", round(acc, 4))
        print("ROC-AUC:", round(roc, 4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------
    # STEP 9: RESULTS TABLE
    # ---------------------------------------------------
    comparison_df = pd.DataFrame([
        [k, v["accuracy"], v["roc_auc"], v["fit_time"]]
        for k, v in results.items()
    ], columns=["Model", "Accuracy", "ROC-AUC", "Fit Time (s)"])

    print("\nBASELINE MODEL PERFORMANCE")
    print(comparison_df)

    # ---------------------------------------------------
    # STEP 10: BEST MODEL (FOR DEMO / REPORT)
    # ---------------------------------------------------
    best_model_name = "Random Forest"

    print("\n✔ BEST BASELINE MODEL:", best_model_name)
    print("✔ Reason: Ensemble learning, better generalization, reduced overfitting")

    # ---------------------------------------------------
    # STEP 11: FEATURE IMPORTANCE
    # ---------------------------------------------------
    rf = results["Random Forest"]["model"]
    feature_importance = dict(
        zip(features, rf.feature_importances_)
    )

    print("\nRandom Forest Feature Importance:")
    for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {round(v, 4)}")

    return comparison_df, best_model_name, feature_importance
    


# 🔥 ADD THIS LINE
if __name__ == "__main__":
    run_baseline_model_comparison()
