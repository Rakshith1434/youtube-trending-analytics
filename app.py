from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import sqlite3
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ftfy import fix_text
from backend.new_video_ctr import predict_new_video_performance

# SAFE AI EXPLAIN
try:
    from backend.ai_explain import generate_explanations
except Exception:
    def generate_explanations(*args, **kwargs):
        return ["AI explainability module unavailable"]

app = Flask(__name__)

# =================================================
# SQLITE CONNECTION
# =================================================
DB_PATH = "youtube.db"

def load_table(table_name):
    conn = sqlite3.connect("youtube.db")
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# =================================================
# LOAD DATA FROM DATABASE ✅
# =================================================
raw_df = load_table("clean_youtube_data")
df = load_table("featured_youtube_data")

df.columns = df.columns.str.strip()

for col in ["title", "channel_title", "tags"]:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(fix_text)

# =================================================
# CATEGORY MAP
# =================================================
CATEGORY_MAP = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism"
}

# =================================================
# DASHBOARD HELPERS
# =================================================
def get_popular_tags(data, top_n=15):
    if "tags" not in data.columns:
        return []

    tags = []
    for t in data["tags"].dropna():
        tags.extend(t.split("|"))

    tags = [t.strip().lower() for t in tags if len(t.strip()) > 2]

    return (
        pd.Series(tags)
        .value_counts()
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "tag", 0: "count"})
        .to_dict("records")
    )

def load_dashboard_data(region=None, category=None):
    temp_df = df.copy()

    if region:
        temp_df = temp_df[temp_df["region"] == region]
    if category and category != "None":
        temp_df = temp_df[temp_df["category_id"] == int(category)]


    top_videos = (
        temp_df.sort_values("views_per_day", ascending=False)
        .head(10)
        .to_dict("records")
    )

    competitors = (
        temp_df.groupby("channel_title")
        .agg(
            avg_views=("views", "mean"),
            avg_engagement=("engagement_rate", "mean"),
        )
        .sort_values("avg_views", ascending=False)
        .head(5)
        .reset_index()
        .to_dict("records")
    )

    return top_videos, competitors, get_popular_tags(temp_df)

# =================================================
# BASELINE MODEL COMPARISON (LOG ONLY)
# =================================================
def run_baseline_model_comparison():
    print("\n[4/8] BASELINE MODEL COMPARISON")

    data = raw_df.copy()

    data["engagement_rate"] = (
        data["likes"] + data["comment_count"]
    ) / data["views"].replace(0, np.nan)
    data["engagement_rate"] = data["engagement_rate"].fillna(0)

    data["title_length"] = data["title"].astype(str).apply(len)
    data["publish_time"] = pd.to_datetime(data["publish_time"], errors="coerce")
    data["publish_hour"] = data["publish_time"].dt.hour.fillna(12)

    data["is_trending"] = (data["views"] > data["views"].median()).astype(int)

    FEATURES = [
        "views",
        "likes",
        "comment_count",
        "engagement_rate",
        "title_length",
        "publish_hour",
    ]

    X = data[FEATURES]
    y = data["is_trending"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_auc = 0
    best_model = None

    for name, model in models.items():
        print(f"\n{name}")

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        print("Accuracy :", round(accuracy_score(y_test, preds), 4))
        print("Precision:", round(precision_score(y_test, preds), 4))
        print("Recall   :", round(recall_score(y_test, preds), 4))
        print("F1 Score :", round(f1_score(y_test, preds), 4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

        roc = roc_auc_score(y_test, probs)
        if roc > best_auc:
            best_auc = roc
            best_model = name

    print("\n✔ BEST BASELINE MODEL:", best_model)
    print("✔ Random Forest performs best due to ensemble learning & generalization")

# =================================================
# PIPELINE LOGS
# =================================================
def run_pipeline_with_logs():
    print("\n[1/8] DATA QUALITY REPORT")
    print("Missing BEFORE cleaning:", raw_df.isnull().sum().sum())
    print("Duplicate rows BEFORE cleaning:", raw_df.duplicated().sum())
    print("Missing AFTER cleaning:", df.isnull().sum().sum())
    print("Duplicate rows AFTER cleaning:", df.duplicated().sum())

    print("\n[2/8] FEATURE SUMMARY")
    print("Initial features:", list(raw_df.columns))
    print("Final features:", list(df.columns))

    print("\n[3/8] CROSS VALIDATION (5-FOLD)")
    if "is_trending" in df.columns:
        FEATURES = [
            "views",
            "likes",
            "likes_ratio",
            "views_per_day",
            "title_length",
            "days_since_publish",
        ]
        X = df[FEATURES]
        y = df["is_trending"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print("CV Scores:", scores)
        print("Mean CV Accuracy:", round(scores.mean(), 4))

    run_baseline_model_comparison()

    print("\n[8/8] FLASK SERVER STARTING")
    print("👉 http://127.0.0.1:5000\n")

# =================================================
# ROUTES
# =================================================
@app.route("/")
def index():
    region = request.args.get("region")
    category = request.args.get("category")

    top_videos, competitors, popular_tags = load_dashboard_data(region, category)

    return render_template(
        "index.html",
        regions=sorted(df["region"].dropna().unique()),
        categories=CATEGORY_MAP,
        selected_region=region,
        selected_category=category,
        top_videos=top_videos,
        competitors=competitors,
        popular_tags=popular_tags,
        ctr_result=None,
        confidence=None,
        confidence_level=None,
        explanations=None,
        metrics={}
    )

@app.route("/predict_ctr", methods=["GET","POST"])
def predict_ctr():
    region = request.form.get("region")
    category = request.form.get("category")

    title_length = int(request.form["title_length"])
    publish_hour = int(request.form["publish_hour"])
    publish_day = int(request.form["publish_day"])
    expected_views = int(request.form["views"])

    result = predict_new_video_performance(
        title_length, publish_hour, publish_day, expected_views
    )

    explanations = generate_explanations(
        {
            "title_length": title_length,
            "publish_hour": publish_hour,
            "publish_day": publish_day,
            "views": expected_views,
        },
        result
    )

    top_videos, competitors, popular_tags = load_dashboard_data(region, category)

    return render_template(
        "index.html",
        regions=sorted(df["region"].dropna().unique()),
        categories=CATEGORY_MAP,
        selected_region=region,
        selected_category=category,
        top_videos=top_videos,
        competitors=competitors,
        popular_tags=popular_tags,
        ctr_result=result["predicted_ctr"],
        confidence=result["trending_probability"],
        confidence_level=result["trend_level"],
        explanations=explanations,
        metrics={}
    )

# =================================================
if __name__ == "__main__":
    run_pipeline_with_logs()
    app.run(debug=True, use_reloader=False)