import pandas as pd
import numpy as np
import sqlite3

from scipy import stats
from scipy import stats
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

# =================================================
# SQLITE HELPERS
# =================================================
DB_PATH = "youtube.db"

def load_table(table_name):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# =================================================
# LOAD DATA (RUN ONCE)
# =================================================
print("\n[0/7] LOADING DATA")
raw_df = load_table("clean_youtube_data")
df = load_table("featured_youtube_data")

df.columns = df.columns.str.strip()

for col in ["title", "channel_title", "tags"]:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(fix_text)

# =================================================
# PIPELINE LOGS (RUN ONCE)
# =================================================
def run_pipeline_with_logs():
    print("\n[1/7] DATA QUALITY REPORT")
    print("Missing BEFORE cleaning:", raw_df.isnull().sum().sum())
    print("Duplicate rows BEFORE cleaning:", raw_df.duplicated().sum())
    print("Missing AFTER cleaning:", df.isnull().sum().sum())
    print("Duplicate rows AFTER cleaning:", df.duplicated().sum())

    print("\n[2/7] FEATURE SUMMARY")
    print("Initial features:", list(raw_df.columns))
    print("Final features:", list(df.columns))

    print("\n[3/7] CROSS VALIDATION (5-FOLD) + STATISTICAL VALIDATION")

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

        mean_acc = scores.mean()
        std_acc = scores.std()

        print("CV Scores:", scores)
        print("Mean Accuracy:", round(mean_acc, 4))
        print("Std Deviation:", round(std_acc, 4))

    # ==============================
    # HYPOTHESIS TESTING
    # ==============================
        print("\nSTATISTICAL HYPOTHESIS TEST")

        mean_acc = scores.mean()
        std_acc = scores.std()

        print("Mean Accuracy:", round(mean_acc, 4))
        print("Std Deviation:", round(std_acc, 6))

# -----------------------------------
# SAFE STATISTICAL TEST
# -----------------------------------
        if std_acc == 0:
            print("\nModel accuracy identical across folds.")
            print("Variance ≈ 0 → Model is highly stable.")
            print("Hypothesis test not required (deterministic performance).")
        else:
            t_stat, p_value = stats.ttest_1samp(scores, 0.5)

            conf_interval = stats.t.interval(
                0.95,
                len(scores) - 1,
                loc=mean_acc,
                scale=stats.sem(scores)
            )

            print("t-statistic:", round(t_stat, 4))
            print("p-value:", round(p_value, 6))
            print("95% Confidence Interval:",
                tuple(round(x, 4) for x in conf_interval))

            if p_value < 0.05:
                print("✔ Reject H0 → Model is statistically significant")
            else:
                print("✘ Fail to reject H0 → Not statistically significant")


            run_baseline_model_comparison()

# =================================================
# BASELINE MODEL COMPARISON (RUN ONCE)
# =================================================
def run_baseline_model_comparison():
    print("\n[4/7] BASELINE MODEL COMPARISON")

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

# =================================================
# LOOPED CTR PREDICTION (UNTIL EXIT)
# =================================================
def run_ctr_prediction_loop():
    print("\n[5/7] CTR PREDICTION MODE")
    print("Type 'exit' at any prompt to stop.\n")

    while True:
        user_input = input("Enter title length (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        title_length = int(user_input)

        user_input = input("Enter publish hour (0-23) (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        publish_hour = int(user_input)

        user_input = input("Enter publish day (1=Mon ... 7=Sun) (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        publish_day = int(user_input)

        user_input = input("Enter expected views (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        expected_views = int(user_input)

        result = predict_new_video_performance(
            title_length,
            publish_hour,
            publish_day,
            expected_views,
        )

        explanations = generate_explanations(
            {
                "title_length": title_length,
                "publish_hour": publish_hour,
                "publish_day": publish_day,
                "views": expected_views,
            },
            result,
        )

        print("\n--- PREDICTION RESULT ---")
        print("Predicted CTR        :", result["predicted_ctr"])
        print("Trending Probability :", result["trending_probability"])
        print("Trend Level          :", result["trend_level"])

        print("\nAI Explanation:")
        for exp in explanations:
            print("-", exp)

        # =====================================
        # BUSINESS OPTIMIZATION INSIGHTS
        # =====================================
        print("\nBUSINESS OPTIMIZATION INSIGHTS")
        print("Higher CTR → higher views → higher ad revenue")
        print("More successful content with less effort")
        print("Optimized publishing improves engagement & ROI")

        print("\n---------------------------------\n")


# =================================================
# ENTRY POINT
# =================================================
if __name__ == "__main__":
    print("\n=== YOUTUBE TRENDING ANALYTICS : TERMINAL MODE ===")
    run_pipeline_with_logs()
    run_ctr_prediction_loop()
    print("\n[7/7] PROGRAM EXITED")
