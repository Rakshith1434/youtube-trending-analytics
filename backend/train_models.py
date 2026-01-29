import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "data/processed/featured_youtube_data.csv",
    encoding="latin1",
    low_memory=False
)

df["views_per_day"] = df["views"] / df["days_since_publish"]

df["trend_score"] = (
    0.5 * df["engagement_rate"] +
    0.3 * (df["views_per_day"] / df["views_per_day"].max()) +
    0.2 * df["likes_ratio"]
)
df["is_trending"] = (df["trend_score"] > df["trend_score"].median()).astype(int)

feature_cols = [
    "views",
    "likes",
    "likes_ratio",
    "views_per_day",
    "title_length",
    "days_since_publish"
]

X = df[feature_cols]
y_trend = df["is_trending"]
y_ctr = df["engagement_rate"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_trend, test_size=0.2, random_state=42, stratify=y_trend
)

trend_model = LogisticRegression(max_iter=1000)
trend_model.fit(X_train, y_train)

ctr_model = RandomForestRegressor(n_estimators=150, random_state=42)
ctr_model.fit(X, y_ctr)

os.makedirs("models", exist_ok=True)

joblib.dump(trend_model, "models/trend_model.pkl")
joblib.dump(ctr_model, "models/ctr_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")


print("✅ Models trained and saved successfully")
