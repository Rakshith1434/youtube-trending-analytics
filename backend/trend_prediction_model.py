import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
def load_dataset():
    print("\n[1/8] LOADING DATASET")

    df = pd.read_csv(
        "data/processed/featured_youtube_data.csv",
        encoding="latin1",
        low_memory=False
    )

    print(f"[INFO] Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    return df

# -------------------------------------------------
# 1. LOAD FEATURE-ENGINEERED DATASET
# -------------------------------------------------
df = pd.read_csv(
    "data/processed/featured_youtube_data.csv",
    encoding="latin1",
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Dataset loaded:", df.shape)

# -------------------------------------------------
# 2. ENSURE REQUIRED FEATURES EXIST
# -------------------------------------------------
required_features = [
    "views",
    "likes",
    "likes_ratio",
    "engagement_rate",
    "days_since_publish",
    "title_length"
]

for col in required_features:
    if col not in df.columns:
        raise ValueError(f"Missing required feature: {col}")

# -------------------------------------------------
# 3. CREATE views_per_day IF MISSING
# -------------------------------------------------
if "views_per_day" not in df.columns:
    df["views_per_day"] = df["views"] / df["days_since_publish"]

# -------------------------------------------------
# 4. CREATE TARGET VARIABLE: is_trending
# -------------------------------------------------
# Trend score based on engagement + velocity
df["trend_score"] = (
    0.5 * df["engagement_rate"] +
    0.3 * (df["views_per_day"] / df["views_per_day"].max()) +
    0.2 * df["likes_ratio"]
)

# Binary target using median threshold
df["is_trending"] = (df["trend_score"] > df["trend_score"].median()).astype(int)

print("Target distribution:")
print(df["is_trending"].value_counts())

# -------------------------------------------------
# 5. DEFINE FEATURES & TARGET
# -------------------------------------------------
feature_columns = [
    "views",
    "likes",
    "likes_ratio",
    "views_per_day",
    "title_length",
    "days_since_publish"
]

X = df[feature_columns]
y = df["is_trending"]

# -------------------------------------------------
# 6. FEATURE SCALING
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# 7. TRAIN–TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 8. TRAIN MODEL
# -------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------------------------
# 9. PREDICTION & EVALUATION
# -------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n📊 Trending Prediction Model Results")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
def train_and_evaluate(df):
    print("\n[4/8] TRAIN-TEST SPLIT")

    X = df.drop("is_trending", axis=1)
    y = df["is_trending"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✔ Train size:", X_train.shape)
    print("✔ Test size:", X_test.shape)

    print("\n[5/8] SCALING FEATURES")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("✔ StandardScaler applied")

    print("\n[6/8] MODEL TRAINING")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("✔ Logistic Regression trained")

    print("\n[7/8] MODEL EVALUATION")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler
