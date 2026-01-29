import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------
# 1. LOAD FEATURE-ENGINEERED DATA
# -------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(
    "data/processed/featured_youtube_data.csv",
    encoding="latin1",
    low_memory=False
)

print("Input shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------------------------------
# 2. FIX PUBLISH TIME (TIMEZONE SAFE)
# -------------------------------------------------
if "publish_time" in df.columns:
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df["publish_time"] = df["publish_time"].dt.tz_localize(None)

    df["days_since_publish"] = (
        pd.Timestamp.now() - df["publish_time"]
    ).dt.days

    df["days_since_publish"] = df["days_since_publish"].replace(0, 1)

    df["publish_hour"] = df["publish_time"].dt.hour
    df["publish_day"] = df["publish_time"].dt.dayofweek
else:
    df["days_since_publish"] = 1
    df["publish_hour"] = 0
    df["publish_day"] = 0

# -------------------------------------------------
# 3. ENCODE CATEGORICAL COLUMNS
# -------------------------------------------------
le_channel = LabelEncoder()
le_category = LabelEncoder()
le_region = LabelEncoder()

if "channel_title" in df.columns:
    df["channel_encoded"] = le_channel.fit_transform(df["channel_title"])
else:
    df["channel_encoded"] = 0

if "category_id" in df.columns:
    df["category_encoded"] = le_category.fit_transform(df["category_id"])
else:
    df["category_encoded"] = 0

if "region" in df.columns:
    df["region_encoded"] = le_region.fit_transform(df["region"])
else:
    df["region_encoded"] = 0

# -------------------------------------------------
# 4. CREATE TARGET VARIABLE (TREND SCORE)
# -------------------------------------------------
# Since YouTube dataset has no explicit label,
# we define a TREND SCORE using engagement logic
# -------------------------------------------------
# ENSURE views_per_day EXISTS
# -------------------------------------------------
if "views_per_day" not in df.columns:
    if "views" in df.columns and "days_since_publish" in df.columns:
        df["views_per_day"] = df["views"] / df["days_since_publish"]
    else:
        df["views_per_day"] = 0


df["trend_score"] = (
    0.5 * df["engagement_rate"] +
    0.3 * df["views_per_day"] / df["views_per_day"].max() +
    0.2 * df["likes_ratio"]
)

# Binary classification label
df["is_trending"] = (df["trend_score"] > df["trend_score"].median()).astype(int)

# -------------------------------------------------
# 5. DROP NON-ML COLUMNS
# -------------------------------------------------
drop_cols = [
    "title",
    "comment_count",
    "description",
    "comments_ratio"
    
]

df.drop(columns=drop_cols, inplace=True, errors="ignore")

# -------------------------------------------------
# 6. HANDLE NaN / INF
# -------------------------------------------------
df.replace([float("inf"), -float("inf")], 0, inplace=True)
df.fillna(0, inplace=True)

# -------------------------------------------------
# 7. FEATURE SELECTION
# -------------------------------------------------
feature_columns = [
    "views",
    "likes",
    "publish_time",
    "engagement_rate",
    "likes_ratio",
    "channel_title",
    "views_per_day",
    "title_length",
    "days_since_publish",
    "publish_hour",
    "publish_day",
    "tags",
    "channel_encoded",
    "category_encoded",
    "region_encoded"
]

X = df[feature_columns]
y = df["is_trending"]

# -------------------------------------------------
# 8. SAVE ML DATASETS
# -------------------------------------------------
X.to_csv("data/processed/X_features.csv", index=False)
y.to_csv("data/processed/y_target.csv", index=False)

print("✅ ML preprocessing completed")
print("X shape:", X.shape)
print("y distribution:\n", y.value_counts())
