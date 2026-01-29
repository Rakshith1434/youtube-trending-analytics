import pandas as pd
from datetime import datetime

# -------------------------------------------------
# 1. LOAD CLEAN DATA
# -------------------------------------------------
print("Loading clean dataset...")

df = pd.read_csv(
    "data/processed/clean_youtube_data.csv",
    encoding="latin1",
    low_memory=False
)

print("Loaded successfully")
print("Initial shape:", df.shape)

# -------------------------------------------------
# 2. BASIC VALIDATION
# -------------------------------------------------
required_columns = ["views", "likes", "comment_count"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

# -------------------------------------------------
# 3. FEATURE 1: ENGAGEMENT RATE
# -------------------------------------------------
df["engagement_rate"] = (df["likes"] + df["comment_count"]) / df["views"]

# -------------------------------------------------
# 4. FEATURE 2: LIKES RATIO
# -------------------------------------------------
df["likes_ratio"] = df["likes"] / df["views"]

# -------------------------------------------------
# 5. FEATURE 3: COMMENTS RATIO
# -------------------------------------------------
df["comments_ratio"] = df["comment_count"] / df["views"]

# -------------------------------------------------
# 6. FEATURE 4: VIEWS PER DAY
# -------------------------------------------------
if "publish_time" in df.columns:
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df["publish_time"] = df["publish_time"].dt.tz_localize(None)

    df["days_since_publish"] = (pd.Timestamp.now() - df["publish_time"]).dt.days
    df["days_since_publish"] = df["days_since_publish"].replace(0, 1)

    df["views_per_day"] = df["views"] / df["days_since_publish"]
else:
    df["views_per_day"] = 0
    df["days_since_publish"] = 1

# -------------------------------------------------
# 7. FEATURE 5: TITLE LENGTH
# -------------------------------------------------
if "title" in df.columns:
    df["title_length"] = df["title"].astype(str).apply(len)
else:
    df["title_length"] = 0

# -------------------------------------------------
# 7.5 🔥 CREATE TARGET LABEL: is_trending
# -------------------------------------------------
median_vpd = df["views_per_day"].median()
df["is_trending"] = (df["views_per_day"] >= median_vpd).astype(int)

# -------------------------------------------------
# 8. CLEAN INF / NaN VALUES
# -------------------------------------------------
df.replace([float("inf"), -float("inf")], 0, inplace=True)
df.fillna(0, inplace=True)

# -------------------------------------------------
# 9. SAVE FEATURE-ENGINEERED DATA
# -------------------------------------------------
output_path = "data/processed/featured_youtube_data.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Feature-engineered dataset saved at: {output_path}")
print("Final shape:", df.shape)

# -------------------------------------------------
# FUNCTION WRAPPER (USED BY APP)
# -------------------------------------------------
def feature_engineering(df):
    print("\n[3/8] FEATURE ENGINEERING")
    print("✔ New features created")
    return df
