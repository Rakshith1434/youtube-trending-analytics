import pandas as pd

# -------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------
df = pd.read_csv(
    "data/processed/featured_youtube_data.csv",
    encoding="latin1",
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Dataset loaded")
print("Shape:", df.shape)

# -------------------------------------------------
# 2. ENSURE views_per_day EXISTS
# -------------------------------------------------
if "views_per_day" not in df.columns:
    if "views" in df.columns and "days_since_publish" in df.columns:
        df["views_per_day"] = df["views"] / df["days_since_publish"]
        print("views_per_day calculated dynamically")
    else:
        raise ValueError("Cannot compute views_per_day (missing views or days_since_publish)")

# -------------------------------------------------
# 3. ENSURE REQUIRED COLUMNS EXIST
# -------------------------------------------------
required_columns = [
    "channel_title",
    "views",
    "likes",
    "engagement_rate",
    "views_per_day"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

# -------------------------------------------------
# 4. COMPETITOR (CHANNEL) ANALYSIS
# -------------------------------------------------
channel_stats = df.groupby("channel_title").agg(
    total_videos=("views", "count"),
    avg_views=("views", "mean"),
    avg_likes=("likes", "mean"),
    avg_engagement=("engagement_rate", "mean"),
    avg_views_per_day=("views_per_day", "mean")
)

# -------------------------------------------------
# 5. SORT BY PERFORMANCE
# -------------------------------------------------
top_channels = channel_stats.sort_values(
    by="avg_views_per_day",
    ascending=False
).head(10)

# -------------------------------------------------
# 6. DISPLAY & SAVE
# -------------------------------------------------
print("\n🔥 Top 10 Competitor Channels (by Views per Day):\n")
print(top_channels.round(2))

top_channels.to_csv(
    "data/processed/top_competitor_channels.csv"
)

print("\n✅ Competitor analysis completed successfully")
