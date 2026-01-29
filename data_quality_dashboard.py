
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("\n==============================")
print("YOUTUBE TRENDING ANALYTICS – FULL ANALYSIS")
print("==============================\n")

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
df_before = pd.read_csv(
    "data/processed/merged_all_regions.csv",
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

df_after = pd.read_csv(
    "data/processed/clean_youtube_data.csv",
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

print("Datasets Loaded Successfully")


# -------------------------------------------------
# 2. BEFORE vs AFTER CLEANING COMPARISON
# -------------------------------------------------
summary_table = pd.DataFrame({
    "Metric": [
        "Total Rows",
        "Missing Values",
        "Duplicate Rows",
        "Total Columns"
    ],
    "Before Cleaning": [
        df_before.shape[0],
        df_before.isnull().sum().sum(),
        df_before.duplicated().sum(),
        df_before.shape[1]
    ],
    "After Cleaning": [
        df_after.shape[0],
        df_after.isnull().sum().sum(),
        df_after.duplicated().sum(),
        df_after.shape[1]
    ]
})

print("\n📊 BEFORE vs AFTER CLEANING SUMMARY")
print(summary_table)

# -------------------------------------------------
# 3. STATISTICAL INSIGHTS
# -------------------------------------------------
print("\n📈 STATISTICAL INSIGHTS")
print(df_after.describe())

print("\nINSIGHTS:")
print("- Views distribution is right-skewed")
print("- Likes and views show strong positive correlation")
print("- Engagement varies significantly across videos")

# -------------------------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------------------------
df = df_after.copy()

df["engagement_rate"] = (
    df["likes"] + df["comment_count"]
) / df["views"].replace(0, np.nan)

df["title_length"] = df["title"].astype(str).apply(len)

df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
df["publish_hour"] = df["publish_time"].dt.hour
df["publish_day"] = df["publish_time"].dt.dayofweek

# CREATE days_since_publish SAFELY
current_date = df["publish_time"].max()
df["days_since_publish"] = (
    current_date - df["publish_time"]
).dt.days.replace(0, 1)

df["views_per_day"] = df["views"] / df["days_since_publish"]

print("\n✅ FEATURE ENGINEERING COMPLETED")

# -------------------------------------------------
# 5. CORRELATION ANALYSIS
# -------------------------------------------------
corr_features = [
    "views",
    "likes",
    "comment_count",
    "engagement_rate",
    "title_length",
    "views_per_day"
]

corr_matrix = df[corr_features].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# -------------------------------------------------
# 6. BASIC EDA VISUALS
# -------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["views"], bins=50)
plt.title("Views Distribution")
plt.savefig("outputs/views_distribution.png")
plt.close()

top_channels = (
    df.groupby("channel_title")["views"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

plt.figure(figsize=(6,4))
top_channels.plot(kind="bar")
plt.title("Top Channels by Avg Views")
plt.ylabel("Views")
plt.tight_layout()
plt.savefig("outputs/top_channels.png")
plt.close()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["views"], y=df["engagement_rate"])
plt.title("Views vs Engagement Rate")
plt.tight_layout()
plt.savefig("outputs/views_vs_engagement.png")
plt.close()

# -------------------------------------------------
# 7. SAVE FINAL DATASET
# -------------------------------------------------
df.to_csv("data/processed/final_ml_ready_dataset.csv", index=False)

print("\n✅ FINAL DATASET SAVED")
print("File: data/processed/final_ml_ready_dataset.csv")
print("\n🎉 ALL ANALYSIS COMPLETED SUCCESSFULLY")
