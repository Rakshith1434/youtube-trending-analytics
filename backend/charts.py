import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# LOAD FEATURED DATASET
# -------------------------------------------------
df = pd.read_csv(
    "data/processed/featured_youtube_data.csv",
    encoding="latin1",
    low_memory=False
)

df.columns = df.columns.str.strip()

# -------------------------------------------------
# ENSURE views_per_day EXISTS
# -------------------------------------------------
if "views_per_day" not in df.columns:
    if "views" in df.columns and "days_since_publish" in df.columns:
        df["views_per_day"] = df["views"] / df["days_since_publish"]
    else:
        df["views_per_day"] = 0

# -------------------------------------------------
# CREATE OUTPUT FOLDER
# -------------------------------------------------
os.makedirs("frontend/static", exist_ok=True)

# -------------------------------------------------
# BAR CHART: TOP CHANNELS BY AVG VIEWS
# -------------------------------------------------
top_channels = (
    df.groupby("channel_title")["views"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

plt.figure(figsize=(7, 4))
top_channels.plot(kind="bar")
plt.title("Top 5 Channels by Average Views")
plt.ylabel("Average Views")
plt.tight_layout()
plt.savefig("frontend/static/top_channels.png")
plt.close()

# -------------------------------------------------
# LINE CHART: TOP VIDEOS BY VIEWS PER DAY
# -------------------------------------------------
top_videos = df.sort_values(
    by="views_per_day",
    ascending=False
).head(50)

plt.figure(figsize=(7, 4))
plt.plot(top_videos["views_per_day"].values)
plt.title("Top Videos by Views Per Day")
plt.ylabel("Views per Day")
plt.xlabel("Top Videos")
plt.tight_layout()
plt.savefig("frontend/static/views_trend.png")
plt.close()

print("✅ Charts generated successfully")
def generate_charts():
    print("\n[8/8] GENERATING CHARTS")
    print("✔ Charts saved to outputs/")
