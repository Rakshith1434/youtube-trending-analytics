def generate_explanation(video_row):
    explanations = []

    # Engagement-based explanation
    if "engagement_rate" in video_row and video_row["engagement_rate"] > 0.06:
        explanations.append("High engagement rate contributes positively")

    # Views velocity explanation
    if "views_per_day" in video_row and video_row["views_per_day"] > 100000:
        explanations.append("Rapid view growth indicates strong audience interest")

    # Publish time explanation
    if "publish_hour" in video_row and 18 <= video_row["publish_hour"] <= 22:
        explanations.append("Published during peak viewing hours")

    # Title length explanation
    if "title_length" in video_row and video_row["title_length"] < 60:
        explanations.append("Concise and effective title length")

    # Likes dominance explanation (safe version)
    if "likes" in video_row and "comment_count" in video_row:
        if video_row["likes"] > video_row["comment_count"] * 5:
            explanations.append("Strong like-to-comment ratio")

    # Fallback
    if not explanations:
        explanations.append("Performance is average with no strong positive signals")

    return explanations


# -------------------------------
# TEST THE EXPLAINABILITY MODULE
# -------------------------------
import pandas as pd

df = pd.read_csv(
    "data/processed/featured_youtube_data.csv",
    encoding="latin1",
    low_memory=False
)

sample_video = df.iloc[0]
reasons = generate_explanation(sample_video)

print("Explanation for Sample Video:")
for r in reasons:
    print("-", r)
