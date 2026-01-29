def predict_new_video_performance(
    title_length,
    publish_hour,
    publish_day,
    expected_views
):
    explanations = []
    base_ctr = 0.035  # 3.5% baseline CTR

    # Title length impact
    if 40 <= title_length <= 60:
        base_ctr += 0.015
        explanations.append("Optimal title length improves click rate")
    elif title_length < 30:
        base_ctr -= 0.005
        explanations.append("Title may be too short")

    # Publish time impact
    if 18 <= publish_hour <= 22:
        base_ctr += 0.02
        explanations.append("Published during peak viewing hours")

    # Publish day impact
    if publish_day in [5, 6]:
        base_ctr += 0.01
        explanations.append("Weekend publishing boosts engagement")

    # Expected views impact
    if expected_views > 100_000:
        base_ctr += 0.02
        explanations.append("High expected reach")
    elif expected_views < 10_000:
        base_ctr -= 0.01

    predicted_ctr = max(0.01, min(base_ctr, 0.15))
    trending_probability = min(95, predicted_ctr * 1200)

    trend_level = (
        "High" if trending_probability > 70 else
        "Medium" if trending_probability > 40 else
        "Low"
    )

    return {
        "predicted_ctr": round(predicted_ctr * 100, 2),
        "trending_probability": round(trending_probability, 2),
        "trend_level": trend_level,
        "explanations": explanations
    }
