def generate_explanations(input_data, prediction):
    explanations = []

    if input_data["title_length"] < 40:
        explanations.append("Title is short; longer titles often improve discoverability.")

    if input_data["publish_hour"] in [18, 19, 20, 21]:
        explanations.append("Publishing during peak hours increases early engagement.")

    if input_data["views"] < 50000:
        explanations.append("Lower expected views may reduce recommendation boost.")

    if prediction["trending_probability"] >= 70:
        explanations.append("High engagement velocity suggests strong trending potential.")
    else:
        explanations.append("Engagement signals are moderate; optimization may help.")

    return explanations
