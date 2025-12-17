import ast
from collections import Counter

def analyze_popularity(df, metric):
    all_tokens = []

    for _, row in df.iterrows():
        try:
            tokens = ast.literal_eval(row["bag_of_phrases"])
        except (SyntaxError, ValueError):
            tokens = []

        entities = row.get("entities")
        if not isinstance(entities, list):
            entities = []

        all_tokens.extend(tokens)
        all_tokens.extend(entities)

    top_10_tokens = Counter(all_tokens).most_common(10)
    top_tweets = df.nlargest(5, metric)
    bottom_tweets = df.nsmallest(5, metric)

    return top_10_tokens, top_tweets, bottom_tweets
