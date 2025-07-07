import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
anime = pd.read_csv("anime.csv")

# Fill missing values
anime['genre'] = anime['genre'].fillna('')
anime['type'] = anime['type'].fillna('')
anime['combinedFeature'] = anime['name'] + " " + anime['genre'] + " " + anime['type']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
anime_vectors = vectorizer.fit_transform(anime['combinedFeature'])

# Compute similarity
anime_similarity = cosine_similarity(anime_vectors)

# Save models
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(anime_similarity, open("anime_similarity.pkl", "wb"))

# Recommendation function
def recommend_anime(anime_name, k=5):
    anime_name = anime_name.lower()
    matched_indices = anime[anime['name'].str.lower() == anime_name].index

    if len(matched_indices) == 0:
        return []

    anime_index = matched_indices[0]
    sim_scores = list(enumerate(anime_similarity[anime_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]
    return [anime['name'][i[0]] for i in sim_scores]

# Evaluation Metrics
def mean_average_precision_at_k(k=5):
    total_precision = 0
    num_animes = len(anime)

    for idx in range(num_animes):
        recommended = recommend_anime(anime['name'][idx], k)
        relevant = [anime['name'][idx]]

        hits = 0
        avg_precision = 0
        for i, rec in enumerate(recommended):
            if rec in relevant:
                hits += 1
                avg_precision += hits / (i + 1)

        avg_precision /= len(relevant)
        total_precision += avg_precision

    return total_precision / num_animes

def coverage():
    recommended_set = set()
    for idx in range(len(anime)):
        recommended_set.update(recommend_anime(anime['name'][idx], k=5))
    return len(recommended_set) / len(anime)

# Print Evaluation Results
print(f"MAP@5: {mean_average_precision_at_k(5):.4f}")
print(f"Coverage: {coverage():.4f}")
