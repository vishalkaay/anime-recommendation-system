import streamlit as st
import pandas as pd
import pickle

# Load data and models
anime = pd.read_csv("anime.csv")
vectorizer = pickle.load(open("PickleFiles/vectorizer.pkl", "rb"))
anime_similarity = pickle.load(open("PickleFiles/anime_similarity.pkl", "rb"))

# Streamlit UI
st.title("Anime Recommendation System")

# Dropdown for anime selection
anime_names = anime['name'].tolist()
selected_anime = st.selectbox("Choose an anime", anime_names)

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

# Display recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_anime(selected_anime)
    if recommendations:
        st.write("### Recommended Anime:")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("No recommendations found.")
