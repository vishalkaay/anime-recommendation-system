import streamlit as st

from recommender import Recommender


@st.cache_resource
def get_recommender() -> Recommender:
    return Recommender.from_csv()


recommender = get_recommender()

st.title("Anime Recommendation System")

selected = st.selectbox("Choose an anime", recommender.names)
count = st.slider("How many recommendations", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    results = recommender.similar(selected, count)
    if results.empty:
        st.write("No recommendations found.")
    else:
        st.subheader("Recommended anime")
        st.dataframe(
            results[["name", "genre", "type", "rating", "similarity"]],
            hide_index=True,
            column_config={
                "name": "Name",
                "genre": "Genres",
                "type": "Type",
                "rating": st.column_config.NumberColumn("Rating", format="%.2f"),
                "similarity": st.column_config.NumberColumn("Similarity", format="%.2f"),
            },
        )
