# Suggesteria-
## Anime recommendation model

This project presents a content-based recommendation system for anime, designed to suggest titles similar to a given input based on textual metadata. The system leverages key features like name, genre, and type to compute similarity scores between anime entries and generate personalized recommendations.

## Key Features

Content-Based Filtering: Uses anime metadata (name, genre, type) to identify similar titles.

TF-IDF Vectorization: Converts textual features into numerical vectors for meaningful comparison.

Cosine Similarity: Measures similarity between anime entries based on their feature vectors.

Pickled Model Components: Saves the TF-IDF vectorizer and similarity matrix for efficient reuse and faster loading.

Interactive Web Interface: Built with Streamlit to provide users a smooth and intuitive experience.

## Technologies Used

Python

Pandas, NumPy

Scikit-learn

Streamlit

Pickle

TfidfVectorizer

Cosine Similarity

## How It Works

Data Preprocessing: The dataset is cleaned, and missing values are handled.

Feature Engineering: Combines relevant metadata fields into a single textual representation.

Vectorization: Applies TF-IDF to convert text into vector format.

Similarity Computation: Calculates cosine similarity between all anime vectors.

Recommendation: For a given input, the system retrieves the top k most similar anime.

Interface: Streamlit is used to allow user input and display recommendations interactively.
