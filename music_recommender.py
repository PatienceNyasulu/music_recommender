import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load and prepare data
@st.cache_data
def load_data():
    songs = pd.read_csv('content based recommedation system/songdata.csv')
    songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
    songs['text'] = songs['text'].str.replace(r'\n', '', regex=True)
    return songs

songs = load_data()

# TF-IDF vectorization
@st.cache_data
def compute_similarity_matrix(songs_df):
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    lyrics_matrix = tfidf.fit_transform(songs_df['text'])
    cosine_similarities = cosine_similarity(lyrics_matrix)

    similarities = {}
    for i in range(len(cosine_similarities)):
        similar_indices = cosine_similarities[i].argsort()[:-50:-1]
        similarities[songs_df['song'].iloc[i]] = [
            (cosine_similarities[i][x], songs_df['song'][x], songs_df['artist'][x])
            for x in similar_indices if x != i
        ]
    return similarities

similarities = compute_similarity_matrix(songs)


# Recommender class
class ContentBasedRecommender:
    def __init__(self, matrix: dict):
        self.matrix_similar = matrix

    def recommend(self, song_title: str):
        recs = self.matrix_similar.get(song_title)
        return recs if recs else []


# Streamlit App
def main():
    st.title("🎵 Content-Based Music Recommendation System")
    st.markdown("Get song suggestions based on lyrics and metadata similarity!")

    song_titles = songs['song'].unique().tolist()
    song_title = st.selectbox("Choose a song title:", song_titles)

    if st.button("Recommend"):
        recommender = ContentBasedRecommender(similarities)
        recommendations = recommender.recommend(song_title)

        if recommendations:
            st.success(f"Top {len(recommendations)} recommended songs for **{song_title}**:")
            for i, (score, title, artist) in enumerate(recommendations):
                st.markdown(f"**{i+1}. {title}** by *{artist}* — Similarity Score: `{round(score, 3)}`")
        else:
            st.warning("No recommendations found for the selected song.")


if __name__ == '__main__':
    main()
