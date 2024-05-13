import numpy as np
import pandas as pd

from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


songs = pd.read_csv('content based recommedation system/songdata.csv')

songs.head()



songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)



songs['text'] = songs['text'].str.replace(r'\n', '')



tfidf = TfidfVectorizer(analyzer='word', stop_words='english')

lyrics_matrix = tfidf.fit_transform(songs['text'])


cosine_similarities = cosine_similarity(lyrics_matrix)



similarities = {}

for i in range(len(cosine_similarities)):
    # Now we'll sort each element in cosine_similarities and get the indexes of the songs.
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    # After that, we'll store in similarities each name of the 50 most similar songs.
    # Except the first one that is the same song.
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]



class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)

        print(f'The {rec_items} recommended song for {song} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score")
            print("--------------------")

    def recommend(self, recommendation):
        # Get song to find recommendations for
        song = recommendation['song']
        # Get number of songs to recommend
        number_songs = recommendation['number_songs']
        # Get the index of the song in the DataFrame
        ##song_index = self.song[self.song['song'] == song].index[0]##
        # Get the number of songs most similars from matrix similarities
        if song in self.matrix_similar:
            recom_song = sorted(self.matrix_similar[song], reverse=True)[:number_songs]
            return recom_song
        else:
            return []




recommedations = ContentBasedRecommender(similarities)


recommendation = {
    "song": songs['song'].iloc[10],
    "number_songs": 4
}

recommedations.recommend(recommendation)



import streamlit as st

def main():
    st.title("Music Recommendation System")

    # Sidebar input
    song_title = st.text_input("Enter a song title")
    number_songs = st.slider("Number of songs to recommend", min_value=1, max_value=10, value=5)

    if st.button("Recommend"):
        # Get the index of the song in the DataFrame
        ###song_index = song[song['song'] == song_title].index[0]##
        # Create an instance of the ContentBasedRecommender class
        recommender = ContentBasedRecommender(cosine_similarities)
        # Define the recommendation input
        recommendation = {'song': song_title, 'number_songs': number_songs}
        # Get recommendations
        recommendations = recommender.recommend(recommendation)
               
        # Display recommendations
        st.write(f"The {number_songs} recommended songs for {song_title} are:")
        for i, recommendation in recommendations:
            st.write('Recommendations:', recommendation)
            #st.write(f"Number {i+1}:")
            #st.write(f"{rec_song[1]} by {rec_song[2]} with {round(rec_song[0], 3)} similarity score")
            st.write("--------------------")


if __name__ == '__main__':
    main()
