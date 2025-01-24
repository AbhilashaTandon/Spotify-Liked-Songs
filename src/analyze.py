import pandas as pd
import json
import umap
import numpy as np
import plotly.express as px
from sklearn.cluster import SpectralClustering
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# I'm going to assume that trying to do unsupervised learning does not violate
# Spotify developer terms section IV 2.a.i:

''' Misuse of the Spotify Platform. Do not misuse the Spotify Platform, including by

    using the Spotify Platform or any Spotify Content to train a machine learning or AI model or otherwise ingesting Spotify Content into a machine learning or AI model;
'''

# hyperparameters
NUM_ARTISTS = 6000
NUM_GENRES = 10


class PlaylistData:

    def __init__(self, json_obj):
        """ Reads data from json and saves helpful metadata variables like num_playlists, num_artists, etc.

        Args:
            json_obj (JSON): Json obj imported from file
        """
        self.artist_occurences_per_playlist = Counter()
        # how many unique playlists an artist occurs in

        self.num_playlists = len(json_obj)

        self.data = []
        # placeholder arr

        for playlist_id in json_obj:
            playlist = json_obj[playlist_id]

            reformatted_playlist = []

            current_playlist_artists = set()
            # all unique artists that occur in this playlist

            for song in playlist:
                if (song['id'] is None):
                    continue

                artists = [artist for artist in song['artists']
                           if not (artist is None or artist == '')]

                reformatted_playlist.append(artists)

                if (len(artists) > 0):
                    current_playlist_artists.add(artists[0])

            self.data.append(reformatted_playlist)

            self.artist_occurences_per_playlist.update(
                current_playlist_artists)

        self.unique_artists = [x[0]
                               for x in self.artist_occurences_per_playlist.most_common(NUM_ARTISTS)]

        self.num_artists = len(self.unique_artists)

        print(f'{self.num_artists} unique artists')

        self.artist_to_idx = {artist: idx for idx,
                              artist in enumerate(self.unique_artists)}

    def get_tf_idf(self):
        term_frequencies = np.zeros((self.num_playlists, self.num_artists))

        document_frequencies = Counter()

        for idx, playlist in enumerate(self.data):

            for song in playlist:
                if (len(song) > 0):
                    artist = song[0]
                    if (artist in self.unique_artists):  # if occurs suitably often
                        artist_idx = self.artist_to_idx[artist]

                        # we dont care much abt how many times an artist occurs in the playlist
                        term_frequencies[idx][artist_idx] = 1

                        document_frequencies.update([artist])

        idf = np.log(
            self.num_artists / np.fromiter(document_frequencies.values(), dtype=float))

        tf_idf = term_frequencies * idf

        return term_frequencies * idf

    def get_cos_sim(self):
        self.cos_sim = cosine_similarity(self.get_tf_idf().T)

    def get_most_similar(self, artists, n):

        similarities = np.zeros_like(self.cos_sim[0])

        for artist in artists:
            similarities += self.cos_sim[self.artist_to_idx[artist]]

        sim_artists = list(sorted(enumerate(similarities),
                                  key=lambda x: x[1], reverse=True))
        # n+1 to exclude similarity with self

        artist_idxs = [self.artist_to_idx[artist] for artist in artists]

        most_sim = [(idx, sim)

                    for idx, sim in sim_artists if idx not in artist_idxs]

        for related_artist, score in most_sim[:n]:

            yield (self.unique_artists[related_artist], score)


def show_plot(data):
    if ('labels' in data):
        fig = px.scatter(data, x='x', y='y',
                         color='labels', hover_data='names', width=1000, height=500)
    else:
        fig = px.scatter(data, x='x', y='y', hover_data='names')

    fig.update_layout(title='Spotify Genre Mapping')
    fig.write_html("interactive_genre_plot.html")
    fig.show()


def main():
    with open('data/output.json', mode='r') as data:
        json_obj = json.load(data)

        plist_data = PlaylistData(json_obj)

        tf_idf = plist_data.get_tf_idf()

        plist_data.get_cos_sim()

        artists = ["Queen", "Modest Mouse"]

        print("| Most Similar Artists to " +
              ', '.join(artists) + " | Cosine Similarity |")

        print("| --------------- | --------------- |")

        for artist, score in plist_data.get_most_similar(artists, 5):
            print(artist, score, sep='\t')

        reduced = np.array(umap.UMAP(n_neighbors=100, n_components=10,
                                     n_jobs=-1, metric='cosine', min_dist=0).fit_transform(tf_idf.T))

        labels_numeric = list(

            SpectralClustering(n_clusters=NUM_GENRES, random_state=5).fit_predict(reduced))

        labels = list(map(str, labels_numeric))

        genre_artist_freq = [Counter() for _ in range(np.min(labels_numeric), np.max(
            labels_numeric)+1)]  # dict of artists partitioned by genre and their popularity
        for artist, label in zip(plist_data.unique_artists, labels_numeric):
            genre_artist_freq[label][artist] = plist_data.artist_occurences_per_playlist[artist]

        genre_artist_freq = sorted(
            genre_artist_freq, key=lambda counter: counter.total(), reverse=True)

        # sorts genres from most to least popular

        genres = {0: "Pop", 1: "Hip Hop", 2: "Indie", 3: "Rock", 4: "Alt",
                  5: "Country", 6: "Spanish", 7: "K-Pop", 8: "CCM", 9: "South Asian"}
        # make sure to fix this manually each time you run the model

        named_genres = [genres[x] for x in labels_numeric]

        for idx, counter in enumerate(genre_artist_freq):
            print(genres[idx] + ': ' + ', '.join([x[0]
                                                 for x in counter.most_common(NUM_GENRES)]))
            print()

        reducer = umap.UMAP(n_neighbors=50, n_components=2,
                            n_jobs=-1, metric='cosine', min_dist=0)
        # 30 is a good value for this

        umap_embedded = np.array(reducer.fit_transform(tf_idf.T))

        df = pd.DataFrame(
            {'x': umap_embedded[:, 0], 'y': umap_embedded[:, 1], 'names': plist_data.unique_artists, "labels": named_genres})

        show_plot(df)


if __name__ == "__main__":
    main()
