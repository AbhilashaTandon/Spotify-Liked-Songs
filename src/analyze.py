import json
import umap

import numpy as np
import plotly.graph_objects as go
from collections import Counter

# i'm going to assume that trying to do unsupervised learning does not violate
# spotify developer terms section IV 2.a.i:

''' Misuse of the Spotify Platform. Do not misuse the Spotify Platform, including by

    using the Spotify Platform or any Spotify Content to train a machine learning or AI model or otherwise ingesting Spotify Content into a machine learning or AI model;
'''


def show_plot(x_data, y_data, labels):
    fig = go.Figure(data=go.Scatter(x=x_data,
                                    y=y_data,
                                    mode='markers',
                                    text=labels))  # hover text goes here

    fig.update_layout(title='Spotify Artists')
    fig.show()


def get_most_similar(artist, cosine_matrix, artist_to_idx, unique_artists, n):
    similarities = cosine_matrix[artist_to_idx[artist]]

    artists_idxs = np.argpartition(similarities, -(n+1))[-(n+1):][::-1]

    # n+1 to exclude similarity with self

    most_sim = [(unique_artists[idx], similarities[idx])
                for idx in artists_idxs if idx != artist_to_idx[artist]]

    print("Most Similar to " + artist)

    for related_artist, score in most_sim:
        print(related_artist, score, sep='\t')


def main():
    with open('data/output.json', mode='r') as data:
        json_obj = json.load(data)

        # how many times an artist occurs in all songs in data

        num_playlists = len(json_obj)

        print(f'{num_playlists} unique playlists')

        artist_occurences_per_playlist = Counter()
        # how many unique playlists an artist occurs in

        for playlist_id in json_obj:
            playlist = json_obj[playlist_id]

            current_playlist_artists = set()
            # all unique artists that occur in this playlist

            for song in playlist:
                if (song['id'] is None):
                    continue

                artists = [artist for artist in song['artists']
                           if not (artist is None)]

                if (len(artists) > 0):
                    current_playlist_artists.add(artists[0])

            artist_occurences_per_playlist.update(current_playlist_artists)

        unique_artists = [
            artist for artist, value in artist_occurences_per_playlist.items() if value > 4]

        num_artists = len(unique_artists)

        print(f'{num_artists} unique artists')

        artist_to_idx = {artist: idx for idx,
                         artist in enumerate(unique_artists)}

        term_frequencies = np.zeros((num_playlists, num_artists))
        document_frequencies = Counter()

        for playlist_idx, playlist_id in enumerate(json_obj):
            playlist = json_obj[playlist_id]

            for song in playlist:
                if (song['id'] is None):
                    continue

                artists = [artist for artist in song['artists']
                           if not (artist is None)]

                if (len(artists) > 0):
                    artist = artists[0]
                    if (artist in unique_artists):  # if occurs suitably often
                        artist_idx = artist_to_idx[artist]

                        term_frequencies[playlist_idx][artist_idx] += 1

                        document_frequencies.update([artist])

        idf = np.log(
            num_artists / np.fromiter(document_frequencies.values(), dtype=float))

        tf_idf = term_frequencies * idf

        from sklearn.metrics.pairwise import cosine_similarity

        cosine_sim = cosine_similarity(tf_idf.T)

        get_most_similar('Queen', cosine_sim,
                         artist_to_idx, unique_artists, 5)

        print(tf_idf.shape)

        reducer = umap.UMAP(n_neighbors=50, n_components=2,
                            n_jobs=-1, metric='cosine', min_dist=0)
        # 30 is a good value for this

        umap_embedded = reducer.fit_transform(tf_idf.T)

        print(umap_embedded.shape)

        show_plot(umap_embedded[:, 0], umap_embedded[:, 1], unique_artists)


if __name__ == "__main__":
    main()
