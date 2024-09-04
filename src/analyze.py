import pandas as pd
import json
from hdbscan import HDBSCAN
import umap

import numpy as np
import plotly.express as px

from collections import Counter

# i'm going to assume that trying to do unsupervised learning does not violate
# spotify developer terms section IV 2.a.i:

''' Misuse of the Spotify Platform. Do not misuse the Spotify Platform, including by

    using the Spotify Platform or any Spotify Content to train a machine learning or AI model or otherwise ingesting Spotify Content into a machine learning or AI model;
'''


def show_plot(data):

    fig = px.scatter(data, x='x', y='y', color='labels', hover_data='names')

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

        unique_artists = [x[0]
                          for x in artist_occurences_per_playlist.most_common(5000)]

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

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # hdbscanner = HDBSCAN(min_cluster_size=10, metric="precomputed")
        umap_for_clustering = umap.UMAP(n_neighbors=40, n_components=10,
                                        n_jobs=-1, metric='cosine', min_dist=0).fit_transform(tf_idf.T)
        # labels = hdbscanner.fit_predict(umap_for_hdbscan.T)
        # for num_clusters in range(2, 12):
        #     for data, type_ in zip([tf_idf.T, cosine_sim, umap_for_clustering], ["tf_idf", "cosine_sim", "umap"]):
        #         labels = KMeans(n_clusters=10).fit_predict(data)

        #         print(type_, num_clusters, labels.shape,
        #               silhouette_score(data, labels), sep='\t')

        labels = list(
            map(str, list(KMeans(n_clusters=8).fit_predict(umap_for_clustering))))

        # print(f'{np.max(labels) - np.min(labels)} labels found')
        reducer = umap.UMAP(n_neighbors=50, n_components=2,
                            n_jobs=-1, metric='cosine', min_dist=0)
        # 30 is a good value for this

        umap_embedded = reducer.fit_transform(tf_idf.T)

        print(umap_embedded.shape)

        df = pd.DataFrame(
            {'x': umap_embedded[:, 0], 'y': umap_embedded[:, 1], 'names': unique_artists, 'labels': labels})

        show_plot(df)


if __name__ == "__main__":
    main()
