import json
import umap

import numpy as np
import plotly.graph_objects as go


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
    with open('output.json', mode='r') as data:
        json_obj = json.load(data)

        artist_occurences = {}  # how many times an artist occurs in all songs in data

        num_playlists = len(json_obj)

        for playlist_id in json_obj:
            playlist = json_obj[playlist_id]

            for song in playlist:
                if (song['id'] is None):
                    continue

                artists = song['artists']

                for artist in artists:
                    if (artist in artist_occurences):
                        artist_occurences[artist] += 1
                    else:
                        artist_occurences[artist] = 1

        unique_artists = list(artist_occurences.keys())

        unique_artists = [
            artist for artist in unique_artists if artist_occurences[artist] > 3]

        num_artists = len(unique_artists)

        artist_to_idx = {artist: idx for idx,
                         artist in enumerate(unique_artists)}

        term_frequencies = np.zeros((num_playlists, num_artists))
        document_frequencies = np.zeros((num_artists))

        for playlist_idx, playlist_id in enumerate(json_obj):
            playlist = json_obj[playlist_id]

            for song in playlist:
                if (song['id'] is None):
                    continue

                artists = song['artists']

                for artist in artists:
                    if (artist in unique_artists):  # if occurs more than once
                        artist_idx = artist_to_idx[artist]

                        term_frequencies[playlist_idx][artist_idx] += 1
                        document_frequencies[artist_idx] += 1

        idf = np.log(num_artists / document_frequencies)

        tf_idf = term_frequencies * idf

        from sklearn.metrics.pairwise import cosine_similarity

        cosine_sim = cosine_similarity(tf_idf.T)

        get_most_similar('Of Monsters And Men', cosine_sim,
                         artist_to_idx, unique_artists, 5)

        # print(tf_idf.shape)

        # reducer = umap.UMAP(n_neighbors=50, n_components=2,
        #                     n_jobs=-1, metric='cosine')

        # umap_embedded = reducer.fit_transform(tf_idf.T)

        # print(umap_embedded.shape)

        # show_plot(umap_embedded[:, 0], umap_embedded[:, 1], unique_artists)


if __name__ == "__main__":
    main()
