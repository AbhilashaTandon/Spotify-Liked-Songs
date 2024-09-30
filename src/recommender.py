import json
from analyze import PlaylistData

import numpy as np


def main():
    with open('data/output.json', mode='r') as data:
        json_obj = json.load(data)

    plist_data = PlaylistData(json_obj)

    tf_idf = plist_data.get_tf_idf()

    # plist_data.get_cos_sim()

    # artists = ["Queen", "Modest Mouse", "Taylor Swift",
    #            "Lil Nas X", "Elvis Presley", "A$AP Rocky"]

    # for artist in artists:

    #     print("| Most Similar Artists to " +
    #           ', '.join([artist]) + " | Cosine Similarity |")

    #     print("| --------------- | --------------- |")

    #     for sim_artist, score in plist_data.get_most_similar([artist], 5):
    #         print(sim_artist, score, sep='\t')

    from sklearn.decomposition import TruncatedSVD

    model = TruncatedSVD(n_components=100)

    reduced = model.fit_transform(tf_idf)

    print(reduced.shape)

    print(np.cumsum(model.explained_variance_ratio_))

    def recommender(artists, n):
        input_vector = np.zeros(plist_data.num_artists)

        artist_idxs = [plist_data.artist_to_idx[artist] for artist in artists]

        for idx in artist_idxs:
            input_vector[idx] = 1

        reconstructed = model.inverse_transform(
            model.transform(input_vector.reshape(1, -1))).reshape(-1)

        artist_scores = sorted(enumerate(reconstructed),
                               key=lambda x: x[1], reverse=True)

        num_artists_returned = 0

        for idx, score in artist_scores:
            if (num_artists_returned >= n):
                return
            if idx in artist_idxs:
                continue
            num_artists_returned += 1
            yield (plist_data.unique_artists[idx], score)

    artists = ["Queen", "Modest Mouse", "Taylor Swift",
               "Lil Nas X", "Elvis Presley", "A$AP Rocky"]

    for artist in artists:

        print("| Most Similar Artists to " +
              ', '.join([artist]) + " | Cosine Similarity |")

        print("| --------------- | --------------- |")

        for sim_artist, score in recommender([artist], 5):
            print(sim_artist, score, sep='\t')


if __name__ == "__main__":
    main()
