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

    print(tf_idf.shape)

    num_artist_to_keep = 5

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(tf_idf, test_size=.2)

    test_copy = test.copy()

    for idx, rating in enumerate(test_copy):
        # we randomly remove artists from rating
        # and then try and use pca to reconstruct them
        non_zero_indices = [idx for idx, val in enumerate(rating) if val > 0]

        if len(non_zero_indices) == 0:
            continue

        num_indices_to_remove = max(
            1, len(non_zero_indices)-num_artist_to_keep)
        indices_to_zero = np.random.choice(
            range(len(non_zero_indices)), replace=False, size=num_indices_to_remove)

        test_copy[idx][indices_to_zero] = 0

    n_components = 100

    # while (n_components <= 4096):

    from sklearn.decomposition import TruncatedSVD

    model = TruncatedSVD(n_components=n_components)

    model.fit(train)

    reduced = model.transform(test_copy)

    reconstructed = model.inverse_transform(reduced)

    mse = ((reconstructed - test)**2).mean(axis=None)

    print(n_components, mse, sep='\t')

    # n_components *= 2

    # from sklearn.decomposition import TruncatedSVD

    # model = TruncatedSVD(n_components=100)

    # reduced = model.fit_transform(tf_idf)

    # print(reduced.shape)

    # print(np.cumsum(model.explained_variance_ratio_))

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
