artists = ["Queen", "Modest Mouse", "Taylor Swift",
               "Lil Nas X", "Elvis Presley", "A$AP Rocky"]

    for artist in artists:

        print("| Most Similar Artists to " +
              ', '.join([artist]) + " | Cosine Similarity |")

        print("| --------------- | --------------- |")

        for sim_artist, score in recommender([artist], 5):
            print(sim_artist, score, sep='\t')