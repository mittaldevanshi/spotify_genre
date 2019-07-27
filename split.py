import numpy as np


def fromstr(s):
    return s


def load_data(size, balance):
    column = [
        "track_id",
        "analysis_sample_rate",
        "artist_id",
        "artist_name",
        "danceability",
        "duration",
        "end_of_fade_in",
        "energy",
        "loudness",
        "song_id",
        "start_of_fade_out",
        "tempo",
        "time_signature",
        "time_signature_confidence",
        "title",
        "track_7digitalid",
        "year",
        "genre",
        "s_acousticness",
        "s_danceability",
        "s_energy",
        "s_instrumentalness",
        "s_liveness",
        "s_loudness",
        "s_speechiness",
        "s_valence",
        "s_tempo"]

    new_column = [
        "track_id",
        "s_acousticness",
        "s_danceability",
        "s_energy",
        "s_instrumentalness",
        "s_liveness",
        "s_loudness",
        "s_speechiness",
        "s_valence",
        "s_tempo",
        "genre"]

    data = {}
    for key in column:
        data[key] = []

    genres = {}

    with open('./data/data_spotify.cls', 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            split = line.split('\t')
            data["track_id"].append(split[0])
            data["analysis_sample_rate"].append(int(split[1]))
            data["artist_id"].append(fromstr(split[2]))
            data["artist_name"].append(fromstr(split[3]))
            data["danceability"].append(float(split[4]))
            data["duration"].append(float(split[5]))
            data["end_of_fade_in"].append(float(split[6]))
            data["energy"].append(float(split[7]))
            data["loudness"].append(float(split[8]))
            data["song_id"].append(fromstr(split[9]))
            data["start_of_fade_out"].append(float(split[10]))
            data["tempo"].append(float(split[11]))
            data["time_signature"].append(float(split[12]))
            data["time_signature_confidence"].append(float(split[13]))
            data["title"].append(fromstr(split[14]))
            data["track_7digitalid"].append(int(split[15]))
            data["year"].append(int(split[16]))
            genre = split[17]
            if genre in genres:
                genres[genre]['num'] += 1
            else:
                genres[genre] = {'num': 1}
            data["genre"].append(split[17])
            data["s_acousticness"].append(float(split[18]))
            data["s_danceability"].append(float(split[19]))
            data["s_energy"].append(float(split[20]))
            data["s_instrumentalness"].append(float(split[21]))
            data["s_liveness"].append(float(split[22]))
            data["s_loudness"].append(float(split[23]))
            data["s_speechiness"].append(float(split[24]))
            data["s_valence"].append(float(split[25]))
            data["s_tempo"].append(float(split[26]))
    for genre in genres:
        genres[genre]['count'] = 0
        num = genres[genre]['num']
        # print("{}: {}".format(genre, num))
        if balance:
            genres[genre]['train'] = np.random.choice(num, size, replace=False)
        else:
            genres[genre]['train'] = np.random.choice(
                num, num * size // 100, replace=False)

    data['split'] = []
    for i, genre in enumerate(data['genre']):
        if genres[genre]['count'] in genres[genre]['train']:
            data['split'].append('train')
        else:
            data['split'].append('test')
        genres[genre]['count'] += 1

    with open("./data/data_{}_train.cls".format(size), "w") as f_train:
        with open("./data/data_{}_test.cls".format(size), "w") as f_test:
            line = ""
            for c in new_column:
                line += c + '\t'
            f_train.write(line[:-1] + '\n')
            f_test.write(line[:-1] + '\n')

            for i in range(len(data['genre'])):
                line = ""
                for c in new_column:
                    line += str(data[c][i]) + "\t"
                if data['split'][i] == 'train':
                    f_train.write(line[:-1] + '\n')
                else:
                    f_test.write(line[:-1] + '\n')


if __name__ == '__main__':
    load_data(1000, True)
    load_data(2000, True)
    load_data(80, False)
