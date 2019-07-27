def fromstr(input):
    return input[2:-1]


def load_data():
    column = [
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
        "year"]
    data = {}
    for key in column:
        data[key] = []
    data['track_id'] = []
    data['genre'] = []

    with open('./data.cls', 'r') as f:
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
    with open('./genre.cls', 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            data["genre"].append(line.split('\t')[1])

    return data


if __name__ == '__main__':
    load_data()
