import json
import urllib.request
import urllib.parse
import urllib.error
import base64
import time
import loader
import datetime


def auth():
    client_id = "66329fd57b9c4e72b3b9c829f80b4708"
    secret_id = "d649f928bcb14d49b874e72535f6a00f"
    key = "{}:{}".format(client_id, secret_id)
    key = base64.b64encode(key.encode()).decode()

    data = urllib.parse.urlencode(
        {"grant_type": "client_credentials"}).encode()

    request = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data,
        {"Authorization": "Basic {}".format(key)})

    response = urllib.request.urlopen(request).read()
    results = json.loads(response.decode())
    header = {"Authorization": "Bearer {}".format(results['access_token'])}
    return header


def get_info(spotify_id, auth_header):
    request = urllib.request.Request(
        "https://api.spotify.com/v1/audio-features/{}".format(spotify_id),
        None, auth_header)
    while True:
        try:
            response = urllib.request.urlopen(request).read()
            results = json.loads(response.decode())
            return results
        except urllib.error.HTTPError as e:
            print(e.headers)
            print("sleep")
            if e.code == 429:
                time.sleep(int(e.headers['Retry-After']))
            else:
                print(e)
                exit()


def get_spotify_info(msd_id, auth_header):
    subdir = msd_id[2:4]
    with open('./idmapping/{}/{}.json'.format(subdir, msd_id)) as f:
        raw = f.read()

    data = json.loads(raw)
    spotify_id = None

    if "response" in data:
        data = data["response"]
    else:
        print(msd_id)
        print("error")
        return None

    if "songs" in data:
        data = data["songs"]
    else:
        print(msd_id)
        print("error")
        return None

    if len(data) > 0:
        data = data[0]
    else:
        print(msd_id)
        print("error")
        return None

    if "tracks" in data:
        data = data["tracks"]
    else:
        print(msd_id)
        print("error")
        return None

    for t in data:
        if t["catalog"] == "spotify":
            spotify_id = t["foreign_id"].split(':')[-1]
            return get_info(spotify_id, auth_header)

    print(msd_id)
    print("error")

    return None


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
    "genre"]
new_column = [
    "s_acousticness",
    "s_danceability",
    "s_energy",
    "s_instrumentalness",
    "s_liveness",
    "s_loudness",
    "s_speechiness",
    "s_valence",
    "s_tempo"]


def main(start, end):
    data = loader.load_data()
    none = 0
    auth_header = auth()
    s = datetime.datetime.now()
    for key in new_column:
        data[key] = []
    for i, msd_id in enumerate(data['song_id']):
        if i < start:
            for key in new_column:
                data[key].append(0)
            continue
        if i > end:
            return
        result = get_spotify_info(msd_id, auth_header)
        if result is None:
            none += 1
            for key in new_column:
                data[key].append(None)
            with open("./data_no_spotify.cls", 'a') as f:
                line = ""
                for k in column + new_column:
                    line += "{}\t".format(data[k][i])
                f.write(line[:-1] + '\n')
            print(i)
            continue

        for key in new_column:
            data[key].append(float(result[key[2:]]))

        with open("./data_spotify.cls", 'a') as f:
            line = ""
            for k in column + new_column:
                line += "{}\t".format(data[k][i])
            f.write(line[:-1] + '\n')
        n = datetime.datetime.now()
        d = (n - s).total_seconds()
        speed = d / (i - start + 1)
        eta = datetime.timedelta(seconds=speed * (end - start)) + s
        print("{}, speed: {}, eta: {:%Y-%m-%d %H:%M}".format(i, speed, eta))

    print("None results:")
    print(none)


if __name__ == '__main__':
    start = 4913
    start = 0
    end = 406427 - 1
    main(start, end)
