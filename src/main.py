import json
from api_requests import FatalError, NonFatalError
from extract import extract_playlists, extract_tracks
from alive_progress import alive_bar
from requests.exceptions import HTTPError

with open('secrets.json', mode='r') as secrets_file:
    secrets = json.load(secrets_file)


past_search_requests: list[str] = [
    'https://api.spotify.com/v1/search?q=liked+songs&type=playlist&market=US&limit=50&offset=0',
    'https://api.spotify.com/v1/search?q=favorite+songs&type=playlist&market=US&limit=50&offset=0',
    'https://api.spotify.com/v1/search?q=best+songs&type=playlist&market=US&limit=50&offset=0',
    'https://api.spotify.com/v1/search?q=my+songs&type=playlist&market=US&limit=50&offset=0',
    'https://api.spotify.com/v1/search?q=my+favorite+playlist&type=playlist&market=US&limit=50&offset=0',
    'https://api.spotify.com/v1/search?q=liked+songs+playlist&type=playlist&market=US&limit=50&offset=0']

search_requests: list[str] = []


def get_playlist_ids(api_token: str):
    with open('../data/playlist_ids.txt', mode='a') as output_file:
        for search_request in search_requests:
            extract_playlists(search_request, api_token, output_file)


def main():

    api_token = 'BQBYkN7oJaOrhZ5jPRgYW5Ru1ZMzw7T8QWOOvBBqDRxO_ESVXiqPoJ7uOl8mBCBouauh1KXK90l0lJqZxcXUY0DcUwL51U-NBgtmoNBQlSnO-7MnuJU'

    print(api_token)

    max_playlists = 6000

    offset = 4793

    with open('data/output.json', mode='w') as output_file:
        with open('data/playlist_ids.txt', mode='r') as input_file:
            ids = input_file.read().splitlines()

            num_ids = len(ids)

            with alive_bar(min(num_ids, max_playlists)) as bar:
                bar(offset, skipped=True)
                for idx, playlist_id in enumerate(ids[offset:max_playlists]):
                    api_req = 'https://api.spotify.com/v1/playlists/' + \
                        playlist_id + '/tracks?limit=50&offset=0'
                    try:
                        tracks = extract_tracks(api_req, api_token)
                    except FatalError as e:
                        print(f"{e.message}. stopped at playlist {idx + offset}")
                        break
                    except NonFatalError as e:
                        with open('log.txt', mode='a') as log_file:
                            log_file.write(
                                f"playlist {idx + offset} {e}\n")
                        bar()
                        continue

                    output_file.write(
                        f'{playlist_id}: {json.dumps(tracks)},\n')

                    bar()


if __name__ == '__main__':
    main()
