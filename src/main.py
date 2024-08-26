import json
from api_requests import FatalError, NonFatalError, request_access_token
from extract import extract_playlists, extract_tracks
from alive_progress import alive_bar

# past_search_requests: list[str] = [
#     'https://api.spotify.com/v1/search?q=liked+songs&type=playlist&market=US&limit=50&offset=0',
#     'https://api.spotify.com/v1/search?q=favorite+songs&type=playlist&market=US&limit=50&offset=0',
#     'https://api.spotify.com/v1/search?q=best+songs&type=playlist&market=US&limit=50&offset=0',
#     'https://api.spotify.com/v1/search?q=my+songs&type=playlist&market=US&limit=50&offset=0',
#     'https://api.spotify.com/v1/search?q=my+favorite+playlist&type=playlist&market=US&limit=50&offset=0',
#     'https://api.spotify.com/v1/search?q=liked+songs+playlist&type=playlist&market=US&limit=50&offset=0']

search_requests: list[str] = []


def get_playlist_ids(api_token: str):
    with open('../data/playlist_ids.txt', mode='a') as output_file:
        for search_request in search_requests:
            extract_playlists(search_request, api_token, output_file)


def main():

    with open('secrets.json', mode='r') as secrets_file:
        secrets = json.load(secrets_file)

    api_token = request_access_token(secrets)

    print(api_token)

    with open('data/output.json', mode='w+') as output_file:
        output = output_file.read()
        # removes comma and whitespace at end of file
        # by wrapping file in curly braces we make it a single json obj

        output_json = json.loads(output)
        past_ids = set(output_json.keys())

        with open('data/playlist_ids.txt', mode='r') as input_file:
            ids = set(input_file.read().splitlines())

            print(len(ids))

            return

            num_playlists = len(ids)

            # remove playlists we've already extracted
            ids = list(ids - past_ids)

            offset = len(past_ids)  # how many we've done so far

            with alive_bar(num_playlists) as bar:
                bar(offset, skipped=True)
                for idx, playlist_id in enumerate(ids):
                    api_req = 'https://api.spotify.com/v1/playlists/' + \
                        playlist_id + '/tracks?limit=50&offset=0'
                    try:
                        tracks = extract_tracks(api_req, api_token)
                        output_json[playlist_id] = tracks
                        bar()
                    except FatalError as e:
                        output_file.write(json.dumps(output_json))
                        with open('log.txt', mode='a') as log_file:
                            log_file.write(
                                f"{e.message}. stopped at playlist {idx + offset}\n")

                        break
                    except NonFatalError as e:
                        with open('log.txt', mode='a') as log_file:
                            log_file.write(
                                f"playlist {idx + offset} {e}\n")
                        bar()
                        continue

        output_file.write(json.dumps(output_json))


if __name__ == '__main__':
    main()
