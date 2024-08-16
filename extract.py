# i'm going to assume that trying to do genre classification does not violate
# spotify developer terms section IV 2.a.i:

''' Misuse of the Spotify Platform. Do not misuse the Spotify Platform, including by

    using the Spotify Platform or any Spotify Content to train a machine learning or AI model or otherwise ingesting Spotify Content into a machine learning or AI model;
'''

from alive_progress import alive_bar
import requests

import json

with open('secrets.json', mode='r') as secrets_file:
    secrets = json.load(secrets_file)

search_requests: list[str] = [
    'https://api.spotify.com/v1/search?q=liked+songs&type=playlist&market=US&limit=50&offset=777',
    'https://api.spotify.com/v1/search?q=favorite+songs&type=playlist&market=US&limit=50&offset=0',
    'https://api.spotify.com/v1/search?q=best+songs&type=playlist&market=US&limit=50&offset=0']


def request_access_token():
    response = requests.post("https://accounts.spotify.com/api/token",
                             headers={'Content-Type': 'application/x-www-form-urlencoded'}, data={'grant_type': 'client_credentials', 'client_id': secrets['client_id'], 'client_secret': secrets['client_secret']})
    # note this info is secret and I should not upload this repo with it visible
    if (response.status_code == '429'):
        raise Exception(
            f"Being Rate Limited")

    if (response):  # automatically checks error code, e.g. 200 -> true, 401 -> false

        return response.json()['access_token']
    else:
        raise Exception(
            f"Access Token Request; Error Code: {response.status_code}")


def make_http_request(url: str, access_token: str):
    response = requests.get(
        url, headers={'Authorization': 'Bearer ' + access_token})
    if (response):  # automatically checks error code, e.g. 200 -> true, 401 -> false
        return response.json()
    else:
        raise Exception(f"HTTP Get: Error code: {response.status_code}")


# returns list of jsons containing track_id, name, and artist


def extract_tracks(api_request: str, access_token: str):
    # consider replacing this with generator
    api_response = make_http_request(api_request, access_token)
    tracks = []

    for item in api_response['items']:
        track = item['track']
        if (track is None):
            continue

        track_item = {}

        # track_item['artists'] = [
        #     [artist['name'] for artist in (track['artists'] or [])]
        #     if ('artists' in track)
        #     else []
        # ]
        # list compression here is less readable

        track_item['artists'] = []

        if 'artists' in track:
            for artist in track['artists']:
                # if ('name' in artist):
                # this above line may not be necessary
                track_item['artists'].append(artist['name'])

        # concatenates artist names together into string
        track_item['title'] = track['name']
        track_item['id'] = track['id']

        tracks.append(track_item)

    return tracks

# there is an average of about 9 kb of data per playlist


def extract_playlists(api_request: str, access_token: str, output_file, max_playlists=3, offset=0):

    playlists: dict[str, list] = {}
    next_page: str = api_request

    num_playlists = 0

    # iterate through pages of search result
    with alive_bar(max_playlists) as bar:
        while (next_page != 'None'):

            # request current page
            api_response = make_http_request(next_page, access_token)
            current_page = api_response['playlists']['items']
            next_page = api_response['playlists']['next']

            if (num_playlists < offset):  # goes to next page without adding any playlists
                continue

            # process all playlists on current page
            for playlist in current_page:
                playlist_id = playlist['id']
                # api url for tracks in playlist
                tracks_url = playlist['tracks']['href']

                output_file.write('{')

                output_file.write(json.dumps(extract_tracks(
                    tracks_url, access_token)))

                output_file.write('}, ')

            num_playlists += 1
            bar()

            if (num_playlists + offset >= max_playlists):
                break

            output_file.write(json.dumps(playlists))

    print("Output written")

    return playlists


def main():
    api_token = 'BQDCHNkep75VdUpdPX1T1AmsPtQNWt5YdanFhBIdKPRi5UdyT_Q8uR2RXrh25QsfgV7XJQH0AJ73-mKXGKCy1SMHoy6k50iCxx0yVeiAzJk6neG1mio'

    print(api_token)

    with open('output.txt', mode='w') as output_file:
        output_file.write('[')
        for search_request in search_requests[:1]:
            extract_playlists(search_request, api_token, output_file)
        output_file.write(']')


if __name__ == '__main__':
    print(main())
