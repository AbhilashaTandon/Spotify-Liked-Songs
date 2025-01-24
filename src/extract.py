

from alive_progress import alive_bar
import requests

import json

from urllib.error import HTTPError
from api_requests import make_http_request


# returns list of jsons containing track_id, name, and artist


def extract_tracks(api_request: str, access_token: str) -> list[dict]:
    # consider replacing this with generator
    api_response = make_http_request(api_request, access_token)
    tracks = []

    for item in api_response['items']:
        track = item['track']
        if (track is None):
            continue

        track_item = {}

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


def extract_playlists(api_request: str, access_token: str, output_file, max_playlists=1000):

    next_page: str = api_request  # api request for next page to be analyzed
    # starts with first page

    num_playlists = 0

    # iterate through pages of search result
    with alive_bar(max_playlists) as bar:
        while (not next_page is None):

            # request current page
            api_response = make_http_request(next_page, access_token)
            current_page = api_response['playlists']['items']
            next_page = api_response['playlists']['next']

            # process all playlists on current page
            for playlist in current_page:
                playlist_id = playlist['id']
                # api url for tracks in playlist
                output_file.write(playlist_id + '\n')
                bar()
                num_playlists += 1

            if (num_playlists >= max_playlists):
                break

    print(f'{num_playlists} found.')
