from alive_progress import alive_bar
import requests

import json


class FatalError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.message = message
        self.code = code


class NonFatalError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.message = message
        self.code = code


def handle_response(response):
    if (response.status_code == 429):
        raise FatalError(
            f"Api Failure: Error Code: {response.status_code}", response.status_code)
    elif (response.status_code == 401):
        raise FatalError(
            f"Unauthorized: Error Code: {response.status_code}", response.status_code)

    if (response):  # automatically checks error code, e.g. 200 -> true, 401 -> false
        return response.json()
    else:
        raise NonFatalError(
            f"HTTP Error: Error Code: {response.status_code}", response.status_code)


def request_access_token(secrets):
    URL = "https://accounts.spotify.com/api/token"
    response = requests.post(URL,
                             headers={'Content-Type': 'application/x-www-form-urlencoded'}, data={'grant_type': 'client_credentials', 'client_id': secrets['client_id'], 'client_secret': secrets['client_secret']})
    # note this info is secret and I should not upload this repo with it visible
    data = handle_response(response)
    return data['access_token']


def make_http_request(url: str, access_token: str):
    response = requests.get(
        url, headers={'Authorization': 'Bearer ' + access_token})

    data = handle_response(response)
    return data
