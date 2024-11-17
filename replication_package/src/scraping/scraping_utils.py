
import os
import requests
import json

def get_paginated_data(url, token_manager):
    """
    Fetches data from a paginated API endpoint, handling token switching and rate limits.

    Args:
        url (str): The API endpoint URL to start fetching data from.
        token_manager (TokenManager): The TokenManager object used for token management and rate limit handling.

    Returns:
        list: A list of data parsed from the paginated API responses.
    """
    data = []
    while url:
        headers = token_manager.get_headers()  # Get headers for the current token
        response = requests.get(url, headers=headers)  # Make the GET request
        if response.status_code == 200:
            response_data = response.json()  # Parse the JSON response
            data.extend(parse_data(response_data))  # Add parsed data to the list
            link_header = response.headers.get("Link")  # Check if there is a "Link" header for pagination
            if link_header and 'rel="next"' in link_header:
                next_url = [
                    url_part.split(";")[0].strip().strip("<>")  # Extract the next URL from the Link header
                    for url_part in link_header.split(",")
                    if 'rel="next"' in url_part
                ]
                url = next_url[0] if next_url else None  # Set the next URL for the next iteration
            else:
                url = None  # No next page, exit the loop
            token_manager.update_rate_limit(response.headers)  # Update rate limit information
            token_manager.print_rate_limit()  # Print the current rate limit
            if token_manager.rate_limits[token_manager.index].get("remaining", 1) <= 10:
                token_manager.switch_token()  # Switch to a new token if remaining requests are low
        else:
            if response.status_code == 403 and "rate limit exceeded" in response.text.lower():
                token_manager.update_rate_limit(response.headers)
                token_manager.switch_token()  # Switch token if rate limit exceeded
                if token_manager.index == 0:
                    token_manager.wait_for_rate_limit_reset()  # Wait for rate limit reset if all tokens are exhausted
                continue  # Retry with the new token
            else:
                print(f"Error: {response.status_code}, {response.text}")
                break
    return data

def parse_data(response_data):
    """
    Parses the API response data to ensure it is in list format.

    Args:
        response_data (dict or list): The data returned from the API response.

    Returns:
        list: A list of parsed data.
    """
    if isinstance(response_data, list):
        return response_data  # Return the list if already in list format
    elif isinstance(response_data, dict):
        return [response_data]  # Wrap the dict in a list
    return []  # Return an empty list if the data format is unrecognized

def save_data_to_json(data, filename="data.json"):
    """
    Saves the data to a JSON file.

    Args:
        data (list): The data to be saved.
        filename (str): The filename for the saved JSON file (default is "data.json").
    """
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)  # Save the data as formatted JSON