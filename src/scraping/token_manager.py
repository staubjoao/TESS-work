import time
from dotenv import load_dotenv

class TokenManager:
    """
    This class is responsible for managing multiple tokens to interact with GitHub's API. It handles
    rate-limiting and token switching to ensure continuous access when rate limits are reached.
    """
    
    def __init__(self, tokens):
        """
        Initializes the TokenManager with a list of tokens.

        Args:
            tokens (list): A list of GitHub API tokens that will be used to authenticate requests.
        """
        self.tokens = tokens  # List of tokens for GitHub API access
        self.index = 0  # Index to keep track of the current token being used
        self.rate_limits = [{} for _ in tokens]  # Rate limits for each token (initialized as empty dictionaries)
        self.current_token = tokens[self.index]  # Set the first token as the current token

    def get_headers(self):
        """
        Returns the headers to be used in the GitHub API requests, including the current token for authorization.

        Returns:
            dict: The headers dictionary with authorization and API version information.
        """
        return {
            "Accept": "application/vnd.github+json",  # GitHub API accepts this content type
            "Authorization": f"Bearer {self.current_token}",  # Bearer token for authentication
            "X-GitHub-Api-Version": "2022-11-28",  # The API version being used
        }

    def update_rate_limit(self, response_headers):
        """
        Updates the rate limit information for the current token based on the response headers.

        Args:
            response_headers (dict): The response headers from the GitHub API call containing rate limit information.
        """
        # Extract the remaining rate limit and reset time from the response headers
        self.rate_limits[self.index] = {
            "remaining": int(response_headers.get("X-RateLimit-Remaining", 0)),
            "reset": int(response_headers.get("X-RateLimit-Reset", 0)),
        }

    def switch_token(self):
        """
        Switches to the next token in the list, updating the current token and rate limit information.
        """
        # Increment the index to switch to the next token (circular)
        self.index = (self.index + 1) % len(self.tokens)
        self.current_token = self.tokens[self.index]  # Set the new token as the current one
        print(f"Switched to token {self.index + 1}")  # Inform the user about the token switch
        self.print_rate_limit()  # Print the rate limit for the new token

    def wait_for_rate_limit_reset(self):
        """
        Waits for the rate limit to reset by sleeping until the earliest reset time across all tokens.
        """
        # Find the earliest reset time among all tokens' rate limits
        reset_times = [rl.get("reset", int(time.time()) + 3600) for rl in self.rate_limits]
        earliest_reset = min(reset_times)
        
        # Calculate the amount of time to sleep until the earliest reset
        sleep_time = max(earliest_reset - int(time.time()), 0) + 10  # Add 10 seconds buffer
        print(f"All tokens rate limited. Waiting for {sleep_time} seconds.")  # Inform the user of the wait time
        time.sleep(sleep_time)  # Sleep for the calculated time

        # After waiting, reset to the first token
        self.index = 0
        self.current_token = self.tokens[self.index]
        print("Resuming with first token after waiting.")  # Inform the user that waiting is over
        self.print_rate_limit()  # Print the rate limit for the first token

    def print_rate_limit(self):
        """
        Prints the current rate limit information for the active token, including remaining requests and reset time.
        """
        # Get the remaining rate limit and reset time for the current token
        remaining = self.rate_limits[self.index].get("remaining", "Unknown")
        reset = self.rate_limits[self.index].get("reset", "Unknown")
        
        # Format the reset time as a human-readable string if it is available
        reset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reset)) if reset != "Unknown" else "Unknown"
        
        # Display the rate limit information for the current token
        print(f"Current Token: {self.index + 1} | Remaining Rate Limit: {remaining} | Rate Limit Resets At: {reset_time}")
