"""NBA Stats API client for PADIM."""

import time
import random
import logging
import requests
from typing import Dict, Any, Optional, List
from ..config.settings import (
    API_BASE_URL,
    ENDPOINTS,
    MAX_RETRIES,
    API_TIMEOUT,
    MIN_SLEEP,
    MAX_SLEEP,
    RETRY_BACKOFF_BASE,
    RETRY_BACKOFF_FACTOR,
    MAX_BACKOFF_SLEEP,
    BASE_HEADERS,
    USER_AGENTS
)

class NBAStatsClient:
    """Client for making requests to the NBA Stats API for PADIM data collection."""

    def __init__(self):
        """Initialize the NBA Stats API client."""
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Host": "stats.nba.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Connection": "keep-alive",
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site"
        })

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API request with random user agent."""
        headers = self.session.headers.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        return headers

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Handle rate limiting by waiting for the specified time."""
        if response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get("Retry-After", 30))
            logging.warning(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff time."""
        backoff = min(
            RETRY_BACKOFF_BASE * (RETRY_BACKOFF_FACTOR ** attempt),
            MAX_BACKOFF_SLEEP
        )
        return backoff + random.uniform(MIN_SLEEP, MAX_SLEEP)

    def make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_retries: int = MAX_RETRIES
    ) -> Optional[Dict[str, Any]]:
        """Make a request to the NBA Stats API with retries and backoff.

        Args:
            endpoint: API endpoint to call
            params: Query parameters for the request
            max_retries: Maximum number of retry attempts

        Returns:
            Optional[Dict[str, Any]]: JSON response data or None if request fails

        Raises:
            requests.exceptions.RequestException: If all retry attempts fail
        """
        url = f"{API_BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            try:
                # Add random delay between requests
                time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

                response = self.session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=API_TIMEOUT
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    self._handle_rate_limit(response)
                    continue
                else:
                    logging.error(
                        f"API request failed with status {response.status_code}"
                    )

                # Calculate and apply backoff
                if attempt < max_retries - 1:
                    backoff = self._calculate_backoff(attempt)
                    logging.warning(f"Retrying in {backoff:.2f} seconds...")
                    time.sleep(backoff)

            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    backoff = self._calculate_backoff(attempt)
                    logging.warning(f"Retrying in {backoff:.2f} seconds...")
                    time.sleep(backoff)
                else:
                    raise

        return None

    def get_play_by_play(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get play-by-play data for stint aggregation.

        Args:
            game_id: NBA game ID

        Returns:
            Optional[Dict[str, Any]]: Play-by-play data or None if request fails
        """
        params = {
            "GameID": game_id,
            "StartPeriod": 1,
            "EndPeriod": 10,
            "StartRange": 0,
            "EndRange": 55800,
            "RangeType": 2
        }

        return self.make_request(ENDPOINTS["play_by_play"], params)

    def get_shot_chart_data(self, player_id: int, season: str, season_type: str = "Regular Season") -> Optional[Dict[str, Any]]:
        """Get shot chart data for rim frequency analysis.

        Args:
            player_id: NBA player ID
            season: Season ID (e.g., "2024-25")
            season_type: Type of season (Regular Season, Playoffs, etc.)

        Returns:
            Optional[Dict[str, Any]]: Shot chart data or None if request fails
        """
        params = {
            "PlayerID": player_id,
            "Season": season,
            "SeasonType": season_type,
            "LeagueID": "00",
            "ContextMeasure": "FGA"
        }

        return self.make_request(ENDPOINTS["shot_chart"], params)

    def get_player_hustle_stats(self, player_id: int, season: str, season_type: str = "Regular Season") -> Optional[Dict[str, Any]]:
        """Get player hustle statistics (steals, blocks, charges).

        Args:
            player_id: NBA player ID
            season: Season ID (e.g., "2024-25")
            season_type: Type of season (Regular Season, Playoffs, etc.)

        Returns:
            Optional[Dict[str, Any]]: Player hustle statistics or None if request fails
        """
        params = {
            "PlayerID": player_id,
            "Season": season,
            "SeasonType": season_type,
            "LeagueID": "00",
            "PerMode": "Totals"
        }

        return self.make_request(ENDPOINTS["player_hustle"], params)

    def get_player_tracking_stats(self, player_id: int, season: str, season_type: str = "Regular Season") -> Optional[Dict[str, Any]]:
        """Get player tracking statistics including defensive metrics.

        Args:
            player_id: NBA player ID
            season: Season ID (e.g., "2024-25")
            season_type: Type of season (Regular Season, Playoffs, etc.)

        Returns:
            Optional[Dict[str, Any]]: Player tracking statistics or None if request fails
        """
        params = {
            "PlayerID": player_id,
            "Season": season,
            "SeasonType": season_type,
            "LeagueID": "00",
            "PerMode": "PerGame"
        }

        return self.make_request(ENDPOINTS["player_tracking"], params)

    def get_player_shooting_splits(self, player_id: int, season: str, season_type: str = "Regular Season") -> Optional[Dict[str, Any]]:
        """Get player shooting splits by distance for rim frequency analysis.

        Args:
            player_id: NBA player ID
            season: Season ID (e.g., "2024-25")
            season_type: Type of season (Regular Season, Playoffs, etc.)

        Returns:
            Optional[Dict[str, Any]]: Player shooting splits or None if request fails
        """
        params = {
            "PlayerID": player_id,
            "Season": season,
            "SeasonType": season_type,
            "LeagueID": "00",
            "PerMode": "Totals"
        }

        return self.make_request(ENDPOINTS["shooting_splits"], params)

    def search_players(self, player_name: str) -> Optional[List[Dict[str, Any]]]:
        """Search for players by name.

        Args:
            player_name: Name to search for

        Returns:
            Optional[List[Dict[str, Any]]]: List of matching players or None if request fails
        """
        # Use the commonplayerinfo endpoint to search for players
        params = {
            "PlayerName": player_name,
            "LeagueID": "00"
        }

        response = self.make_request(ENDPOINTS.get("common_player_info", "/stats/commonplayerinfo"), params)

        if not response or "resultSets" not in response:
            return None

        # Extract player data from the response
        players = []
        for result_set in response["resultSets"]:
            if result_set.get("name") == "CommonPlayerInfo":
                headers = result_set.get("headers", [])
                rows = result_set.get("rowSet", [])

                for row in rows:
                    player_data = dict(zip(headers, row))
                    if player_data.get("PERSON_ID"):  # Only include players with valid IDs
                        players.append({
                            "id": player_data.get("PERSON_ID"),
                            "full_name": player_data.get("DISPLAY_FIRST_LAST"),
                            "first_name": player_data.get("FIRST_NAME"),
                            "last_name": player_data.get("LAST_NAME"),
                            "is_active": player_data.get("IS_ACTIVE", False)
                        })
                break

        return players
