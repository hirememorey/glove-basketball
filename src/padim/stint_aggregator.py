"""Stint aggregation module for RAPM modeling."""

import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from nba_api.stats.endpoints import GameRotation, PlayByPlayV3, BoxScoreHustleV2, BoxScoreSummaryV2
from .db.database import DatabaseConnection

logger = logging.getLogger(__name__)

class StintAggregator:
    """Aggregates defensive statistics by stint for RAPM modeling."""

    def __init__(self):
        """Initialize the stint aggregator."""
        self.db = DatabaseConnection()

    def aggregate_game_stints(self, game_id: str) -> Dict[str, any]:
        """Aggregate stints for a single game.

        Args:
            game_id: NBA game ID (e.g., '0022200001')

        Returns:
            Dict containing stint data and metadata
        """
        try:
            logger.info(f"Aggregating stints for game {game_id}")

            # Get rotation data for both teams
            logger.info("Getting rotation data...")
            rotation_data = self._get_game_rotations(game_id)
            if not rotation_data:
                return {'error': 'Could not retrieve rotation data'}

            # Get play-by-play data for defensive stats
            logger.info("Getting play-by-play data...")
            pbp_data = self._get_play_by_play_data(game_id)
            if pbp_data is None:
                return {'error': 'Could not retrieve play-by-play data'}

            # Identify all stint boundaries (substitution time points)
            logger.info("Finding stint boundaries...")
            stint_boundaries = self._find_stint_boundaries(rotation_data)
            logger.info(f"Found {len(stint_boundaries)} stint boundaries")

            # Aggregate stints with defensive statistics
            logger.info("Aggregating stint stats...")
            stints = self._aggregate_stint_stats(game_id, stint_boundaries, rotation_data, pbp_data)

            result = {
                'game_id': game_id,
                'stints': stints,
                'total_stints': len(stints),
                'rotation_data': rotation_data,
                'stint_boundaries': stint_boundaries
            }

            logger.info(f"Successfully aggregated {len(stints)} stints for game {game_id}")
            return result

        except Exception as e:
            logger.error(f"Error aggregating stints for game {game_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}

    def _get_game_rotations(self, game_id: str) -> Optional[List[pd.DataFrame]]:
        """Get rotation data for both teams in a game."""
        try:
            endpoint = GameRotation(game_id=game_id)
            dfs = endpoint.get_data_frames()

            if len(dfs) != 2:
                logger.warning(f"Expected 2 teams, got {len(dfs)} for game {game_id}")
                return None

            # Validate data structure
            for i, df in enumerate(dfs):
                required_cols = ['PERSON_ID', 'IN_TIME_REAL', 'OUT_TIME_REAL']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.error(f"Missing columns {missing_cols} in team {i+1} rotation data")
                    return None

            return dfs

        except Exception as e:
            logger.error(f"Error getting rotation data for game {game_id}: {e}")
            return None

    def _get_play_by_play_data(self, game_id: str) -> Optional[pd.DataFrame]:
        """Get play-by-play data for defensive statistics."""
        try:
            endpoint = PlayByPlayV3(game_id=game_id)
            df = endpoint.get_data_frames()[0]

            # Filter to only opponent field goal attempts for defensive stats
            # This is a simplified approach - in production we'd want more sophisticated filtering
            return df

        except Exception as e:
            logger.error(f"Error getting play-by-play data for game {game_id}: {e}")
            return None

    def _find_stint_boundaries(self, rotation_data: List[pd.DataFrame]) -> List[float]:
        """Find all time points where substitutions occur across both teams."""
        all_times = set()

        for team_df in rotation_data:
            for _, player in team_df.iterrows():
                if pd.notna(player['IN_TIME_REAL']):
                    all_times.add(player['IN_TIME_REAL'])
                if pd.notna(player['OUT_TIME_REAL']):
                    all_times.add(player['OUT_TIME_REAL'])

        # Sort and return unique time boundaries
        return sorted(list(all_times))

    def _get_lineup_at_time(self, time_seconds: float, rotation_data: List[pd.DataFrame]) -> Dict[str, List[int]]:
        """Get the player lineups for both teams at a specific time, resolving substitution overlaps."""
        lineups = {'home': [], 'away': []}

        for team_idx, team_df in enumerate(rotation_data):
            team_key = 'home' if team_idx == 0 else 'away'
            team_lineup = self._resolve_team_lineup_at_time(team_df, time_seconds)
            lineups[team_key] = team_lineup

        return lineups

    def _resolve_team_lineup_at_time(self, team_df: pd.DataFrame, time_seconds: float) -> List[int]:
        """Resolve the correct 5 players on court for a team at a specific time, handling substitution overlaps."""
        # Get all players who could potentially be on court
        potential_players = []
        for _, player in team_df.iterrows():
            in_time = player['IN_TIME_REAL']
            out_time = player['OUT_TIME_REAL']

            if pd.notna(in_time) and pd.notna(out_time):
                in_time = float(in_time)
                out_time = float(out_time)

                # Check if player is active at this time
                if in_time <= time_seconds <= out_time:
                    potential_players.append({
                        'player_id': int(player['PERSON_ID']),
                        'in_time': in_time,
                        'out_time': out_time,
                        'is_entering': in_time == time_seconds,
                        'is_exiting': out_time == time_seconds
                    })

        # If we have exactly 5 players, return them
        if len(potential_players) == 5:
            return [p['player_id'] for p in potential_players]

        # If we have more than 5, we need to resolve overlaps
        if len(potential_players) > 5:
            return self._resolve_overlapping_substitutions(potential_players, time_seconds)

        # If we have fewer than 5, something is wrong with the data
        logger.warning(f"Only {len(potential_players)} players found at time {time_seconds}, expected 5")
        return [p['player_id'] for p in potential_players]

    def _resolve_overlapping_substitutions(self, players: List[Dict], time_seconds: float) -> List[int]:
        """Resolve overlapping substitutions by prioritizing entering players over exiting players."""
        entering_players = [p for p in players if p['is_entering']]
        exiting_players = [p for p in players if p['is_exiting']]
        continuing_players = [p for p in players if not p['is_entering'] and not p['is_exiting']]

        # Start with continuing players (definitely on court)
        lineup = continuing_players.copy()

        # Add entering players (they're coming on)
        lineup.extend(entering_players)

        # If we still need more players, add non-exiting players
        # (this handles edge cases where the logic above doesn't give us exactly 5)
        if len(lineup) < 5:
            remaining_players = [p for p in players if p not in lineup and not p['is_exiting']]
            lineup.extend(remaining_players[:5 - len(lineup)])

        # Take only the first 5 players
        resolved_lineup = lineup[:5]

        if len(resolved_lineup) != 5:
            logger.warning(f"Could not resolve to exactly 5 players at time {time_seconds}. Got {len(resolved_lineup)}")

        return [p['player_id'] for p in resolved_lineup]

    def _aggregate_stint_stats(self, game_id: str, stint_boundaries: List[float],
                             rotation_data: List[pd.DataFrame], pbp_data: pd.DataFrame) -> List[Dict]:
        """Aggregate defensive statistics for each stint."""
        stints = []

        # Process each time interval between boundaries
        for i in range(len(stint_boundaries) - 1):
            start_time = stint_boundaries[i]
            end_time = stint_boundaries[i + 1]

            # Skip stints that are too short (less than 30 seconds)
            if end_time - start_time < 30:
                continue

            # Get lineups at the start of the stint
            lineups = self._get_lineup_at_time(start_time, rotation_data)

            # Only process stints with proper lineups (5 players per team)
            if len(lineups['home']) != 5 or len(lineups['away']) != 5:
                continue

            # Calculate defensive statistics for this stint
            stint_stats = self._calculate_stint_defensive_stats(
                game_id, start_time, end_time, lineups, pbp_data
            )

            if stint_stats:
                stint_data = {
                    'game_id': game_id,
                    'stint_start': start_time,
                    'stint_end': end_time,
                    'duration': end_time - start_time,
                    'home_players': lineups['home'],
                    'away_players': lineups['away'],
                    **stint_stats
                }
                stints.append(stint_data)

        return stints

    def _calculate_stint_defensive_stats(self, game_id: str, start_time: float, end_time: float,
                                       lineups: Dict[str, List[int]], pbp_data: pd.DataFrame) -> Optional[Dict]:
        """Calculate defensive statistics for a stint."""
        try:
            # Convert stint time range to PBP-compatible format
            stint_events = self._filter_pbp_events_to_stint(pbp_data, start_time, end_time)

            # Get opponent shooting data during this stint
            home_defensive_stats = self._calculate_opponent_shooting_stats(
                game_id, stint_events, 'home', lineups
            )
            away_defensive_stats = self._calculate_opponent_shooting_stats(
                game_id, stint_events, 'away', lineups
            )

            return {
                'home_defensive_stats': home_defensive_stats,
                'away_defensive_stats': away_defensive_stats
            }

        except Exception as e:
            logger.error(f"Error calculating stint stats: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _filter_pbp_events_to_stint(self, pbp_data: pd.DataFrame, start_time: float, end_time: float) -> pd.DataFrame:
        """Filter PBP events to those occurring within the stint time range."""
        try:
            # Convert PBP clock format to total seconds
            pbp_seconds = []
            for _, event in pbp_data.iterrows():
                period = event.get('period', 1)
                clock = event.get('clock', '12:00')
                total_seconds = self._convert_pbp_clock_to_seconds(period, clock)
                pbp_seconds.append(total_seconds)

            pbp_data = pbp_data.copy()
            pbp_data['total_seconds'] = pbp_seconds

            # Filter to events within stint time range
            mask = (pbp_data['total_seconds'] >= start_time) & (pbp_data['total_seconds'] <= end_time)
            return pbp_data[mask]

        except Exception as e:
            logger.error(f"Error filtering PBP events: {e}")
            return pd.DataFrame()

    def _convert_pbp_clock_to_seconds(self, period: int, clock: str) -> float:
        """Convert PBP period + clock format to total seconds from game start."""
        try:
            if not clock or clock == '':
                return 0.0

            # Handle different clock formats (e.g., "PT12M00.00S" or "12:00")
            if 'PT' in clock:
                # ISO 8601 duration format: PT12M00.00S
                clock_clean = clock.replace('PT', '').replace('S', '')
                if 'M' in clock_clean:
                    parts = clock_clean.split('M')
                    minutes = int(float(parts[0]))
                    seconds = float(parts[1]) if len(parts) > 1 else 0.0
                else:
                    minutes = 0
                    seconds = float(clock_clean)
            else:
                # MM:SS format
                if ':' in clock:
                    parts = clock.split(':')
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                else:
                    minutes = 0
                    seconds = float(clock)

            # Time remaining in this period
            time_remaining = minutes * 60 + seconds

            # Convert to total game seconds elapsed
            base_period_length = 720  # 12 minutes
            ot_period_length = 300    # 5 minutes

            if period <= 4:
                periods_completed = period - 1
                time_elapsed_this_period = base_period_length - time_remaining
                total_seconds = periods_completed * base_period_length + time_elapsed_this_period
            else:
                regular_time_elapsed = 4 * base_period_length
                ot_periods_completed = period - 5  # period 5 is first OT
                time_elapsed_this_ot = ot_period_length - time_remaining
                total_seconds = regular_time_elapsed + ot_periods_completed * ot_period_length + time_elapsed_this_ot

            return total_seconds

        except (ValueError, IndexError, AttributeError) as e:
            logger.warning(f"Could not parse clock '{clock}' for period {period}: {e}")
            return 0.0

    def _calculate_opponent_shooting_stats(self, game_id: str, stint_events: pd.DataFrame, team_key: str,
                                         lineups: Dict[str, List[int]]) -> Dict[str, int]:
        """Calculate opponent shooting statistics during the stint."""
        try:
            stats = {
                'opponent_fga': 0,
                'opponent_fgm': 0,
                'opponent_fg3a': 0,
                'opponent_fg3m': 0,
                'opponent_rim_attempts': 0,
            }

            if stint_events.empty:
                return stats

            # Get team IDs for this game from the stint events
            team_ids = stint_events['teamId'].unique()
            team_ids = [tid for tid in team_ids if tid != 0]  # Remove null team ID

            if len(team_ids) != 2:
                logger.warning(f"Expected 2 teams, found {len(team_ids)} in stint events")
                return stats

            # Get proper home/away team mapping from game data
            home_team_id, away_team_id = self._get_home_away_team_ids(game_id, team_ids)

            if not home_team_id or not away_team_id:
                logger.warning(f"Could not determine home/away teams for game {game_id}")
                return stats

            # Filter to field goal attempts (both made and missed)
            fg_events = stint_events[stint_events['isFieldGoal'] == 1].copy()

            if fg_events.empty:
                return stats

            if team_key == 'home':
                opponent_team_id = away_team_id
            else:  # team_key == 'away'
                opponent_team_id = home_team_id

            # Filter to shots by the opposing team
            opponent_shots = fg_events[fg_events['teamId'] == opponent_team_id]

            if opponent_shots.empty:
                return stats

            # Calculate basic shooting stats
            made_shots = opponent_shots[opponent_shots['shotResult'] == 'Made']
            missed_shots = opponent_shots[opponent_shots['shotResult'] == 'Missed']

            stats['opponent_fga'] = len(opponent_shots)
            stats['opponent_fgm'] = len(made_shots)

            # Identify 3-point attempts (shots with value 3 or distance > 22 feet)
            three_pointers = opponent_shots[
                (opponent_shots['shotValue'] == 3) |
                (opponent_shots['shotDistance'] >= 22)
            ]
            three_pointers_made = three_pointers[three_pointers['shotResult'] == 'Made']

            stats['opponent_fg3a'] = len(three_pointers)
            stats['opponent_fg3m'] = len(three_pointers_made)

            # Identify rim attempts (shots within 4 feet of basket)
            rim_attempts = opponent_shots[opponent_shots['shotDistance'] <= 4]
            stats['opponent_rim_attempts'] = len(rim_attempts)

            return stats

        except Exception as e:
            logger.error(f"Error calculating opponent shooting stats: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'opponent_fga': 0,
                'opponent_fgm': 0,
                'opponent_fg3a': 0,
                'opponent_fg3m': 0,
                'opponent_rim_attempts': 0,
            }

    def _get_home_away_team_ids(self, game_id: str, team_ids: List[int]) -> Tuple[Optional[int], Optional[int]]:
        """Get the correct home and away team IDs for a game."""
        try:
            # Get game summary to identify home/away teams
            summary = BoxScoreSummaryV2(game_id=game_id)
            game_summary = summary.get_data_frames()[0]

            if game_summary.empty:
                logger.warning(f"No game summary found for {game_id}")
                return None, None

            home_team_id = int(game_summary['HOME_TEAM_ID'].iloc[0])
            away_team_id = int(game_summary['VISITOR_TEAM_ID'].iloc[0])

            # Validate that these match the team IDs we found in PBP
            if home_team_id not in team_ids or away_team_id not in team_ids:
                logger.warning(f"Game summary team IDs {home_team_id}, {away_team_id} don't match PBP team IDs {team_ids}")
                return None, None

            return home_team_id, away_team_id

        except Exception as e:
            logger.error(f"Error getting home/away team IDs for game {game_id}: {e}")
            return None, None

    def save_stints_to_db(self, stint_data: Dict) -> bool:
        """Save aggregated stint data to database."""
        try:
            if 'error' in stint_data:
                logger.error(f"Cannot save stints with error: {stint_data['error']}")
                return False

            game_id = stint_data['game_id']
            stints = stint_data['stints']

            logger.info(f"Saving {len(stints)} stints for game {game_id}")

            # Create stints table if it doesn't exist
            self._create_stints_table()

            # Save each stint
            for stint in stints:
                self._save_single_stint(stint)

            self.db.commit()
            logger.info(f"Successfully saved {len(stints)} stints for game {game_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving stints to database: {e}")
            return False

    def _create_stints_table(self):
        """Create the stints table if it doesn't exist."""
        create_sql = '''
            CREATE TABLE IF NOT EXISTS stints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                stint_start REAL NOT NULL,
                stint_end REAL NOT NULL,
                duration REAL NOT NULL,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_player_1 INTEGER,
                home_player_2 INTEGER,
                home_player_3 INTEGER,
                home_player_4 INTEGER,
                home_player_5 INTEGER,
                away_player_1 INTEGER,
                away_player_2 INTEGER,
                away_player_3 INTEGER,
                away_player_4 INTEGER,
                away_player_5 INTEGER,
                home_opp_fga INTEGER,
                home_opp_fgm INTEGER,
                home_opp_fg3a INTEGER,
                home_opp_fg3m INTEGER,
                home_opp_rim_attempts INTEGER,
                away_opp_fga INTEGER,
                away_opp_fgm INTEGER,
                away_opp_fg3a INTEGER,
                away_opp_fg3m INTEGER,
                away_opp_rim_attempts INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_id, stint_start)
            )
        '''
        self.db.execute(create_sql)

    def _save_single_stint(self, stint: Dict):
        """Save a single stint to the database."""
        # Extract team IDs from player data (simplified)
        # In production, we'd look up team IDs properly

        logger.info(f"Saving stint: {len(stint['home_players'])} home players, {len(stint['away_players'])} away players")

        insert_sql = '''
            INSERT OR REPLACE INTO stints
            (game_id, stint_start, stint_end, duration,
             home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
             away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
             home_opp_fga, home_opp_fgm, home_opp_fg3a, home_opp_fg3m, home_opp_rim_attempts,
             away_opp_fga, away_opp_fgm, away_opp_fg3a, away_opp_fg3m, away_opp_rim_attempts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        home_players = stint['home_players'][:5]
        away_players = stint['away_players'][:5]

        logger.info(f"Home players: {home_players} (len: {len(home_players)})")
        logger.info(f"Away players: {away_players} (len: {len(away_players)})")

        values = (
            stint['game_id'],
            stint['stint_start'],
            stint['stint_end'],
            stint['duration'],
            *home_players,  # Should be 5 values
            *away_players,  # Should be 5 values
            stint['home_defensive_stats']['opponent_fga'],
            stint['home_defensive_stats']['opponent_fgm'],
            stint['home_defensive_stats']['opponent_fg3a'],
            stint['home_defensive_stats']['opponent_fg3m'],
            stint['home_defensive_stats']['opponent_rim_attempts'],
            stint['away_defensive_stats']['opponent_fga'],
            stint['away_defensive_stats']['opponent_fgm'],
            stint['away_defensive_stats']['opponent_fg3a'],
            stint['away_defensive_stats']['opponent_fg3m'],
            stint['away_defensive_stats']['opponent_rim_attempts']
        )

        placeholders_count = insert_sql.count('?')
        logger.info(f"Values tuple length: {len(values)}, Placeholders: {placeholders_count}")
        logger.info(f"Values: {list(values)}")
        self.db.execute(insert_sql, values)

    def close(self):
        """Close database connection."""
        self.db.close()
