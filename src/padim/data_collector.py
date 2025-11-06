"""Data collection module for PADIM NBA data collection."""

import logging
import time
import random
from typing import Optional, Dict, Any, List
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguegamelog,
    playergamelogs,
    shotchartdetail,
    playercareerstats,
    LeagueHustleStatsPlayer,
    CommonTeamYears
)
from .db.database import DatabaseConnection

logger = logging.getLogger(__name__)

class NBADataCollector:
    """Collect NBA data for PADIM analysis."""

    def __init__(self):
        """Initialize the data collector."""
        self.db = DatabaseConnection()

    def collect_team_mappings(self) -> bool:
        """Collect team ID to abbreviation mappings.

        Returns:
            bool: Success status
        """
        try:
            logger.info("Collecting team mappings...")

            endpoint = CommonTeamYears()
            df = endpoint.get_data_frames()[0]

            # Filter for teams with abbreviations
            teams_with_abbrev = df[df['ABBREVIATION'].notna() & (df['ABBREVIATION'] != '')]

            team_data = []
            for _, row in teams_with_abbrev.iterrows():
                team_data.append({
                    'team_id': int(row['TEAM_ID']),
                    'abbreviation': row['ABBREVIATION'],
                    'nickname': row['ABBREVIATION'],  # Use abbreviation as name for now
                    'city': None
                })

            if team_data:
                self.db.executemany('''
                    INSERT OR REPLACE INTO teams (team_id, team_abbreviation, team_name, team_city)
                    VALUES (?, ?, ?, ?)
                ''', [(d['team_id'], d['abbreviation'], d['nickname'], d['city']) for d in team_data])

                self.db.commit()
                logger.info(f"Inserted {len(team_data)} team mappings")
                return True
            else:
                logger.warning("No team data to insert")
                return False

        except Exception as e:
            logger.error(f"Error collecting team mappings: {e}")
            return False

    def collect_player_hustle_stats(self, season: str = "2022-23") -> bool:
        """Collect hustle stats for all players in a season.

        Args:
            season: Season ID (e.g., "2022-23")

        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Collecting player hustle stats for {season}...")

            endpoint = LeagueHustleStatsPlayer(season=season)
            df = endpoint.get_data_frames()[0]

            hustle_data = []
            for _, row in df.iterrows():
                hustle_data.append({
                    'player_id': int(row['PLAYER_ID']),
                    'season_id': season,
                    'player_name': row.get('PLAYER_NAME'),
                    'team_abbreviation': row.get('TEAM_ABBREVIATION'),
                    'contested_shots': int(row['CONTESTED_SHOTS']) if pd.notna(row.get('CONTESTED_SHOTS')) else None,
                    'contested_shots_2pt': int(row['CONTESTED_SHOTS_2PT']) if pd.notna(row.get('CONTESTED_SHOTS_2PT')) else None,
                    'contested_shots_3pt': int(row['CONTESTED_SHOTS_3PT']) if pd.notna(row.get('CONTESTED_SHOTS_3PT')) else None,
                    'deflections': int(row['DEFLECTIONS']) if pd.notna(row.get('DEFLECTIONS')) else None,
                    'charges_drawn': int(row['CHARGES_DRAWN']) if pd.notna(row.get('CHARGES_DRAWN')) else None,
                    'screen_assists': int(row['SCREEN_ASSISTS']) if pd.notna(row.get('SCREEN_ASSISTS')) else None,
                    'screen_ast_pts': float(row['SCREEN_AST_PTS']) if pd.notna(row.get('SCREEN_AST_PTS')) else None,
                    'loose_balls_recovered_off': int(row['OFF_LOOSE_BALLS_RECOVERED']) if pd.notna(row.get('OFF_LOOSE_BALLS_RECOVERED')) else None,
                    'loose_balls_recovered_def': int(row['DEF_LOOSE_BALLS_RECOVERED']) if pd.notna(row.get('DEF_LOOSE_BALLS_RECOVERED')) else None,
                    'loose_balls_recovered': int(row['LOOSE_BALLS_RECOVERED']) if pd.notna(row.get('LOOSE_BALLS_RECOVERED')) else None,
                    'pct_loose_balls_recovered_off': float(row['PCT_LOOSE_BALLS_RECOVERED_OFF']) if pd.notna(row.get('PCT_LOOSE_BALLS_RECOVERED_OFF')) else None,
                    'pct_loose_balls_recovered_def': float(row['PCT_LOOSE_BALLS_RECOVERED_DEF']) if pd.notna(row.get('PCT_LOOSE_BALLS_RECOVERED_DEF')) else None,
                })

            if hustle_data:
                self.db.executemany('''
                    INSERT OR REPLACE INTO player_hustle_stats
                    (player_id, season_id, player_name, team_abbreviation, contested_shots,
                     contested_shots_2pt, contested_shots_3pt, deflections, charges_drawn,
                     screen_assists, screen_ast_pts, loose_balls_recovered_off,
                     loose_balls_recovered_def, loose_balls_recovered,
                     pct_loose_balls_recovered_off, pct_loose_balls_recovered_def)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [(d['player_id'], d['season_id'], d['player_name'], d['team_abbreviation'],
                       d['contested_shots'], d['contested_shots_2pt'], d['contested_shots_3pt'],
                       d['deflections'], d['charges_drawn'], d['screen_assists'], d['screen_ast_pts'],
                       d['loose_balls_recovered_off'], d['loose_balls_recovered_def'],
                       d['loose_balls_recovered'], d['pct_loose_balls_recovered_off'],
                       d['pct_loose_balls_recovered_def']) for d in hustle_data])

                self.db.commit()
                logger.info(f"Inserted {len(hustle_data)} player hustle records")
                return True
            else:
                logger.warning("No hustle data to insert")
                return False

        except Exception as e:
            logger.error(f"Error collecting hustle stats: {e}")
            return False

    def collect_player_season_stats(self, season: str = "2022-23") -> bool:
        """Collect season stats for all players in a season.

        Args:
            season: Season ID (e.g., "2022-23")

        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Collecting player season stats for {season}...")

            # Get season stats
            endpoint = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
            df = endpoint.get_data_frames()[0]

            # Prepare data for insertion
            season_data = []
            for _, row in df.iterrows():
                season_data.append({
                    'player_id': int(row['PLAYER_ID']),
                    'season_id': season,
                    'player_name': row['PLAYER_NAME'],
                    'team_abbreviation': row['TEAM_ABBREVIATION'],
                    'age': float(row['AGE']) if pd.notna(row['AGE']) else None,
                    'gp': int(row['GP']) if pd.notna(row['GP']) else None,
                    'w': int(row['W']) if pd.notna(row['W']) else None,
                    'l': int(row['L']) if pd.notna(row['L']) else None,
                    'w_pct': float(row['W_PCT']) if pd.notna(row['W_PCT']) else None,
                    'min': float(row['MIN']) if pd.notna(row['MIN']) else None,
                    'pts': float(row['PTS']) if pd.notna(row['PTS']) else None,
                    'reb': float(row['REB']) if pd.notna(row['REB']) else None,
                    'ast': float(row['AST']) if pd.notna(row['AST']) else None,
                    'stl': float(row['STL']) if pd.notna(row['STL']) else None,
                    'blk': float(row['BLK']) if pd.notna(row['BLK']) else None,
                    'fg_pct': float(row['FG_PCT']) if pd.notna(row['FG_PCT']) else None,
                    'fg3_pct': float(row['FG3_PCT']) if pd.notna(row['FG3_PCT']) else None,
                    'ft_pct': float(row['FT_PCT']) if pd.notna(row['FT_PCT']) else None,
                })

            # Insert data
            if season_data:
                self.db.executemany('''
                    INSERT OR REPLACE INTO player_season_stats
                    (player_id, season_id, player_name, team_abbreviation, age, gp, w, l, w_pct,
                     min, pts, reb, ast, stl, blk, fg_pct, fg3_pct, ft_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [(d['player_id'], d['season_id'], d['player_name'], d['team_abbreviation'],
                       d['age'], d['gp'], d['w'], d['l'], d['w_pct'], d['min'], d['pts'],
                       d['reb'], d['ast'], d['stl'], d['blk'], d['fg_pct'], d['fg3_pct'], d['ft_pct'])
                      for d in season_data])

                self.db.commit()
                logger.info(f"Inserted {len(season_data)} player season records")
                return True
            else:
                logger.warning("No season data to insert")
                return False

        except Exception as e:
            logger.error(f"Error collecting season stats: {e}")
            return False

    def collect_single_player_data(self, player_id: int, season: str = "2022-23") -> bool:
        """Collect all data for a single player.

        Args:
            player_id: NBA player ID
            season: Season ID

        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Collecting data for player {player_id} in {season}...")

            success = True

            # 1. Collect player game logs
            if not self._collect_player_game_logs(player_id, season):
                logger.warning(f"Failed to collect game logs for player {player_id}")
                success = False

            # 2. Collect player shot data
            if not self._collect_player_shots(player_id, season):
                logger.warning(f"Failed to collect shots for player {player_id}")
                success = False

            # 3. Collect player info (if not already in season stats)
            self._collect_player_info(player_id)

            return success

        except Exception as e:
            logger.error(f"Error collecting player data: {e}")
            return False

    def _collect_player_game_logs(self, player_id: int, season: str) -> bool:
        """Collect game-by-game stats for a player."""
        try:
            endpoint = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=season
            )
            df = endpoint.get_data_frames()[0]

            if df.empty:
                logger.warning(f"No game logs found for player {player_id}")
                return False

            # Prepare data for insertion
            game_data = []
            for _, row in df.iterrows():
                game_data.append({
                    'player_id': player_id,
                    'game_id': row['GAME_ID'],
                    'team_id': int(row['TEAM_ID']) if pd.notna(row['TEAM_ID']) else None,
                    'team_abbreviation': row['TEAM_ABBREVIATION'],
                    'matchup': row['MATCHUP'],
                    'game_date': row['GAME_DATE'],
                    'wl': row['WL'],
                    'min': float(row['MIN']) if pd.notna(row['MIN']) else None,
                    'pts': int(row['PTS']) if pd.notna(row['PTS']) else None,
                    'reb': int(row['REB']) if pd.notna(row['REB']) else None,
                    'ast': int(row['AST']) if pd.notna(row['AST']) else None,
                    'stl': int(row['STL']) if pd.notna(row['STL']) else None,
                    'blk': int(row['BLK']) if pd.notna(row['BLK']) else None,
                    'fg_pct': float(row['FG_PCT']) if pd.notna(row['FG_PCT']) else None,
                    'fg3_pct': float(row['FG3_PCT']) if pd.notna(row['FG3_PCT']) else None,
                    'ft_pct': float(row['FT_PCT']) if pd.notna(row['FT_PCT']) else None,
                })

            # Insert data
            if game_data:
                self.db.executemany('''
                    INSERT OR REPLACE INTO player_game_stats
                    (player_id, game_id, team_id, team_abbreviation, matchup, game_date, wl,
                     min, pts, reb, ast, stl, blk, fg_pct, fg3_pct, ft_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [(d['player_id'], d['game_id'], d['team_id'], d['team_abbreviation'],
                       d['matchup'], d['game_date'], d['wl'], d['min'], d['pts'], d['reb'],
                       d['ast'], d['stl'], d['blk'], d['fg_pct'], d['fg3_pct'], d['ft_pct'])
                      for d in game_data])

                self.db.commit()
                logger.info(f"Inserted {len(game_data)} game records for player {player_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error collecting game logs: {e}")
            return False

    def _collect_player_shots(self, player_id: int, season: str) -> bool:
        """Collect shot chart data for a player."""
        try:
            # First get player's team for the season
            self.db.execute("SELECT team_abbreviation FROM player_season_stats WHERE player_id = ? AND season_id = ?", (player_id, season))
            team_result = self.db.fetchone()
            if not team_result:
                logger.warning(f"No team found for player {player_id} in {season}")
                return False

            team_abbrev = team_result['team_abbreviation'] if team_result else None
            if not team_abbrev:
                logger.warning(f"No team abbreviation for player {player_id}")
                return False

            # Look up team ID from teams table
            self.db.execute("SELECT team_id FROM teams WHERE team_abbreviation = ?", (team_abbrev,))
            team_id_result = self.db.fetchone()
            team_id = team_id_result['team_id'] if team_id_result else None

            if not team_id:
                logger.warning(f"No team ID found for abbreviation {team_abbrev}")
                return False

            endpoint = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=player_id,
                season_nullable=season
            )
            df = endpoint.get_data_frames()[0]

            if df.empty:
                logger.warning(f"No shot data found for player {player_id}")
                return False

            # Prepare data for insertion
            shot_data = []
            for _, row in df.iterrows():
                shot_data.append({
                    'grid_type': row['GRID_TYPE'],
                    'game_id': row['GAME_ID'],
                    'game_event_id': int(row['GAME_EVENT_ID']) if pd.notna(row['GAME_EVENT_ID']) else None,
                    'player_id': player_id,
                    'player_name': row['PLAYER_NAME'],
                    'team_id': int(row['TEAM_ID']) if pd.notna(row['TEAM_ID']) else None,
                    'team_name': row['TEAM_NAME'],
                    'period': int(row['PERIOD']) if pd.notna(row['PERIOD']) else None,
                    'minutes_remaining': int(row['MINUTES_REMAINING']) if pd.notna(row['MINUTES_REMAINING']) else None,
                    'seconds_remaining': int(row['SECONDS_REMAINING']) if pd.notna(row['SECONDS_REMAINING']) else None,
                    'action_type': row['ACTION_TYPE'],
                    'shot_type': row['SHOT_TYPE'],
                    'shot_zone_basic': row['SHOT_ZONE_BASIC'],
                    'shot_zone_area': row['SHOT_ZONE_AREA'],
                    'shot_zone_range': row['SHOT_ZONE_RANGE'],
                    'shot_distance': float(row['SHOT_DISTANCE']) if pd.notna(row['SHOT_DISTANCE']) else None,
                    'loc_x': int(row['LOC_X']) if pd.notna(row['LOC_X']) else None,
                    'loc_y': int(row['LOC_Y']) if pd.notna(row['LOC_Y']) else None,
                    'shot_attempted_flag': int(row['SHOT_ATTEMPTED_FLAG']) if pd.notna(row['SHOT_ATTEMPTED_FLAG']) else None,
                    'shot_made_flag': int(row['SHOT_MADE_FLAG']) if pd.notna(row['SHOT_MADE_FLAG']) else None,
                })

            # Insert data
            if shot_data:
                self.db.executemany('''
                    INSERT OR IGNORE INTO shots
                    (grid_type, game_id, game_event_id, player_id, player_name, team_id, team_name,
                     period, minutes_remaining, seconds_remaining, action_type, shot_type,
                     shot_zone_basic, shot_zone_area, shot_zone_range, shot_distance, loc_x, loc_y,
                     shot_attempted_flag, shot_made_flag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [(d['grid_type'], d['game_id'], d['game_event_id'], d['player_id'], d['player_name'],
                       d['team_id'], d['team_name'], d['period'], d['minutes_remaining'], d['seconds_remaining'],
                       d['action_type'], d['shot_type'], d['shot_zone_basic'], d['shot_zone_area'],
                       d['shot_zone_range'], d['shot_distance'], d['loc_x'], d['loc_y'],
                       d['shot_attempted_flag'], d['shot_made_flag']) for d in shot_data])

                self.db.commit()
                logger.info(f"Inserted {len(shot_data)} shot records for player {player_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error collecting shots: {e}")
            return False

    def _collect_player_info(self, player_id: int) -> bool:
        """Collect basic player information."""
        try:
            # Get player info from season stats if available
            self.db.execute("SELECT * FROM player_season_stats WHERE player_id = ?", (player_id,))
            result = self.db.fetchone()
            if result:
                self.db.execute('''
                    INSERT OR REPLACE INTO players
                    (player_id, full_name, first_name, last_name, is_active)
                    VALUES (?, ?, ?, ?, ?)
                ''', (result['player_id'], result['player_name'], None, None, 1))
                return True

            return False

        except Exception as e:
            logger.error(f"Error collecting player info: {e}")
            return False

    def collect_multiple_players(self, player_ids: List[int], season: str = "2022-23",
                               batch_size: int = 5, delay_between_batches: float = 2.0) -> Dict[str, Any]:
        """Collect data for multiple players with batching and rate limiting.

        Args:
            player_ids: List of NBA player IDs to collect
            season: Season ID (e.g., "2022-23")
            batch_size: Number of players to process per batch
            delay_between_batches: Seconds to wait between batches

        Returns:
            Dict with success/failure counts and timing info
        """
        total_players = len(player_ids)
        successful = 0
        failed = 0
        start_time = time.time()

        logger.info(f"Starting batch collection for {total_players} players in {season}")

        # Process in batches
        for i in range(0, total_players, batch_size):
            batch = player_ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_players + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} players)")

            batch_success = 0
            batch_failed = 0

            for player_id in batch:
                try:
                    success = self.collect_single_player_data(player_id, season)
                    if success:
                        batch_success += 1
                        successful += 1
                    else:
                        batch_failed += 1
                        failed += 1

                    # Small delay between individual players
                    time.sleep(random.uniform(0.5, 1.5))

                except Exception as e:
                    logger.error(f"Failed to collect data for player {player_id}: {e}")
                    failed += 1
                    batch_failed += 1

            logger.info(f"Batch {batch_num} complete: {batch_success} success, {batch_failed} failed")

            # Delay between batches (except for the last batch)
            if i + batch_size < total_players:
                logger.info(f"Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)

        elapsed_time = time.time() - start_time

        results = {
            'total_players': total_players,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_players if total_players > 0 else 0,
            'elapsed_time': elapsed_time,
            'avg_time_per_player': elapsed_time / total_players if total_players > 0 else 0
        }

        logger.info(f"Batch collection complete: {successful}/{total_players} successful ({results['success_rate']:.1%}) in {elapsed_time:.1f}s")
        return results

    def collect_team_players(self, team_abbreviation: str, season: str = "2022-23",
                           min_minutes: int = 500) -> Dict[str, Any]:
        """Collect data for all players on a specific team above minimum minutes threshold.

        Args:
            team_abbreviation: Team abbreviation (e.g., "LAL")
            season: Season ID
            min_minutes: Minimum minutes played to include player

        Returns:
            Dict with collection results
        """
        try:
            logger.info(f"Collecting players for {team_abbreviation} in {season} (min {min_minutes} minutes)")

            # Get team players from season stats
            self.db.execute('''
                SELECT player_id, player_name, min
                FROM player_season_stats
                WHERE team_abbreviation = ? AND season_id = ? AND min >= ?
                ORDER BY min DESC
            ''', (team_abbreviation, season, min_minutes))

            team_players = self.db.fetchall()

            if not team_players:
                logger.warning(f"No players found for {team_abbreviation} in {season} with >= {min_minutes} minutes")
                return {'players_found': 0, 'collection_results': None}

            player_ids = [int(row['player_id']) for row in team_players]
            player_names = [row['player_name'] for row in team_players]

            logger.info(f"Found {len(player_ids)} players for {team_abbreviation}: {player_names[:3]}{'...' if len(player_names) > 3 else ''}")

            # Collect data for these players
            collection_results = self.collect_multiple_players(
                player_ids=player_ids,
                season=season,
                batch_size=3,  # Smaller batches for team collections
                delay_between_batches=1.0
            )

            return {
                'team': team_abbreviation,
                'players_found': len(player_ids),
                'player_names': player_names,
                'collection_results': collection_results
            }

        except Exception as e:
            logger.error(f"Error collecting team {team_abbreviation}: {e}")
            return {'error': str(e)}

    def get_collection_status(self) -> Dict[str, Any]:
        """Get current data collection status across all tables."""
        status = {}

        tables = {
            'player_season_stats': 'Season stats',
            'player_game_stats': 'Game-by-game stats',
            'shots': 'Shot chart data',
            'player_hustle_stats': 'Hustle stats',
            'teams': 'Team mappings'
        }

        for table, description in tables.items():
            try:
                self.db.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = self.db.fetchone()
                status[table] = {
                    'description': description,
                    'count': result['count'] if result else 0
                }
            except Exception as e:
                status[table] = {
                    'description': description,
                    'error': str(e)
                }

        # Add cross-table insights
        try:
            # Players with complete data
            self.db.execute('''
                SELECT COUNT(DISTINCT pss.player_id) as players_with_complete_data
                FROM player_season_stats pss
                JOIN player_hustle_stats phs ON pss.player_id = phs.player_id
                WHERE pss.season_id = '2022-23' AND phs.season_id = '2022-23'
            ''')
            complete_result = self.db.fetchone()
            status['data_completeness'] = {
                'players_with_hustle_data': complete_result['players_with_complete_data'] if complete_result else 0
            }
        except Exception as e:
            status['data_completeness'] = {'error': str(e)}

        return status

    def close(self):
        """Close database connection."""
        self.db.close()
