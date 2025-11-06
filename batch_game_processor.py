"""
Batch Game Processor for PADIM Stint Aggregation

Processes large numbers of NBA games to build sufficient data volume for RAPM modeling.
Handles rate limiting, error recovery, and progress tracking.
"""

import logging
import time
import random
from typing import List, Dict, Optional, Any, Set
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import LeagueGameLog

from src.padim.stint_aggregator import StintAggregator
from src.padim.data_collector import NBADataCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchGameProcessor:
    """Batch processor for NBA games with rate limiting and error recovery."""

    def __init__(self, max_workers: int = 4, batch_size: int = 50):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum concurrent threads
            batch_size: Games to process before saving progress
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.stint_agg = StintAggregator()
        self.data_collector = NBADataCollector()

        # Rate limiting: NBA API allows ~200 requests/minute
        self.requests_per_minute = 180
        self.min_delay = 60.0 / self.requests_per_minute  # ~0.33 seconds between requests

        # Progress tracking
        self.progress_file = "data/batch_progress.json"
        self.processed_games: Set[str] = set()
        self.failed_games: Dict[str, str] = {}
        self.start_time = None

        self._load_progress()

    def _load_progress(self):
        """Load previous processing progress."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.processed_games = set(progress.get('processed_games', []))
                    self.failed_games = progress.get('failed_games', {})
                    logger.info(f"Loaded progress: {len(self.processed_games)} games processed, {len(self.failed_games)} failed")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")

    def _save_progress(self):
        """Save current processing progress."""
        progress = {
            'processed_games': list(self.processed_games),
            'failed_games': self.failed_games,
            'last_updated': datetime.now().isoformat(),
            'total_processed': len(self.processed_games),
            'total_failed': len(self.failed_games)
        }

        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

    def get_season_games(self, season: str = "2022-23") -> List[str]:
        """
        Get all unique game IDs for a season.

        Args:
            season: Season ID (e.g., "2022-23")

        Returns:
            List of unique game IDs
        """
        try:
            logger.info(f"Fetching game IDs for season {season}")
            endpoint = LeagueGameLog(season=season)
            df = endpoint.get_data_frames()[0]

            # Get unique game IDs (remove duplicates from home/away entries)
            unique_games = df['GAME_ID'].unique().tolist()
            logger.info(f"Found {len(unique_games)} unique games for {season}")

            return unique_games

        except Exception as e:
            logger.error(f"Error fetching season games: {e}")
            return []

    def process_single_game(self, game_id: str) -> Dict[str, Any]:
        """
        Process a single game with error handling.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with processing result
        """
        if game_id in self.processed_games:
            return {'game_id': game_id, 'status': 'already_processed', 'stints': 0}

        try:
            logger.debug(f"Processing game {game_id}")

            # Add random delay for rate limiting
            time.sleep(random.uniform(self.min_delay * 0.8, self.min_delay * 1.2))

            # Process the game
            result = self.stint_agg.aggregate_game_stints(game_id)

            if 'error' in result:
                logger.warning(f"Failed to process game {game_id}: {result['error']}")
                self.failed_games[game_id] = result['error']
                return {'game_id': game_id, 'status': 'failed', 'error': result['error'], 'stints': 0}

            # Validate that stints were actually created
            stints_data = result.get('stints', [])
            stints_count = len(stints_data) if isinstance(stints_data, list) else 0

            if stints_count == 0:
                logger.warning(f"No stints created for game {game_id}")
                self.failed_games[game_id] = "No stints created"
                return {'game_id': game_id, 'status': 'no_stints', 'error': 'No stints created', 'stints': 0}

            # Validate stint count is reasonable (should be 15-40 stints per game)
            if stints_count < 10:
                logger.warning(f"Very low stint count for game {game_id}: {stints_count} (expected 15-40)")
                self.failed_games[game_id] = f"Low stint count: {stints_count}"
                return {'game_id': game_id, 'status': 'low_stints', 'error': f'Low stint count: {stints_count}', 'stints': stints_count}

            # Save stints to database
            save_success = self.stint_agg.save_stints_to_db(result)

            if not save_success:
                logger.error(f"Failed to save stints for game {game_id}")
                self.failed_games[game_id] = "Database save failed"
                return {'game_id': game_id, 'status': 'save_failed', 'stints': stints_count}

            # Verify stints were actually saved to database
            saved_count = self._verify_stints_saved(game_id, stints_count)
            if saved_count != stints_count:
                logger.error(f"Stint count mismatch for game {game_id}: expected {stints_count}, saved {saved_count}")
                self.failed_games[game_id] = f"Save verification failed: {saved_count}/{stints_count} stints saved"
                return {'game_id': game_id, 'status': 'save_verification_failed', 'stints': saved_count}

            self.processed_games.add(game_id)

            logger.info(f"Successfully processed game {game_id}: {stints_count} stints created and saved")
            return {'game_id': game_id, 'status': 'success', 'stints': stints_count}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error processing game {game_id}: {error_msg}")
            self.failed_games[game_id] = error_msg
            return {'game_id': game_id, 'status': 'error', 'error': error_msg, 'stints': 0}

    def process_games_batch(self, game_ids: List[str], max_games: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a batch of games with progress tracking.

        Args:
            game_ids: List of game IDs to process
            max_games: Maximum number of games to process (None = all)

        Returns:
            Dict with batch processing results
        """
        self.start_time = datetime.now()

        # Filter out already processed games
        games_to_process = [gid for gid in game_ids if gid not in self.processed_games]
        if max_games:
            games_to_process = games_to_process[:max_games]

        logger.info(f"Starting batch processing of {len(games_to_process)} games")
        logger.info(f"Already processed: {len(self.processed_games)} games")

        results = []
        processed_count = 0

        # Process in smaller batches to save progress regularly
        for i in range(0, len(games_to_process), self.batch_size):
            batch = games_to_process[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(games_to_process) + self.batch_size - 1)//self.batch_size} ({len(batch)} games)")

            # Sequential processing for now (API rate limits)
            # Could be parallelized with proper rate limiting
            for game_id in batch:
                result = self.process_single_game(game_id)
                results.append(result)

                if result['status'] in ['success', 'already_processed']:
                    processed_count += 1

                # Save progress every 10 games
                if processed_count % 10 == 0:
                    self._save_progress()

            # Brief pause between batches
            time.sleep(1)

        # Final progress save
        self._save_progress()

        # Calculate statistics with detailed breakdown
        status_counts = {}
        for result in results:
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        successful = status_counts.get('success', 0)
        failed = sum([count for status, count in status_counts.items()
                     if status in ['failed', 'error', 'save_failed', 'no_stints', 'low_stints', 'save_verification_failed']])
        already_processed = status_counts.get('already_processed', 0)
        total_stints = sum([r.get('stints', 0) for r in results])

        end_time = datetime.now()
        duration = end_time - self.start_time

        summary = {
            'total_games_attempted': len(results),
            'successful': successful,
            'failed': failed,
            'already_processed': already_processed,
            'total_stints_created': total_stints,
            'success_rate': successful / len(results) if results else 0,
            'processing_time': str(duration),
            'games_per_hour': len(results) / (duration.total_seconds() / 3600) if duration.total_seconds() > 0 else 0,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'status_breakdown': status_counts,
            'failed_games': list(self.failed_games.keys())[:10],  # First 10 failures
            'failure_reasons': list(self.failed_games.values())[:10]  # First 10 failure reasons
        }

        logger.info(f"Batch processing complete: {successful} successful, {failed} failed, {total_stints} stints created")
        logger.info(f"Status breakdown: {status_counts}")

        if failed > 0:
            logger.warning(f"Top failure reasons: {list(self.failed_games.values())[:3]}")
        return summary

    def _verify_stints_saved(self, game_id: str, expected_count: int) -> int:
        """
        Verify that stints were actually saved to the database.

        Args:
            game_id: Game ID to check
            expected_count: Expected number of stints

        Returns:
            Actual number of stints found in database
        """
        try:
            # Query database to count stints for this game
            self.stint_agg.db.execute("SELECT COUNT(*) as count FROM stints WHERE game_id = ?", (game_id,))
            result = self.stint_agg.db.fetchone()
            actual_count = result['count'] if result else 0

            logger.debug(f"Verified stints for game {game_id}: expected {expected_count}, found {actual_count}")
            return actual_count

        except Exception as e:
            logger.error(f"Failed to verify stints saved for game {game_id}: {e}")
            return 0

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            'processed_games': len(self.processed_games),
            'failed_games': len(self.failed_games),
            'total_attempted': len(self.processed_games) + len(self.failed_games),
            'success_rate': len(self.processed_games) / (len(self.processed_games) + len(self.failed_games)) if (len(self.processed_games) + len(self.failed_games)) > 0 else 0,
            'progress_file': self.progress_file
        }

    def retry_failed_games(self, max_retries: int = 3) -> Dict[str, Any]:
        """Retry processing failed games."""
        if not self.failed_games:
            logger.info("No failed games to retry")
            return {'status': 'no_failures'}

        failed_list = list(self.failed_games.keys())
        logger.info(f"Retrying {len(failed_list)} failed games")

        # Clear failed status for retry
        for game_id in failed_list:
            if game_id in self.failed_games:
                del self.failed_games[game_id]

        return self.process_games_batch(failed_list)

    def validate_existing_games(self) -> Dict[str, Any]:
        """
        Validate which games in processed_games actually have stints.

        Returns:
            Dict with validation results
        """
        logger.info("Validating existing processed games...")

        validation = {
            'total_processed_games': len(self.processed_games),
            'games_with_stints': 0,
            'games_without_stints': 0,
            'total_stints_found': 0,
            'games_with_stints_list': [],
            'games_without_stints_list': []
        }

        for game_id in self.processed_games:
            stint_count = self._verify_stints_saved(game_id, 0)  # Expected count doesn't matter here

            if stint_count > 0:
                validation['games_with_stints'] += 1
                validation['total_stints_found'] += stint_count
                validation['games_with_stints_list'].append({
                    'game_id': game_id,
                    'stints': stint_count
                })
            else:
                validation['games_without_stints'] += 1
                validation['games_without_stints_list'].append(game_id)

        validation['true_success_rate'] = validation['games_with_stints'] / len(self.processed_games) if self.processed_games else 0

        logger.info(f"Validation complete: {validation['games_with_stints']}/{len(self.processed_games)} games have stints")
        return validation

def main():
    """Main batch processing function."""
    import argparse

    parser = argparse.ArgumentParser(description='Batch process NBA games for PADIM')
    parser.add_argument('--season', default='2022-23', help='Season to process (default: 2022-23)')
    parser.add_argument('--max-games', type=int, help='Maximum games to process')
    parser.add_argument('--batch-size', type=int, default=50, help='Games per batch before saving progress')
    parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed games')
    parser.add_argument('--status', action='store_true', help='Show processing status only')
    parser.add_argument('--validate-existing', action='store_true', help='Validate which existing games actually have stints')

    args = parser.parse_args()

    processor = BatchGameProcessor(batch_size=args.batch_size)

    if args.status:
        status = processor.get_processing_status()
        print("\n=== PROCESSING STATUS ===")
        print(json.dumps(status, indent=2))
        return

    if args.validate_existing:
        validation_result = processor.validate_existing_games()
        print("\n=== EXISTING GAMES VALIDATION ===")
        print(json.dumps(validation_result, indent=2))
        return

    if args.retry_failed:
        print("Retrying failed games...")
        result = processor.retry_failed_games()
        print("\n=== RETRY RESULTS ===")
        print(json.dumps(result, indent=2))
        return

    # Get games for the season
    game_ids = processor.get_season_games(args.season)
    if not game_ids:
        print(f"No games found for season {args.season}")
        return

    # Process games
    print(f"Starting batch processing of up to {args.max_games or len(game_ids)} games from {args.season}")
    result = processor.process_games_batch(game_ids, args.max_games)

    print("\n=== PROCESSING RESULTS ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
