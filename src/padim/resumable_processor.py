"""
Resumable Batch Processor for PADIM

Provides robust resumability for large-scale NBA data collection across multiple seasons.
Handles interruptions gracefully, prevents duplicate work, and enables recovery from failures.
"""

import logging
import time
import json
import signal
import os
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading

from .stint_aggregator import StintAggregator
from .data_collector import NBADataCollector
from .config.logging_config import get_logger, log_performance_metric
# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from batch_game_processor import BatchGameProcessor

logger = get_logger(__name__)


class ProcessingState(Enum):
    """Enumeration of possible processing states for a game."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GameProcessingRecord:
    """Record of processing state for a single game."""

    def __init__(self, game_id: str, state: ProcessingState = ProcessingState.NOT_STARTED):
        self.game_id = game_id
        self.state = state
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.attempts = 0
        self.last_attempt_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.error_category: Optional[str] = None
        self.stints_created: int = 0
        self.processing_time_seconds: float = 0.0
        self.api_calls_made: int = 0
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return {
            'game_id': self.game_id,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'attempts': self.attempts,
            'last_attempt_at': self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'error_category': self.error_category,
            'stints_created': self.stints_created,
            'processing_time_seconds': self.processing_time_seconds,
            'api_calls_made': self.api_calls_made,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameProcessingRecord':
        """Create record from dictionary."""
        record = cls(data['game_id'], ProcessingState(data['state']))
        record.created_at = datetime.fromisoformat(data['created_at'])
        record.updated_at = datetime.fromisoformat(data['updated_at'])
        record.attempts = data.get('attempts', 0)
        if data.get('last_attempt_at'):
            record.last_attempt_at = datetime.fromisoformat(data['last_attempt_at'])
        if data.get('completed_at'):
            record.completed_at = datetime.fromisoformat(data['completed_at'])
        record.error_message = data.get('error_message')
        record.error_category = data.get('error_category')
        record.stints_created = data.get('stints_created', 0)
        record.processing_time_seconds = data.get('processing_time_seconds', 0.0)
        record.api_calls_made = data.get('api_calls_made', 0)
        record.metadata = data.get('metadata', {})
        return record


class ResumableProcessor:
    """
    Resumable batch processor for PADIM data collection.

    Key features:
    - Atomic game-level processing with state persistence
    - Graceful interruption handling with checkpointing
    - Intelligent retry logic with exponential backoff
    - Detailed progress tracking and reporting
    - Prevention of duplicate processing
    """

    def __init__(self, state_file: str = "data/processing_state.json", checkpoint_interval: int = 10):
        """
        Initialize the resumable processor.

        Args:
            state_file: Path to the state file for persistence
            checkpoint_interval: Save state after every N games processed
        """
        self.state_file = Path(state_file)
        self.checkpoint_interval = checkpoint_interval

        # Initialize components
        self.stint_agg = StintAggregator()
        self.data_collector = NBADataCollector()

        # Processing state
        self.game_records: Dict[str, GameProcessingRecord] = {}
        self.current_season: Optional[str] = None
        self.target_games: List[str] = []
        self.start_time: Optional[datetime] = None
        self.is_running = False
        self.should_stop = False

        # Configuration
        self.max_retry_attempts = 3
        self.retry_backoff_base = 2.0  # Exponential backoff multiplier
        self.retry_backoff_max = 300  # Max backoff time in seconds

        # Progress tracking
        self.games_processed_since_checkpoint = 0
        self.total_games_processed = 0
        self.total_stints_created = 0

        # Load existing state if available
        self._load_state()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info("Resumable processor initialized", extra={
            'extra_data': {
                'state_file': str(state_file),
                'checkpoint_interval': checkpoint_interval,
                'existing_games_loaded': len(self.game_records)
            }
        })

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown on interrupts."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.should_stop = True
            if not self.is_running:
                self._save_state()
                logger.info("Shutdown complete")
                exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _load_state(self):
        """Load processing state from file."""
        if not self.state_file.exists():
            logger.info("No existing state file found, starting fresh")
            return

        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            # Load game records
            for game_data in state_data.get('game_records', []):
                record = GameProcessingRecord.from_dict(game_data)
                self.game_records[record.game_id] = record

            # Load session state
            self.current_season = state_data.get('current_season')
            self.target_games = state_data.get('target_games', [])
            self.total_games_processed = state_data.get('total_games_processed', 0)
            self.total_stints_created = state_data.get('total_stints_created', 0)

            if state_data.get('start_time'):
                self.start_time = datetime.fromisoformat(state_data['start_time'])

            logger.info("Loaded processing state", extra={
                'extra_data': {
                    'games_loaded': len(self.game_records),
                    'current_season': self.current_season,
                    'total_games_processed': self.total_games_processed,
                    'total_stints_created': self.total_stints_created
                }
            })

        except Exception as e:
            logger.error(f"Failed to load state file: {e}")
            # Continue with empty state rather than crashing

    def _save_state(self):
        """Save current processing state to file."""
        try:
            # Ensure directory exists
            self.state_file.parent.mkdir(exist_ok=True)

            state_data = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'current_season': self.current_season,
                'target_games': self.target_games,
                'total_games_processed': self.total_games_processed,
                'total_stints_created': self.total_stints_created,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'game_records': [record.to_dict() for record in self.game_records.values()]
            }

            # Atomic write (write to temp file, then rename)
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            temp_file.replace(self.state_file)

            logger.debug("Processing state saved", extra={
                'extra_data': {
                    'state_file': str(self.state_file),
                    'games_saved': len(self.game_records)
                }
            })

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _checkpoint_if_needed(self):
        """Save state if checkpoint interval reached."""
        self.games_processed_since_checkpoint += 1

        if self.games_processed_since_checkpoint >= self.checkpoint_interval:
            logger.info("Checkpoint: Saving processing state")
            self._save_state()
            self.games_processed_since_checkpoint = 0

    def start_season_processing(self, season: str, max_games: Optional[int] = None,
                              resume: bool = True) -> Dict[str, Any]:
        """
        Start or resume processing for a season.

        Args:
            season: Season ID (e.g., "2022-23")
            max_games: Maximum games to process (None = all)
            resume: Whether to resume from previous state

        Returns:
            Processing results summary
        """
        self.start_time = datetime.now()
        self.is_running = True
        self.should_stop = False

        try:
            logger.info(f"Starting season processing for {season}", extra={
                'extra_data': {
                    'season': season,
                    'max_games': max_games,
                    'resume': resume,
                    'existing_games': len(self.game_records)
                }
            })

            # Get all games for the season
            all_season_games = self._get_season_games(season)
            if not all_season_games:
                return {'error': f'No games found for season {season}'}

            # Determine which games to process
            if resume and self.current_season == season and self.target_games:
                # Resume from previous session
                games_to_process = self.target_games
                logger.info(f"Resuming from previous session: {len(games_to_process)} games remaining")
            else:
                # Start fresh for this season
                self.current_season = season
                self.target_games = all_season_games[:max_games] if max_games else all_season_games

                # Initialize records for new games
                for game_id in self.target_games:
                    if game_id not in self.game_records:
                        self.game_records[game_id] = GameProcessingRecord(game_id)

                games_to_process = self.target_games.copy()
                logger.info(f"Starting fresh processing: {len(games_to_process)} games to process")

            # Process games
            results = self._process_games_batch(games_to_process)

            # Final save
            self._save_state()

            # Generate summary
            summary = self._generate_processing_summary(results)
            return summary

        except Exception as e:
            logger.error(f"Season processing failed: {e}")
            self._save_state()  # Save state even on failure
            return {'error': str(e)}

        finally:
            self.is_running = False

    def _process_games_batch(self, game_ids: List[str]) -> Dict[str, Any]:
        """Process a batch of games with resumability."""
        results = {
            'total_attempted': len(game_ids),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_stints_created': 0,
            'processing_time_seconds': 0,
            'games_processed': []
        }

        batch_start_time = time.time()

        for i, game_id in enumerate(game_ids):
            if self.should_stop:
                logger.info("Processing interrupted by user")
                break

            logger.info(f"Processing game {game_id} ({i+1}/{len(game_ids)})")

            # Check if already completed
            record = self.game_records.get(game_id)
            if record and record.state == ProcessingState.COMPLETED:
                logger.debug(f"Game {game_id} already completed, skipping")
                results['skipped'] += 1
                continue

            # Process the game
            game_result = self._process_single_game_safe(game_id)
            results['games_processed'].append(game_result)

            # Update counters
            if game_result['success']:
                results['successful'] += 1
                results['total_stints_created'] += game_result.get('stints_created', 0)
                self.total_games_processed += 1
                self.total_stints_created += game_result.get('stints_created', 0)
            else:
                results['failed'] += 1

            # Periodic checkpoint
            self._checkpoint_if_needed()

            # Progress logging
            if (i + 1) % 10 == 0 or i == len(game_ids) - 1:
                progress = (i + 1) / len(game_ids) * 100
                logger.info(".1f")

        results['processing_time_seconds'] = time.time() - batch_start_time
        return results

    def _process_single_game_safe(self, game_id: str) -> Dict[str, Any]:
        """Process a single game with comprehensive error handling."""
        record = self.game_records.get(game_id)
        if not record:
            record = GameProcessingRecord(game_id)
            self.game_records[game_id] = record

        # Check if we should retry this game BEFORE marking as in progress
        if not self._should_retry_game(record):
            logger.info(f"Skipping game {game_id} - max retries exceeded or permanent failure")
            record.state = ProcessingState.SKIPPED
            record.error_message = "Max retries exceeded or permanent failure"
            record.error_category = "max_retries"
            return {
                'game_id': game_id,
                'success': False,
                'error': 'Max retries exceeded or permanent failure',
                'stints_created': 0
            }

        # Mark as in progress and increment attempts
        record.state = ProcessingState.IN_PROGRESS
        record.attempts += 1
        record.last_attempt_at = datetime.now()
        record.updated_at = datetime.now()

        start_time = time.time()

        try:
            # Process the game
            result = self.stint_agg.aggregate_game_stints(game_id)

            processing_time = time.time() - start_time

            if 'error' in result:
                # Processing failed
                record.state = ProcessingState.FAILED
                record.error_message = result['error']
                record.error_category = self._categorize_error(result['error'])
                record.processing_time_seconds = processing_time

                logger.warning(f"Game {game_id} processing failed: {result['error']}", extra={
                    'extra_data': {
                        'game_id': game_id,
                        'error': result['error'],
                        'attempts': record.attempts
                    }
                })

                return {
                    'game_id': game_id,
                    'success': False,
                    'error': result['error'],
                    'error_category': record.error_category,
                    'stints_created': 0,
                    'processing_time_seconds': processing_time
                }

            # Validate results
            stints_data = result.get('stints', [])
            stints_count = len(stints_data) if isinstance(stints_data, list) else 0

            if stints_count == 0:
                record.state = ProcessingState.FAILED
                record.error_message = "No stints created"
                record.error_category = "no_stints"
                return {
                    'game_id': game_id,
                    'success': False,
                    'error': 'No stints created',
                    'stints_created': 0
                }

            if stints_count < 10:
                record.state = ProcessingState.FAILED
                record.error_message = f"Low stint count: {stints_count}"
                record.error_category = "low_stints"
                return {
                    'game_id': game_id,
                    'success': False,
                    'error': f'Low stint count: {stints_count}',
                    'stints_created': stints_count
                }

            # Save to database
            save_success = self.stint_agg.save_stints_to_db(result)
            if not save_success:
                record.state = ProcessingState.FAILED
                record.error_message = "Database save failed"
                record.error_category = "save_failed"
                return {
                    'game_id': game_id,
                    'success': False,
                    'error': 'Database save failed',
                    'stints_created': stints_count
                }

            # Verify save
            saved_count = self._verify_stints_saved(game_id, stints_count)
            if saved_count != stints_count:
                record.state = ProcessingState.FAILED
                record.error_message = f"Save verification failed: {saved_count}/{stints_count}"
                record.error_category = "save_verification_failed"
                return {
                    'game_id': game_id,
                    'success': False,
                    'error': f'Save verification failed: {saved_count}/{stints_count}',
                    'stints_created': saved_count
                }

            # Success!
            record.state = ProcessingState.COMPLETED
            record.completed_at = datetime.now()
            record.stints_created = stints_count
            record.processing_time_seconds = processing_time

            logger.info(f"Game {game_id} processed successfully: {stints_count} stints", extra={
                'extra_data': {
                    'game_id': game_id,
                    'stints_created': stints_count,
                    'processing_time_seconds': processing_time
                }
            })

            return {
                'game_id': game_id,
                'success': True,
                'stints_created': stints_count,
                'processing_time_seconds': processing_time
            }

        except Exception as e:
            processing_time = time.time() - start_time
            record.state = ProcessingState.FAILED
            record.error_message = str(e)
            record.error_category = "unexpected_error"
            record.processing_time_seconds = processing_time

            logger.error(f"Unexpected error processing game {game_id}: {e}")

            return {
                'game_id': game_id,
                'success': False,
                'error': str(e),
                'error_category': 'unexpected_error',
                'stints_created': 0,
                'processing_time_seconds': processing_time
            }

    def _should_retry_game(self, record: GameProcessingRecord) -> bool:
        """Determine if a failed game should be retried."""
        # If this is the first attempt (attempts == 0), always try
        if record.attempts == 0:
            return True

        # If we've exceeded max retries, don't retry
        if record.attempts >= self.max_retry_attempts:
            return False

        # Don't retry certain permanent failures
        permanent_categories = ['invalid_game_id']
        if record.error_category in permanent_categories:
            return False

        # For transient errors, implement exponential backoff
        if record.last_attempt_at and record.state == ProcessingState.FAILED:
            time_since_last_attempt = datetime.now() - record.last_attempt_at
            backoff_time = min(
                self.retry_backoff_base ** record.attempts,
                self.retry_backoff_max
            )

            if time_since_last_attempt.total_seconds() < backoff_time:
                return False

        return True

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages for better handling."""
        error_lower = error_message.lower()

        if 'timeout' in error_lower or 'read timed out' in error_lower:
            return 'api_timeout'
        elif 'not found' in error_lower or '404' in error_message:
            return 'not_found'
        elif 'rate limit' in error_lower or '429' in error_message:
            return 'rate_limited'
        elif 'no rotation data' in error_lower:
            return 'rotation_data_missing'
        elif 'no pbp data' in error_lower:
            return 'pbp_data_missing'
        elif 'database' in error_lower:
            return 'database_error'
        else:
            return 'unknown'

    def _verify_stints_saved(self, game_id: str, expected_count: int) -> int:
        """Verify stints were saved to database."""
        try:
            self.stint_agg.db.execute("SELECT COUNT(*) as count FROM stints WHERE game_id = ?", (game_id,))
            result = self.stint_agg.db.fetchone()
            return result['count'] if result else 0
        except Exception:
            return 0

    def _get_season_games(self, season: str) -> List[str]:
        """Get all game IDs for a season."""
        try:
            from nba_api.stats.endpoints import LeagueGameLog
            endpoint = LeagueGameLog(season=season)
            df = endpoint.get_data_frames()[0]
            unique_games = df['GAME_ID'].unique().tolist()
            return sorted(unique_games)  # Sort for consistent ordering
        except Exception as e:
            logger.error(f"Failed to get season games for {season}: {e}")
            return []

    def _generate_processing_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive processing summary."""
        total_time = results.get('processing_time_seconds', 0)
        successful = results.get('successful', 0)
        total_attempted = results.get('total_attempted', 0)

        # Calculate rates
        success_rate = successful / total_attempted if total_attempted > 0 else 0
        games_per_hour = total_attempted / (total_time / 3600) if total_time > 0 else 0
        stints_per_hour = results.get('total_stints_created', 0) / (total_time / 3600) if total_time > 0 else 0

        # Count errors by category
        error_categories = {}
        for game_result in results.get('games_processed', []):
            if not game_result.get('success', False):
                category = game_result.get('error_category', 'unknown')
                error_categories[category] = error_categories.get(category, 0) + 1

        summary = {
            'season': self.current_season,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now().isoformat(),
            'total_games_attempted': total_attempted,
            'successful_games': successful,
            'failed_games': results.get('failed', 0),
            'skipped_games': results.get('skipped', 0),
            'total_stints_created': results.get('total_stints_created', 0),
            'processing_time_seconds': total_time,
            'success_rate': success_rate,
            'games_per_hour': games_per_hour,
            'stints_per_hour': stints_per_hour,
            'error_categories': error_categories,
            'all_time_stats': {
                'total_games_processed': self.total_games_processed,
                'total_stints_created': self.total_stints_created
            }
        }

        logger.info("Processing summary", extra={'extra_data': summary})
        return summary

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        state_counts = {}
        for record in self.game_records.values():
            state_counts[record.state.value] = state_counts.get(record.state.value, 0) + 1

        return {
            'current_season': self.current_season,
            'is_running': self.is_running,
            'state_counts': state_counts,
            'total_games_tracked': len(self.game_records),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'all_time_stats': {
                'total_games_processed': self.total_games_processed,
                'total_stints_created': self.total_stints_created
            }
        }

    def reset_season(self, season: str):
        """Reset all processing state for a season."""
        logger.warning(f"Resetting all processing state for season {season}")

        # Remove all records for this season's games
        games_to_remove = []
        for game_id, record in self.game_records.items():
            if game_id.startswith(season.replace('-', '')):
                games_to_remove.append(game_id)

        for game_id in games_to_remove:
            del self.game_records[game_id]

        if self.current_season == season:
            self.current_season = None
            self.target_games = []

        self._save_state()
        logger.info(f"Reset complete: removed {len(games_to_remove)} game records")


def main():
    """Command line interface for resumable processing."""
    import argparse

    parser = argparse.ArgumentParser(description='Resumable PADIM Batch Processor')
    parser.add_argument('season', help='Season to process (e.g., 2022-23)')
    parser.add_argument('--max-games', type=int, help='Maximum games to process')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh instead of resuming')
    parser.add_argument('--status', action='store_true', help='Show current processing status')
    parser.add_argument('--reset', action='store_true', help='Reset processing state for the season')

    args = parser.parse_args()

    processor = ResumableProcessor()

    if args.status:
        status = processor.get_processing_status()
        print("\n=== PROCESSING STATUS ===")
        print(json.dumps(status, indent=2))
        return

    if args.reset:
        processor.reset_season(args.season)
        print(f"Reset complete for season {args.season}")
        return

    # Start processing
    print(f"Starting resumable processing for season {args.season}")
    resume = not args.no_resume
    results = processor.start_season_processing(args.season, args.max_games, resume)

    print("\n=== PROCESSING RESULTS ===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
