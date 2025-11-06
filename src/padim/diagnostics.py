"""
Single-Game Deep Diagnostics Tool for PADIM

Comprehensive diagnostic tool for testing PADIM pipeline on individual games.
Provides detailed validation and error analysis for each step of the process.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd

from .stint_aggregator import StintAggregator
from .data_collector import NBADataCollector
from .config.logging_config import get_logger, log_performance_metric, log_data_quality_metric
from .db.database import DatabaseConnection

logger = get_logger(__name__)


class GameDiagnostics:
    """Comprehensive diagnostics for individual NBA games."""

    def __init__(self):
        """Initialize diagnostics tool."""
        self.stint_agg = StintAggregator()
        self.data_collector = NBADataCollector()
        self.db = DatabaseConnection()

    def run_full_diagnostics(self, game_id: str, save_report: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics on a single game.

        Args:
            game_id: NBA game ID to diagnose
            save_report: Whether to save detailed report to file

        Returns:
            Dict containing full diagnostic results
        """
        start_time = time.time()
        logger.info(f"Starting full diagnostics for game {game_id}", extra={
            'extra_data': {'game_id': game_id, 'operation': 'full_diagnostics_start'}
        })

        diagnostics = {
            'game_id': game_id,
            'timestamp': datetime.now().isoformat(),
            'overall_success': False,
            'processing_time_seconds': 0,
            'stages': {},
            'errors': [],
            'warnings': [],
            'recommendations': []
        }

        try:
            # Stage 1: Data Availability Check
            diagnostics['stages']['data_availability'] = self._diagnose_data_availability(game_id)

            # Stage 2: API Endpoint Validation
            diagnostics['stages']['api_validation'] = self._diagnose_api_endpoints(game_id)

            # Stage 3: Data Quality Assessment
            diagnostics['stages']['data_quality'] = self._diagnose_data_quality(game_id)

            # Stage 4: Pipeline Processing
            diagnostics['stages']['pipeline_processing'] = self._diagnose_pipeline_processing(game_id)

            # Stage 5: Stint Aggregation Deep Dive
            diagnostics['stages']['stint_aggregation'] = self._diagnose_stint_aggregation(game_id)

            # Stage 6: Database Integration
            diagnostics['stages']['database_integration'] = self._diagnose_database_integration(game_id)

            # Analyze results and generate recommendations
            diagnostics['overall_success'] = self._analyze_overall_success(diagnostics)
            diagnostics['recommendations'] = self._generate_recommendations(diagnostics)

            diagnostics['processing_time_seconds'] = time.time() - start_time

            log_performance_metric('run_full_diagnostics', diagnostics['processing_time_seconds'],
                                 success=diagnostics['overall_success'],
                                 extra_data={'game_id': game_id})

            if save_report:
                self._save_diagnostic_report(diagnostics)

            logger.info(f"Completed diagnostics for game {game_id}: {'SUCCESS' if diagnostics['overall_success'] else 'FAILED'}")

        except Exception as e:
            diagnostics['processing_time_seconds'] = time.time() - start_time
            diagnostics['errors'].append(f"Diagnostic failure: {str(e)}")
            logger.error(f"Diagnostics failed for game {game_id}: {e}", extra={
                'extra_data': {'game_id': game_id, 'error': str(e)}
            })

        return diagnostics

    def _diagnose_data_availability(self, game_id: str) -> Dict[str, Any]:
        """Check if all required data sources are available for the game."""
        stage_result = {
            'stage_name': 'data_availability',
            'success': False,
            'checks': {},
            'issues': []
        }

        try:
            # Check rotation data availability
            rotation_data = self.stint_agg._get_game_rotations(game_id)
            stage_result['checks']['rotation_data'] = {
                'available': rotation_data is not None,
                'teams_count': len(rotation_data) if rotation_data else 0,
                'total_players': sum(len(df) for df in rotation_data) if rotation_data else 0
            }

            # Check PBP data availability
            pbp_data = self.stint_agg._get_play_by_play_data(game_id)
            stage_result['checks']['pbp_data'] = {
                'available': pbp_data is not None,
                'events_count': len(pbp_data) if pbp_data is not None else 0,
                'periods_covered': len(pbp_data['period'].unique()) if pbp_data is not None else 0
            }

            # Check team data
            if pbp_data is not None:
                team_ids = pbp_data['teamId'].unique()
                team_ids = [int(tid) for tid in team_ids if tid != 0]
                stage_result['checks']['team_ids'] = {
                    'count': len(team_ids),
                    'ids': team_ids
                }

                # Try to get home/away teams
                home_team_id, away_team_id = self.stint_agg._get_home_away_team_ids(game_id, team_ids)
                stage_result['checks']['home_away_teams'] = {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'resolved': home_team_id is not None and away_team_id is not None
                }

            stage_result['success'] = all([
                stage_result['checks']['rotation_data']['available'],
                stage_result['checks']['pbp_data']['available'],
                stage_result['checks'].get('home_away_teams', {}).get('resolved', False)
            ])

        except Exception as e:
            stage_result['issues'].append(f"Data availability check failed: {str(e)}")

        return stage_result

    def _diagnose_api_endpoints(self, game_id: str) -> Dict[str, Any]:
        """Test API endpoints and measure response times."""
        stage_result = {
            'stage_name': 'api_validation',
            'success': True,
            'endpoint_tests': {},
            'performance_metrics': {},
            'issues': []
        }

        endpoints_to_test = [
            ('GameRotation', lambda: self.stint_agg._get_game_rotations(game_id)),
            ('PlayByPlayV3', lambda: self.stint_agg._get_play_by_play_data(game_id))
        ]

        for endpoint_name, endpoint_func in endpoints_to_test:
            start_time = time.time()
            try:
                result = endpoint_func()
                duration = time.time() - start_time

                stage_result['endpoint_tests'][endpoint_name] = {
                    'success': result is not None,
                    'response_time_seconds': duration,
                    'data_size': len(result) if hasattr(result, '__len__') else 'N/A'
                }

                stage_result['performance_metrics'][f'{endpoint_name}_response_time'] = duration

                if duration > 10:  # API calls taking > 10 seconds are concerning
                    stage_result['issues'].append(f"{endpoint_name} response time: {duration:.2f}s (slow)")
                    stage_result['success'] = False

            except Exception as e:
                duration = time.time() - start_time
                stage_result['endpoint_tests'][endpoint_name] = {
                    'success': False,
                    'response_time_seconds': duration,
                    'error': str(e)
                }
                stage_result['issues'].append(f"{endpoint_name} failed: {str(e)}")
                stage_result['success'] = False

        return stage_result

    def _diagnose_data_quality(self, game_id: str) -> Dict[str, Any]:
        """Assess quality of retrieved data."""
        stage_result = {
            'stage_name': 'data_quality',
            'success': True,
            'quality_metrics': {},
            'issues': []
        }

        try:
            # Get data
            rotation_data = self.stint_agg._get_game_rotations(game_id)
            pbp_data = self.stint_agg._get_play_by_play_data(game_id)

            if rotation_data:
                # Check rotation data quality
                total_players = sum(len(df) for df in rotation_data)
                total_stints = sum(len(df.dropna(subset=['IN_TIME_REAL', 'OUT_TIME_REAL'])) for df in rotation_data)

                stage_result['quality_metrics']['rotation'] = {
                    'total_players': total_players,
                    'total_stints': total_stints,
                    'avg_stints_per_player': total_stints / total_players if total_players > 0 else 0
                }

                # Flag potential issues
                if total_players < 20:  # Should be ~24-30 players (12 per team + bench)
                    stage_result['issues'].append(f"Low player count: {total_players} (expected ~24-30)")
                    stage_result['success'] = False

            if pbp_data is not None:
                # Check PBP data quality
                total_events = len(pbp_data)
                periods = len(pbp_data['period'].unique())
                teams_in_pbp = len(pbp_data['teamId'].unique())

                stage_result['quality_metrics']['pbp'] = {
                    'total_events': total_events,
                    'periods_covered': periods,
                    'teams_represented': teams_in_pbp
                }

                # Flag potential issues
                if total_events < 300:  # Typical game has 400-600 events
                    stage_result['issues'].append(f"Low event count: {total_events} (expected 400-600)")
                    stage_result['success'] = False

                if periods < 4:  # Should cover at least 4 periods
                    stage_result['issues'].append(f"Incomplete periods: {periods} (expected 4+)")
                    stage_result['success'] = False

                # Check for missing critical columns
                critical_cols = ['period', 'clock', 'actionType', 'teamId']
                missing_cols = [col for col in critical_cols if col not in pbp_data.columns]
                if missing_cols:
                    stage_result['issues'].extend([f"Missing critical column: {col}" for col in missing_cols])
                    stage_result['success'] = False

        except Exception as e:
            stage_result['issues'].append(f"Data quality assessment failed: {str(e)}")
            stage_result['success'] = False

        return stage_result

    def _diagnose_pipeline_processing(self, game_id: str) -> Dict[str, Any]:
        """Test the core stint aggregation pipeline."""
        stage_result = {
            'stage_name': 'pipeline_processing',
            'success': False,
            'processing_steps': {},
            'performance': {},
            'issues': []
        }

        try:
            start_time = time.time()

            # Step 1: Get rotation data
            step_start = time.time()
            rotation_data = self.stint_agg._get_game_rotations(game_id)
            rotation_time = time.time() - step_start

            stage_result['processing_steps']['get_rotations'] = {
                'success': rotation_data is not None,
                'duration_seconds': rotation_time,
                'teams_returned': len(rotation_data) if rotation_data else 0
            }

            if not rotation_data:
                stage_result['issues'].append("Failed to get rotation data")
                return stage_result

            # Step 2: Get PBP data
            step_start = time.time()
            pbp_data = self.stint_agg._get_play_by_play_data(game_id)
            pbp_time = time.time() - step_start

            stage_result['processing_steps']['get_pbp'] = {
                'success': pbp_data is not None,
                'duration_seconds': pbp_time,
                'events_count': len(pbp_data) if pbp_data is not None else 0
            }

            if not pbp_data:
                stage_result['issues'].append("Failed to get PBP data")
                return stage_result

            # Step 3: Extract team IDs
            step_start = time.time()
            team_ids = pbp_data['teamId'].unique()
            team_ids = [int(tid) for tid in team_ids if tid != 0]
            team_extraction_time = time.time() - step_start

            stage_result['processing_steps']['extract_team_ids'] = {
                'success': len(team_ids) >= 2,
                'duration_seconds': team_extraction_time,
                'team_ids_found': team_ids
            }

            # Step 4: Get home/away teams
            step_start = time.time()
            home_team_id, away_team_id = self.stint_agg._get_home_away_team_ids(game_id, team_ids)
            home_away_time = time.time() - step_start

            stage_result['processing_steps']['get_home_away'] = {
                'success': home_team_id is not None and away_team_id is not None,
                'duration_seconds': home_away_time,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id
            }

            # Step 5: Find stint boundaries
            step_start = time.time()
            stint_boundaries = self.stint_agg._find_stint_boundaries(rotation_data)
            boundary_time = time.time() - step_start

            stage_result['processing_steps']['find_boundaries'] = {
                'success': len(stint_boundaries) > 0,
                'duration_seconds': boundary_time,
                'boundaries_found': len(stint_boundaries),
                'first_few_boundaries': stint_boundaries[:5] if stint_boundaries else []
            }

            # Step 6: Aggregate stints (partial test - just check if it runs without crashing)
            step_start = time.time()
            try:
                stints = self.stint_agg._aggregate_stint_stats(
                    game_id, stint_boundaries, rotation_data, pbp_data, home_team_id, away_team_id
                )
                aggregation_success = stints is not None
                stints_created = len(stints) if stints is not None else 0
            except Exception as e:
                aggregation_success = False
                stints_created = 0
                stage_result['issues'].append(f"Stint aggregation failed: {str(e)}")

            aggregation_time = time.time() - step_start

            stage_result['processing_steps']['aggregate_stints'] = {
                'success': aggregation_success,
                'duration_seconds': aggregation_time,
                'stints_created': stints_created
            }

            total_time = time.time() - start_time
            stage_result['performance'] = {
                'total_duration_seconds': total_time,
                'steps_breakdown': {k: v['duration_seconds'] for k, v in stage_result['processing_steps'].items()}
            }

            # Overall success: all steps must succeed
            stage_result['success'] = all(step['success'] for step in stage_result['processing_steps'].values())

        except Exception as e:
            stage_result['issues'].append(f"Pipeline processing failed: {str(e)}")

        return stage_result

    def _diagnose_stint_aggregation(self, game_id: str) -> Dict[str, Any]:
        """Deep dive into stint aggregation logic."""
        stage_result = {
            'stage_name': 'stint_aggregation',
            'success': False,
            'stint_analysis': {},
            'lineup_validation': {},
            'defensive_stats': {},
            'issues': []
        }

        try:
            # Run full stint aggregation
            result = self.stint_agg.aggregate_game_stints(game_id)

            if 'error' in result:
                stage_result['issues'].append(f"Stint aggregation error: {result['error']}")
                return stage_result

            stints = result.get('stints', [])
            stage_result['stint_analysis'] = {
                'total_stints': len(stints),
                'stints_with_data': len([s for s in stints if s.get('home_defensive_stats', {}).get('opponent_fga', 0) > 0]),
                'avg_stint_duration': sum(s.get('duration', 0) for s in stints) / len(stints) if stints else 0
            }

            # Analyze lineup consistency
            lineup_issues = []
            for i, stint in enumerate(stints[:5]):  # Check first 5 stints
                home_players = stint.get('home_players', [])
                away_players = stint.get('away_players', [])

                if len(home_players) != 5:
                    lineup_issues.append(f"Stint {i}: Home team has {len(home_players)} players (expected 5)")
                if len(away_players) != 5:
                    lineup_issues.append(f"Stint {i}: Away team has {len(away_players)} players (expected 5)")

                # Check for duplicate players (shouldn't happen in valid lineups)
                if len(set(home_players)) != len(home_players):
                    lineup_issues.append(f"Stint {i}: Duplicate players in home lineup")
                if len(set(away_players)) != len(away_players):
                    lineup_issues.append(f"Stint {i}: Duplicate players in away lineup")

            stage_result['lineup_validation'] = {
                'issues_found': len(lineup_issues),
                'sample_issues': lineup_issues[:3]  # First 3 issues
            }

            # Analyze defensive statistics
            defensive_stats = []
            for stint in stints:
                home_stats = stint.get('home_defensive_stats', {})
                away_stats = stint.get('away_defensive_stats', {})

                if home_stats.get('opponent_fga', 0) > 0:
                    defensive_stats.append(home_stats)
                if away_stats.get('opponent_fga', 0) > 0:
                    defensive_stats.append(away_stats)

            if defensive_stats:
                avg_fga = sum(s.get('opponent_fga', 0) for s in defensive_stats) / len(defensive_stats)
                avg_fg_pct = sum(s.get('opponent_fgm', 0) / s.get('opponent_fga', 1) for s in defensive_stats) / len(defensive_stats)

                stage_result['defensive_stats'] = {
                    'stints_with_defensive_data': len(defensive_stats),
                    'avg_opponent_fga_per_stint': avg_fga,
                    'avg_opponent_fg_pct': avg_fg_pct
                }

            stage_result['issues'].extend(lineup_issues)
            stage_result['success'] = len(stints) > 0 and not lineup_issues

        except Exception as e:
            stage_result['issues'].append(f"Stint aggregation diagnostics failed: {str(e)}")

        return stage_result

    def _diagnose_database_integration(self, game_id: str) -> Dict[str, Any]:
        """Test database save/load operations."""
        stage_result = {
            'stage_name': 'database_integration',
            'success': True,
            'database_operations': {},
            'issues': []
        }

        try:
            # Test stint saving
            result = self.stint_agg.aggregate_game_stints(game_id)

            if 'error' not in result:
                save_success = self.stint_agg.save_stints_to_db(result)
                stage_result['database_operations']['save_stints'] = {
                    'success': save_success,
                    'stints_attempted': len(result.get('stints', []))
                }

                if not save_success:
                    stage_result['issues'].append("Failed to save stints to database")
                    stage_result['success'] = False

                # Test stint retrieval
                stints_saved = self.db.execute(
                    "SELECT COUNT(*) as count FROM stints WHERE game_id = ?", (game_id,)
                ).fetchone()

                stage_result['database_operations']['verify_saved'] = {
                    'stints_in_db': stints_saved['count'] if stints_saved else 0,
                    'matches_attempted': stints_saved['count'] == len(result.get('stints', [])) if stints_saved else False
                }

        except Exception as e:
            stage_result['issues'].append(f"Database integration test failed: {str(e)}")
            stage_result['success'] = False

        return stage_result

    def _analyze_overall_success(self, diagnostics: Dict) -> bool:
        """Analyze if the overall diagnostics were successful."""
        stages = diagnostics.get('stages', {})

        # Critical stages that must succeed
        critical_stages = ['data_availability', 'pipeline_processing']
        for stage_name in critical_stages:
            if stage_name in stages and not stages[stage_name].get('success', False):
                return False

        # Must have created at least some stints
        stint_stage = stages.get('stint_aggregation', {})
        if stint_stage.get('stint_analysis', {}).get('total_stints', 0) == 0:
            return False

        return True

    def _generate_recommendations(self, diagnostics: Dict) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []

        stages = diagnostics.get('stages', {})

        # Data availability issues
        data_avail = stages.get('data_availability', {})
        if not data_avail.get('success', False):
            checks = data_avail.get('checks', {})
            if not checks.get('rotation_data', {}).get('available', False):
                recommendations.append("Fix GameRotation API access - rotation data unavailable")
            if not checks.get('pbp_data', {}).get('available', False):
                recommendations.append("Fix PlayByPlayV3 API access - PBP data unavailable")
            if not checks.get('home_away_teams', {}).get('resolved', False):
                recommendations.append("Fix team ID resolution logic")

        # Performance issues
        api_stage = stages.get('api_validation', {})
        if not api_stage.get('success', False):
            recommendations.append("Address slow API response times (>10 seconds)")

        # Data quality issues
        quality_stage = stages.get('data_quality', {})
        if not quality_stage.get('success', False):
            issues = quality_stage.get('issues', [])
            recommendations.extend([f"Fix data quality: {issue}" for issue in issues[:3]])

        # Pipeline issues
        pipeline_stage = stages.get('pipeline_processing', {})
        if not pipeline_stage.get('success', False):
            recommendations.append("Debug stint aggregation pipeline - check each processing step")

        # Stint issues
        stint_stage = stages.get('stint_aggregation', {})
        if not stint_stage.get('success', False):
            lineup_issues = stint_stage.get('lineup_validation', {}).get('issues_found', 0)
            if lineup_issues > 0:
                recommendations.append(f"Fix lineup resolution logic - {lineup_issues} lineup validation issues")

        if not recommendations:
            recommendations.append("Game processed successfully - ready for batch processing")

        return recommendations

    def _save_diagnostic_report(self, diagnostics: Dict):
        """Save detailed diagnostic report to file."""
        try:
            filename = f"diagnostics_{diagnostics['game_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = f"data/{filename}"

            with open(filepath, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)

            logger.info(f"Diagnostic report saved to {filepath}", extra={
                'extra_data': {'game_id': diagnostics['game_id'], 'report_file': filepath}
            })

        except Exception as e:
            logger.error(f"Failed to save diagnostic report: {e}")

    def compare_games_diagnostics(self, game_ids: List[str]) -> Dict[str, Any]:
        """
        Run diagnostics on multiple games and compare results.

        Args:
            game_ids: List of game IDs to compare

        Returns:
            Dict with comparative analysis
        """
        logger.info(f"Running comparative diagnostics on {len(game_ids)} games")

        all_results = {}
        for game_id in game_ids:
            all_results[game_id] = self.run_full_diagnostics(game_id, save_report=False)

        # Analyze patterns across games
        comparison = {
            'games_analyzed': len(game_ids),
            'success_rate': sum(1 for r in all_results.values() if r['overall_success']) / len(game_ids),
            'avg_processing_time': sum(r['processing_time_seconds'] for r in all_results.values()) / len(game_ids),
            'common_failures': self._identify_common_failures(all_results),
            'performance_patterns': self._analyze_performance_patterns(all_results)
        }

        return comparison

    def _identify_common_failures(self, all_results: Dict[str, Dict]) -> List[str]:
        """Identify the most common failure patterns across games."""
        failure_counts = {}

        for game_id, result in all_results.items():
            if not result['overall_success']:
                for stage_name, stage_result in result.get('stages', {}).items():
                    if not stage_result.get('success', True):
                        issues = stage_result.get('issues', [])
                        for issue in issues:
                            key = f"{stage_name}: {issue}"
                            failure_counts[key] = failure_counts.get(key, 0) + 1

        # Return top 5 most common failures
        sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        return [failure for failure, count in sorted_failures[:5]]

    def _analyze_performance_patterns(self, all_results: Dict[str, Dict]) -> Dict[str, float]:
        """Analyze performance patterns across games."""
        processing_times = [r['processing_time_seconds'] for r in all_results.values()]

        return {
            'avg_processing_time': sum(processing_times) / len(processing_times),
            'min_processing_time': min(processing_times),
            'max_processing_time': max(processing_times),
            'processing_time_std': pd.Series(processing_times).std()
        }


def run_game_diagnostics(game_id: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run diagnostics on a single game.

    Args:
        game_id: NBA game ID to diagnose
        verbose: Whether to print results to console

    Returns:
        Diagnostic results
    """
    diagnostics = GameDiagnostics()
    results = diagnostics.run_full_diagnostics(game_id)

    if verbose:
        print(f"\n=== DIAGNOSTIC RESULTS FOR GAME {game_id} ===")
        print(f"Overall Success: {'✅ YES' if results['overall_success'] else '❌ NO'}")
        print(".2f")
        print(f"Stages Completed: {sum(1 for s in results['stages'].values() if s.get('success', False))}/{len(results['stages'])}")

        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors'][:3]:
                print(f"  • {error}")

        if results['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['recommendations'][:3]:
                print(f"  • {rec}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PADIM Game Diagnostics Tool')
    parser.add_argument('game_id', help='NBA game ID to diagnose (e.g., 0022200001)')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')

    args = parser.parse_args()

    results = run_game_diagnostics(args.game_id, verbose=not args.quiet)
