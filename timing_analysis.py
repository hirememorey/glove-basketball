"""
Timing Analysis for PADIM Stint-Level Defensive Outcome Collection

This script analyzes timing precision and synchronization issues between:
- GameRotation endpoint (player substitutions and stints)
- PlayByPlay endpoint (game events and defensive outcomes)

Key findings from initial exploration:
- Rotation data uses milliseconds from game start
- PBP data uses period + countdown clock format
- Need to align these for accurate stint-level analysis
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import GameRotation, PlayByPlayV3
from typing import Dict, List, Tuple, Optional, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_period_clock_to_seconds(period: int, clock_str: str) -> float:
    """Convert period + clock format to total seconds from game start."""
    if not clock_str or clock_str == '':
        return 0.0

    try:
        # Remove PT prefix and S suffix
        clock_clean = clock_str.replace('PT', '').replace('S', '')

        # Split by M to get minutes and seconds remaining in period
        if 'M' in clock_clean:
            parts = clock_clean.split('M')
            minutes = int(parts[0])
            seconds = float(parts[1]) if len(parts) > 1 else 0.0
        else:
            minutes = 0
            seconds = float(clock_clean)

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
            ot_periods_completed = period - 4
            time_elapsed_this_ot = ot_period_length - time_remaining
            total_seconds = regular_time_elapsed + (ot_periods_completed - 1) * ot_period_length + time_elapsed_this_ot

        return total_seconds

    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse clock string '{clock_str}': {e}")
        return 0.0

def analyze_game_timing_precision(game_id: str) -> Dict[str, Any]:
    """
    Analyze timing precision for a single game.

    Args:
        game_id: NBA game ID

    Returns:
        Dict with detailed timing analysis
    """
    logger.info(f"Analyzing timing precision for game {game_id}")

    results = {
        'game_id': game_id,
        'timing_precision': {},
        'stint_boundaries': {},
        'event_alignment': {},
        'issues_identified': []
    }

    try:
        # Get rotation data
        rotation_endpoint = GameRotation(game_id=game_id)
        rotation_dfs = rotation_endpoint.get_data_frames()

        # Get PBP data
        pbp_endpoint = PlayByPlayV3(game_id=game_id)
        pbp_df = pbp_endpoint.get_data_frames()[0]

        # Convert rotation times from milliseconds to seconds
        rotation_times_seconds = []
        for team_df in rotation_dfs:
            for _, row in team_df.iterrows():
                if pd.notna(row['IN_TIME_REAL']):
                    rotation_times_seconds.append(row['IN_TIME_REAL'] / 10)
                if pd.notna(row['OUT_TIME_REAL']):
                    rotation_times_seconds.append(row['OUT_TIME_REAL'] / 10)

        # Convert PBP times to seconds
        pbp_times_seconds = []
        for _, row in pbp_df.iterrows():
            if pd.notna(row['clock']) and pd.notna(row['period']):
                pbp_seconds = convert_period_clock_to_seconds(int(row['period']), row['clock'])
                pbp_times_seconds.append(pbp_seconds)

        results['timing_precision'] = {
            'rotation_times_range': (min(rotation_times_seconds), max(rotation_times_seconds)),
            'pbp_times_range': (min(pbp_times_seconds), max(pbp_times_seconds)),
            'total_rotation_events': len(rotation_times_seconds),
            'total_pbp_events': len(pbp_times_seconds),
            'time_ranges_match': abs(max(rotation_times_seconds) - max(pbp_times_seconds)) < 1
        }

        # Analyze stint boundaries
        stint_boundaries = sorted(set(rotation_times_seconds))
        results['stint_boundaries'] = {
            'boundaries': stint_boundaries,
            'num_stints': len(stint_boundaries) - 1,  # Number of intervals
            'boundary_gaps': [stint_boundaries[i+1] - stint_boundaries[i] for i in range(len(stint_boundaries)-1)]
        }

        # Analyze event distribution around stint boundaries
        results['event_alignment'] = analyze_event_alignment(stint_boundaries, pbp_df)

        # Identify timing issues
        results['issues_identified'] = identify_timing_issues(rotation_dfs, pbp_df, stint_boundaries)

        logger.info(f"Successfully analyzed timing precision for game {game_id}")

    except Exception as e:
        logger.error(f"Error analyzing timing precision for game {game_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return results

def analyze_event_alignment(stint_boundaries: List[float], pbp_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how PBP events align with stint boundaries.

    Args:
        stint_boundaries: List of stint boundary times in seconds
        pbp_df: Play-by-play DataFrame

    Returns:
        Dict with alignment analysis
    """
    alignment_analysis = {
        'events_near_boundaries': [],
        'boundary_event_distribution': {},
        'possession_continuity': {}
    }

    # Check events within 10 seconds of each boundary
    boundary_window = 10  # seconds

    for boundary_time in stint_boundaries:
        events_near_boundary = []
        for _, event in pbp_df.iterrows():
            if pd.notna(event['clock']) and pd.notna(event['period']):
                event_time = convert_period_clock_to_seconds(int(event['period']), event['clock'])
                if abs(event_time - boundary_time) <= boundary_window:
                    events_near_boundary.append({
                        'time_diff': event_time - boundary_time,
                        'action_type': event['actionType'],
                        'description': event['description'][:50] if event['description'] else ''
                    })

        alignment_analysis['events_near_boundaries'].append({
            'boundary_time': boundary_time,
            'events_count': len(events_near_boundary),
            'events': events_near_boundary[:5]  # Keep first 5 for brevity
        })

    return alignment_analysis

def identify_timing_issues(rotation_dfs: List[pd.DataFrame], pbp_df: pd.DataFrame,
                          stint_boundaries: List[float]) -> List[str]:
    """
    Identify specific timing issues that could affect stint analysis.

    Returns:
        List of identified issues
    """
    issues = []

    # Check for overlapping stints
    for i in range(len(stint_boundaries) - 1):
        start_time = stint_boundaries[i]
        end_time = stint_boundaries[i + 1]

        # Count players on court at start time
        players_at_start = 0
        for team_df in rotation_dfs:
            for _, row in team_df.iterrows():
                if (pd.notna(row['IN_TIME_REAL']) and pd.notna(row['OUT_TIME_REAL']) and
                    row['IN_TIME_REAL']/10 <= start_time <= row['OUT_TIME_REAL']/10):
                    players_at_start += 1

        if players_at_start != 10:  # Should be 5 per team
            issues.append(f"Stint {i}: {players_at_start} players on court (expected 10)")

    # Check for PBP events without timing data
    missing_timing = pbp_df['clock'].isnull().sum() + pbp_df['period'].isnull().sum()
    if missing_timing > 0:
        issues.append(f"{missing_timing} PBP events missing timing data")

    # Check for unrealistic time gaps in stints
    for i, gap in enumerate([stint_boundaries[i+1] - stint_boundaries[i] for i in range(len(stint_boundaries)-1)]):
        if gap < 10:  # Less than 10 seconds
            issues.append(f"Stint {i}: Very short duration ({gap:.1f} seconds)")
        elif gap > 600:  # More than 10 minutes
            issues.append(f"Stint {i}: Very long duration ({gap:.1f} seconds)")

    return issues

def analyze_multiple_games(game_ids: List[str]) -> Dict[str, Any]:
    """
    Analyze timing precision across multiple games.

    Args:
        game_ids: List of NBA game IDs

    Returns:
        Dict with multi-game analysis
    """
    logger.info(f"Analyzing timing precision for {len(game_ids)} games")

    all_results = []
    for game_id in game_ids:
        result = analyze_game_timing_precision(game_id)
        all_results.append(result)

    # Aggregate results
    aggregated = {
        'games_analyzed': len(all_results),
        'timing_consistency': {},
        'common_issues': {},
        'recommendations': []
    }

    # Analyze timing consistency
    ranges_match = [r['timing_precision']['time_ranges_match'] for r in all_results if 'time_ranges_match' in r['timing_precision']]
    aggregated['timing_consistency'] = {
        'ranges_match_percentage': (sum(ranges_match) / len(ranges_match)) * 100 if ranges_match else 0,
        'total_games_with_range_match': sum(ranges_match)
    }

    # Collect common issues
    all_issues = []
    for result in all_results:
        all_issues.extend(result['issues_identified'])

    # Count issue frequencies
    issue_counts = {}
    for issue in all_issues:
        issue_type = issue.split(':')[0] if ':' in issue else issue
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    aggregated['common_issues'] = issue_counts

    # Generate recommendations
    if aggregated['timing_consistency']['ranges_match_percentage'] < 80:
        aggregated['recommendations'].append("Timing alignment issues detected - investigate time conversion logic")

    if issue_counts.get('Stint', 0) > 0:
        aggregated['recommendations'].append("Player lineup inconsistencies found - validate rotation data integrity")

    return aggregated

def create_timing_visualization(game_id: str, save_path: Optional[str] = None):
    """
    Create visualizations of timing data for a game.

    Args:
        game_id: NBA game ID
        save_path: Optional path to save the plot
    """
    try:
        # Get data
        rotation_endpoint = GameRotation(game_id=game_id)
        rotation_dfs = rotation_endpoint.get_data_frames()

        pbp_endpoint = PlayByPlayV3(game_id=game_id)
        pbp_df = pbp_endpoint.get_data_frames()[0]

        # Convert times
        rotation_times = []
        for team_df in rotation_dfs:
            for _, row in team_df.iterrows():
                if pd.notna(row['IN_TIME_REAL']):
                    rotation_times.append(row['IN_TIME_REAL'] / 10 / 60)  # Convert to minutes
                if pd.notna(row['OUT_TIME_REAL']):
                    rotation_times.append(row['OUT_TIME_REAL'] / 10 / 60)

        pbp_times = []
        for _, row in pbp_df.iterrows():
            if pd.notna(row['clock']) and pd.notna(row['period']):
                seconds = convert_period_clock_to_seconds(int(row['period']), row['clock'])
                pbp_times.append(seconds / 60)  # Convert to minutes

        # Create visualization
        plt.figure(figsize=(12, 6))

        # Plot histograms
        plt.subplot(1, 2, 1)
        plt.hist(rotation_times, bins=50, alpha=0.7, label='Rotation Events', color='blue')
        plt.hist(pbp_times, bins=50, alpha=0.7, label='PBP Events', color='red')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Event Count')
        plt.title(f'Event Timing Distribution - Game {game_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot cumulative distribution
        plt.subplot(1, 2, 2)
        rotation_sorted = sorted(rotation_times)
        pbp_sorted = sorted(pbp_times)

        plt.plot(rotation_sorted, range(len(rotation_sorted)), label='Rotation Events', color='blue')
        plt.plot(pbp_sorted, range(len(pbp_sorted)), label='PBP Events', color='red')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Cumulative Events')
        plt.title('Cumulative Event Timing')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved timing visualization to {save_path}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error creating timing visualization: {e}")

if __name__ == "__main__":
    # Analyze timing precision for sample games
    sample_games = ["0022200001", "0022200002", "0022200003"]

    print("TIMING PRECISION ANALYSIS")
    print("=" * 50)

    # Single game analysis
    print("\n1. SINGLE GAME ANALYSIS:")
    game_result = analyze_game_timing_precision(sample_games[0])
    precision = game_result['timing_precision']
    boundaries = game_result['stint_boundaries']

    print(f"Game: {game_result['game_id']}")
    print(f"Rotation time range: {precision['rotation_times_range']}")
    print(f"PBP time range: {precision['pbp_times_range']}")
    print(f"Time ranges match: {precision['time_ranges_match']}")
    print(f"Number of stints: {boundaries['num_stints']}")
    print(f"Stint duration stats: min={min(boundaries['boundary_gaps']):.1f}s, max={max(boundaries['boundary_gaps']):.1f}s, avg={np.mean(boundaries['boundary_gaps']):.1f}s")

    if game_result['issues_identified']:
        print(f"Issues identified: {len(game_result['issues_identified'])}")
        for issue in game_result['issues_identified'][:3]:  # Show first 3
            print(f"  - {issue}")

    # Multi-game analysis
    print("\n2. MULTI-GAME ANALYSIS:")
    multi_results = analyze_multiple_games(sample_games)
    consistency = multi_results['timing_consistency']
    print(f"Games analyzed: {multi_results['games_analyzed']}")
    print(f"Timing ranges match: {consistency['ranges_match_percentage']:.1f}%")

    if multi_results['common_issues']:
        print("Common issues:")
        for issue_type, count in sorted(multi_results['common_issues'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {issue_type}: {count} occurrences")

    if multi_results['recommendations']:
        print("Recommendations:")
        for rec in multi_results['recommendations']:
            print(f"  - {rec}")

    # Create visualization
    print("\n3. CREATING VISUALIZATION...")
    create_timing_visualization(sample_games[0], f"timing_analysis_{sample_games[0]}.png")

    print("\nTiming analysis complete!")
