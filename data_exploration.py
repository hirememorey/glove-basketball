"""
Data Exploration Script for PADIM Stint-Level Defensive Outcome Collection

This script analyzes raw NBA API data to understand:
1. Timing precision between GameRotation and PlayByPlay endpoints
2. Data quality issues and missing events
3. Basic possession logic patterns
4. Mapping between different time formats

Part of Step 1: Domain Understanding & Data Exploration
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import GameRotation, PlayByPlayV3, BoxScoreSummaryV2
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_game_timing_and_data(game_id: str) -> Dict[str, Any]:
    """
    Analyze timing precision and data quality for a single game.

    Args:
        game_id: NBA game ID (e.g., '0022200001')

    Returns:
        Dict containing analysis results
    """
    logger.info(f"Analyzing game {game_id}")

    results = {
        'game_id': game_id,
        'rotation_data': None,
        'pbp_data': None,
        'timing_analysis': {},
        'data_quality': {},
        'possession_patterns': []
    }

    try:
        # Get rotation data
        logger.info("Fetching rotation data...")
        rotation_endpoint = GameRotation(game_id=game_id)
        rotation_dfs = rotation_endpoint.get_data_frames()

        if len(rotation_dfs) != 2:
            logger.warning(f"Expected 2 teams, got {len(rotation_dfs)} for game {game_id}")
            return results

        results['rotation_data'] = rotation_dfs

        # Get play-by-play data
        logger.info("Fetching play-by-play data...")
        pbp_endpoint = PlayByPlayV3(game_id=game_id)
        pbp_df = pbp_endpoint.get_data_frames()[0]
        results['pbp_data'] = pbp_df

        # Get box score for validation
        logger.info("Fetching box score data...")
        box_score_endpoint = BoxScoreSummaryV2(game_id=game_id)
        box_score_df = box_score_endpoint.get_data_frames()[0]

        # Analyze timing precision
        results['timing_analysis'] = analyze_timing_precision(rotation_dfs, pbp_df)

        # Analyze data quality
        results['data_quality'] = analyze_data_quality(pbp_df, box_score_df)

        # Analyze possession patterns
        results['possession_patterns'] = analyze_possession_patterns(pbp_df)

        logger.info(f"Successfully analyzed game {game_id}")

    except Exception as e:
        logger.error(f"Error analyzing game {game_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return results

def convert_period_clock_to_seconds(period: int, clock_str: str) -> float:
    """
    Convert period + clock format to total seconds from game start.

    IMPORTANT: NBA PBP clock counts DOWN from 12:00 to 0:00 within each period.

    Args:
        period: Period number (1-4 regular, 5+ overtime)
        clock_str: Clock in format "PT12M00.00S" (counts down)

    Returns:
        Total seconds from game start (counting up)
    """
    # Parse clock string (e.g., "PT12M00.00S" -> 12*60 + 0 = 720 seconds remaining)
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
        # Regular periods: 12 minutes each (720 seconds)
        # Overtime periods: 5 minutes each (300 seconds)
        base_period_length = 720  # 12 minutes
        ot_period_length = 300    # 5 minutes

        if period <= 4:
            # Regular quarters: time elapsed = (period-1)*720 + (720 - time_remaining)
            periods_completed = period - 1
            time_elapsed_this_period = base_period_length - time_remaining
            total_seconds = periods_completed * base_period_length + time_elapsed_this_period
        else:
            # Overtime periods
            regular_time_elapsed = 4 * base_period_length
            ot_periods_completed = period - 4
            time_elapsed_this_ot = ot_period_length - time_remaining
            total_seconds = regular_time_elapsed + (ot_periods_completed - 1) * ot_period_length + time_elapsed_this_ot

        return total_seconds

    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse clock string '{clock_str}': {e}")
        return 0.0

def analyze_timing_precision(rotation_dfs: List[pd.DataFrame], pbp_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze timing precision between rotation and PBP data.

    Returns:
        Dict with timing analysis results
    """
    analysis = {
        'rotation_time_format': None,
        'pbp_time_format': None,
        'time_mapping_issues': [],
        'sample_time_conversions': [],
        'timing_alignment_check': {}
    }

    # Analyze rotation data timing
    logger.info("Analyzing rotation timing format...")
    if len(rotation_dfs) >= 1:
        sample_rotation = rotation_dfs[0]
        analysis['rotation_columns'] = list(sample_rotation.columns)
        analysis['rotation_time_format'] = 'seconds_since_start'  # Based on IN_TIME_REAL/OUT_TIME_REAL

        # Sample rotation times
        in_times = sample_rotation['IN_TIME_REAL'].dropna().head(5).tolist()
        out_times = sample_rotation['OUT_TIME_REAL'].dropna().head(5).tolist()
        analysis['sample_rotation_times'] = {
            'in_times': in_times,
            'out_times': out_times,
            'max_time': sample_rotation['OUT_TIME_REAL'].max(),
            'min_time': sample_rotation['IN_TIME_REAL'].min()
        }

    # Analyze PBP data timing
    logger.info("Analyzing PBP timing format...")
    analysis['pbp_columns'] = list(pbp_df.columns)

    # Look for timing columns
    timing_cols = [col for col in pbp_df.columns if 'time' in col.lower() or 'clock' in col.lower() or 'period' in col.lower()]
    analysis['pbp_timing_columns'] = timing_cols

    if 'clock' in pbp_df.columns and 'period' in pbp_df.columns:
        analysis['pbp_time_format'] = 'period_clock_format'
        sample_clocks = pbp_df['clock'].dropna().head(10).tolist()
        sample_periods = pbp_df['period'].dropna().head(10).tolist()
        analysis['sample_pbp_clocks'] = sample_clocks
        analysis['sample_pbp_periods'] = sample_periods

        # Test time conversions
        conversions = []
        for i in range(min(10, len(pbp_df))):
            row = pbp_df.iloc[i]
            if pd.notna(row['clock']) and pd.notna(row['period']):
                converted = convert_period_clock_to_seconds(int(row['period']), row['clock'])
                conversions.append({
                    'original': f"P{row['period']} {row['clock']}",
                    'converted_seconds': converted
                })
        analysis['sample_time_conversions'] = conversions

    # Check timing alignment between rotation and PBP
    logger.info("Checking timing alignment...")
    if len(rotation_dfs) >= 1 and 'clock' in pbp_df.columns:
        rotation_max = rotation_dfs[0]['OUT_TIME_REAL'].max()

        # Convert first PBP event time
        first_pbp = pbp_df.iloc[0]
        if pd.notna(first_pbp['clock']) and pd.notna(first_pbp['period']):
            first_pbp_seconds = convert_period_clock_to_seconds(int(first_pbp['period']), first_pbp['clock'])

        # Convert last PBP event time
        last_pbp = pbp_df.iloc[-1]
        if pd.notna(last_pbp['clock']) and pd.notna(last_pbp['period']):
            last_pbp_seconds = convert_period_clock_to_seconds(int(last_pbp['period']), last_pbp['clock'])

        analysis['timing_alignment_check'] = {
            'rotation_game_duration': rotation_max,
            'pbp_first_event_seconds': first_pbp_seconds if 'first_pbp_seconds' in locals() else None,
            'pbp_last_event_seconds': last_pbp_seconds if 'last_pbp_seconds' in locals() else None,
            'alignment_status': 'To be determined'
        }

    # Analyze time conversion challenges
    analysis['time_mapping_issues'] = [
        "Rotation uses total seconds from game start",
        "PBP uses period + clock format (PT12M00.00S)",
        "Need to handle overtime periods (5 min vs 12 min)",
        "Quarter breaks and timeouts affect continuous time",
        "Clock runs backwards within each period"
    ]

    return analysis

def analyze_data_quality(pbp_df: pd.DataFrame, box_score_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze data quality issues in PBP data.

    Returns:
        Dict with data quality analysis
    """
    analysis = {
        'total_events': len(pbp_df),
        'missing_data': {},
        'event_type_distribution': {},
        'validation_against_box_score': {}
    }

    # Check for missing data
    logger.info("Checking for missing data in PBP...")
    for col in pbp_df.columns:
        null_count = pbp_df[col].isnull().sum()
        if null_count > 0:
            analysis['missing_data'][col] = {
                'null_count': null_count,
                'null_percentage': (null_count / len(pbp_df)) * 100
            }

    # Analyze event types
    if 'EVENTMSGTYPE' in pbp_df.columns:
        event_counts = pbp_df['EVENTMSGTYPE'].value_counts().to_dict()
        analysis['event_type_distribution'] = event_counts

        # Map event types (based on NBA API documentation)
        event_type_map = {
            1: 'Made Field Goal',
            2: 'Missed Field Goal',
            3: 'Free Throw',
            4: 'Rebound',
            5: 'Turnover',
            6: 'Foul',
            7: 'Violation',
            8: 'Substitution',
            9: 'Timeout',
            10: 'Jump Ball',
            11: 'Ejection',
            12: 'Start Period',
            13: 'End Period'
        }
        analysis['event_type_descriptions'] = {k: event_type_map.get(k, f'Unknown ({k})') for k in event_counts.keys()}

    # Validate against box score
    if not box_score_df.empty:
        try:
            # Compare total field goals
            home_team_stats = box_score_df[box_score_df['TEAM_ID'] == box_score_df.iloc[0]['TEAM_ID']].iloc[0]
            away_team_stats = box_score_df[box_score_df['TEAM_ID'] == box_score_df.iloc[1]['TEAM_ID']].iloc[0]

            analysis['validation_against_box_score'] = {
                'home_team_fgm': home_team_stats.get('FGM', 'N/A'),
                'away_team_fgm': away_team_stats.get('FGM', 'N/A'),
                'pbp_field_goals_found': 'To be calculated'
            }
        except Exception as e:
            logger.warning(f"Could not validate against box score: {e}")

    return analysis

def analyze_possession_patterns(pbp_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Analyze basic possession patterns in PBP data.

    Returns:
        List of possession pattern observations
    """
    patterns = []

    # Look for field goal attempts (events that change possession)
    if 'EVENTMSGTYPE' in pbp_df.columns:
        fg_events = pbp_df[pbp_df['EVENTMSGTYPE'].isin([1, 2])]  # Made/missed field goals

        patterns.append({
            'pattern_type': 'field_goal_attempts',
            'count': len(fg_events),
            'description': 'Events that typically change possession (made/missed FGs)'
        })

        # Analyze shot types if available
        if 'EVENTMSGACTIONTYPE' in pbp_df.columns:
            shot_type_counts = fg_events['EVENTMSGACTIONTYPE'].value_counts().head(5).to_dict()
            patterns.append({
                'pattern_type': 'shot_type_distribution',
                'data': shot_type_counts,
                'description': 'Most common shot types in field goal attempts'
            })

    # Look for rebounds (events that continue possession)
    if 'EVENTMSGTYPE' in pbp_df.columns:
        rebound_events = pbp_df[pbp_df['EVENTMSGTYPE'] == 4]
        patterns.append({
            'pattern_type': 'rebound_events',
            'count': len(rebound_events),
            'description': 'Events that continue possession (rebounds)'
        })

    # Look for turnovers
    if 'EVENTMSGTYPE' in pbp_df.columns:
        turnover_events = pbp_df[pbp_df['EVENTMSGTYPE'] == 5]
        patterns.append({
            'pattern_type': 'turnover_events',
            'count': len(turnover_events),
            'description': 'Events that change possession (turnovers)'
        })

    return patterns

def print_analysis_summary(analysis_results: Dict[str, Any]):
    """Print a summary of the analysis results."""
    print("\n" + "="*80)
    print(f"ANALYSIS SUMMARY FOR GAME {analysis_results['game_id']}")
    print("="*80)

    # Timing Analysis
    print("\n1. TIMING ANALYSIS:")
    timing = analysis_results['timing_analysis']
    print(f"   Rotation Time Format: {timing.get('rotation_time_format', 'Unknown')}")
    print(f"   PBP Time Format: {timing.get('pbp_time_format', 'Unknown')}")

    if 'sample_rotation_times' in timing:
        rt = timing['sample_rotation_times']
        print(f"   Sample Rotation Times: {rt['in_times'][:3]}... (max: {rt['max_time']})")

    if 'sample_pbp_clocks' in timing:
        print(f"   Sample PBP Clocks: {timing['sample_pbp_clocks'][:5]}")

    print(f"   Time Mapping Issues: {len(timing.get('time_mapping_issues', []))}")
    for issue in timing.get('time_mapping_issues', []):
        print(f"     - {issue}")

    # Data Quality
    print("\n2. DATA QUALITY:")
    quality = analysis_results['data_quality']
    print(f"   Total PBP Events: {quality.get('total_events', 'N/A')}")

    missing = quality.get('missing_data', {})
    if missing:
        print(f"   Missing Data in {len(missing)} columns:")
        for col, stats in list(missing.items())[:5]:  # Show first 5
            print(f"     {col}: {stats['null_count']} nulls ({stats['null_percentage']:.1f}%)")
    event_dist = quality.get('event_type_distribution', {})
    if event_dist:
        print(f"   Event Type Distribution (top 5):")
        for event_type, count in sorted(event_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            desc = quality.get('event_type_descriptions', {}).get(event_type, f'Unknown ({event_type})')
            print(f"     {event_type}: {count} events ({desc})")

    # Possession Patterns
    print("\n3. POSSESSION PATTERNS:")
    patterns = analysis_results['possession_patterns']
    for pattern in patterns:
        print(f"   {pattern['pattern_type']}: {pattern['count']} events")
        print(f"     {pattern['description']}")
        if 'data' in pattern:
            print(f"     Data: {pattern['data']}")

    print("\n" + "="*80)

if __name__ == "__main__":
    # Analyze a sample game
    sample_game_id = "0022200001"  # First game from our database

    print("Starting data exploration for PADIM stint-level defensive outcome collection...")
    print(f"Analyzing game: {sample_game_id}")

    results = analyze_game_timing_and_data(sample_game_id)
    print_analysis_summary(results)

    # Save detailed results for further analysis
    import json
    output_file = f"data_exploration_{sample_game_id}.json"
    with open(output_file, 'w') as f:
        # Convert DataFrames to dicts for JSON serialization
        serializable_results = results.copy()
        if serializable_results['rotation_data']:
            serializable_results['rotation_data'] = [df.to_dict('records') for df in serializable_results['rotation_data']]
        if serializable_results['pbp_data'] is not None:
            serializable_results['pbp_data'] = serializable_results['pbp_data'].to_dict('records')

        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nDetailed results saved to {output_file}")
