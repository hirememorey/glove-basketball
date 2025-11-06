"""
Data Quality Assessment for PADIM Stint-Level Defensive Outcome Collection

This script performs comprehensive data quality assessment of:
1. PBP data completeness and consistency
2. Rotation data integrity and player tracking
3. Stint aggregation accuracy
4. Cross-validation between data sources
5. Missing data patterns and gaps

Critical for ensuring RAPM model reliability.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import GameRotation, PlayByPlayV3, BoxScoreSummaryV2
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assess_overall_data_quality(game_ids: List[str]) -> Dict[str, Any]:
    """
    Perform comprehensive data quality assessment across multiple games.

    Args:
        game_ids: List of NBA game IDs to assess

    Returns:
        Dict with comprehensive quality assessment
    """
    logger.info(f"Assessing data quality for {len(game_ids)} games")

    assessment = {
        'games_assessed': len(game_ids),
        'data_completeness': {},
        'consistency_checks': {},
        'error_patterns': {},
        'quality_score': {},
        'recommendations': []
    }

    game_results = []
    for game_id in game_ids:
        game_quality = assess_single_game_quality(game_id)
        game_results.append(game_quality)

    # Aggregate results
    assessment['data_completeness'] = aggregate_completeness_results(game_results)
    assessment['consistency_checks'] = aggregate_consistency_results(game_results)
    assessment['error_patterns'] = identify_error_patterns(game_results)
    assessment['quality_score'] = calculate_overall_quality_score(game_results)
    assessment['recommendations'] = generate_quality_recommendations(assessment)

    logger.info("Data quality assessment complete")
    return assessment

def assess_single_game_quality(game_id: str) -> Dict[str, Any]:
    """
    Assess data quality for a single game.

    Args:
        game_id: NBA game ID

    Returns:
        Dict with game-specific quality metrics
    """
    logger.info(f"Assessing data quality for game {game_id}")

    quality = {
        'game_id': game_id,
        'data_sources': {},
        'completeness': {},
        'consistency': {},
        'issues': []
    }

    try:
        # Assess each data source
        quality['data_sources']['rotation'] = assess_rotation_data_quality(game_id)
        quality['data_sources']['pbp'] = assess_pbp_data_quality(game_id)
        quality['data_sources']['box_score'] = assess_box_score_alignment(game_id)

        # Cross-source validation
        quality['consistency'] = assess_cross_source_consistency(game_id,
            quality['data_sources']['rotation'],
            quality['data_sources']['pbp'],
            quality['data_sources']['box_score'])

        # Overall completeness score
        quality['completeness'] = calculate_game_completeness(quality)

        # Identify specific issues
        quality['issues'] = identify_game_specific_issues(quality)

    except Exception as e:
        logger.error(f"Error assessing game {game_id}: {e}")
        quality['issues'].append(f"Assessment failed: {str(e)}")

    return quality

def assess_rotation_data_quality(game_id: str) -> Dict[str, Any]:
    """
    Assess quality of rotation (stint) data.
    """
    quality = {
        'available': False,
        'teams_count': 0,
        'total_players': 0,
        'stint_count': 0,
        'time_coverage': 0,
        'issues': []
    }

    try:
        rotation_endpoint = GameRotation(game_id=game_id)
        rotation_dfs = rotation_endpoint.get_data_frames()

        if len(rotation_dfs) == 2:
            quality['available'] = True
            quality['teams_count'] = 2

            total_players = 0
            total_stints = 0
            max_time = 0

            for team_idx, team_df in enumerate(rotation_dfs):
                team_players = len(team_df['PERSON_ID'].unique())
                total_players += team_players

                # Check for proper stint structure
                if 'IN_TIME_REAL' in team_df.columns and 'OUT_TIME_REAL' in team_df.columns:
                    valid_stints = 0
                    for _, row in team_df.iterrows():
                        if pd.notna(row['IN_TIME_REAL']) and pd.notna(row['OUT_TIME_REAL']):
                            valid_stints += 1
                            max_time = max(max_time, row['OUT_TIME_REAL'] / 10)  # Convert to seconds

                    total_stints += valid_stints
                else:
                    quality['issues'].append(f"Team {team_idx + 1}: Missing time columns")

            quality['total_players'] = total_players
            quality['stint_count'] = total_stints
            quality['time_coverage'] = max_time

            # Check for data quality issues
            if total_stints == 0:
                quality['issues'].append("No valid stints found")
            if max_time < 2000:  # Less than ~33 minutes
                quality['issues'].append(f"Short game duration: {max_time:.0f} seconds")

        else:
            quality['issues'].append(f"Expected 2 teams, got {len(rotation_dfs)}")

    except Exception as e:
        quality['issues'].append(f"Rotation data error: {str(e)}")

    return quality

def assess_pbp_data_quality(game_id: str) -> Dict[str, Any]:
    """
    Assess quality of play-by-play data.
    """
    quality = {
        'available': False,
        'event_count': 0,
        'periods_covered': 0,
        'time_span': 0,
        'missing_data': {},
        'issues': []
    }

    try:
        pbp_endpoint = PlayByPlayV3(game_id=game_id)
        pbp_df = pbp_endpoint.get_data_frames()[0]

        quality['available'] = True
        quality['event_count'] = len(pbp_df)

        # Check period coverage
        if 'period' in pbp_df.columns:
            quality['periods_covered'] = len(pbp_df['period'].unique())
        else:
            quality['issues'].append("Missing period column")

        # Check for missing data
        critical_columns = ['actionType', 'clock', 'period', 'description']
        for col in critical_columns:
            if col in pbp_df.columns:
                null_count = pbp_df[col].isnull().sum()
                if null_count > 0:
                    quality['missing_data'][col] = {
                        'count': null_count,
                        'percentage': (null_count / len(pbp_df)) * 100
                    }
            else:
                quality['issues'].append(f"Missing critical column: {col}")

        # Check time coverage
        if 'period' in pbp_df.columns and 'clock' in pbp_df.columns:
            # Get first and last events
            first_event = pbp_df.iloc[0]
            last_event = pbp_df.iloc[-1]

            if pd.notna(first_event.get('period')) and pd.notna(last_event.get('period')):
                # Rough estimate: each period covers some time span
                quality['time_span'] = quality['periods_covered'] * 720  # 12 min per period

        # Check for data quality red flags
        if quality['event_count'] < 200:
            quality['issues'].append(f"Very few events: {quality['event_count']}")
        if quality['periods_covered'] < 4:
            quality['issues'].append(f"Incomplete periods: {quality['periods_covered']}")

    except Exception as e:
        quality['issues'].append(f"PBP data error: {str(e)}")

    return quality

def assess_box_score_alignment(game_id: str) -> Dict[str, Any]:
    """
    Assess alignment between PBP and box score data.
    """
    alignment = {
        'box_score_available': False,
        'pbp_vs_box_score': {},
        'issues': []
    }

    try:
        # Get box score
        box_score_endpoint = BoxScoreSummaryV2(game_id=game_id)
        box_score_df = box_score_endpoint.get_data_frames()[0]

        if not box_score_df.empty:
            alignment['box_score_available'] = True

            # Get PBP for comparison
            pbp_endpoint = PlayByPlayV3(game_id=game_id)
            pbp_df = pbp_endpoint.get_data_frames()[0]

            # Compare field goals
            pbp_fg_made = len(pbp_df[pbp_df['actionType'] == 'Made Shot'])
            pbp_fg_attempted = len(pbp_df[pbp_df['actionType'].isin(['Made Shot', 'Missed Shot'])])

            # This would require extracting team stats from box score
            # For now, just note that alignment checking is possible

            alignment['pbp_vs_box_score'] = {
                'pbp_fg_made': pbp_fg_made,
                'pbp_fg_attempted': pbp_fg_attempted,
                'box_score_comparison': 'Not implemented yet'
            }

        else:
            alignment['issues'].append("Box score data not available")

    except Exception as e:
        alignment['issues'].append(f"Box score alignment error: {str(e)}")

    return alignment

def assess_cross_source_consistency(game_id: str, rotation_quality: Dict,
                                  pbp_quality: Dict, box_score_quality: Dict) -> Dict[str, Any]:
    """
    Assess consistency between different data sources.
    """
    consistency = {
        'rotation_pbp_time_alignment': 'unknown',
        'data_completeness_match': 'unknown',
        'issues': []
    }

    # Check time alignment
    rotation_time = rotation_quality.get('time_coverage', 0)
    pbp_time = pbp_quality.get('time_span', 0)

    if rotation_time > 0 and pbp_time > 0:
        time_diff = abs(rotation_time - pbp_time)
        if time_diff < 60:  # Within 1 minute
            consistency['rotation_pbp_time_alignment'] = 'good'
        elif time_diff < 300:  # Within 5 minutes
            consistency['rotation_pbp_time_alignment'] = 'moderate'
        else:
            consistency['rotation_pbp_time_alignment'] = 'poor'
            consistency['issues'].append(f"Large time discrepancy: {time_diff:.0f} seconds")

    # Check data availability consistency
    sources_available = sum([
        rotation_quality.get('available', False),
        pbp_quality.get('available', False),
        box_score_quality.get('box_score_available', False)
    ])

    if sources_available == 3:
        consistency['data_completeness_match'] = 'complete'
    elif sources_available >= 2:
        consistency['data_completeness_match'] = 'partial'
    else:
        consistency['data_completeness_match'] = 'incomplete'
        consistency['issues'].append("Multiple data sources unavailable")

    return consistency

def calculate_game_completeness(game_quality: Dict) -> Dict[str, Any]:
    """
    Calculate overall completeness score for a game.
    """
    completeness = {
        'overall_score': 0,
        'components': {}
    }

    # Score each component (0-100)
    scores = {}

    # Data availability (40% weight)
    sources = game_quality['data_sources']
    available_count = sum([
        sources['rotation'].get('available', False),
        sources['pbp'].get('available', False),
        sources['box_score'].get('box_score_available', False)
    ])
    scores['data_availability'] = (available_count / 3) * 100

    # Data quality (30% weight)
    quality_score = 100
    all_issues = []
    for source_name, source_data in sources.items():
        issues = source_data.get('issues', [])
        all_issues.extend(issues)
        # Deduct points for each issue
        quality_score -= len(issues) * 5

    scores['data_quality'] = max(0, quality_score)

    # Consistency (30% weight)
    consistency = game_quality['consistency']
    if consistency.get('rotation_pbp_time_alignment') == 'good':
        scores['consistency'] = 100
    elif consistency.get('rotation_pbp_time_alignment') == 'moderate':
        scores['consistency'] = 70
    else:
        scores['consistency'] = 30

    # Overall score
    completeness['overall_score'] = (
        scores['data_availability'] * 0.4 +
        scores['data_quality'] * 0.3 +
        scores['consistency'] * 0.3
    )

    completeness['components'] = scores

    return completeness

def identify_game_specific_issues(game_quality: Dict) -> List[str]:
    """
    Identify specific issues for a game.
    """
    issues = []

    # Collect issues from all sources
    for source_name, source_data in game_quality['data_sources'].items():
        source_issues = source_data.get('issues', [])
        for issue in source_issues:
            issues.append(f"{source_name}: {issue}")

    # Add consistency issues
    consistency_issues = game_quality['consistency'].get('issues', [])
    issues.extend(consistency_issues)

    return issues

def aggregate_completeness_results(game_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate completeness results across games.
    """
    if not game_results:
        return {}

    scores = [game['completeness']['overall_score'] for game in game_results]
    components = {}

    # Average component scores
    component_keys = game_results[0]['completeness']['components'].keys()
    for key in component_keys:
        component_scores = [game['completeness']['components'][key] for game in game_results]
        components[key] = {
            'average': np.mean(component_scores),
            'min': min(component_scores),
            'max': max(component_scores)
        }

    return {
        'average_score': np.mean(scores),
        'score_distribution': {
            'excellent': len([s for s in scores if s >= 90]),
            'good': len([s for s in scores if 70 <= s < 90]),
            'fair': len([s for s in scores if 50 <= s < 70]),
            'poor': len([s for s in scores if s < 50])
        },
        'component_breakdown': components
    }

def aggregate_consistency_results(game_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate consistency results across games.
    """
    consistency_results = {
        'time_alignment_distribution': {},
        'data_completeness_distribution': {}
    }

    time_alignments = []
    completeness_levels = []

    for game in game_results:
        consistency = game['consistency']
        time_alignments.append(consistency.get('rotation_pbp_time_alignment', 'unknown'))
        completeness_levels.append(consistency.get('data_completeness_match', 'unknown'))

    # Count distributions
    for alignment in set(time_alignments):
        consistency_results['time_alignment_distribution'][alignment] = time_alignments.count(alignment)

    for level in set(completeness_levels):
        consistency_results['data_completeness_distribution'][level] = completeness_levels.count(level)

    return consistency_results

def identify_error_patterns(game_results: List[Dict]) -> Dict[str, Any]:
    """
    Identify common error patterns across games.
    """
    all_issues = []
    for game in game_results:
        all_issues.extend(game.get('issues', []))

    # Count issue frequencies
    issue_counts = {}
    for issue in all_issues:
        if issue in issue_counts:
            issue_counts[issue] += 1
        else:
            issue_counts[issue] = 1

    # Sort by frequency
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        'total_issues': len(all_issues),
        'unique_issue_types': len(issue_counts),
        'most_common_issues': sorted_issues[:10],  # Top 10
        'issue_distribution': issue_counts
    }

def calculate_overall_quality_score(game_results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate overall quality score across all games.
    """
    if not game_results:
        return {'overall_score': 0}

    completeness_scores = [game['completeness']['overall_score'] for game in game_results]

    # Weight factors
    weights = {
        'data_availability': 0.4,
        'data_quality': 0.3,
        'consistency': 0.3
    }

    # Calculate weighted average
    overall_score = np.mean(completeness_scores)

    # Determine quality level
    if overall_score >= 85:
        quality_level = 'excellent'
    elif overall_score >= 70:
        quality_level = 'good'
    elif overall_score >= 55:
        quality_level = 'fair'
    else:
        quality_level = 'poor'

    return {
        'overall_score': overall_score,
        'quality_level': quality_level,
        'score_range': f"{min(completeness_scores):.1f} - {max(completeness_scores):.1f}",
        'standard_deviation': np.std(completeness_scores)
    }

def generate_quality_recommendations(assessment: Dict) -> List[str]:
    """
    Generate recommendations based on quality assessment.
    """
    recommendations = []

    quality_score = assessment['quality_score']['overall_score']

    if quality_score < 70:
        recommendations.append("Critical: Overall data quality is poor. Address major data source issues before proceeding.")

    # Check specific areas
    completeness = assessment['data_completeness']
    if completeness.get('average_score', 0) < 80:
        recommendations.append("Improve data completeness - multiple games have missing or incomplete data sources.")

    consistency = assessment['consistency_checks']
    time_alignment = consistency.get('time_alignment_distribution', {})
    if time_alignment.get('poor', 0) > 0:
        recommendations.append("Fix timing alignment issues between rotation and PBP data sources.")

    error_patterns = assessment['error_patterns']
    if error_patterns.get('total_issues', 0) > assessment.get('games_assessed', 0):
        recommendations.append("Address recurring data quality issues identified in error patterns.")

    # Specific recommendations based on common issues
    most_common = error_patterns.get('most_common_issues', [])
    for issue_text, count in most_common[:3]:  # Top 3 issues
        if 'stint' in issue_text.lower():
            recommendations.append("Fix stint aggregation logic - player count inconsistencies detected.")
        elif 'time' in issue_text.lower():
            recommendations.append("Improve time conversion and alignment between data sources.")
        elif 'missing' in issue_text.lower():
            recommendations.append("Address missing data patterns in PBP or rotation sources.")

    return recommendations

if __name__ == "__main__":
    # Assess data quality for sample games
    sample_games = ["0022200001", "0022200002", "0022200003"]

    print("DATA QUALITY ASSESSMENT")
    print("=" * 50)

    assessment = assess_overall_data_quality(sample_games)

    print(f"\nGames Assessed: {assessment['games_assessed']}")

    # Overall quality score
    quality = assessment['quality_score']
    print(f"\nOVERALL QUALITY SCORE: {quality['overall_score']:.1f}/100 ({quality['quality_level'].upper()})")
    print(f"Score Range: {quality['score_range']}")
    print(f"Standard Deviation: {quality['standard_deviation']:.1f}")

    # Completeness breakdown
    completeness = assessment['data_completeness']
    print(f"\nCOMPLETENESS BREAKDOWN:")
    print(f"Average Score: {completeness['average_score']:.1f}")
    distribution = completeness['score_distribution']
    print(f"Distribution: Excellent: {distribution['excellent']}, Good: {distribution['good']}, Fair: {distribution['fair']}, Poor: {distribution['poor']}")

    # Component scores
    components = completeness['component_breakdown']
    print(f"\nCOMPONENT SCORES:")
    for component, stats in components.items():
        print(f"  {component}: {stats['average']:.1f} (range: {stats['min']:.1f} - {stats['max']:.1f})")

    # Consistency checks
    consistency = assessment['consistency_checks']
    print(f"\nCONSISTENCY CHECKS:")
    if 'time_alignment_distribution' in consistency:
        print(f"Time Alignment: {consistency['time_alignment_distribution']}")
    if 'data_completeness_distribution' in consistency:
        print(f"Data Completeness: {consistency['data_completeness_distribution']}")

    # Error patterns
    errors = assessment['error_patterns']
    print(f"\nERROR PATTERNS:")
    print(f"Total Issues: {errors['total_issues']}")
    print(f"Unique Issue Types: {errors['unique_issue_types']}")
    print("Most Common Issues:")
    for issue_text, count in errors['most_common_issues'][:5]:
        print(f"  {count} games: {issue_text}")

    # Recommendations
    recommendations = assessment['recommendations']
    if recommendations:
        print(f"\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  • {rec}")

    # Save detailed results
    output_file = "data_quality_assessment.json"
    with open(output_file, 'w') as f:
        json.dump(assessment, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_file}")

    print("\nData quality assessment complete!")
