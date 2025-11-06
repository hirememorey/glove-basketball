"""
Free Throw Logic Analysis for PADIM Defensive Impact Modeling

This script systematically analyzes free throw patterns to resolve critical unknowns
in possession attribution and defensive outcome calculation.

Key Unknowns to Resolve:
1. Possession attribution after free throw sequences
2. Sequence completion logic (which FT determines possession)
3. Technical foul vs regular foul differences
4. And-one situation handling
5. Free throw rebound attribution
6. Free throw timing and stint boundaries

Based on first principles analysis of basketball rules and PBP data patterns.
"""

import pandas as pd
from nba_api.stats.endpoints import PlayByPlayV3
from typing import Dict, List, Optional, Tuple, Any
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreeThrowAnalyzer:
    """
    Comprehensive analyzer for free throw patterns and possession logic.
    """

    def __init__(self):
        self.ft_patterns = {
            'sequences': [],
            'possession_outcomes': [],
            'rebound_patterns': [],
            'and_one_situations': [],
            'technical_fouls': []
        }

    def analyze_game_free_throws(self, game_id: str) -> Dict[str, Any]:
        """
        Analyze all free throw patterns in a single game.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with comprehensive free throw analysis
        """
        logger.info(f"Analyzing free throw patterns for game {game_id}")

        try:
            # Get PBP data
            pbp_endpoint = PlayByPlayV3(game_id=game_id)
            pbp_df = pbp_endpoint.get_data_frames()[0]

            # Extract free throw sequences
            ft_sequences = self._extract_free_throw_sequences(pbp_df)

            # Analyze each sequence
            analyzed_sequences = []
            for seq in ft_sequences:
                analysis = self._analyze_ft_sequence(seq, pbp_df)
                analyzed_sequences.append(analysis)

            return {
                'game_id': game_id,
                'total_ft_events': len(pbp_df[pbp_df['actionType'] == 'Free Throw']),
                'sequences_found': len(ft_sequences),
                'analyzed_sequences': analyzed_sequences,
                'patterns': self._summarize_patterns(analyzed_sequences)
            }

        except Exception as e:
            logger.error(f"Error analyzing free throws for game {game_id}: {e}")
            return {'error': str(e)}

    def _extract_free_throw_sequences(self, pbp_df: pd.DataFrame) -> List[List[pd.Series]]:
        """
        Extract free throw sequences from PBP data.
        A sequence is consecutive free throws by the same player.
        """
        ft_events = pbp_df[pbp_df['actionType'] == 'Free Throw'].copy()
        sequences = []

        if ft_events.empty:
            return sequences

        current_sequence = []
        prev_player = None
        prev_period = None

        for idx, ft in ft_events.iterrows():
            current_player = ft.get('personId')
            current_period = ft.get('period')

            # Check if this FT continues the current sequence
            if (current_player == prev_player and current_period == prev_period):
                current_sequence.append(ft)
            else:
                # Process previous sequence if it exists
                if current_sequence:
                    sequences.append(current_sequence)

                # Start new sequence
                current_sequence = [ft]

            prev_player = current_player
            prev_period = current_period

        # Process final sequence
        if current_sequence:
            sequences.append(current_sequence)

        return sequences

    def _analyze_ft_sequence(self, ft_sequence: List[pd.Series], full_pbp: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single free throw sequence for possession implications.
        """
        if not ft_sequence:
            return {}

        first_ft = ft_sequence[0]
        last_ft = ft_sequence[-1]

        # Basic sequence info
        sequence_info = {
            'player_id': first_ft.get('personId'),
            'player_name': first_ft.get('playerName'),
            'period': first_ft.get('period'),
            'sequence_length': len(ft_sequence),
            'start_time': first_ft.get('clock'),
            'end_time': last_ft.get('clock'),
            'team_id': first_ft.get('teamId'),
            'team_tricode': first_ft.get('teamTricode'),
        }

        # Analyze makes vs misses
        makes = sum(1 for ft in ft_sequence if ft.get('pointsTotal', 0) > 0)
        misses = len(ft_sequence) - makes
        sequence_info.update({
            'makes': makes,
            'misses': misses,
            'final_ft_made': ft_sequence[-1].get('pointsTotal', 0) > 0
        })

        # Determine foul type from descriptions
        sequence_info['foul_type'] = self._classify_foul_type(ft_sequence)

        # Analyze what happens after the sequence
        post_sequence_events = self._analyze_post_sequence_events(last_ft, full_pbp, sequence_info['team_id'])
        sequence_info['possession_outcome'] = post_sequence_events

        return sequence_info

    def _classify_foul_type(self, ft_sequence: List[pd.Series]) -> str:
        """
        Classify the type of foul that led to free throws.
        """
        descriptions = [ft.get('description', '') for ft in ft_sequence]

        # Check for technical fouls
        if any('technical' in desc.lower() or 'flagrant' in desc.lower() for desc in descriptions):
            return 'technical_flagrant'

        # Check for and-one situations
        if any('and one' in desc.lower() or 'and-one' in desc.lower() for desc in descriptions):
            return 'and_one'

        # Check for shooting fouls vs other fouls
        if any(re.search(r'\d+\s+of\s+\d+', desc) for desc in descriptions):
            return 'shooting_foul'

        return 'unknown'

    def _analyze_post_sequence_events(self, last_ft: pd.Series, full_pbp: pd.DataFrame, ft_team_id: int) -> Dict[str, Any]:
        """
        Analyze what happens immediately after a free throw sequence ends.
        This is critical for understanding possession attribution.
        """
        outcome = {
            'next_event_type': None,
            'next_event_description': None,
            'possession_recipient': None,
            'rebound_type': None,
            'possession_continuity': None,  # 'same_team', 'opponent', 'unknown'
            'time_to_next_event': None
        }

        # Find the last FT event in the full PBP
        last_ft_idx = None
        for idx, event in full_pbp.iterrows():
            if (event.get('actionNumber') == last_ft.get('actionNumber') and
                event.get('actionType') == 'Free Throw'):
                last_ft_idx = idx
                break

        if last_ft_idx is None:
            return outcome

        # Look at next few events
        next_events = full_pbp.iloc[last_ft_idx + 1 : last_ft_idx + 6]

        for _, next_event in next_events.iterrows():
            event_type = next_event.get('actionType')

            # Skip timeouts and other non-possession events initially
            if event_type in ['Timeout', 'Instant Replay', 'Video Review']:
                continue

            outcome['next_event_type'] = event_type
            outcome['next_event_description'] = next_event.get('description', '')

            # Analyze possession implications
            next_team_id = next_event.get('teamId')

            if event_type == 'Rebound':
                outcome['rebound_type'] = self._classify_rebound(next_event)
                # CRITICAL UPDATE: Offensive rebounds after FTs continue offensive possession
                if next_team_id == ft_team_id:
                    outcome['possession_continuity'] = 'same_team'
                    outcome['possession_recipient'] = 'offensive_rebound'
                else:
                    outcome['possession_continuity'] = 'opponent'
                    outcome['possession_recipient'] = 'defensive_rebound'
            elif event_type in ['Made Shot', 'Missed Shot']:
                outcome['possession_recipient'] = 'continuing_possession'
                if next_team_id == ft_team_id:
                    outcome['possession_continuity'] = 'same_team'
                else:
                    outcome['possession_continuity'] = 'opponent'
            elif event_type == 'Turnover':
                outcome['possession_recipient'] = 'turnover'
                outcome['possession_continuity'] = 'opponent'  # Turnovers change possession
            elif event_type == 'Jump Ball':
                outcome['possession_recipient'] = 'jump_ball'
                outcome['possession_continuity'] = 'unknown'  # Jump balls establish new possession

            break  # Only analyze the first meaningful event

        return outcome

    def _classify_rebound(self, rebound_event: pd.Series) -> str:
        """
        Classify rebound type after free throws.
        For free throws, rebounds need context about which team was shooting.
        """
        description = rebound_event.get('description', '')

        # Check for explicit offensive/defensive in description
        if 'Offensive Rebound' in description:
            return 'offensive'
        elif 'Defensive Rebound' in description:
            return 'defensive'
        else:
            # For free throw context, we need to know which team was shooting the FTs
            # This requires comparing team IDs, which we don't have in this simplified analysis
            # Return the raw description for later analysis
            return f'team_rebound: {description}'

    def _summarize_patterns(self, analyzed_sequences: List[Dict]) -> Dict[str, Any]:
        """
        Summarize patterns across all analyzed sequences.
        """
        if not analyzed_sequences:
            return {}

        patterns = {
            'total_sequences': len(analyzed_sequences),
            'foul_type_distribution': {},
            'final_ft_outcomes': {'made': 0, 'missed': 0},
            'possession_outcomes': {},
            'rebound_patterns': {},
            'sequence_lengths': {},
            'possession_continuity_analysis': {
                'sequences_followed_by_same_team_action': 0,
                'sequences_followed_by_opponent_action': 0,
                'sequences_followed_by_rebound': 0,
                'total_analyzed': 0
            }
        }

        for seq in analyzed_sequences:
            # Foul types
            foul_type = seq.get('foul_type', 'unknown')
            patterns['foul_type_distribution'][foul_type] = patterns['foul_type_distribution'].get(foul_type, 0) + 1

            # Final FT outcomes
            if seq.get('final_ft_made'):
                patterns['final_ft_outcomes']['made'] += 1
            else:
                patterns['final_ft_outcomes']['missed'] += 1

            # Possession outcomes
            outcome = seq.get('possession_outcome', {})
            next_event = outcome.get('next_event_type')

            if next_event:
                patterns['possession_outcomes'][next_event] = patterns['possession_outcomes'].get(next_event, 0) + 1

            # Analyze possession continuity
            patterns['possession_continuity_analysis']['total_analyzed'] += 1

            continuity = outcome.get('possession_continuity')
            if continuity == 'same_team':
                patterns['possession_continuity_analysis']['sequences_followed_by_same_team_action'] += 1
            elif continuity == 'opponent':
                patterns['possession_continuity_analysis']['sequences_followed_by_opponent_action'] += 1
            elif next_event == 'Rebound':
                patterns['possession_continuity_analysis']['sequences_followed_by_rebound'] += 1

            # Rebound patterns
            rebound_type = outcome.get('rebound_type')
            if rebound_type and not rebound_type.startswith('team_rebound'):
                patterns['rebound_patterns'][rebound_type] = patterns['rebound_patterns'].get(rebound_type, 0) + 1
            elif rebound_type and rebound_type.startswith('team_rebound'):
                # Count team rebounds by description
                patterns['rebound_patterns'][rebound_type] = patterns['rebound_patterns'].get(rebound_type, 0) + 1

            # Sequence lengths
            seq_len = seq.get('sequence_length', 0)
            patterns['sequence_lengths'][seq_len] = patterns['sequence_lengths'].get(seq_len, 0) + 1

        return patterns

def analyze_multiple_games(game_ids: List[str]) -> Dict[str, Any]:
    """
    Analyze free throw patterns across multiple games.
    """
    analyzer = FreeThrowAnalyzer()
    all_results = []

    for game_id in game_ids:
        result = analyzer.analyze_game_free_throws(game_id)
        if 'error' not in result:
            all_results.append(result)

    # Aggregate patterns across games
    if not all_results:
        return {'error': 'No games successfully analyzed'}

    aggregated = {
        'games_analyzed': len(all_results),
        'total_sequences': sum(r['sequences_found'] for r in all_results),
        'total_ft_events': sum(r['total_ft_events'] for r in all_results),
        'aggregated_patterns': {}
    }

    # Aggregate pattern summaries
    foul_distributions = {}
    possession_outcomes = {}
    rebound_patterns = {}

    for result in all_results:
        patterns = result.get('patterns', {})

        # Aggregate foul type distributions
        for foul_type, count in patterns.get('foul_type_distribution', {}).items():
            foul_distributions[foul_type] = foul_distributions.get(foul_type, 0) + count

        # Aggregate possession outcomes
        for outcome, count in patterns.get('possession_outcomes', {}).items():
            possession_outcomes[outcome] = possession_outcomes.get(outcome, 0) + count

        # Aggregate rebound patterns
        for rebound_type, count in patterns.get('rebound_patterns', {}).items():
            if rebound_type == 'unknown_descriptions':
                # Handle list of unknown descriptions
                if 'unknown_descriptions' not in rebound_patterns:
                    rebound_patterns['unknown_descriptions'] = []
                rebound_patterns['unknown_descriptions'].extend(count)
            else:
                rebound_patterns[rebound_type] = rebound_patterns.get(rebound_type, 0) + count

    aggregated['aggregated_patterns'] = {
        'foul_type_distribution': foul_distributions,
        'possession_outcomes': possession_outcomes,
        'rebound_patterns': rebound_patterns
    }

    return aggregated

if __name__ == "__main__":
    # Test on a few games from the processed dataset
    test_games = ['0022200001', '0022200002', '0022200003']

    print("FREE THROW LOGIC ANALYSIS")
    print("=" * 50)

    results = analyze_multiple_games(test_games)

    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Analyzed {results['games_analyzed']} games")
        print(f"Total free throw sequences: {results['total_sequences']}")
        print(f"Total free throw events: {results['total_ft_events']}")

        patterns = results['aggregated_patterns']

        print("\nFOUL TYPE DISTRIBUTION:")
        for foul_type, count in patterns['foul_type_distribution'].items():
            print(f"  {foul_type}: {count}")

        print("\nPOSSESSION OUTCOMES AFTER FT SEQUENCES:")
        for outcome, count in patterns['possession_outcomes'].items():
            print(f"  {outcome}: {count}")

        print("\nREBOUND PATTERNS AFTER MISSED FTs:")
        for rebound_type, count in patterns['rebound_patterns'].items():
            if rebound_type != 'unknown_descriptions':
                print(f"  {rebound_type}: {count}")
            else:
                print(f"  unknown descriptions: {len(count)} examples")

        continuity = patterns.get('possession_continuity_analysis', {})
        if continuity.get('total_analyzed', 0) > 0:
            print("\nPOSSESSION CONTINUITY ANALYSIS:")
            same_team_pct = (continuity['sequences_followed_by_same_team_action'] / continuity['total_analyzed']) * 100
            rebound_pct = (continuity['sequences_followed_by_rebound'] / continuity['total_analyzed']) * 100
            opponent_pct = (continuity['sequences_followed_by_opponent_action'] / continuity['total_analyzed']) * 100

            print(f"  Same team continues: {same_team_pct:.1f}% ({continuity['sequences_followed_by_same_team_action']}/{continuity['total_analyzed']})")
            print(f"  Followed by rebound: {rebound_pct:.1f}% ({continuity['sequences_followed_by_rebound']}/{continuity['total_analyzed']})")
            print(f"  Opponent gets possession: {opponent_pct:.1f}% ({continuity['sequences_followed_by_opponent_action']}/{continuity['total_analyzed']})")

        print("\nSEQUENCE LENGTHS:")
        for length, count in patterns.get('sequence_lengths', {}).items():
            print(f"  {length} FT attempts: {count} sequences")

    print("\nFree throw analysis complete!")
