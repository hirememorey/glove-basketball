"""
Possession Logic Study for PADIM Defensive Impact Modeling

This script analyzes basketball possession rules and creates a conceptual model
for tracking possessions in NBA play-by-play data.

Key Concepts:
- Possession: A team's opportunity to score before turnover or shot
- Possession Changes: Shots, turnovers, rebounds, fouls, timeouts
- Defensive Possessions: When opponent has the ball

Based on NBA rule interpretations for possession counting.
"""

import pandas as pd
from nba_api.stats.endpoints import PlayByPlayV3
from typing import Dict, List, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PossessionTracker:
    """
    Conceptual model for tracking basketball possessions from PBP data.
    """

    def __init__(self):
        self.possession_rules = {
            # Events that change possession
            'possession_changing_events': {
                1: 'Made Field Goal',      # Team scores, gets ball back
                2: 'Missed Field Goal',    # Team misses, opponent gets rebound opportunity
                3: 'Free Throw',          # Can change possession if last FT is made/missed
                5: 'Turnover',            # Direct possession change
                6: 'Foul',               # May lead to possession change via FTs
                7: 'Violation',          # Possession change
                10: 'Jump Ball',         # Possession established
            },

            # Events that continue possession
            'possession_continuing_events': {
                4: 'Rebound',            # Same team keeps possession
                9: 'Timeout',            # Possession continues after timeout
                13: 'End Period',        # Period ends but possession concept continues
            }
        }

        # Free throw possession logic
        self.free_throw_logic = {
            'and_one_situations': ['shooting', 'offensive', 'loose ball'],
            'last_ft_made': 'possession_changes',  # Team gets ball after made final FT
            'last_ft_missed': 'opponent_gets_possession'  # Opponent gets ball after missed final FT
        }

    def analyze_possession_events(self, pbp_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze PBP events for possession implications.

        Args:
            pbp_df: Play-by-play DataFrame

        Returns:
            Dict with possession analysis
        """
        analysis = {
            'event_distribution': {},
            'possession_changing_events': [],
            'free_throw_sequences': [],
            'turnover_patterns': [],
            'shot_possession_flow': []
        }

        # Analyze event distribution
        if 'actionType' in pbp_df.columns:
            event_counts = pbp_df['actionType'].value_counts().to_dict()
            analysis['event_distribution'] = event_counts

            # actionType descriptions are already strings in this API
            analysis['event_descriptions'] = {k: k for k in event_counts.keys()}

        # Analyze possession changing events
        possession_changers = []
        possession_changing_actions = ['Made Shot', 'Missed Shot', 'Turnover', 'Violation', 'Jump Ball']
        for idx, event in pbp_df.iterrows():
            if event.get('actionType') in possession_changing_actions:
                possession_changers.append({
                    'event_num': event.get('actionNumber', idx),
                    'event_type': event.get('actionType'),
                    'description': event.get('description', ''),
                    'period': event.get('period'),
                    'clock': event.get('clock')
                })

        analysis['possession_changing_events'] = possession_changers[:20]  # First 20 for analysis

        # Analyze free throw sequences
        ft_analysis = self.analyze_free_throw_sequences(pbp_df)
        analysis['free_throw_sequences'] = ft_analysis

        # Analyze turnover patterns
        to_analysis = self.analyze_turnover_patterns(pbp_df)
        analysis['turnover_patterns'] = to_analysis

        return analysis

    def analyze_free_throw_sequences(self, pbp_df: pd.DataFrame) -> List[Dict]:
        """
        Analyze free throw sequences and their possession implications.
        """
        ft_sequences = []

        # Find sequences of free throws
        ft_events = pbp_df[pbp_df['actionType'] == 'Free Throw'].copy()

        if len(ft_events) == 0:
            return ft_sequences

        # Group consecutive FTs by player/period
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
                    sequence_info = self._analyze_ft_sequence(current_sequence)
                    if sequence_info:
                        ft_sequences.append(sequence_info)

                # Start new sequence
                current_sequence = [ft]

            prev_player = current_player
            prev_period = current_period

        # Process final sequence
        if current_sequence:
            sequence_info = self._analyze_ft_sequence(current_sequence)
            if sequence_info:
                ft_sequences.append(sequence_info)

        return ft_sequences

    def _analyze_ft_sequence(self, ft_sequence: List[pd.Series]) -> Optional[Dict]:
        """Analyze a single free throw sequence."""
        if not ft_sequence:
            return None

        first_ft = ft_sequence[0]
        last_ft = ft_sequence[-1]

        # Determine if this is the final FT in a sequence
        # Look for the next non-FT event to see possession outcome
        sequence_end_idx = None
        possession_outcome = 'unknown'

        # This would require looking at events after the FT sequence
        # For now, return basic sequence info

        return {
            'player_id': first_ft.get('personId'),
            'period': first_ft.get('period'),
            'num_ft_attempts': len(ft_sequence),
            'sequence_start_time': first_ft.get('clock'),
            'possession_outcome': possession_outcome,  # Would need more analysis
            'description': f"{len(ft_sequence)} FT attempts by player {first_ft.get('personId')}"
        }

    def analyze_turnover_patterns(self, pbp_df: pd.DataFrame) -> List[Dict]:
        """
        Analyze turnover patterns and causes.
        """
        to_patterns = []

        turnover_events = pbp_df[pbp_df['actionType'] == 'Turnover'].copy()

        for idx, to_event in turnover_events.iterrows():
            # Look for steal events that might precede turnovers
            steal_info = self._find_associated_steal(pbp_df, idx, to_event)

            pattern = {
                'period': to_event.get('period'),
                'clock': to_event.get('clock'),
                'player_id': to_event.get('personId'),
                'turnover_type': to_event.get('subType', 'unknown'),
                'description': to_event.get('description', ''),
                'steal_associated': steal_info is not None,
                'steal_player': steal_info.get('player_id') if steal_info else None
            }

            to_patterns.append(pattern)

        return to_patterns[:20]  # Return first 20 for analysis

    def _find_associated_steal(self, pbp_df: pd.DataFrame, to_idx: int, to_event: pd.Series) -> Optional[Dict]:
        """
        Look for steal events associated with a turnover.
        """
        # Check a few events before the turnover for steals
        start_idx = max(0, to_idx - 5)

        for check_idx in range(start_idx, to_idx):
            event = pbp_df.iloc[check_idx]
            if event.get('actionType') == 'Turnover' and 'steal' in (event.get('description', '') or '').lower():
                return {
                    'player_id': event.get('personId'),
                    'description': event.get('description', '')
                }

        return None

def conceptual_possession_model() -> Dict[str, Any]:
    """
    Create a conceptual model for possession tracking.

    Returns:
        Dict describing the possession tracking approach
    """
    model = {
        'core_principles': [
            'Possession begins when a team gains the ball',
            'Possession ends when the ball changes teams or a score occurs',
            'Defensive possessions are counted when the opponent has the ball',
            'Each possession should have one primary outcome'
        ],

        'possession_change_events': [
            'Made Field Goals (team retains possession after scoring)',
            'Missed Field Goals (opponent gets rebound opportunity)',
            'Turnovers (direct possession change)',
            'Shot Clock Violations (possession change)',
            'Final Free Throws (depending on make/miss)',
            'End of Quarter (possession continues to next quarter)'
        ],

        'challenges': [
            'Free throw sequences require tracking make/miss of final FT',
            'Rebounds can be offensive (continue possession) or defensive (change possession)',
            'Technical fouls and flagrant fouls have complex possession rules',
            'Quarter breaks and timeouts preserve possession',
            'Jump balls establish new possessions'
        ],

        'implementation_approach': {
            'phase_1_basic': [
                'Track made/missed shots as possession changes',
                'Track turnovers as possession changes',
                'Simple rebound logic (assume defensive rebounds change possession)'
            ],

            'phase_2_intermediate': [
                'Implement free throw sequence logic',
                'Differentiate offensive vs defensive rebounds',
                'Handle timeouts and quarter breaks'
            ],

            'phase_3_advanced': [
                'Track technical fouls and complex situations',
                'Handle overtime periods correctly',
                'Validate possession counts against game totals'
            ]
        }
    }

    return model

def analyze_sample_game_possessions(game_id: str) -> Dict[str, Any]:
    """
    Analyze possession patterns in a sample game.

    Args:
        game_id: NBA game ID

    Returns:
        Dict with possession analysis
    """
    logger.info(f"Analyzing possession patterns for game {game_id}")

    try:
        # Get PBP data
        pbp_endpoint = PlayByPlayV3(game_id=game_id)
        pbp_df = pbp_endpoint.get_data_frames()[0]

        # Initialize possession tracker
        tracker = PossessionTracker()

        # Analyze possession events
        possession_analysis = tracker.analyze_possession_events(pbp_df)

        # Add conceptual model
        possession_analysis['conceptual_model'] = conceptual_possession_model()

        # Calculate basic possession statistics
        possession_analysis['basic_stats'] = {
            'total_events': len(pbp_df),
            'field_goal_attempts': len(pbp_df[pbp_df['actionType'].isin(['Made Shot', 'Missed Shot'])]),
            'turnovers': len(pbp_df[pbp_df['actionType'] == 'Turnover']),
            'rebounds': len(pbp_df[pbp_df['actionType'] == 'Rebound']),
            'free_throws': len(pbp_df[pbp_df['actionType'] == 'Free Throw']),
            'timeouts': len(pbp_df[pbp_df['actionType'] == 'Timeout']),
            'fouls': len(pbp_df[pbp_df['actionType'] == 'Foul'])
        }

        logger.info(f"Successfully analyzed possession patterns for game {game_id}")

    except Exception as e:
        logger.error(f"Error analyzing possession patterns: {e}")
        possession_analysis = {'error': str(e)}

    return possession_analysis

if __name__ == "__main__":
    # Analyze possession patterns for a sample game
    sample_game = "0022200001"

    print("POSSESSION LOGIC STUDY")
    print("=" * 50)

    results = analyze_sample_game_possessions(sample_game)

    if 'error' not in results:
        print(f"\nAnalyzing game: {sample_game}")

        # Basic statistics
        stats = results['basic_stats']
        print("\nBASIC STATISTICS:")
        print(f"  Total events: {stats['total_events']}")
        print(f"  Field goal attempts: {stats['field_goal_attempts']}")
        print(f"  Turnovers: {stats['turnovers']}")
        print(f"  Rebounds: {stats['rebounds']}")
        print(f"  Free throws: {stats['free_throws']}")
        print(f"  Timeouts: {stats['timeouts']}")

        # Event distribution
        event_dist = results['event_distribution']
        print("\nEVENT DISTRIBUTION (top 5):")
        for event_type, count in sorted(event_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            desc = results.get('event_descriptions', {}).get(event_type, f'Unknown {event_type}')
            print(f"  {event_type}: {count} events ({desc})")

        # Possession changing events sample
        possession_events = results['possession_changing_events']
        print(f"\nPOSSESSION CHANGING EVENTS (sample of {len(possession_events)}):")
        for event in possession_events[:5]:
            print(f"  Period {event['period']} {event['clock']}: {event['event_type']} - {event['description'][:60]}...")

        # Conceptual model summary
        model = results['conceptual_model']
        print("\nCONCEPTUAL MODEL PRINCIPLES:")
        for principle in model['core_principles']:
            print(f"  • {principle}")

        print("\nIMPLEMENTATION PHASES:")
        for phase, tasks in model['implementation_approach'].items():
            print(f"  {phase.upper()}:")
            for task in tasks:
                print(f"    - {task}")

        print("\nCHALLENGES TO ADDRESS:")
        for challenge in model['challenges']:
            print(f"  • {challenge}")

    else:
        print(f"Error analyzing game: {results['error']}")

    print("\nPossession logic study complete!")
