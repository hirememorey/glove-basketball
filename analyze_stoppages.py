"""
Analyze PlayByPlay data to understand when stoppages occur and how they relate to substitutions.
"""

import pandas as pd
from nba_api.stats.endpoints import PlayByPlayV3
import json

def analyze_stoppage_timing(game_id: str = "0022200001"):
    """Analyze PBP data around the 7200 second overlap to understand stoppage timing."""

    print(f"=== ANALYZING STOPPAGE TIMING FOR GAME {game_id} ===\n")

    # Get PBP data
    endpoint = PlayByPlayV3(game_id=game_id)
    pbp_df = endpoint.get_data_frames()[0]

    # Convert clock to total seconds elapsed
    def clock_to_seconds(clock_str, period):
        if pd.isna(clock_str) or clock_str == '':
            return None
        try:
            minutes, seconds = clock_str.split(':')
            total_seconds = (int(period) - 1) * 720 + (int(minutes) * 60) + int(seconds)
            return total_seconds
        except:
            return None

    pbp_df['seconds_elapsed'] = pbp_df.apply(
        lambda row: clock_to_seconds(row.get('clock'), row.get('period')), axis=1
    )

    # Focus on the 7200 second area where we saw the overlap
    print("FOCUSING ON 7200 SECOND AREA (where 7 players were 'active'):")
    print()

    # Look at PBP events around 7200 seconds
    around_7200 = pbp_df[
        (pbp_df['seconds_elapsed'] >= 7100) &
        (pbp_df['seconds_elapsed'] <= 7300) &
        pd.notna(pbp_df['seconds_elapsed'])
    ].copy()

    around_7200 = around_7200.sort_values('seconds_elapsed')

    print("PBP events around 7200 seconds:")
    for _, event in around_7200.iterrows():
        action = event.get('actionType', 'Unknown')
        desc = event.get('description', 'No description')
        clock = event.get('clock', 'Unknown')
        period = event.get('period', 'Unknown')
        seconds = event.get('seconds_elapsed', 'Unknown')

        print("<10")
    print()

    # Look for substitution-related events
    sub_events = pbp_df[pbp_df['actionType'].str.contains('substitution|Sub', na=False, case=False)]
    print(f"Found {len(sub_events)} substitution events in the entire game")
    print()

    # Look for timeout/foul events that could create stoppages
    stoppage_events = pbp_df[
        pbp_df['actionType'].isin(['Timeout', 'Foul', 'Violation', 'Jump Ball']) |
        pbp_df['description'].str.contains('timeout|foul|violation', na=False, case=False)
    ]

    print(f"Found {len(stoppage_events)} potential stoppage events")
    print()

    # Check for events very close to 7200 seconds
    very_close = stoppage_events[
        (stoppage_events['seconds_elapsed'] >= 7150) &
        (stoppage_events['seconds_elapsed'] <= 7250) &
        pd.notna(stoppage_events['seconds_elapsed'])
    ]

    if len(very_close) > 0:
        print("STOPPAGE EVENTS VERY CLOSE TO 7200 SECONDS:")
        for _, event in very_close.iterrows():
            action = event.get('actionType', 'Unknown')
            desc = event.get('description', 'No description')
            seconds = event.get('seconds_elapsed', 'Unknown')
            print("<10")
    else:
        print("No obvious stoppage events near 7200 seconds")
    print()

    # Check the broader context - what happens right before and after 7200
    print("BROADER CONTEXT: Events from 7000-7400 seconds")
    broader = pbp_df[
        (pbp_df['seconds_elapsed'] >= 7000) &
        (pbp_df['seconds_elapsed'] <= 7400) &
        pd.notna(pbp_df['seconds_elapsed'])
    ].copy()

    broader = broader.sort_values('seconds_elapsed')

    last_event_time = None
    for _, event in broader.iterrows():
        seconds = event['seconds_elapsed']
        action = event.get('actionType', 'Unknown')
        desc = event.get('description', 'No description')[:80]  # Truncate long descriptions

        time_gap = f" (+{seconds - last_event_time:.1f}s)" if last_event_time else ""
        print("<10")

        last_event_time = seconds

    print()
    print("KEY QUESTION: Is there a stoppage/stop in play around 7200 seconds?")
    print("If so, this would explain why 7 players appear 'active' - they're all")
    print("administratively recorded during the substitution processing period.")

if __name__ == "__main__":
    analyze_stoppage_timing()
