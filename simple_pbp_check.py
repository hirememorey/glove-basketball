"""
Simple check of PBP events at the overlap times.
"""

import pandas as pd
from nba_api.stats.endpoints import PlayByPlayV3

def check_pbp_at_time(game_id: str = "0022200001", target_seconds: int = 7200):
    """Check what PBP events happen near the overlap time."""

    print(f"=== PBP CHECK AT {target_seconds} SECONDS ===\n")

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

    # Look for events within 60 seconds of target
    window_start = target_seconds - 60
    window_end = target_seconds + 60

    window_events = pbp_df[
        (pbp_df['seconds_elapsed'] >= window_start) &
        (pbp_df['seconds_elapsed'] <= window_end) &
        pd.notna(pbp_df['seconds_elapsed'])
    ].sort_values('seconds_elapsed')

    print(f"PBP events between {window_start}-{window_end} seconds:")
    print()

    for _, event in window_events.iterrows():
        event_num = event.get('actionNumber', 'Unknown')
        action = event.get('actionType', 'Unknown')
        desc = str(event.get('description', 'No description'))[:120]
        clock = event.get('clock', 'Unknown')
        period = event.get('period', 'Unknown')
        seconds = event.get('seconds_elapsed', 'Unknown')

        time_diff = f"{seconds - target_seconds:+.1f}s" if seconds != 'Unknown' else 'Unknown'

        print("<10")
    print()

    # Count different event types
    event_counts = window_events['actionType'].value_counts()
    print("Event type breakdown in this window:")
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count}")
    print()

if __name__ == "__main__":
    check_pbp_at_time(target_seconds=7200)
    print("\n" + "="*60 + "\n")
    check_pbp_at_time(target_seconds=24850)
