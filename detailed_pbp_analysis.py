"""
Detailed PBP analysis to understand what happens during the overlap periods.
"""

import pandas as pd
from nba_api.stats.endpoints import PlayByPlayV3

def detailed_pbp_at_time(game_id: str = "0022200001", target_seconds: int = 7200):
    """Get detailed PBP events at specific time points where overlaps occur."""

    print(f"=== DETAILED PBP ANALYSIS AT {target_seconds} SECONDS ===\n")

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

    # Filter to valid time entries and calculate time differences
    valid_times = pbp_df.dropna(subset=['seconds_elapsed']).copy()
    valid_times['time_diff'] = abs(valid_times['seconds_elapsed'] - target_seconds)

    # Find the closest PBP events to our target time
    closest_events = valid_times.nsmallest(10, 'time_diff').sort_values('seconds_elapsed')

    print(f"10 closest PBP events to {target_seconds} seconds:")
    print("(where we saw 7 Philadelphia players 'active')")
    print()

    for _, event in closest_events.iterrows():
        event_num = event.get('actionNumber', 'Unknown')
        action = event.get('actionType', 'Unknown')
        desc = event.get('description', 'No description')
        clock = event.get('clock', 'Unknown')
        period = event.get('period', 'Unknown')
        seconds = event.get('seconds_elapsed', 'Unknown')
        time_diff = event.get('time_diff', 'Unknown')

        print("<10")
    print()

    # Now let's check what the actual game state is at 7200 seconds
    # Convert 7200 seconds back to period and clock
    period = (target_seconds // 720) + 1
    remaining_seconds = target_seconds % 720
    clock_minutes = remaining_seconds // 60
    clock_seconds = remaining_seconds % 60
    clock_str = f"{clock_minutes}:{clock_seconds:02d}"

    print(f"At {target_seconds} seconds: This corresponds to Period {period}, Clock {clock_str}")
    print()

    # Look for any events in a wider window
    wider_window = valid_times[
        (valid_times['seconds_elapsed'] >= target_seconds - 30) &
        (valid_times['seconds_elapsed'] <= target_seconds + 30)
    ].sort_values('seconds_elapsed')

    if len(wider_window) > 0:
        print("All PBP events in ±30 second window:")
        for _, event in wider_window.iterrows():
            event_num = event.get('actionNumber', 'Unknown')
            action = event.get('actionType', 'Unknown')
            desc = event.get('description', 'No description')[:100]
            seconds = event.get('seconds_elapsed', 'Unknown')
            time_diff = f"{seconds - target_seconds:+.1f}s"

            print("<10")
    else:
        print("No PBP events in the ±30 second window")
    print()

    # Check if this time corresponds to a substitution event
    sub_events = pbp_df[pbp_df['actionType'].str.contains('substitution|Sub', na=False, case=False)]
    if len(sub_events) > 0:
        sub_times = sub_events['seconds_elapsed'].dropna().sort_values()

        closest_sub = None
        min_diff = float('inf')
        for sub_time in sub_times:
            diff = abs(sub_time - target_seconds)
            if diff < min_diff:
                min_diff = diff
                closest_sub = sub_time

        if closest_sub is not None:
            print(f"Closest substitution event is at {closest_sub} seconds ({closest_sub - target_seconds:+.1f}s from target)")
    else:
        print("No substitution events found in PBP data")

if __name__ == "__main__":
    # Check both overlap time points
    detailed_pbp_at_time(target_seconds=7200)
    print("\n" + "="*80 + "\n")
    detailed_pbp_at_time(target_seconds=24850)