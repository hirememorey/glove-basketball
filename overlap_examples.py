"""
Show specific examples of overlapping player time periods in GameRotation data.
"""

import json
from nba_api.stats.endpoints import GameRotation
import pandas as pd

def show_overlap_examples(game_id: str = "0022200001"):
    """Show specific examples of overlapping time periods."""

    print(f"=== SPECIFIC OVERLAP EXAMPLES FOR GAME {game_id} ===\n")

    # Get rotation data
    endpoint = GameRotation(game_id=game_id)
    dfs = endpoint.get_data_frames()

    if len(dfs) != 2:
        print("Expected 2 teams")
        return

    # Focus on Philadelphia 76ers (first team)
    team_df = dfs[0]
    team_name = f"{team_df['TEAM_CITY'].iloc[0]} {team_df['TEAM_NAME'].iloc[0]}"

    print(f"Analyzing overlaps for {team_name}:\n")

    # Example 1: Start of game overlaps
    print("EXAMPLE 1: Start of Game (Time 0-5000 seconds)")
    print("These players all have overlapping time periods at the beginning:")
    print()

    starters = []
    for _, player in team_df.iterrows():
        in_time = player['IN_TIME_REAL']
        out_time = player['OUT_TIME_REAL']
        name = f"{player['PLAYER_FIRST']} {player['PLAYER_LAST']}"

        # Only show players active in the first 5000 seconds
        if pd.notna(in_time) and pd.notna(out_time) and in_time <= 5000:
            starters.append({
                'name': name,
                'in': in_time,
                'out': min(out_time, 5000)  # Cap at 5000 for display
            })

    starters.sort(key=lambda x: x['in'])
    for player in starters:
        print("<10")
    print()

    # Example 2: Mid-game substitution cluster
    print("EXAMPLE 2: Mid-Game Substitution Period (around 7200-10000 seconds)")
    print("Multiple substitutions happening in a short time window:")
    print()

    mid_game = []
    for _, player in team_df.iterrows():
        in_time = player['IN_TIME_REAL']
        out_time = player['OUT_TIME_REAL']
        name = f"{player['PLAYER_FIRST']} {player['PLAYER_LAST']}"

        # Look for players active around 7200-10000
        if pd.notna(in_time) and pd.notna(out_time):
            # Check if this player's period overlaps with the 7200-10000 window
            if in_time <= 10000 and out_time >= 7200:
                mid_game.append({
                    'name': name,
                    'in': max(in_time, 7200),  # Don't show times before 7200
                    'out': min(out_time, 10000)  # Don't show times after 10000
                })

    mid_game.sort(key=lambda x: x['in'])
    for player in mid_game:
        print("<10")
    print()

    # Example 3: Count players at specific time points
    print("EXAMPLE 3: Player Count at Specific Time Points")
    print()

    test_times = [1000, 3000, 7200, 9000, 24850]

    for test_time in test_times:
        active_players = []
        for _, player in team_df.iterrows():
            in_time = player['IN_TIME_REAL']
            out_time = player['OUT_TIME_REAL']
            name = f"{player['PLAYER_FIRST']} {player['PLAYER_LAST']}"

            if (pd.notna(in_time) and pd.notna(out_time) and
                float(in_time) <= test_time <= float(out_time)):
                active_players.append(name)

        print(f"At {test_time} seconds: {len(active_players)} active players")
        if len(active_players) > 5:
            print(f"  OVERFLOW: {active_players}")
        else:
            print(f"  Active: {active_players}")
        print()

if __name__ == "__main__":
    show_overlap_examples()
