"""
Debug script to analyze GameRotation data and understand the stint logic issue.
"""

import pandas as pd
from nba_api.stats.endpoints import GameRotation
import json

def analyze_game_rotation(game_id: str = "0022200001"):
    """Analyze the raw GameRotation data to understand the overlap issue."""

    print(f"Analyzing GameRotation data for game {game_id}")

    # Get rotation data
    endpoint = GameRotation(game_id=game_id)
    dfs = endpoint.get_data_frames()

    if len(dfs) != 2:
        print(f"Expected 2 teams, got {len(dfs)}")
        return

    analysis = {
        'game_id': game_id,
        'teams': []
    }

    for team_idx, team_df in enumerate(dfs):
        team_name = team_df['TEAM_CITY'].iloc[0] + " " + team_df['TEAM_NAME'].iloc[0]
        print(f"\n=== {team_name} ===")

        team_analysis = {
            'team_name': team_name,
            'players': []
        }

        # Analyze each player's time periods
        for _, player in team_df.iterrows():
            player_id = player['PERSON_ID']
            player_name = f"{player['PLAYER_FIRST']} {player['PLAYER_LAST']}"
            in_time = player['IN_TIME_REAL']
            out_time = player['OUT_TIME_REAL']

            print(f"{player_name}: {in_time} -> {out_time}")

            team_analysis['players'].append({
                'player_id': player_id,
                'player_name': player_name,
                'in_time': in_time,
                'out_time': out_time
            })

        analysis['teams'].append(team_analysis)

    # Check for overlaps within each team
    print("\n=== OVERLAP ANALYSIS ===")

    for team_idx, team_data in enumerate(analysis['teams']):
        print(f"\n{team_data['team_name']}:")
        overlaps_found = 0

        # Check every pair of players for time overlaps
        players = team_data['players']
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players):
                if i >= j:  # Don't check same player or reverse pairs
                    continue

                # Check if p1 and p2 have overlapping time periods
                if (p1['in_time'] < p2['out_time'] and p2['in_time'] < p1['out_time']):
                    print(f"  OVERLAP: {p1['player_name']} ({p1['in_time']}-{p1['out_time']}) overlaps with {p2['player_name']} ({p2['in_time']}-{p2['out_time']})")
                    overlaps_found += 1

        if overlaps_found == 0:
            print("  No overlaps found")
        else:
            print(f"  Total overlaps: {overlaps_found}")

    return analysis

if __name__ == "__main__":
    result = analyze_game_rotation()
    with open('gamerotation_analysis.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
