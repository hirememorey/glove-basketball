"""
Detailed analysis of the specific overlaps causing >5 players per team.
"""

import pandas as pd
from nba_api.stats.endpoints import GameRotation

def analyze_specific_overflows(game_id: str = "0022200001"):
    """Analyze the specific time periods causing player count overflows."""

    print(f"=== DETAILED OVERLAP ANALYSIS FOR GAME {game_id} ===\n")

    endpoint = GameRotation(game_id=game_id)
    dfs = endpoint.get_data_frames()

    team_df = dfs[0]  # Philadelphia 76ers
    team_name = f"{team_df['TEAM_CITY'].iloc[0]} {team_df['TEAM_NAME'].iloc[0]}"

    print(f"Analyzing overflows for {team_name}:\n")

    # Analyze the 7200 second overflow (7 players)
    print("OVERFLOW AT 7200 SECONDS: 7 players active")
    print("Players active at 7200s and their time periods:")
    print()

    overflow_7200 = []
    for _, player in team_df.iterrows():
        in_time = player['IN_TIME_REAL']
        out_time = player['OUT_TIME_REAL']
        name = f"{player['PLAYER_FIRST']} {player['PLAYER_LAST']}"

        if (pd.notna(in_time) and pd.notna(out_time) and
            float(in_time) <= 7200 <= float(out_time)):
            overflow_7200.append({
                'name': name,
                'in_time': in_time,
                'out_time': out_time
            })

    for player in sorted(overflow_7200, key=lambda x: x['in_time']):
        print("<10")
    print()

    # Find the specific overlapping pairs at 7200s
    print("SPECIFIC OVERLAPPING PAIRS AT 7200 SECONDS:")
    for i, p1 in enumerate(overflow_7200):
        for j, p2 in enumerate(overflow_7200):
            if i < j:  # Avoid duplicate pairs
                if p1['in_time'] < p2['out_time'] and p2['in_time'] < p1['out_time']:
                    print(f"  {p1['name']} ({p1['in_time']}-{p1['out_time']}) overlaps with {p2['name']} ({p2['in_time']}-{p2['out_time']})")
    print()

    # Analyze the 24850 second overflow (8 players)
    print("OVERFLOW AT 24850 SECONDS: 8 players active")
    print("Players active at 24850s and their time periods:")
    print()

    overflow_24850 = []
    for _, player in team_df.iterrows():
        in_time = player['IN_TIME_REAL']
        out_time = player['OUT_TIME_REAL']
        name = f"{player['PLAYER_FIRST']} {player['PLAYER_LAST']}"

        if (pd.notna(in_time) and pd.notna(out_time) and
            float(in_time) <= 24850 <= float(out_time)):
            overflow_24850.append({
                'name': name,
                'in_time': in_time,
                'out_time': out_time
            })

    for player in sorted(overflow_24850, key=lambda x: x['in_time']):
        print("<10")
    print()

    # What should be the correct 5 players?
    print("QUESTION: If there should only be 5 players on court at any time,")
    print("which 5 of these 8 should actually be considered 'on court'?")
    print()
    print("Current logic counts ALL players where: IN_TIME <= current_time <= OUT_TIME")
    print("But this creates", len(overflow_24850), "active players instead of 5")
    print()

    # Show a timeline of when players should transition
    print("TIMELINE ANALYSIS: Around 24850 seconds")
    print("Looking at players entering/exiting around this time:")
    print()

    around_24850 = []
    for _, player in team_df.iterrows():
        in_time = player['IN_TIME_REAL']
        out_time = player['OUT_TIME_REAL']
        name = f"{player['PLAYER_FIRST']} {player['PLAYER_LAST']}"

        # Look at players with activity around 24000-26000
        if pd.notna(in_time) and pd.notna(out_time):
            if (in_time >= 24000 or out_time >= 24000) and (in_time <= 26000 or out_time <= 26000):
                around_24850.append({
                    'name': name,
                    'in_time': in_time,
                    'out_time': out_time
                })

    for player in sorted(around_24850, key=lambda x: x['in_time']):
        status = "ON COURT" if player['in_time'] <= 24850 <= player['out_time'] else "NOT ON COURT"
        print("<10")

if __name__ == "__main__":
    analyze_specific_overflows()
