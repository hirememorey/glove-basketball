"""
Analyze the substitution patterns at the exact overlap times.
"""

import pandas as pd
from nba_api.stats.endpoints import GameRotation

def analyze_substitution_patterns(game_id: str = "0022200001", target_seconds: int = 7200):
    """Analyze what substitution patterns occur at the exact overlap time."""

    print(f"=== SUBSTITUTION PATTERN ANALYSIS AT {target_seconds} SECONDS ===\n")

    endpoint = GameRotation(game_id=game_id)
    dfs = endpoint.get_data_frames()

    team_df = dfs[0]  # Philadelphia 76ers
    team_name = f"{team_df['TEAM_CITY'].iloc[0]} {team_df['TEAM_NAME'].iloc[0]}"

    print(f"Analyzing {team_name} at {target_seconds} seconds:\n")

    # Get all players active at the target time
    active_players = []
    for _, player in team_df.iterrows():
        in_time = player['IN_TIME_REAL']
        out_time = player['OUT_TIME_REAL']
        name = f"{player['PLAYER_FIRST']} {player['PLAYER_LAST']}"

        if (pd.notna(in_time) and pd.notna(out_time) and
            float(in_time) <= target_seconds <= float(out_time)):
            active_players.append({
                'name': name,
                'in_time': in_time,
                'out_time': out_time,
                'in_diff': target_seconds - in_time,  # How long ago they entered
                'out_diff': out_time - target_seconds  # How long until they exit
            })

    print(f"At exactly {target_seconds} seconds, {len(active_players)} players are recorded as active:")
    for player in active_players:
        print("<10")
    print()

    # Analyze the pattern
    print("PATTERN ANALYSIS:")
    print()

    # Group by when their stints start/end relative to target time
    entering_now = [p for p in active_players if p['in_diff'] == 0]
    exiting_now = [p for p in active_players if p['out_diff'] == 0]
    continuing = [p for p in active_players if p['in_diff'] > 0 and p['out_diff'] > 0]
    recently_entered = [p for p in active_players if 0 < p['in_diff'] <= 60]  # Entered in last minute
    will_exit_soon = [p for p in active_players if 0 < p['out_diff'] <= 60]  # Will exit in next minute

    print(f"Players ENTERING at exactly {target_seconds}: {len(entering_now)}")
    for p in entering_now:
        print(f"  → {p['name']}")
    print()

    print(f"Players EXITING at exactly {target_seconds}: {len(exiting_now)}")
    for p in exiting_now:
        print(f"  ← {p['name']}")
    print()

    print(f"Players CONTINUING through {target_seconds}: {len(continuing)}")
    for p in continuing:
        print("<10")
    print()

    # The key insight: if this is a clean substitution, we should have equal numbers entering/exiting
    if len(entering_now) == len(exiting_now):
        print(f"🎯 PERFECT SUBSTITUTION PATTERN: {len(entering_now)} out, {len(entering_now)} in")
        print("This suggests a clean substitution where the exiting players should be")
        print("considered 'no longer active' and entering players 'now active'")
        print()

        print("PROPOSED RESOLUTION:")
        print("Remove exiting players from active list, keep entering + continuing players")
        resolved_active = [p for p in active_players if p not in exiting_now]
        print(f"Resolved active players: {len(resolved_active)}")
        for p in resolved_active:
            print(f"  ✓ {p['name']}")
    else:
        print(f"⚠️  UNEVEN PATTERN: {len(exiting_now)} exiting, {len(entering_now)} entering")
        print("This doesn't look like a clean substitution")

    print("\n" + "="*60)
    print("FIRST PRINCIPLES REASONING:")
    print("="*60)
    print()
    print("1. Basketball allows only 5 players per team on court")
    print("2. Substitutions happen instantaneously during stoppages")
    print("3. If data shows more than 5 players, it's tracking entry/exit timing")
    print("4. Players exiting at time T should be considered 'already subbed out'")
    print("5. Players entering at time T should be considered 'now active'")
    print()
    print("ELEGANT SOLUTION: At any exact time T, prioritize the entering players")
    print("over the exiting players to resolve overlaps.")

if __name__ == "__main__":
    analyze_substitution_patterns(target_seconds=7200)
    print("\n" + "="*80 + "\n")
    analyze_substitution_patterns(target_seconds=24850)
