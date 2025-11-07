#!/usr/bin/env python3
"""
RAPM vs Traditional Metrics Comparison

This script compares RAPM coefficients against traditional defensive metrics
to validate RAPM's ability to capture defensive impact beyond simple box score stats.

Key comparisons:
1. RAPM coefficients vs. traditional defensive stats (STL, BLK, DEFLECTIONS)
2. RAPM lineup predictions vs. traditional lineup aggregations
3. Correlation analysis between RAPM and traditional approaches
4. Validation that RAPM captures additional defensive signal

Author: PADIM Analysis Team
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from padim.db.database import DatabaseConnection

class RAPMvsTraditionalComparator:
    """Compares RAPM against traditional defensive metrics."""

    def __init__(self):
        """Initialize the comparator."""
        self.db = DatabaseConnection()
        self.rapm_results = None
        self.traditional_metrics = None

    def load_player_traditional_stats(self, season: str = "2022-23") -> pd.DataFrame:
        """
        Load traditional defensive player stats.

        Args:
            season: NBA season string

        Returns:
            DataFrame with player defensive stats
        """
        print(f"🔍 Loading traditional defensive stats for {season}...")

        # Get player season stats with defensive metrics
        query = """
        SELECT
            ps.player_id,
            ps.player_name,
            ps.gp,
            ps.min,
            -- Traditional defensive stats
            COALESCE(ph.contested_shots, 0) as contested_shots,
            COALESCE(ph.contested_shots_2pt, 0) as contested_shots_2pt,
            COALESCE(ph.contested_shots_3pt, 0) as contested_shots_3pt,
            COALESCE(ph.deflections, 0) as deflections,
            COALESCE(ph.charges_drawn, 0) as charges_drawn,
            COALESCE(ph.loose_balls_recovered_def, 0) as loose_balls_recovered_def,
            -- Normalize by minutes played
            CASE WHEN ps.min > 0 THEN COALESCE(ph.contested_shots, 0) * 36.0 / ps.min ELSE 0 END as contested_shots_per_36,
            CASE WHEN ps.min > 0 THEN COALESCE(ph.deflections, 0) * 36.0 / ps.min ELSE 0 END as deflections_per_36,
            CASE WHEN ps.min > 0 THEN COALESCE(ph.charges_drawn, 0) * 36.0 / ps.min ELSE 0 END as charges_per_36
        FROM player_season_stats ps
        LEFT JOIN player_hustle_stats ph ON ps.player_id = ph.player_id AND ps.season_id = ph.season_id
        WHERE ps.season_id = ?
        ORDER BY ps.min DESC
        """

        df = pd.read_sql_query(query, self.db._connection, params=[season])
        df = df[df['min'] >= 500]  # Minimum minutes played

        print(f"📊 Loaded traditional stats for {len(df)} players (min 500 min)")

        return df

    def train_rapm_on_larger_dataset(self, max_games: int = 200) -> Dict[str, any]:
        """
        Train RAPM on a larger dataset for more stable coefficients.

        Args:
            max_games: Maximum games to use for training

        Returns:
            RAPM training results
        """
        print(f"🎯 Training RAPM on larger dataset ({max_games} games)...")

        # Load training data
        query = f"""
        SELECT
            game_id,
            stint_start,
            stint_end,
            duration,
            home_team_id,
            away_team_id,
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
            home_opp_fga, home_opp_fgm, home_opp_fg3a, home_opp_fg3m,
            away_opp_fga, away_opp_fgm, away_opp_fg3a, away_opp_fg3m
        FROM stints
        WHERE duration >= 120
        ORDER BY game_id, stint_start
        LIMIT {max_games * 25}
        """

        df = pd.read_sql_query(query, self.db._connection)

        # Limit to specified games
        unique_games = df['game_id'].unique()[:max_games]
        df = df[df['game_id'].isin(unique_games)]

        print(f"📊 Training on {len(df):,} stints from {len(unique_games)} games")

        # Calculate defensive targets
        df['home_def_eFG'] = np.where(
            df['home_opp_fga'] > 0,
            (df['home_opp_fgm'] + 0.5 * df['home_opp_fg3m']) / df['home_opp_fga'],
            np.nan
        )

        df['away_def_eFG'] = np.where(
            df['away_opp_fga'] > 0,
            (df['away_opp_fgm'] + 0.5 * df['away_opp_fg3m']) / df['away_opp_fga'],
            np.nan
        )

        # Create design matrix
        all_players = set()
        for col in ['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5',
                   'away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5']:
            all_players.update(df[col].dropna().astype(int).unique())

        player_ids = sorted(list(all_players))
        player_to_idx = {pid: i for i, pid in enumerate(player_ids)}

        n_stints = len(df)
        n_players = len(player_ids)
        X = np.zeros((n_stints, n_players))
        y = np.zeros(n_stints)

        for i, (_, stint) in enumerate(df.iterrows()):
            # Add +1 for home team players (defending)
            for j in range(1, 6):
                player_col = f'home_player_{j}'
                if pd.notna(stint[player_col]):
                    player_id = int(stint[player_col])
                    if player_id in player_to_idx:
                        X[i, player_to_idx[player_id]] = 1

            # Add +1 for away team players (defending)
            for j in range(1, 6):
                player_col = f'away_player_{j}'
                if pd.notna(stint[player_col]):
                    player_id = int(stint[player_col])
                    if player_id in player_to_idx:
                        X[i, player_to_idx[player_id]] = 1

            # Target: defensive eFG allowed
            if pd.notna(stint['home_def_eFG']):
                y[i] = stint['home_def_eFG']
            elif pd.notna(stint['away_def_eFG']):
                y[i] = stint['away_def_eFG']
            else:
                y[i] = np.nan

        # Remove NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"📊 Final design matrix: {X.shape} (stints x players)")

        # Train RAPM
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Ridge(alpha=100.0, random_state=42)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        results = {
            'model': model,
            'coefficients': model.coef_,
            'player_ids': player_ids,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'n_stints': len(X),
            'n_players': len(player_ids),
            'n_games': len(unique_games)
        }

        print(".4f")
        print(".4f")

        return results

    def create_player_comparison_dataset(self, rapm_results: Dict,
                                       traditional_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset comparing RAPM coefficients with traditional metrics.

        Args:
            rapm_results: RAPM training results
            traditional_stats: Traditional defensive stats

        Returns:
            DataFrame with RAPM vs traditional comparisons
        """
        print("🔗 Creating player comparison dataset...")

        comparisons = []

        for i, player_id in enumerate(rapm_results['player_ids']):
            rapm_coef = rapm_results['coefficients'][i]

            # Find traditional stats for this player
            trad_stats = traditional_stats[traditional_stats['player_id'] == player_id]

            if len(trad_stats) > 0:
                trad = trad_stats.iloc[0]
                comparisons.append({
                    'player_id': player_id,
                    'player_name': trad['player_name'],
                    'rapm_coefficient': rapm_coef,
                    'contested_shots_per_36': trad['contested_shots_per_36'],
                    'deflections_per_36': trad['deflections_per_36'],
                    'charges_per_36': trad['charges_per_36'],
                    'minutes_played': trad['min'],
                    'games_played': trad['gp']
                })

        comparison_df = pd.DataFrame(comparisons)

        print(f"📊 Created comparisons for {len(comparison_df)} players")

        return comparison_df

    def analyze_correlations(self, comparison_df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze correlations between RAPM and traditional metrics.

        Args:
            comparison_df: Player comparison data

        Returns:
            Correlation analysis results
        """
        print("📈 Analyzing correlations between RAPM and traditional metrics...")

        correlations = {}

        # RAPM vs each traditional metric
        traditional_metrics = ['contested_shots_per_36', 'deflections_per_36', 'charges_per_36']

        for metric in traditional_metrics:
            valid_data = comparison_df.dropna(subset=['rapm_coefficient', metric])
            if len(valid_data) >= 10:
                corr, p_value = stats.pearsonr(valid_data['rapm_coefficient'], valid_data[metric])
                correlations[metric] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_players': len(valid_data)
                }

        # Overall analysis
        analysis = {
            'correlations': correlations,
            'summary': {
                'total_players': len(comparison_df),
                'rapm_range': comparison_df['rapm_coefficient'].max() - comparison_df['rapm_coefficient'].min(),
                'rapm_std': comparison_df['rapm_coefficient'].std(),
                'most_defensive_rapm': comparison_df.nlargest(5, 'rapm_coefficient')[['player_name', 'rapm_coefficient']].to_dict('records'),
                'least_defensive_rapm': comparison_df.nsmallest(5, 'rapm_coefficient')[['player_name', 'rapm_coefficient']].to_dict('records')
            }
        }

        return analysis

    def create_traditional_lineup_predictions(self, lineup_data: pd.DataFrame,
                                            traditional_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Create traditional metric predictions for lineups.

        Args:
            lineup_data: Lineup performance data
            traditional_stats: Traditional defensive stats

        Returns:
            DataFrame with traditional lineup predictions
        """
        print("🏀 Creating traditional lineup predictions...")

        predictions = []

        for _, lineup in lineup_data.iterrows():
            lineup_sig = lineup['lineup_sig']
            player_ids = [int(pid) for pid in lineup_sig.split(',')]

            # Calculate traditional metrics for lineup
            lineup_trad_stats = traditional_stats[traditional_stats['player_id'].isin(player_ids)]

            if len(lineup_trad_stats) >= 3:  # At least 3 players with data
                avg_contested = lineup_trad_stats['contested_shots_per_36'].mean()
                avg_deflections = lineup_trad_stats['deflections_per_36'].mean()
                avg_charges = lineup_trad_stats['charges_per_36'].mean()

                predictions.append({
                    'lineup_sig': lineup_sig,
                    'actual_def_eFG': lineup['def_eFG_mean'],
                    'traditional_contested_shots': avg_contested,
                    'traditional_deflections': avg_deflections,
                    'traditional_charges': avg_charges,
                    'players_with_data': len(lineup_trad_stats)
                })

        pred_df = pd.DataFrame(predictions)

        print(f"📊 Created traditional predictions for {len(pred_df)} lineups")

        return pred_df

    def compare_predictive_power(self, rapm_predictions: pd.DataFrame,
                               traditional_predictions: pd.DataFrame) -> Dict[str, any]:
        """
        Compare predictive power of RAPM vs traditional metrics.

        Args:
            rapm_predictions: RAPM lineup predictions
            traditional_predictions: Traditional lineup predictions

        Returns:
            Predictive power comparison results
        """
        print("⚖️  Comparing predictive power...")

        # Debug: Print available columns
        print(f"RAPM predictions columns: {list(rapm_predictions.columns)}")
        print(f"Traditional predictions columns: {list(traditional_predictions.columns)}")

        # Merge predictions on lineup_sig
        merged = pd.merge(
            rapm_predictions[['lineup_sig', 'rapm_predicted_impact', 'actual_def_eFG']],
            traditional_predictions,
            on='lineup_sig',
            how='inner'
        )

        print(f"Merged columns: {list(merged.columns)}")

        if len(merged) < 10:
            return {'error': 'Insufficient overlapping predictions'}

        # Use the correct column name - pandas creates _x and _y suffixes for duplicates
        actual_col = 'actual_def_eFG_x'  # From RAPM predictions

        # Calculate correlations and R² for each approach
        results = {
            'sample_size': len(merged),
            'rapm': {
                'correlation': merged['rapm_predicted_impact'].corr(merged[actual_col]),
                'r_squared': r2_score(merged[actual_col], merged['rapm_predicted_impact']),
                'mae': abs(merged['rapm_predicted_impact'] - merged[actual_col]).mean()
            },
            'traditional_contested': {
                'correlation': merged['traditional_contested_shots'].corr(merged[actual_col]),
                'r_squared': r2_score(merged[actual_col], merged['traditional_contested_shots']),
                'mae': abs(merged['traditional_contested_shots'] - merged[actual_col]).mean()
            },
            'traditional_deflections': {
                'correlation': merged['traditional_deflections'].corr(merged[actual_col]),
                'r_squared': r2_score(merged[actual_col], merged['traditional_deflections']),
                'mae': abs(merged['traditional_deflections'] - merged[actual_col]).mean()
            }
        }

        return results

    def run_full_comparison(self, max_games: int = 200) -> Dict[str, any]:
        """
        Run complete RAPM vs traditional metrics comparison.

        Args:
            max_games: Games to use for RAPM training

        Returns:
            Comprehensive comparison results
        """
        print("🚀 Starting RAPM vs Traditional Metrics Comparison")
        print("=" * 70)

        # Load traditional stats
        traditional_stats = self.load_player_traditional_stats()

        # Train RAPM on larger dataset
        rapm_results = self.train_rapm_on_larger_dataset(max_games=max_games)

        # Create player-level comparisons
        player_comparisons = self.create_player_comparison_dataset(rapm_results, traditional_stats)
        correlation_analysis = self.analyze_correlations(player_comparisons)

        # Create lineup-level predictions
        from lineup_performance_comparison import LineupPerformanceComparator
        comparator = LineupPerformanceComparator()
        lineup_data = comparator.load_lineup_data(min_stints=5, min_duration=120)

        # RAPM lineup predictions
        rapm_lineup_preds = self.predict_lineup_impacts(rapm_results, lineup_data)

        # Traditional lineup predictions
        traditional_lineup_preds = self.create_traditional_lineup_predictions(lineup_data, traditional_stats)

        # Compare predictive power
        predictive_comparison = self.compare_predictive_power(rapm_lineup_preds, traditional_lineup_preds)

        # Compile results
        results = {
            'training_info': {
                'games_used': rapm_results['n_games'],
                'stints_used': rapm_results['n_stints'],
                'players_in_rapm': rapm_results['n_players'],
                'rapm_train_r2': rapm_results['train_r2'],
                'rapm_test_r2': rapm_results['test_r2']
            },
            'player_level_analysis': {
                'players_compared': len(player_comparisons),
                'correlation_results': correlation_analysis['correlations'],
                'rapm_distribution': {
                    'mean': player_comparisons['rapm_coefficient'].mean(),
                    'std': player_comparisons['rapm_coefficient'].std(),
                    'range': player_comparisons['rapm_coefficient'].max() - player_comparisons['rapm_coefficient'].min()
                }
            },
            'predictive_comparison': predictive_comparison,
            'conclusions': self._draw_conclusions(correlation_analysis, predictive_comparison)
        }

        # Print summary
        self._print_summary(results)

        return results

    def predict_lineup_impacts(self, rapm_results: Dict, lineup_data: pd.DataFrame) -> pd.DataFrame:
        """Predict RAPM impacts for lineups."""
        predictions = []

        for _, lineup in lineup_data.iterrows():
            lineup_sig = lineup['lineup_sig']
            player_ids = [int(pid) for pid in lineup_sig.split(',')]

            # Calculate average RAPM coefficient for lineup
            rapm_impact = 0
            players_found = 0

            for player_id in player_ids:
                if player_id in rapm_results['player_ids']:
                    idx = rapm_results['player_ids'].index(player_id)
                    rapm_impact += rapm_results['coefficients'][idx]
                    players_found += 1

            if players_found > 0:
                rapm_impact /= players_found

            predictions.append({
                'lineup_sig': lineup_sig,
                'rapm_predicted_impact': rapm_impact,
                'actual_def_eFG': lineup['def_eFG_mean'],
                'players_found': players_found
            })

        return pd.DataFrame(predictions)

    def _draw_conclusions(self, correlation_analysis: Dict, predictive_comparison: Dict) -> List[str]:
        """Draw conclusions from the comparison."""
        conclusions = []

        # Check if RAPM correlates with traditional metrics
        rapm_traditional_corr = []
        for metric, results in correlation_analysis['correlations'].items():
            if results['significant']:
                rapm_traditional_corr.append(f"{metric}: {results['correlation']:.3f}")

        if rapm_traditional_corr:
            conclusions.append(f"✅ RAPM-TRADITIONAL CORRELATION: RAPM correlates with {', '.join(rapm_traditional_corr)}")
        else:
            conclusions.append("⚠️  LIMITED RAPM-TRADITIONAL CORRELATION: RAPM captures different defensive aspects than traditional metrics")

        # Compare predictive power
        if 'error' not in predictive_comparison:
            rapm_r2 = predictive_comparison['rapm']['r_squared']
            contested_r2 = predictive_comparison['traditional_contested']['r_squared']

            if rapm_r2 > contested_r2 + 0.05:
                conclusions.append("✅ SUPERIOR PREDICTIVE POWER: RAPM outperforms traditional contested shots metrics")
            elif rapm_r2 > contested_r2:
                conclusions.append("⚠️  SLIGHT PREDICTIVE EDGE: RAPM marginally outperforms traditional metrics")
            else:
                conclusions.append("❌ INFERIOR PREDICTIVE POWER: Traditional metrics outperform RAPM (likely due to sample size)")

        # Overall assessment
        if correlation_analysis['correlations']:
            significant_corrs = sum(1 for r in correlation_analysis['correlations'].values() if r['significant'])
            if significant_corrs >= 2:
                conclusions.append("✅ COMPLEMENTARY APPROACH: RAPM both correlates with and extends beyond traditional metrics")
            elif significant_corrs >= 1:
                conclusions.append("⚠️  PARTIAL OVERLAP: RAPM shares some signal with traditional metrics but adds value")
            else:
                conclusions.append("🎯 DISTINCT APPROACH: RAPM captures fundamentally different defensive information")

        return conclusions

    def _print_summary(self, results: Dict) -> None:
        """Print comparison summary."""
        print("\n" + "="*70)
        print("⚖️  RAPM vs TRADITIONAL METRICS COMPARISON SUMMARY")
        print("="*70)

        ti = results['training_info']
        pc = results['predictive_comparison']

        print(f"RAPM Training: {ti['games_used']} games, {ti['stints_used']:,} stints, {ti['players_in_rapm']} players")
        print(".4f")
        print(f"Players compared: {results['player_level_analysis']['players_compared']}")

        if 'error' not in pc:
            print(".3f")
            print(".3f")

        print("\n🎯 CONCLUSIONS:")
        for conclusion in results['conclusions']:
            print(f"  {conclusion}")

        print("\n" + "="*70)


def main():
    """Main analysis function."""
    comparator = RAPMvsTraditionalComparator()

    try:
        results = comparator.run_full_comparison(max_games=200)

        # Save simplified results
        output_results = {
            'training_info': results['training_info'],
            'players_compared': results['player_level_analysis']['players_compared'],
            'predictive_comparison': results.get('predictive_comparison', {}),
            'conclusions': results['conclusions']
        }

        import json

        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        output_file = Path("rapm_vs_traditional_comparison_results.json")
        with open(output_file, 'w') as f:
            json.dump(output_results, f, indent=2, default=convert_numpy_types)

        print(f"\n💾 Results saved to {output_file}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
