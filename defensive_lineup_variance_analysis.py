#!/usr/bin/env python3
"""
Defensive Lineup Variance Analysis

This script analyzes whether lineup differences reveal meaningful defensive impacts.
Tests the fundamental assumption that different defensive lineups produce
measurably different defensive outcomes.

Key questions:
1. Do different lineups show meaningful variance in defensive performance?
2. Can we identify lineups with significantly different defensive impacts?
3. Does lineup composition correlate with defensive outcomes?
4. Are lineup differences larger than random variation?

Author: PADIM Analysis Team
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from padim.db.database import DatabaseConnection

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class DefensiveLineupVarianceAnalyzer:
    """Analyzes variance in defensive performance across different lineups."""

    def __init__(self):
        """Initialize the analyzer."""
        self.db = DatabaseConnection()
        self.stint_data = None
        self.lineup_stats = None

    def load_stint_data(self, min_duration: float = 60.0) -> pd.DataFrame:
        """
        Load stint data with defensive metrics.

        Args:
            min_duration: Minimum stint duration in seconds to include

        Returns:
            DataFrame with stint data and defensive metrics
        """
        print("🔍 Loading stint data...")

        query = f"""
        SELECT
            game_id,
            stint_start,
            stint_end,
            duration,
            home_team_id,
            away_team_id,
            -- Home team defensive lineups (defending against away team)
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            -- Home team defensive metrics (opponent = away team)
            home_opp_fga, home_opp_fgm, home_opp_fg3a, home_opp_fg3m, home_opp_rim_attempts,
            -- Away team defensive lineups (defending against home team)
            away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
            -- Away team defensive metrics (opponent = home team)
            away_opp_fga, away_opp_fgm, away_opp_fg3a, away_opp_fg3m, away_opp_rim_attempts
        FROM stints
        WHERE duration >= {min_duration}
        ORDER BY game_id, stint_start
        """

        self.stint_data = pd.read_sql_query(query, self.db._connection)
        print(f"📊 Loaded {len(self.stint_data):,} stints with duration >= {min_duration}s")

        return self.stint_data

    def calculate_defensive_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defensive efficiency metrics for each stint.

        Args:
            df: Stint data DataFrame

        Returns:
            DataFrame with calculated defensive metrics
        """
        print("🧮 Calculating defensive metrics...")

        # Create a copy to avoid modifying original
        df_calc = df.copy()

        # Calculate home team defensive efficiency (when they're defending)
        df_calc['home_def_eFG'] = np.where(
            df_calc['home_opp_fga'] > 0,
            (df_calc['home_opp_fgm'] + 0.5 * df_calc['home_opp_fg3m']) / df_calc['home_opp_fga'],
            0
        )

        df_calc['home_def_FG_pct'] = np.where(
            df_calc['home_opp_fga'] > 0,
            df_calc['home_opp_fgm'] / df_calc['home_opp_fga'],
            0
        )

        df_calc['home_def_rim_rate'] = np.where(
            df_calc['home_opp_fga'] > 0,
            df_calc['home_opp_rim_attempts'] / df_calc['home_opp_fga'],
            0
        )

        # Calculate away team defensive efficiency (when they're defending)
        df_calc['away_def_eFG'] = np.where(
            df_calc['away_opp_fga'] > 0,
            (df_calc['away_opp_fgm'] + 0.5 * df_calc['away_opp_fg3m']) / df_calc['away_opp_fga'],
            0
        )

        df_calc['away_def_FG_pct'] = np.where(
            df_calc['away_opp_fga'] > 0,
            df_calc['away_opp_fgm'] / df_calc['away_opp_fga'],
            0
        )

        df_calc['away_def_rim_rate'] = np.where(
            df_calc['away_opp_fga'] > 0,
            df_calc['away_opp_rim_attempts'] / df_calc['away_opp_fga'],
            0
        )

        # Calculate possession-normalized metrics (per minute)
        df_calc['home_opp_fga_per_min'] = df_calc['home_opp_fga'] / (df_calc['duration'] / 60)
        df_calc['away_opp_fga_per_min'] = df_calc['away_opp_fga'] / (df_calc['duration'] / 60)

        return df_calc

    def create_lineup_signatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lineup signature strings for grouping and analysis.

        Args:
            df: DataFrame with player lineup data

        Returns:
            DataFrame with lineup signature columns
        """
        print("🏃 Creating lineup signatures...")

        df_sig = df.copy()

        # Create sorted lineup signatures (for consistent grouping regardless of order)
        df_sig['home_lineup_sig'] = df_sig.apply(
            lambda row: ','.join(sorted([
                str(int(row['home_player_1'])), str(int(row['home_player_2'])),
                str(int(row['home_player_3'])), str(int(row['home_player_4'])),
                str(int(row['home_player_5']))
            ])),
            axis=1
        )

        df_sig['away_lineup_sig'] = df_sig.apply(
            lambda row: ','.join(sorted([
                str(int(row['away_player_1'])), str(int(row['away_player_2'])),
                str(int(row['away_player_3'])), str(int(row['away_player_4'])),
                str(int(row['away_player_5']))
            ])),
            axis=1
        )

        return df_sig

    def analyze_lineup_variance(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze variance in defensive performance across different lineups.

        Args:
            df: DataFrame with defensive metrics and lineup signatures

        Returns:
            Dictionary with variance analysis results
        """
        print("📈 Analyzing lineup variance...")

        results = {}

        # Analyze home team lineups (defending role)
        home_lineup_stats = df.groupby('home_lineup_sig').agg({
            'home_def_eFG': ['mean', 'std', 'count', 'min', 'max'],
            'home_def_FG_pct': ['mean', 'std', 'count'],
            'home_def_rim_rate': ['mean', 'std', 'count'],
            'home_opp_fga_per_min': ['mean', 'std'],
            'duration': ['sum', 'mean']
        }).round(4)

        # Flatten column names
        home_lineup_stats.columns = ['_'.join(col).strip() for col in home_lineup_stats.columns.values]
        home_lineup_stats = home_lineup_stats.reset_index()

        # Filter for lineups with sufficient sample size
        home_lineup_stats = home_lineup_stats[home_lineup_stats['home_def_eFG_count'] >= 5]

        results['home_lineups'] = home_lineup_stats

        # Analyze away team lineups (defending role)
        away_lineup_stats = df.groupby('away_lineup_sig').agg({
            'away_def_eFG': ['mean', 'std', 'count', 'min', 'max'],
            'away_def_FG_pct': ['mean', 'std', 'count'],
            'away_def_rim_rate': ['mean', 'std', 'count'],
            'away_opp_fga_per_min': ['mean', 'std'],
            'duration': ['sum', 'mean']
        }).round(4)

        # Flatten column names
        away_lineup_stats.columns = ['_'.join(col).strip() for col in away_lineup_stats.columns.values]
        away_lineup_stats = away_lineup_stats.reset_index()

        # Filter for lineups with sufficient sample size
        away_lineup_stats = away_lineup_stats[away_lineup_stats['away_def_eFG_count'] >= 5]

        results['away_lineups'] = away_lineup_stats

        # Calculate overall variance statistics
        results['variance_summary'] = {
            'total_unique_home_lineups': len(home_lineup_stats),
            'total_unique_away_lineups': len(away_lineup_stats),
            'home_def_eFG_range': (
                home_lineup_stats['home_def_eFG_mean'].min(),
                home_lineup_stats['home_def_eFG_mean'].max()
            ),
            'home_def_eFG_std_between_lineups': home_lineup_stats['home_def_eFG_mean'].std(),
            'away_def_eFG_range': (
                away_lineup_stats['away_def_eFG_mean'].min(),
                away_lineup_stats['away_def_eFG_mean'].max()
            ),
            'away_def_eFG_std_between_lineups': away_lineup_stats['away_def_eFG_mean'].std(),
        }

        return results

    def analyze_extreme_lineups(self, lineup_stats: pd.DataFrame,
                               metric_col: str, min_stints: int = 10) -> Dict[str, any]:
        """
        Identify lineups with extreme defensive performance.

        Args:
            lineup_stats: DataFrame with lineup statistics
            metric_col: Column name for the metric to analyze
            min_stints: Minimum number of stints required

        Returns:
            Dictionary with extreme lineup analysis
        """
        print(f"🔍 Analyzing extreme lineups for {metric_col}...")

        # Filter for sufficient sample size
        filtered_stats = lineup_stats[lineup_stats[f'{metric_col}_count'] >= min_stints].copy()

        if len(filtered_stats) < 10:
            return {'error': f'Insufficient lineups with {min_stints}+ stints'}

        # Sort by mean performance
        sorted_stats = filtered_stats.sort_values(f'{metric_col}_mean')

        # Get top and bottom performers
        n_extreme = min(5, len(sorted_stats) // 4)  # Top/bottom quartile

        best_lineups = sorted_stats.head(n_extreme)
        worst_lineups = sorted_stats.tail(n_extreme)

        # Calculate effect size (Cohen's d) between best and worst
        best_mean = best_lineups[f'{metric_col}_mean'].mean()
        worst_mean = worst_lineups[f'{metric_col}_mean'].mean()
        best_std = best_lineups[f'{metric_col}_mean'].std()
        worst_std = worst_lineups[f'{metric_col}_mean'].std()
        pooled_std = np.sqrt((best_std**2 + worst_std**2) / 2)

        if pooled_std > 0:
            effect_size = abs(best_mean - worst_mean) / pooled_std
        else:
            effect_size = 0

        return {
            'best_lineups': best_lineups,
            'worst_lineups': worst_lineups,
            'performance_range': worst_mean - best_mean,
            'effect_size': effect_size,
            'n_lineups_analyzed': len(filtered_stats),
            'avg_stints_per_lineup': filtered_stats[f'{metric_col}_count'].mean()
        }

    def plot_lineup_distributions(self, lineup_stats: pd.DataFrame,
                                 metric_col: str, title: str) -> None:
        """
        Plot distribution of lineup performance for a metric.

        Args:
            lineup_stats: DataFrame with lineup statistics
            metric_col: Column name for the metric to plot
            title: Plot title
        """
        plt.figure(figsize=(12, 6))

        # Plot histogram of lineup means
        plt.hist(lineup_stats[f'{metric_col}_mean'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(lineup_stats[f'{metric_col}_mean'].mean(),
                   color='red', linestyle='--', linewidth=2,
                   label='.3f')

        plt.xlabel(f'{metric_col.replace("_", " ").title()}')
        plt.ylabel('Number of Lineups')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self) -> Dict[str, any]:
        """
        Run complete lineup variance analysis.

        Returns:
            Comprehensive analysis results
        """
        print("🚀 Starting Defensive Lineup Variance Analysis")
        print("=" * 60)

        # Load and process data
        raw_data = self.load_stint_data(min_duration=120)  # 2+ minutes for meaningful comparison
        processed_data = self.calculate_defensive_metrics(raw_data)
        lineup_data = self.create_lineup_signatures(processed_data)

        # Analyze variance
        variance_results = self.analyze_lineup_variance(lineup_data)

        # Analyze extreme performers
        home_extreme = self.analyze_extreme_lineups(
            variance_results['home_lineups'], 'home_def_eFG', min_stints=10
        )

        away_extreme = self.analyze_extreme_lineups(
            variance_results['away_lineups'], 'away_def_eFG', min_stints=10
        )

        # Compile results
        results = {
            'summary': {
                'total_stints_analyzed': len(lineup_data),
                'unique_home_lineups': variance_results['variance_summary']['total_unique_home_lineups'],
                'unique_away_lineups': variance_results['variance_summary']['total_unique_away_lineups'],
                'home_def_eFG_range': variance_results['variance_summary']['home_def_eFG_range'],
                'away_def_eFG_range': variance_results['variance_summary']['away_def_eFG_range'],
                'between_lineup_std_home': variance_results['variance_summary']['home_def_eFG_std_between_lineups'],
                'between_lineup_std_away': variance_results['variance_summary']['away_def_eFG_std_between_lineups'],
            },
            'variance_analysis': variance_results,
            'extreme_analysis': {
                'home_teams_defending': home_extreme,
                'away_teams_defending': away_extreme
            },
            'conclusions': self._draw_conclusions(variance_results, home_extreme, away_extreme)
        }

        # Print summary
        self._print_summary(results)

        return results

    def _draw_conclusions(self, variance_results: Dict, home_extreme: Dict, away_extreme: Dict) -> List[str]:
        """
        Draw conclusions from the analysis results.

        Args:
            variance_results: Results from variance analysis
            home_extreme: Results from home team extreme analysis
            away_extreme: Results from away team extreme analysis

        Returns:
            List of conclusion statements
        """
        conclusions = []

        # Check if lineup differences are meaningful
        home_std = variance_results['variance_summary']['home_def_eFG_std_between_lineups']
        away_std = variance_results['variance_summary']['away_def_eFG_std_between_lineups']

        if home_std > 0.02 or away_std > 0.02:  # More than 2% difference between lineup means
            conclusions.append("✅ SIGNIFICANT LINEUP VARIANCE: Lineup differences show meaningful defensive impact variance")
        else:
            conclusions.append("❌ MINIMAL LINEUP VARIANCE: Lineup differences may not be practically significant")

        # Check effect sizes for extreme lineups
        home_effect = home_extreme.get('effect_size', 0) if isinstance(home_extreme, dict) else 0
        away_effect = away_extreme.get('effect_size', 0) if isinstance(away_extreme, dict) else 0

        if home_effect > 0.5 or away_effect > 0.5:  # Medium effect size or larger
            conclusions.append("✅ LARGE EFFECT SIZES: Extreme lineups show substantial defensive performance differences")
        elif home_effect > 0.2 or away_effect > 0.2:  # Small effect size
            conclusions.append("⚠️ SMALL EFFECT SIZES: Lineup differences exist but are modest")
        else:
            conclusions.append("❌ NEGLIGIBLE EFFECTS: Lineup differences may be due to random variation")

        # Check sample sizes
        home_lineups = variance_results['variance_summary']['total_unique_home_lineups']
        away_lineups = variance_results['variance_summary']['total_unique_away_lineups']

        if home_lineups > 100 and away_lineups > 100:
            conclusions.append("✅ SUFFICIENT SAMPLE SIZE: Large number of unique lineups provides robust analysis")
        elif home_lineups > 50 and away_lineups > 50:
            conclusions.append("⚠️ MODERATE SAMPLE SIZE: Analysis is reasonable but could be improved with more data")
        else:
            conclusions.append("❌ LIMITED SAMPLE SIZE: More data needed for definitive conclusions")

        return conclusions

    def _print_summary(self, results: Dict) -> None:
        """
        Print analysis summary to console.

        Args:
            results: Complete analysis results
        """
        print("\n" + "="*60)
        print("📊 DEFENSIVE LINEUP VARIANCE ANALYSIS SUMMARY")
        print("="*60)

        summary = results['summary']
        print(f"Total stints analyzed: {summary['total_stints_analyzed']:,}")
        print(f"Unique home lineups: {summary['unique_home_lineups']:,}")
        print(f"Unique away lineups: {summary['unique_away_lineups']:,}")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")

        print("\n🎯 CONCLUSIONS:")
        for conclusion in results['conclusions']:
            print(f"  {conclusion}")

        print("\n" + "="*60)


def main():
    """Main analysis function."""
    analyzer = DefensiveLineupVarianceAnalyzer()

    try:
        results = analyzer.run_full_analysis()

        # Save results to file
        output_file = Path("defensive_lineup_variance_results.json")
        import json

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            else:
                return obj

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy_types)

        print(f"\n💾 Results saved to {output_file}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
