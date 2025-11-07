#!/usr/bin/env python3
"""
Lineup Performance Comparison Analysis

This script compares defensive performance across different lineups to understand
the practical significance of lineup differences and validate RAPM assumptions.

Key analyses:
1. Compare top-performing vs bottom-performing lineups
2. Calculate effect sizes and practical significance
3. Analyze consistency of lineup performance
4. Validate that lineup differences exceed random variation

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

class LineupPerformanceComparator:
    """Compares defensive performance across different lineups."""

    def __init__(self):
        """Initialize the comparator."""
        self.db = DatabaseConnection()
        self.lineup_data = None

    def load_lineup_data(self, min_stints: int = 10, min_duration: float = 120.0) -> pd.DataFrame:
        """
        Load and aggregate lineup performance data.

        Args:
            min_stints: Minimum number of stints required for a lineup
            min_duration: Minimum stint duration in seconds

        Returns:
            DataFrame with lineup performance metrics
        """
        print("🔍 Loading lineup performance data...")

        # Load stint data with defensive metrics
        query = f"""
        SELECT
            game_id,
            stint_start,
            stint_end,
            duration,
            -- Home team lineups and defensive performance
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            home_opp_fga, home_opp_fgm, home_opp_fg3a, home_opp_fg3m, home_opp_rim_attempts,
            -- Away team lineups and defensive performance
            away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
            away_opp_fga, away_opp_fgm, away_opp_fg3a, away_opp_fg3m, away_opp_rim_attempts
        FROM stints
        WHERE duration >= {min_duration}
        ORDER BY game_id, stint_start
        """

        df = pd.read_sql_query(query, self.db._connection)
        print(f"📊 Loaded {len(df):,} stints")

        # Calculate defensive metrics
        df = self._calculate_defensive_metrics(df)

        # Create lineup signatures
        df = self._create_lineup_signatures(df)

        # Aggregate by lineup
        lineup_stats = self._aggregate_lineup_performance(df, min_stints)

        self.lineup_data = lineup_stats
        print(f"📈 Created performance data for {len(lineup_stats)} lineups (min {min_stints} stints)")

        return lineup_stats

    def _calculate_defensive_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate defensive efficiency metrics for each stint."""
        df_calc = df.copy()

        # Home team defensive metrics (defending against away team)
        df_calc['home_def_eFG'] = np.where(
            df_calc['home_opp_fga'] > 0,
            (df_calc['home_opp_fgm'] + 0.5 * df_calc['home_opp_fg3m']) / df_calc['home_opp_fga'],
            np.nan
        )

        df_calc['home_def_FG_pct'] = np.where(
            df_calc['home_opp_fga'] > 0,
            df_calc['home_opp_fgm'] / df_calc['home_opp_fga'],
            np.nan
        )

        df_calc['home_def_3pt_pct'] = np.where(
            df_calc['home_opp_fg3a'] > 0,
            df_calc['home_opp_fg3m'] / df_calc['home_opp_fg3a'],
            np.nan
        )

        df_calc['home_def_rim_rate'] = np.where(
            df_calc['home_opp_fga'] > 0,
            df_calc['home_opp_rim_attempts'] / df_calc['home_opp_fga'],
            np.nan
        )

        # Away team defensive metrics (defending against home team)
        df_calc['away_def_eFG'] = np.where(
            df_calc['away_opp_fga'] > 0,
            (df_calc['away_opp_fgm'] + 0.5 * df_calc['away_opp_fg3m']) / df_calc['away_opp_fga'],
            np.nan
        )

        df_calc['away_def_FG_pct'] = np.where(
            df_calc['away_opp_fga'] > 0,
            df_calc['away_opp_fgm'] / df_calc['away_opp_fga'],
            np.nan
        )

        df_calc['away_def_3pt_pct'] = np.where(
            df_calc['away_opp_fg3a'] > 0,
            df_calc['away_opp_fg3m'] / df_calc['away_opp_fg3a'],
            np.nan
        )

        df_calc['away_def_rim_rate'] = np.where(
            df_calc['away_opp_fga'] > 0,
            df_calc['away_opp_rim_attempts'] / df_calc['away_opp_fga'],
            np.nan
        )

        return df_calc

    def _create_lineup_signatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create consistent lineup signatures for grouping."""
        df_sig = df.copy()

        # Create sorted lineup signatures
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

    def _aggregate_lineup_performance(self, df: pd.DataFrame, min_stints: int) -> pd.DataFrame:
        """Aggregate performance metrics by lineup."""
        # Aggregate home lineups
        home_agg = df.groupby('home_lineup_sig').agg({
            'home_def_eFG': ['mean', 'std', 'count', 'sem'],
            'home_def_FG_pct': ['mean', 'std'],
            'home_def_3pt_pct': ['mean', 'std'],
            'home_def_rim_rate': ['mean', 'std'],
            'home_opp_fga': ['sum', 'mean'],
            'duration': ['sum', 'mean']
        }).round(4)

        # Flatten column names for home
        home_agg.columns = ['_'.join(col).strip() for col in home_agg.columns.values]
        home_agg = home_agg.reset_index()
        home_agg = home_agg[home_agg['home_def_eFG_count'] >= min_stints]
        home_agg['team_role'] = 'home'

        # Aggregate away lineups
        away_agg = df.groupby('away_lineup_sig').agg({
            'away_def_eFG': ['mean', 'std', 'count', 'sem'],
            'away_def_FG_pct': ['mean', 'std'],
            'away_def_3pt_pct': ['mean', 'std'],
            'away_def_rim_rate': ['mean', 'std'],
            'away_opp_fga': ['sum', 'mean'],
            'duration': ['sum', 'mean']
        }).round(4)

        # Flatten column names for away
        away_agg.columns = ['_'.join(col).strip() for col in away_agg.columns.values]
        away_agg = away_agg.reset_index()
        away_agg = away_agg[away_agg['away_def_eFG_count'] >= min_stints]
        away_agg['team_role'] = 'away'

        # Rename columns to be consistent
        home_agg = home_agg.rename(columns={
            'home_lineup_sig': 'lineup_sig',
            'home_def_eFG_mean': 'def_eFG_mean',
            'home_def_eFG_std': 'def_eFG_std',
            'home_def_eFG_count': 'stint_count',
            'home_def_eFG_sem': 'def_eFG_sem',
            'home_def_FG_pct_mean': 'def_FG_pct_mean',
            'home_def_FG_pct_std': 'def_FG_pct_std',
            'home_def_3pt_pct_mean': 'def_3pt_pct_mean',
            'home_def_3pt_pct_std': 'def_3pt_pct_std',
            'home_def_rim_rate_mean': 'def_rim_rate_mean',
            'home_def_rim_rate_std': 'def_rim_rate_std',
            'home_opp_fga_sum': 'total_opp_fga',
            'home_opp_fga_mean': 'opp_fga_per_stint',
            'duration_sum': 'total_duration',
            'duration_mean': 'avg_duration'
        })

        away_agg = away_agg.rename(columns={
            'away_lineup_sig': 'lineup_sig',
            'away_def_eFG_mean': 'def_eFG_mean',
            'away_def_eFG_std': 'def_eFG_std',
            'away_def_eFG_count': 'stint_count',
            'away_def_eFG_sem': 'def_eFG_sem',
            'away_def_FG_pct_mean': 'def_FG_pct_mean',
            'away_def_FG_pct_std': 'def_FG_pct_std',
            'away_def_3pt_pct_mean': 'def_3pt_pct_mean',
            'away_def_3pt_pct_std': 'def_3pt_pct_std',
            'away_def_rim_rate_mean': 'def_rim_rate_mean',
            'away_def_rim_rate_std': 'def_rim_rate_std',
            'away_opp_fga_sum': 'total_opp_fga',
            'away_opp_fga_mean': 'opp_fga_per_stint',
            'duration_sum': 'total_duration',
            'duration_mean': 'avg_duration'
        })

        # Combine home and away lineups
        combined = pd.concat([home_agg, away_agg], ignore_index=True)

        # Calculate additional metrics
        combined['total_possessions'] = combined['total_opp_fga']  # Approximation
        combined['possessions_per_stint'] = combined['opp_fga_per_stint']

        return combined

    def compare_performance_quartiles(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Compare performance between top and bottom quartile lineups.

        Args:
            df: Lineup performance DataFrame

        Returns:
            Dictionary with quartile comparison results
        """
        print("📊 Comparing performance quartiles...")

        # Sort by defensive efficiency
        sorted_lineups = df.sort_values('def_eFG_mean')

        # Get quartile boundaries
        n_lineups = len(sorted_lineups)
        quartile_size = n_lineups // 4

        bottom_quartile = sorted_lineups.head(quartile_size)
        top_quartile = sorted_lineups.tail(quartile_size)

        # Calculate statistics
        results = {
            'total_lineups': n_lineups,
            'quartile_size': quartile_size,
            'bottom_quartile': {
                'def_eFG_range': (bottom_quartile['def_eFG_mean'].min(),
                                bottom_quartile['def_eFG_mean'].max()),
                'def_eFG_mean': bottom_quartile['def_eFG_mean'].mean(),
                'def_eFG_std': bottom_quartile['def_eFG_mean'].std(),
                'avg_stints': bottom_quartile['stint_count'].mean(),
                'total_stints': bottom_quartile['stint_count'].sum()
            },
            'top_quartile': {
                'def_eFG_range': (top_quartile['def_eFG_mean'].min(),
                                top_quartile['def_eFG_mean'].max()),
                'def_eFG_mean': top_quartile['def_eFG_mean'].mean(),
                'def_eFG_std': top_quartile['def_eFG_mean'].std(),
                'avg_stints': top_quartile['stint_count'].mean(),
                'total_stints': top_quartile['stint_count'].sum()
            }
        }

        # Calculate effect size (Cohen's d)
        mean_diff = results['top_quartile']['def_eFG_mean'] - results['bottom_quartile']['def_eFG_mean']
        pooled_std = np.sqrt(
            (results['bottom_quartile']['def_eFG_std']**2 + results['top_quartile']['def_eFG_std']**2) / 2
        )

        if pooled_std > 0:
            results['effect_size'] = mean_diff / pooled_std
        else:
            results['effect_size'] = 0

        results['performance_gap'] = mean_diff
        results['gap_interpretation'] = self._interpret_performance_gap(mean_diff)

        return results

    def _interpret_performance_gap(self, gap: float) -> str:
        """Interpret the magnitude of performance gap."""
        abs_gap = abs(gap)

        if abs_gap >= 0.10:
            return "VERY LARGE: Lineup differences represent major defensive impact"
        elif abs_gap >= 0.08:
            return "LARGE: Lineup differences show substantial defensive effects"
        elif abs_gap >= 0.05:
            return "MODERATE: Lineup differences are practically significant"
        elif abs_gap >= 0.02:
            return "SMALL: Lineup differences exist but are modest"
        else:
            return "NEGLIGIBLE: Lineup differences may not be meaningful"

    def analyze_performance_stability(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze how consistently lineups perform across their stints.

        Args:
            df: Lineup performance DataFrame

        Returns:
            Dictionary with stability analysis results
        """
        print("🔄 Analyzing performance stability...")

        # Calculate within-lineup consistency (coefficient of variation)
        df['def_eFG_cv'] = df['def_eFG_std'] / df['def_eFG_mean']
        df['def_eFG_cv'] = df['def_eFG_cv'].fillna(0)

        # Calculate stability metrics
        stability_results = {
            'mean_within_lineup_cv': df['def_eFG_cv'].mean(),
            'median_within_lineup_cv': df['def_eFG_cv'].median(),
            'cv_distribution': {
                'q25': df['def_eFG_cv'].quantile(0.25),
                'q75': df['def_eFG_cv'].quantile(0.75),
                'min': df['def_eFG_cv'].min(),
                'max': df['def_eFG_cv'].max()
            },
            'consistency_interpretation': self._interpret_consistency(df['def_eFG_cv'].mean())
        }

        # Identify most and least consistent lineups
        most_consistent = df.nsmallest(5, 'def_eFG_cv')[['lineup_sig', 'def_eFG_mean', 'def_eFG_std', 'def_eFG_cv', 'stint_count']]
        least_consistent = df.nlargest(5, 'def_eFG_cv')[['lineup_sig', 'def_eFG_mean', 'def_eFG_std', 'def_eFG_cv', 'stint_count']]

        stability_results['most_consistent_lineups'] = most_consistent.to_dict('records')
        stability_results['least_consistent_lineups'] = least_consistent.to_dict('records')

        return stability_results

    def _interpret_consistency(self, mean_cv: float) -> str:
        """Interpret the consistency of lineup performance."""
        if mean_cv <= 0.10:
            return "HIGH CONSISTENCY: Lineups perform very consistently"
        elif mean_cv <= 0.15:
            return "MODERATE CONSISTENCY: Lineups show reasonable consistency"
        elif mean_cv <= 0.20:
            return "VARIABLE CONSISTENCY: Lineup performance varies noticeably"
        else:
            return "LOW CONSISTENCY: Lineup performance is highly variable"

    def validate_lineup_differences(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that lineup differences are statistically and practically significant.

        Args:
            df: Lineup performance DataFrame

        Returns:
            Dictionary with validation results
        """
        print("✅ Validating lineup differences...")

        # Statistical test: ANOVA across performance quartiles
        sorted_lineups = df.sort_values('def_eFG_mean')
        n_lineups = len(sorted_lineups)
        quartile_size = n_lineups // 4

        # Create quartile groups
        groups = []
        for i in range(4):
            start_idx = i * quartile_size
            end_idx = (i + 1) * quartile_size if i < 3 else n_lineups
            group_data = sorted_lineups.iloc[start_idx:end_idx].copy()
            group_data['quartile'] = f'Q{i+1}'
            groups.append(group_data)

        quartile_df = pd.concat(groups)

        # Perform ANOVA
        anova_result = stats.f_oneway(
            *[group['def_eFG_mean'] for group in groups]
        )

        # Calculate practical significance metrics
        overall_mean = df['def_eFG_mean'].mean()
        overall_std = df['def_eFG_mean'].std()

        validation_results = {
            'statistical_significance': {
                'anova_f_statistic': anova_result.statistic,
                'anova_p_value': anova_result.pvalue,
                'significant': anova_result.pvalue < 0.05,
                'highly_significant': anova_result.pvalue < 0.01
            },
            'practical_significance': {
                'overall_mean_def_eFG': overall_mean,
                'overall_std_def_eFG': overall_std,
                'coefficient_of_variation': overall_std / overall_mean if overall_mean > 0 else 0,
                'range_of_means': df['def_eFG_mean'].max() - df['def_eFG_mean'].min()
            },
            'sample_adequacy': {
                'total_lineups': len(df),
                'avg_stints_per_lineup': df['stint_count'].mean(),
                'total_stints': df['stint_count'].sum(),
                'min_stints': df['stint_count'].min(),
                'max_stints': df['stint_count'].max()
            }
        }

        return validation_results

    def plot_performance_distributions(self, df: pd.DataFrame) -> None:
        """Create visualizations of lineup performance distributions."""
        # Use non-interactive backend to avoid hanging
        plt.switch_backend('Agg')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Distribution of defensive eFG%
        axes[0,0].hist(df['def_eFG_mean'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,0].axvline(df['def_eFG_mean'].mean(), color='red', linestyle='--', linewidth=2,
                         label='.3f')
        axes[0,0].set_xlabel('Defensive eFG%')
        axes[0,0].set_ylabel('Number of Lineups')
        axes[0,0].set_title('Distribution of Lineup Defensive eFG%')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Performance vs Sample Size
        scatter = axes[0,1].scatter(df['stint_count'], df['def_eFG_mean'],
                                  alpha=0.6, s=50, c=df['def_eFG_std'], cmap='viridis')
        axes[0,1].set_xlabel('Number of Stints')
        axes[0,1].set_ylabel('Defensive eFG%')
        axes[0,1].set_title('Performance vs Sample Size\n(Color = Std Dev)')
        plt.colorbar(scatter, ax=axes[0,1], label='Standard Deviation')

        # 3. Consistency (CV) distribution
        df['def_eFG_cv'] = df['def_eFG_std'] / df['def_eFG_mean']
        df['def_eFG_cv'] = df['def_eFG_cv'].fillna(0)

        axes[1,0].hist(df['def_eFG_cv'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].axvline(df['def_eFG_cv'].mean(), color='red', linestyle='--', linewidth=2,
                         label='.3f')
        axes[1,0].set_xlabel('Coefficient of Variation')
        axes[1,0].set_ylabel('Number of Lineups')
        axes[1,0].set_title('Lineup Performance Consistency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 4. Performance quartiles boxplot
        sorted_lineups = df.sort_values('def_eFG_mean')
        n_lineups = len(sorted_lineups)
        quartile_size = n_lineups // 4

        quartile_data = []
        labels = []
        for i in range(4):
            start_idx = i * quartile_size
            end_idx = (i + 1) * quartile_size if i < 3 else n_lineups
            quartile_data.append(sorted_lineups.iloc[start_idx:end_idx]['def_eFG_mean'])
            labels.append(f'Q{i+1}')

        axes[1,1].boxplot(quartile_data, tick_labels=labels)
        axes[1,1].set_ylabel('Defensive eFG%')
        axes[1,1].set_title('Defensive Performance by Quartile')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot instead of showing
        output_file = Path("lineup_performance_distributions.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 Plot saved to {output_file}")

    def run_full_analysis(self) -> Dict[str, any]:
        """
        Run complete lineup performance comparison analysis.

        Returns:
            Comprehensive analysis results
        """
        print("🚀 Starting Lineup Performance Comparison Analysis")
        print("=" * 60)

        # Load data
        lineup_data = self.load_lineup_data(min_stints=10, min_duration=120)

        # Run analyses
        quartile_comparison = self.compare_performance_quartiles(lineup_data)
        stability_analysis = self.analyze_performance_stability(lineup_data)
        validation_results = self.validate_lineup_differences(lineup_data)

        # Compile results
        results = {
            'summary': {
                'total_lineups_analyzed': len(lineup_data),
                'avg_stints_per_lineup': lineup_data['stint_count'].mean(),
                'performance_range': (
                    lineup_data['def_eFG_mean'].min(),
                    lineup_data['def_eFG_mean'].max()
                ),
                'mean_defensive_efficiency': lineup_data['def_eFG_mean'].mean()
            },
            'quartile_comparison': quartile_comparison,
            'stability_analysis': stability_analysis,
            'validation_results': validation_results,
            'conclusions': self._draw_conclusions(quartile_comparison, stability_analysis, validation_results)
        }

        # Print summary
        self._print_summary(results)

        # Create plots
        try:
            self.plot_performance_distributions(lineup_data)
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")

        return results

    def _draw_conclusions(self, quartile_comp: Dict, stability: Dict, validation: Dict) -> List[str]:
        """Draw conclusions from all analyses."""
        conclusions = []

        # Statistical significance
        if validation['statistical_significance']['highly_significant']:
            conclusions.append("✅ HIGH STATISTICAL SIGNIFICANCE: Lineup differences are highly significant (p < 0.01)")
        elif validation['statistical_significance']['significant']:
            conclusions.append("✅ STATISTICAL SIGNIFICANCE: Lineup differences are statistically significant (p < 0.05)")
        else:
            conclusions.append("❌ NO STATISTICAL SIGNIFICANCE: Lineup differences may be due to random variation")

        # Practical significance
        gap = quartile_comp['performance_gap']
        if abs(gap) >= 0.08:
            conclusions.append("✅ LARGE PRACTICAL SIGNIFICANCE: Performance gap between best/worst lineups is substantial")
        elif abs(gap) >= 0.05:
            conclusions.append("⚠️  MODERATE PRACTICAL SIGNIFICANCE: Performance differences are noticeable but not dramatic")
        else:
            conclusions.append("❌ SMALL PRACTICAL SIGNIFICANCE: Performance differences may not be practically meaningful")

        # Sample adequacy
        total_stints = validation['sample_adequacy']['total_stints']
        if total_stints >= 1000:
            conclusions.append("✅ EXCELLENT SAMPLE SIZE: Large dataset provides robust analysis")
        elif total_stints >= 500:
            conclusions.append("✅ GOOD SAMPLE SIZE: Sufficient data for reliable conclusions")
        elif total_stints >= 200:
            conclusions.append("⚠️  ADEQUATE SAMPLE SIZE: Analysis is reasonable but could be improved")
        else:
            conclusions.append("❌ LIMITED SAMPLE SIZE: More data needed for definitive conclusions")

        # Consistency assessment
        mean_cv = stability['mean_within_lineup_cv']
        if mean_cv <= 0.15:
            conclusions.append("✅ HIGH LINEUP CONSISTENCY: Lineups perform predictably across stints")
        elif mean_cv <= 0.20:
            conclusions.append("⚠️  MODERATE LINEUP CONSISTENCY: Some variation in lineup performance")
        else:
            conclusions.append("❌ LOW LINEUP CONSISTENCY: Lineup performance is highly variable")

        return conclusions

    def _print_summary(self, results: Dict) -> None:
        """Print analysis summary."""
        print("\n" + "="*60)
        print("📊 LINEUP PERFORMANCE COMPARISON SUMMARY")
        print("="*60)

        summary = results['summary']
        qc = results['quartile_comparison']

        print(f"Total lineups analyzed: {summary['total_lineups_analyzed']:,}")
        print(".1f")
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
    comparator = LineupPerformanceComparator()

    try:
        results = comparator.run_full_analysis()

        # Save results
        output_file = Path("lineup_performance_comparison_results.json")
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

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy_types)

        print(f"\n💾 Results saved to {output_file}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
