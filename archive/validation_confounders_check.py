#!/usr/bin/env python3
"""
Validation Confounders Check

This script checks for potential confounding factors that could lead to false
conclusions in our defensive impact validation.

Checks for:
1. Home court advantage effects
2. Pace differences between lineups
3. Opponent quality differences
4. Game situation effects
5. Systematic lineup assignment biases
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from padim.db.database import DatabaseConnection

# Set up plotting
plt.switch_backend('Agg')

class ValidationConfoundersChecker:
    """Checks for confounding factors in validation results."""

    def __init__(self):
        """Initialize the checker."""
        self.db = DatabaseConnection()

    def check_home_court_advantage(self) -> Dict[str, any]:
        """
        Check if home court advantage could confound lineup differences.

        Returns:
            Analysis of home court effects
        """
        print("🏠 Checking home court advantage effects...")

        query = """
        SELECT
            game_id,
            home_team_id,
            away_team_id,
            -- Home team defensive performance
            CASE WHEN home_opp_fga > 0 THEN (home_opp_fgm + 0.5 * home_opp_fg3m) / home_opp_fga ELSE NULL END as home_def_eFG,
            home_opp_fga,
            -- Away team defensive performance
            CASE WHEN away_opp_fga > 0 THEN (away_opp_fgm + 0.5 * away_opp_fg3m) / away_opp_fga ELSE NULL END as away_def_eFG,
            away_opp_fga,
            duration
        FROM stints
        WHERE duration >= 120
        """

        df = pd.read_sql_query(query, self.db._connection)

        # Remove rows with NaN values
        df = df.dropna()

        # Calculate home court advantage effect
        home_performance = df['home_def_eFG'].mean()
        away_performance = df['away_def_eFG'].mean()
        home_advantage = home_performance - away_performance

        # Statistical test
        t_stat, p_value = stats.ttest_ind(df['home_def_eFG'], df['away_def_eFG'])

        return {
            'home_advantage_effect': home_advantage,
            'home_mean_eFG': home_performance,
            'away_mean_eFG': away_performance,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': abs(home_advantage) / df['home_def_eFG'].std(),
            'interpretation': self._interpret_home_advantage(home_advantage, p_value)
        }

    def _interpret_home_advantage(self, advantage: float, p_value: float) -> str:
        """Interpret home court advantage findings."""
        abs_advantage = abs(advantage)

        if p_value >= 0.05:
            return "NO SIGNIFICANT HOME ADVANTAGE: Results not confounded by venue effects"
        elif abs_advantage < 0.01:
            return "NEGLIGIBLE HOME ADVANTAGE: Minimal venue confounding"
        elif abs_advantage < 0.02:
            return "SMALL HOME ADVANTAGE: Some venue confounding possible"
        else:
            return "SUBSTANTIAL HOME ADVANTAGE: Results may be confounded by venue effects"

    def check_pace_differences(self) -> Dict[str, any]:
        """
        Check if lineups differ systematically in pace/tempo.

        Returns:
            Analysis of pace differences across lineups
        """
        print("🏃 Checking pace differences between lineups...")

        query = """
        SELECT
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
            home_opp_fga, away_opp_fga, duration
        FROM stints
        WHERE duration >= 120
        """

        df = pd.read_sql_query(query, self.db._connection)

        # Create lineup signatures
        df['home_lineup'] = df.apply(
            lambda row: ','.join(sorted([
                str(int(row['home_player_1'])), str(int(row['home_player_2'])),
                str(int(row['home_player_3'])), str(int(row['home_player_4'])),
                str(int(row['home_player_5']))
            ])),
            axis=1
        )

        df['away_lineup'] = df.apply(
            lambda row: ','.join(sorted([
                str(int(row['away_player_1'])), str(int(row['away_player_2'])),
                str(int(row['away_player_3'])), str(int(row['away_player_4'])),
                str(int(row['away_player_5']))
            ])),
            axis=1
        )

        # Calculate pace metrics
        df['home_pace'] = df['home_opp_fga'] / (df['duration'] / 60)  # FGA per minute
        df['away_pace'] = df['away_opp_fga'] / (df['duration'] / 60)

        # Analyze pace by lineup
        home_pace_by_lineup = df.groupby('home_lineup')['home_pace'].agg(['mean', 'count']).reset_index()
        away_pace_by_lineup = df.groupby('away_lineup')['away_pace'].agg(['mean', 'count']).reset_index()

        # Filter for lineups with sufficient sample size
        home_pace_by_lineup = home_pace_by_lineup[home_pace_by_lineup['count'] >= 10]
        away_pace_by_lineup = away_pace_by_lineup[away_pace_by_lineup['count'] >= 10]

        # Calculate pace variance
        home_pace_variance = home_pace_by_lineup['mean'].std()
        away_pace_variance = away_pace_by_lineup['mean'].std()

        # Overall pace statistics
        overall_home_pace = df['home_pace'].mean()
        overall_away_pace = df['away_pace'].mean()

        return {
            'pace_variance_analysis': {
                'home_lineups_analyzed': len(home_pace_by_lineup),
                'away_lineups_analyzed': len(away_pace_by_lineup),
                'home_pace_std_between_lineups': home_pace_variance,
                'away_pace_std_between_lineups': away_pace_variance,
                'overall_home_pace': overall_home_pace,
                'overall_away_pace': overall_away_pace
            },
            'pace_range_analysis': {
                'home_pace_range': home_pace_by_lineup['mean'].max() - home_pace_by_lineup['mean'].min(),
                'away_pace_range': away_pace_by_lineup['mean'].max() - away_pace_by_lineup['mean'].min()
            },
            'interpretation': self._interpret_pace_differences(home_pace_variance, away_pace_variance)
        }

    def _interpret_pace_differences(self, home_std: float, away_std: float) -> str:
        """Interpret pace difference findings."""
        avg_std = (home_std + away_std) / 2

        if avg_std < 0.1:
            return "MINIMAL PACE VARIANCE: Lineups play at similar paces"
        elif avg_std < 0.2:
            return "MODERATE PACE VARIANCE: Some pace differences exist"
        else:
            return "SUBSTANTIAL PACE VARIANCE: Pace differences may confound defensive comparisons"

    def check_sample_size_bias(self) -> Dict[str, any]:
        """
        Check if sample size differences could bias results.

        Returns:
            Analysis of sample size effects
        """
        print("📊 Checking sample size bias effects...")

        query = """
        SELECT
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
            CASE WHEN home_opp_fga > 0 THEN (home_opp_fgm + 0.5 * home_opp_fg3m) / home_opp_fga ELSE NULL END as home_def_eFG,
            CASE WHEN away_opp_fga > 0 THEN (away_opp_fgm + 0.5 * away_opp_fg3m) / away_opp_fga ELSE NULL END as away_def_eFG,
            duration
        FROM stints
        WHERE duration >= 120
        """

        df = pd.read_sql_query(query, self.db._connection)
        df = df.dropna()

        # Create lineup signatures
        df['home_lineup'] = df.apply(
            lambda row: ','.join(sorted([
                str(int(row['home_player_1'])), str(int(row['home_player_2'])),
                str(int(row['home_player_3'])), str(int(row['home_player_4'])),
                str(int(row['home_player_5']))
            ])),
            axis=1
        )

        df['away_lineup'] = df.apply(
            lambda row: ','.join(sorted([
                str(int(row['away_player_1'])), str(int(row['away_player_2'])),
                str(int(row['away_player_3'])), str(int(row['away_player_4'])),
                str(int(row['away_player_5']))
            ])),
            axis=1
        )

        # Analyze sample sizes
        home_lineup_counts = df.groupby('home_lineup').size()
        away_lineup_counts = df.groupby('away_lineup').size()

        # Analyze performance by sample size
        home_performance_by_size = df.groupby('home_lineup').agg({
            'home_def_eFG': ['mean', 'std'],
            'home_lineup': 'size'
        }).reset_index()

        home_performance_by_size.columns = ['lineup', 'mean_eFG', 'std_eFG', 'count']

        # Check correlation between sample size and performance
        corr_count_performance = home_performance_by_size['count'].corr(home_performance_by_size['mean_eFG'])

        return {
            'sample_size_distribution': {
                'home_lineups_total': len(home_lineup_counts),
                'away_lineups_total': len(away_lineup_counts),
                'home_lineups_min_count': home_lineup_counts.min(),
                'home_lineups_max_count': home_lineup_counts.max(),
                'home_lineups_median_count': home_lineup_counts.median(),
                'away_lineups_min_count': away_lineup_counts.min(),
                'away_lineups_max_count': away_lineup_counts.max(),
                'away_lineups_median_count': away_lineup_counts.median()
            },
            'sample_size_bias_check': {
                'correlation_sample_size_performance': corr_count_performance,
                'significant_bias': abs(corr_count_performance) > 0.1
            },
            'interpretation': self._interpret_sample_bias(corr_count_performance, home_lineup_counts.std())
        }

    def _interpret_sample_bias(self, correlation: float, count_std: float) -> str:
        """Interpret sample size bias findings."""
        abs_corr = abs(correlation)

        if abs_corr < 0.05:
            return "NO SAMPLE SIZE BIAS: Performance uncorrelated with sample size"
        elif abs_corr < 0.1:
            return "MINIMAL SAMPLE SIZE BIAS: Weak correlation detected"
        elif abs_corr < 0.2:
            return "MODERATE SAMPLE SIZE BIAS: Some correlation with sample size"
        else:
            return "SUBSTANTIAL SAMPLE SIZE BIAS: Results may be confounded by sample size differences"

    def check_lineup_assignment_systematics(self) -> Dict[str, any]:
        """
        Check if lineup assignments are systematic rather than random.

        Returns:
            Analysis of lineup assignment patterns
        """
        print("🎯 Checking lineup assignment systematics...")

        # This is a simplified check - we'd need game context to do this properly
        # For now, just check if certain player combinations appear more frequently

        query = """
        SELECT
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            COUNT(*) as lineup_frequency
        FROM stints
        WHERE duration >= 120
        GROUP BY home_player_1, home_player_2, home_player_3, home_player_4, home_player_5
        ORDER BY lineup_frequency DESC
        LIMIT 20
        """

        df = pd.read_sql_query(query, self.db._connection)

        # Calculate concentration metrics
        total_stints = df['lineup_frequency'].sum()
        top_5_percentage = df.head(5)['lineup_frequency'].sum() / total_stints * 100
        gini_coefficient = self._calculate_gini(df['lineup_frequency'].values)

        return {
            'lineup_concentration': {
                'total_unique_lineups': len(df),
                'top_5_lineups_percentage': top_5_percentage,
                'gini_coefficient': gini_coefficient,
                'most_frequent_lineup_count': df['lineup_frequency'].max()
            },
            'interpretation': self._interpret_lineup_concentration(top_5_percentage, gini_coefficient)
        }

    def _calculate_gini(self, array: np.ndarray) -> float:
        """Calculate Gini coefficient for concentration analysis."""
        array = np.sort(array)
        n = len(array)
        cumsum = np.cumsum(array)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _interpret_lineup_concentration(self, top_5_pct: float, gini: float) -> str:
        """Interpret lineup concentration findings."""
        if top_5_pct < 10 and gini < 0.3:
            return "EVEN LINEUP DISTRIBUTION: Lineups appear randomly assigned"
        elif top_5_pct < 20 and gini < 0.5:
            return "MODERATE LINEUP CONCENTRATION: Some systematic assignment patterns"
        else:
            return "HIGH LINEUP CONCENTRATION: Lineup assignments may be systematic, not random"

    def check_statistical_assumptions(self) -> Dict[str, any]:
        """
        Check if our statistical tests meet their assumptions.

        Returns:
            Analysis of statistical assumption violations
        """
        print("📏 Checking statistical assumptions...")

        query = """
        SELECT
            CASE WHEN home_opp_fga > 0 THEN (home_opp_fgm + 0.5 * home_opp_fg3m) / home_opp_fga ELSE NULL END as home_def_eFG,
            CASE WHEN away_opp_fga > 0 THEN (away_opp_fgm + 0.5 * away_opp_fg3m) / away_opp_fga ELSE NULL END as away_def_eFG
        FROM stints
        WHERE duration >= 120
        """

        df = pd.read_sql_query(query, self.db._connection)
        df = df.dropna()

        # Test for normality
        home_shapiro = stats.shapiro(df['home_def_eFG'].sample(min(5000, len(df))))
        away_shapiro = stats.shapiro(df['away_def_eFG'].sample(min(5000, len(df))))

        # Test for equal variances
        levene_test = stats.levene(df['home_def_eFG'], df['away_def_eFG'])

        return {
            'normality_tests': {
                'home_eFG_normal': home_shapiro.pvalue > 0.05,
                'home_eFG_shapiro_p': home_shapiro.pvalue,
                'away_eFG_normal': away_shapiro.pvalue > 0.05,
                'away_eFG_shapiro_p': away_shapiro.pvalue
            },
            'variance_homogeneity': {
                'equal_variances': levene_test.pvalue > 0.05,
                'levene_p_value': levene_test.pvalue
            },
            'interpretation': self._interpret_statistical_assumptions(
                home_shapiro.pvalue, away_shapiro.pvalue, levene_test.pvalue
            )
        }

    def _interpret_statistical_assumptions(self, home_norm_p: float, away_norm_p: float,
                                         levene_p: float) -> str:
        """Interpret statistical assumption test results."""
        violations = []

        if home_norm_p < 0.05:
            violations.append("home team eFG not normally distributed")
        if away_norm_p < 0.05:
            violations.append("away team eFG not normally distributed")
        if levene_p < 0.05:
            violations.append("unequal variances between home/away")

        if not violations:
            return "ASSUMPTIONS MET: Statistical tests are appropriate"
        else:
            return f"ASSUMPTIONS VIOLATED: {', '.join(violations)} - consider non-parametric tests"

    def run_comprehensive_confounders_check(self) -> Dict[str, any]:
        """
        Run comprehensive check for confounding factors.

        Returns:
            Complete confounders analysis
        """
        print("🔍 Running comprehensive confounders check...")
        print("=" * 60)

        # Run all checks
        home_advantage = self.check_home_court_advantage()
        pace_differences = self.check_pace_differences()
        sample_bias = self.check_sample_size_bias()
        lineup_systematics = self.check_lineup_assignment_systematics()
        statistical_assumptions = self.check_statistical_assumptions()

        # Synthesize findings
        results = {
            'home_court_advantage': home_advantage,
            'pace_differences': pace_differences,
            'sample_size_bias': sample_bias,
            'lineup_assignment_systematics': lineup_systematics,
            'statistical_assumptions': statistical_assumptions,
            'overall_confounders_assessment': self._assess_overall_confounding_risk(
                home_advantage, pace_differences, sample_bias, lineup_systematics, statistical_assumptions
            )
        }

        # Print summary
        self._print_confounders_summary(results)

        return results

    def _assess_overall_confounding_risk(self, home_adv: Dict, pace: Dict, sample: Dict,
                                       lineup_sys: Dict, stats: Dict) -> Dict[str, any]:
        """Assess overall risk of confounding factors."""

        # Count high-risk factors
        high_risk_factors = 0
        moderate_risk_factors = 0

        # Home advantage
        if 'SUBSTANTIAL' in home_adv['interpretation']:
            high_risk_factors += 1
        elif 'SMALL' in home_adv['interpretation']:
            moderate_risk_factors += 1

        # Pace differences
        if 'SUBSTANTIAL' in pace['interpretation']:
            high_risk_factors += 1
        elif 'MODERATE' in pace['interpretation']:
            moderate_risk_factors += 1

        # Sample bias
        if 'SUBSTANTIAL' in sample['interpretation']:
            high_risk_factors += 1
        elif 'MODERATE' in sample['interpretation']:
            moderate_risk_factors += 1

        # Lineup systematics
        if 'HIGH' in lineup_sys['interpretation']:
            high_risk_factors += 1
        elif 'MODERATE' in lineup_sys['interpretation']:
            moderate_risk_factors += 1

        # Statistical assumptions
        if 'VIOLATED' in stats['interpretation']:
            moderate_risk_factors += 1

        # Overall assessment
        if high_risk_factors >= 2:
            risk_level = "HIGH"
            conclusion = "MULTIPLE HIGH-RISK CONFOUNDERS: Results may be substantially biased"
        elif high_risk_factors >= 1 or moderate_risk_factors >= 3:
            risk_level = "MODERATE"
            conclusion = "SOME CONFOUNDING RISK: Results should be interpreted cautiously"
        elif moderate_risk_factors >= 1:
            risk_level = "LOW"
            conclusion = "MINIMAL CONFOUNDING RISK: Results likely robust"
        else:
            risk_level = "VERY LOW"
            conclusion = "NO SIGNIFICANT CONFOUNDERS: High confidence in results"

        return {
            'risk_level': risk_level,
            'conclusion': conclusion,
            'high_risk_factors': high_risk_factors,
            'moderate_risk_factors': moderate_risk_factors,
            'recommendations': self._generate_confounding_recommendations(risk_level)
        }

    def _generate_confounding_recommendations(self, risk_level: str) -> List[str]:
        """Generate recommendations based on confounding risk."""
        if risk_level == "HIGH":
            return [
                "Re-analyze with controls for venue, pace, and sample size",
                "Consider propensity score matching or regression adjustment",
                "Validate findings on additional datasets",
                "Use more sophisticated statistical methods"
            ]
        elif risk_level == "MODERATE":
            return [
                "Interpret results with appropriate caution",
                "Consider sensitivity analyses with different controls",
                "Validate key findings with alternative methods",
                "Report confidence intervals and effect sizes"
            ]
        else:
            return [
                "Results appear robust to common confounding factors",
                "Continue with planned analyses",
                "Monitor for additional confounding factors in future work"
            ]

    def _print_confounders_summary(self, results: Dict) -> None:
        """Print comprehensive confounders summary."""
        print("\n" + "="*60)
        print("🔍 CONFOUNDERS ANALYSIS SUMMARY")
        print("="*60)

        assessment = results['overall_confounders_assessment']

        print(f"🎯 OVERALL RISK ASSESSMENT: {assessment['risk_level']}")
        print(f"📝 CONCLUSION: {assessment['conclusion']}")

        print(f"\n🔴 HIGH-RISK FACTORS: {assessment['high_risk_factors']}")
        print(f"🟡 MODERATE-RISK FACTORS: {assessment['moderate_risk_factors']}")

        print("\n📋 KEY CONFOUNDERS CHECKED:")
        print(f"🏠 Home Court Advantage: {results['home_court_advantage']['interpretation']}")
        print(f"🏃 Pace Differences: {results['pace_differences']['interpretation']}")
        print(f"📊 Sample Size Bias: {results['sample_size_bias']['interpretation']}")
        print(f"🎯 Lineup Assignment: {results['lineup_assignment_systematics']['interpretation']}")
        print(f"📏 Statistical Assumptions: {results['statistical_assumptions']['interpretation']}")

        print("\n💡 RECOMMENDATIONS:")
        for rec in assessment['recommendations']:
            print(f"  • {rec}")

        print("\n" + "="*60)


def main():
    """Main confounders check function."""
    checker = ValidationConfoundersChecker()

    try:
        results = checker.run_comprehensive_confounders_check()

        # Save results
        import json
        output_file = Path("validation_confounders_check_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to {output_file}")

    except Exception as e:
        print(f"❌ Confounders check failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
