#!/usr/bin/env python3
"""
RAPM Discrimination Validation

This script tests RAPM's ability to distinguish between lineups with known
defensive differences by training on a subset and validating predictions.

Key validations:
1. Train RAPM on subset of data
2. Use RAPM to predict lineup defensive impacts
3. Compare RAPM predictions against actual observed performance
4. Assess RAPM's discriminatory power and predictive accuracy

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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from padim.db.database import DatabaseConnection

class RAPMDiscriminationValidator:
    """Validates RAPM's ability to distinguish between defensive lineups."""

    def __init__(self):
        """Initialize the validator."""
        self.db = DatabaseConnection()
        self.rapm_coefficients = None
        self.lineup_predictions = None

    def load_training_data(self, max_games: int = 50) -> pd.DataFrame:
        """
        Load stint data for RAPM training.

        Args:
            max_games: Maximum number of games to include for faster training

        Returns:
            DataFrame with stint data for training
        """
        print(f"🔍 Loading training data (max {max_games} games)...")

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
        LIMIT {max_games * 25}  -- Approximate stints per game
        """

        df = pd.read_sql_query(query, self.db._connection)

        # Limit to specified number of games
        unique_games = df['game_id'].unique()[:max_games]
        df = df[df['game_id'].isin(unique_games)]

        print(f"📊 Loaded {len(df):,} stints from {len(unique_games)} games")

        # Calculate defensive eFG for target variable
        df = self._calculate_defensive_targets(df)

        return df

    def _calculate_defensive_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate defensive efficiency targets for RAPM."""
        df_calc = df.copy()

        # Home team defensive eFG (defending against away team)
        df_calc['home_def_eFG'] = np.where(
            df_calc['home_opp_fga'] > 0,
            (df_calc['home_opp_fgm'] + 0.5 * df_calc['home_opp_fg3m']) / df_calc['home_opp_fga'],
            np.nan
        )

        # Away team defensive eFG (defending against home team)
        df_calc['away_def_eFG'] = np.where(
            df_calc['away_opp_fga'] > 0,
            (df_calc['away_opp_fgm'] + 0.5 * df_calc['away_opp_fg3m']) / df_calc['away_opp_fga'],
            np.nan
        )

        return df_calc

    def create_design_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create RAPM design matrix and target vector.

        Args:
            df: Stint data DataFrame

        Returns:
            Tuple of (design_matrix, target_vector, player_ids)
        """
        print("🔢 Creating RAPM design matrix...")

        # Get all unique players
        all_players = set()
        for col in ['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5',
                   'away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5']:
            all_players.update(df[col].dropna().astype(int).unique())

        player_ids = sorted(list(all_players))
        player_to_idx = {pid: i for i, pid in enumerate(player_ids)}

        print(f"👥 Found {len(player_ids)} unique players")

        # Create design matrix
        n_stints = len(df)
        n_players = len(player_ids)
        X = np.zeros((n_stints, n_players))

        # Create target vector (defensive eFG allowed)
        y = np.zeros(n_stints)

        for i, (_, stint) in enumerate(df.iterrows()):
            # Home team players (defending) get +1
            for j in range(1, 6):
                player_col = f'home_player_{j}'
                if pd.notna(stint[player_col]):
                    player_id = int(stint[player_col])
                    if player_id in player_to_idx:
                        X[i, player_to_idx[player_id]] = 1

            # Away team players (defending) get +1
            for j in range(1, 6):
                player_col = f'away_player_{j}'
                if pd.notna(stint[player_col]):
                    player_id = int(stint[player_col])
                    if player_id in player_to_idx:
                        X[i, player_to_idx[player_id]] = 1

            # Target is the defensive eFG allowed by the home team (when they're defending)
            if pd.notna(stint['home_def_eFG']):
                y[i] = stint['home_def_eFG']
            elif pd.notna(stint['away_def_eFG']):
                y[i] = stint['away_def_eFG']
            else:
                y[i] = np.nan  # Will be filtered out

        # Remove rows with NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"📊 Design matrix shape: {X.shape} (stints x players)")
        print(".3f")

        return X, y, player_ids

    def train_rapm_model(self, X: np.ndarray, y: np.ndarray,
                        alpha: float = 100.0) -> Dict[str, any]:
        """
        Train RAPM model using Ridge regression.

        Args:
            X: Design matrix
            y: Target vector
            alpha: Regularization parameter

        Returns:
            Dictionary with model results
        """
        print(f"🎯 Training RAPM model (alpha={alpha})...")

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        results = {
            'model': model,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'alpha': alpha,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'n_train': len(y_train),
            'n_test': len(y_test)
        }

        print(".4f")
        print(".4f")

        return results

    def predict_lineup_impacts(self, model_results: Dict, player_ids: List[str],
                             lineup_data: pd.DataFrame) -> pd.DataFrame:
        """
        Use trained RAPM model to predict defensive impact of lineups.

        Args:
            model_results: Trained RAPM model results
            player_ids: List of player IDs in coefficient order
            lineup_data: DataFrame with lineup information

        Returns:
            DataFrame with predicted impacts
        """
        print("🔮 Predicting lineup defensive impacts...")

        predictions = []

        for _, lineup in lineup_data.iterrows():
            lineup_sig = lineup['lineup_sig']
            team_role = lineup['team_role']

            # Extract player IDs from lineup signature
            player_ids_in_lineup = [int(pid) for pid in lineup_sig.split(',')]

            # Calculate RAPM-predicted impact
            rapm_impact = 0
            players_found = 0

            for player_id in player_ids_in_lineup:
                if player_id in player_ids:
                    idx = player_ids.index(player_id)
                    rapm_impact += model_results['coefficients'][idx]
                    players_found += 1

            if players_found > 0:
                rapm_impact = rapm_impact / players_found  # Average per player
            else:
                rapm_impact = np.nan

            predictions.append({
                'lineup_sig': lineup_sig,
                'team_role': team_role,
                'rapm_predicted_impact': rapm_impact,
                'actual_def_eFG': lineup['def_eFG_mean'],
                'players_found': players_found,
                'stint_count': lineup['stint_count']
            })

        pred_df = pd.DataFrame(predictions)
        pred_df = pred_df.dropna(subset=['rapm_predicted_impact'])

        print(f"📊 Generated predictions for {len(pred_df)} lineups")

        return pred_df

    def validate_discriminatory_power(self, predictions: pd.DataFrame) -> Dict[str, any]:
        """
        Validate RAPM's ability to discriminate between lineups.

        Args:
            predictions: DataFrame with RAPM predictions and actual performance

        Returns:
            Dictionary with validation results
        """
        print("✅ Validating RAPM discriminatory power...")

        # Remove any remaining NaN values
        valid_preds = predictions.dropna()

        if len(valid_preds) < 10:
            return {'error': 'Insufficient valid predictions for analysis'}

        # Correlation between RAPM predictions and actual performance
        correlation = valid_preds['rapm_predicted_impact'].corr(valid_preds['actual_def_eFG'])

        # Sort by RAPM prediction and compare quartiles
        sorted_preds = valid_preds.sort_values('rapm_predicted_impact')

        n_lineups = len(sorted_preds)
        quartile_size = n_lineups // 4

        rapm_quartiles = []
        actual_quartiles = []

        for i in range(4):
            start_idx = i * quartile_size
            end_idx = (i + 1) * quartile_size if i < 3 else n_lineups

            quartile_data = sorted_preds.iloc[start_idx:end_idx]
            rapm_quartiles.append(quartile_data['rapm_predicted_impact'].mean())
            actual_quartiles.append(quartile_data['actual_def_eFG'].mean())

        # Calculate quartile discrimination
        rapm_range = rapm_quartiles[-1] - rapm_quartiles[0]
        actual_range = actual_quartiles[-1] - actual_quartiles[0]

        # Test if RAPM quartiles are significantly different
        from scipy import stats
        quartile_groups = []
        for i in range(4):
            start_idx = i * quartile_size
            end_idx = (i + 1) * quartile_size if i < 3 else n_lineups
            quartile_groups.append(sorted_preds.iloc[start_idx:end_idx]['actual_def_eFG'])

        anova_result = stats.f_oneway(*quartile_groups)

        validation_results = {
            'correlation_analysis': {
                'pearson_correlation': correlation,
                'correlation_interpretation': self._interpret_correlation(correlation),
                'n_lineups': len(valid_preds)
            },
            'quartile_analysis': {
                'rapm_quartile_means': rapm_quartiles,
                'actual_quartile_means': actual_quartiles,
                'rapm_range': rapm_range,
                'actual_range': actual_range,
                'quartile_discrimination_ratio': actual_range / rapm_range if rapm_range > 0 else 0
            },
            'statistical_tests': {
                'anova_f_statistic': anova_result.statistic,
                'anova_p_value': anova_result.pvalue,
                'quartiles_significantly_different': anova_result.pvalue < 0.05
            },
            'predictive_accuracy': {
                'mean_absolute_error': abs(valid_preds['rapm_predicted_impact'] - valid_preds['actual_def_eFG']).mean(),
                'root_mean_squared_error': np.sqrt(mean_squared_error(
                    valid_preds['actual_def_eFG'], valid_preds['rapm_predicted_impact']
                )),
                'r_squared': r2_score(valid_preds['actual_def_eFG'], valid_preds['rapm_predicted_impact'])
            }
        }

        return validation_results

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(corr)

        if abs_corr >= 0.8:
            return "VERY STRONG correlation"
        elif abs_corr >= 0.6:
            return "STRONG correlation"
        elif abs_corr >= 0.4:
            return "MODERATE correlation"
        elif abs_corr >= 0.2:
            return "WEAK correlation"
        else:
            return "VERY WEAK or NO correlation"

    def run_discrimination_analysis(self, max_games: int = 50,
                                  alpha: float = 100.0) -> Dict[str, any]:
        """
        Run complete RAPM discrimination validation.

        Args:
            max_games: Maximum games for training
            alpha: Regularization parameter

        Returns:
            Comprehensive validation results
        """
        print("🚀 Starting RAPM Discrimination Validation")
        print("=" * 60)

        # Load training data
        training_data = self.load_training_data(max_games=max_games)

        # Create design matrix
        X, y, player_ids = self.create_design_matrix(training_data)

        # Train RAPM model
        model_results = self.train_rapm_model(X, y, alpha=alpha)

        # Load lineup performance data for comparison
        from lineup_performance_comparison import LineupPerformanceComparator
        comparator = LineupPerformanceComparator()
        lineup_data = comparator.load_lineup_data(min_stints=5, min_duration=120)

        # Filter to players who appear in our RAPM training
        player_ids_set = set(player_ids)
        lineup_data['players_in_training'] = lineup_data['lineup_sig'].apply(
            lambda sig: sum(1 for pid in sig.split(',') if int(pid) in player_ids_set)
        )
        lineup_data = lineup_data[lineup_data['players_in_training'] >= 3]  # At least 3 players in training

        # Predict lineup impacts
        predictions = self.predict_lineup_impacts(model_results, player_ids, lineup_data)

        # Validate discriminatory power
        validation_results = self.validate_discriminatory_power(predictions)

        # Compile comprehensive results
        results = {
            'training_summary': {
                'games_used': len(training_data['game_id'].unique()),
                'stints_used': len(X),
                'players_in_model': len(player_ids),
                'alpha_used': alpha
            },
            'model_performance': {
                'train_r2': model_results['train_r2'],
                'test_r2': model_results['test_r2'],
                'train_mse': model_results['train_mse'],
                'test_mse': model_results['test_mse']
            },
            'prediction_summary': {
                'lineups_predicted': len(predictions),
                'correlation': validation_results['correlation_analysis']['pearson_correlation'],
                'predictive_r2': validation_results['predictive_accuracy']['r_squared'],
                'mae': validation_results['predictive_accuracy']['mean_absolute_error']
            },
            'validation_details': validation_results,
            'conclusions': self._draw_conclusions(validation_results)
        }

        # Print summary
        self._print_summary(results)

        return results

    def _draw_conclusions(self, validation: Dict) -> List[str]:
        """Draw conclusions from validation results."""
        conclusions = []

        # Correlation strength
        corr = validation['correlation_analysis']['pearson_correlation']
        if abs(corr) >= 0.4:
            conclusions.append("✅ STRONG CORRELATION: RAPM predictions correlate well with actual lineup performance")
        elif abs(corr) >= 0.2:
            conclusions.append("⚠️  MODERATE CORRELATION: RAPM shows some discriminatory power but room for improvement")
        else:
            conclusions.append("❌ WEAK CORRELATION: RAPM predictions poorly match actual performance")

        # Statistical significance
        if validation['statistical_tests']['quartiles_significantly_different']:
            conclusions.append("✅ STATISTICAL SIGNIFICANCE: RAPM quartiles show significantly different actual performance")
        else:
            conclusions.append("❌ NO STATISTICAL SIGNIFICANCE: RAPM fails to distinguish lineup performance statistically")

        # Predictive accuracy
        r2 = validation['predictive_accuracy']['r_squared']
        if r2 > 0.1:
            conclusions.append("✅ PREDICTIVE POWER: RAPM explains meaningful variance in lineup performance")
        elif r2 > 0:
            conclusions.append("⚠️  LIMITED PREDICTIVE POWER: RAPM shows some signal but weak explanatory power")
        else:
            conclusions.append("❌ NO PREDICTIVE POWER: RAPM predictions worse than random")

        # Practical significance
        mae = validation['predictive_accuracy']['mean_absolute_error']
        if mae < 0.03:  # Less than 3% error in eFG prediction
            conclusions.append("✅ PRACTICAL ACCURACY: RAPM predictions are practically useful")
        elif mae < 0.05:
            conclusions.append("⚠️  MODERATE ACCURACY: RAPM predictions have reasonable precision")
        else:
            conclusions.append("❌ POOR ACCURACY: RAPM predictions have large errors")

        return conclusions

    def _print_summary(self, results: Dict) -> None:
        """Print analysis summary."""
        print("\n" + "="*60)
        print("🎯 RAPM DISCRIMINATION VALIDATION SUMMARY")
        print("="*60)

        ts = results['training_summary']
        ps = results['prediction_summary']

        print(f"Training data: {ts['games_used']} games, {ts['stints_used']:,} stints, {ts['players_in_model']} players")
        print(".4f")
        print(f"Lineups predicted: {ps['lineups_predicted']:,}")
        print(".3f")
        print(".3f")
        print(".3f")

        print("\n🎯 CONCLUSIONS:")
        for conclusion in results['conclusions']:
            print(f"  {conclusion}")

        print("\n" + "="*60)


def main():
    """Main analysis function."""
    validator = RAPMDiscriminationValidator()

    try:
        results = validator.run_discrimination_analysis(max_games=50, alpha=100.0)

        # Save results (avoid circular reference issues)
        output_results = {
            k: v for k, v in results.items()
            if k not in ['validation_details']  # Skip complex nested dict
        }
        output_results['correlation'] = results['prediction_summary']['correlation']
        output_results['predictive_r2'] = results['prediction_summary']['predictive_r2']
        output_results['mae'] = results['prediction_summary']['mae']
        output_results['conclusions'] = results['conclusions']

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

        output_file = Path("rapm_discrimination_validation_results.json")
        with open(output_file, 'w') as f:
            json.dump(output_results, f, indent=2, default=convert_numpy_types)

        print(f"\n💾 Results saved to {output_file}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
