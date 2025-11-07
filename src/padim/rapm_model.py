#!/usr/bin/env python3
"""
RAPM (Regularized Adjusted Plus-Minus) Model Implementation for PADIM.

This module implements the core RAPM methodology for evaluating individual
player defensive impact using ridge regression with regularization.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from padim.db.database import DatabaseConnection
from padim.config.logging_config import get_logger

logger = get_logger(__name__)


class RAPMModel:
    """
    Regularized Adjusted Plus-Minus model for defensive player evaluation.

    This class implements RAPM methodology using ridge regression to isolate
    individual player defensive contributions while controlling for teammates
    and opponents on the court.
    """

    def __init__(self, alpha: float = 1.0, cv_folds: int = 5):
        """
        Initialize RAPM model.

        Args:
            alpha: Regularization parameter for Ridge regression (default: 1.0)
            cv_folds: Number of cross-validation folds (default: 5)
        """
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.db = DatabaseConnection()
        self.model = None
        self.player_ids = None
        self.player_names = None
        self.coefficients = {}
        self.cv_scores = {}

        logger.info(f"Initialized RAPM model with alpha={alpha}, cv_folds={cv_folds}")

    def extract_stint_data(self) -> pd.DataFrame:
        """
        Extract stint data from database for RAPM modeling.

        Returns:
            DataFrame with stint data including lineups and defensive outcomes
        """
        logger.info("Extracting stint data from database...")

        query = """
        SELECT
            s.*,
            -- Home team defensive outcomes
            CASE WHEN s.home_opp_fga > 0
                 THEN (s.home_opp_fgm + 0.5 * s.home_opp_fg3m) / s.home_opp_fga
                 ELSE NULL END as home_def_eFG,
            CASE WHEN s.home_opp_fga > 0
                 THEN s.home_opp_rim_attempts / s.home_opp_fga
                 ELSE NULL END as home_rim_rate,
            -- Away team defensive outcomes
            CASE WHEN s.away_opp_fga > 0
                 THEN (s.away_opp_fgm + 0.5 * s.away_opp_fg3m) / s.away_opp_fga
                 ELSE NULL END as away_def_eFG,
            CASE WHEN s.away_opp_fga > 0
                 THEN s.away_opp_rim_attempts / s.away_opp_fga
                 ELSE NULL END as away_rim_rate
        FROM stints s
        WHERE s.duration >= 120  -- At least 2 minutes
        ORDER BY s.game_id, s.stint_start
        """

        df = pd.read_sql_query(query, self.db._connection)

        # Remove rows with missing defensive data
        df = df.dropna(subset=['home_def_eFG', 'away_def_eFG'])

        logger.info(f"Extracted {len(df):,} stints with complete defensive data")
        logger.info(f"Games covered: {df['game_id'].nunique()}")
        logger.info(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")

        return df

    def build_design_matrix(self, stint_data: pd.DataFrame) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build sparse design matrix for RAPM regression.

        Args:
            stint_data: DataFrame with stint information

        Returns:
            Tuple of (design_matrix, player_ids)
        """
        logger.info("Building RAPM design matrix...")

        # Get all unique player IDs
        home_players = [f'home_player_{i}' for i in range(1, 6)]
        away_players = [f'away_player_{i}' for i in range(1, 6)]
        all_player_cols = home_players + away_players

        # Collect all unique player IDs
        all_player_ids = set()
        for col in all_player_cols:
            all_player_ids.update(stint_data[col].dropna().astype(int))

        self.player_ids = sorted(list(all_player_ids))
        player_id_to_idx = {pid: idx for idx, pid in enumerate(self.player_ids)}

        logger.info(f"Found {len(self.player_ids):,} unique players")

        # Initialize sparse matrix
        n_stints = len(stint_data)
        n_players = len(self.player_ids)
        design_matrix = sparse.lil_matrix((n_stints, n_players))

        # Build design matrix: +1 for home team players, -1 for away team players
        for stint_idx in range(n_stints):
            stint = stint_data.iloc[stint_idx]

            # Home team players (+1)
            for col in home_players:
                player_id = stint[col]
                if not pd.isna(player_id) and int(player_id) in player_id_to_idx:
                    player_idx = player_id_to_idx[int(player_id)]
                    design_matrix[stint_idx, player_idx] = 1

            # Away team players (-1)
            for col in away_players:
                player_id = stint[col]
                if not pd.isna(player_id) and int(player_id) in player_id_to_idx:
                    player_idx = player_id_to_idx[int(player_id)]
                    design_matrix[stint_idx, player_idx] = -1

        # Convert to CSR format for efficient computation
        design_matrix = design_matrix.tocsr()

        logger.info(f"Design matrix shape: {design_matrix.shape}")
        logger.info(f"Non-zero elements: {design_matrix.nnz:,} ({design_matrix.nnz/design_matrix.shape[0]/design_matrix.shape[1]*100:.1f}% density)")

        return design_matrix, np.array(self.player_ids)

    def calculate_target_variable(self, stint_data: pd.DataFrame, domain: str) -> np.ndarray:
        """
        Calculate target variable (defensive outcome) for RAPM regression.

        Args:
            stint_data: DataFrame with stint data
            domain: Defensive domain ('shot_influence' or 'shot_suppression')

        Returns:
            Target variable array weighted by stint duration
        """
        if domain == 'shot_influence':
            # Target: opponent eFG% (lower is better defense)
            target = stint_data['home_def_eFG'] - stint_data['away_def_eFG']
        elif domain == 'shot_suppression':
            # Target: opponent rim attempt rate (lower is better defense)
            target = stint_data['home_rim_rate'] - stint_data['away_rim_rate']
        else:
            raise ValueError(f"Unknown domain: {domain}")

        # Weight by stint duration (square root for stability)
        weights = np.sqrt(stint_data['duration'])

        # Apply weights to target
        weighted_target = target * weights

        logger.info(f"Target variable for {domain}: mean={weighted_target.mean():.4f}, std={weighted_target.std():.4f}")

        return weighted_target.values

    def train_rapm_model(self, design_matrix: sparse.csr_matrix,
                        target: np.ndarray, domain: str) -> Dict[str, Any]:
        """
        Train RAPM model using Ridge regression with cross-validation.

        Args:
            design_matrix: Sparse design matrix (n_stints x n_players)
            target: Target variable array
            domain: Defensive domain being trained

        Returns:
            Dictionary with model results and coefficients
        """
        logger.info(f"Training RAPM model for {domain} domain...")

        # Initialize Ridge regression
        model = Ridge(alpha=self.alpha, fit_intercept=False)

        # Cross-validation
        cv_scores = cross_val_score(
            model, design_matrix, target,
            cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
            scoring='r2'
        )

        # Fit on full dataset
        model.fit(design_matrix, target)

        # Store results
        self.model = model
        self.cv_scores[domain] = cv_scores

        # Extract coefficients
        coefficients = model.coef_
        self.coefficients[domain] = coefficients

        results = {
            'domain': domain,
            'coefficients': coefficients,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_r2_scores': cv_scores,
            'intercept': model.intercept_,
            'n_players': len(coefficients),
            'alpha': self.alpha
        }

        logger.info(f"Training complete - CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return results

    def run_subset_pipeline(self, min_stints: int = 100) -> Dict[str, Any]:
        """
        Run RAPM pipeline on players with sufficient observations.

        Args:
            min_stints: Minimum stints required for player inclusion

        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Running RAPM subset pipeline (min_stints={min_stints})...")

        # Extract data
        stint_data = self.extract_stint_data()

        # Filter players by minimum stints
        all_player_cols = [f'home_player_{i}' for i in range(1, 6)] + [f'away_player_{i}' for i in range(1, 6)]

        player_stint_counts = {}
        for col in all_player_cols:
            counts = stint_data[col].value_counts()
            for player_id, count in counts.items():
                player_stint_counts[player_id] = player_stint_counts.get(player_id, 0) + count

        # Keep only players with sufficient stints
        qualified_players = {pid for pid, count in player_stint_counts.items() if count >= min_stints}

        # Filter stints to only include qualified players
        def has_qualified_players(row):
            players = []
            for i in range(1, 6):
                players.extend([row[f'home_player_{i}'], row[f'away_player_{i}']])
            return all(pid in qualified_players or pd.isna(pid) for pid in players)

        stint_data_filtered = stint_data[stint_data.apply(has_qualified_players, axis=1)]

        logger.info(f"Filtered to {len(qualified_players)} qualified players")
        logger.info(f"Remaining stints: {len(stint_data_filtered)}")

        # Build design matrix with filtered data
        design_matrix, player_ids = self.build_design_matrix(stint_data_filtered)
        self.player_ids = player_ids

        # Train models for each domain
        domains = ['shot_influence', 'shot_suppression']
        results = {}

        for domain in domains:
            target = self.calculate_target_variable(stint_data_filtered, domain)
            domain_results = self.train_rapm_model(design_matrix, target, domain)
            results[domain] = domain_results

        pipeline_results = {
            'total_stints': len(stint_data_filtered),
            'total_players': len(qualified_players),
            'min_stints': min_stints,
            'domains_trained': domains,
            'results': results,
            'player_ids': player_ids.tolist()
        }

        logger.info("Subset pipeline completed successfully")
        return pipeline_results

    def run_top_players_pipeline(self, n_players: int = 50) -> Dict[str, Any]:
        """
        Run RAPM pipeline on top N players by stint count.

        Args:
            n_players: Number of top players to include

        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Running RAPM top-{n_players} players pipeline...")

        # Extract data
        stint_data = self.extract_stint_data()

        # Get player stint counts
        all_player_cols = [f'home_player_{i}' for i in range(1, 6)] + [f'away_player_{i}' for i in range(1, 6)]

        player_stint_counts = {}
        for col in all_player_cols:
            counts = stint_data[col].value_counts()
            for player_id, count in counts.items():
                player_stint_counts[player_id] = player_stint_counts.get(player_id, 0) + count

        # Get top N players
        top_players = sorted(player_stint_counts.items(), key=lambda x: x[1], reverse=True)[:n_players]
        top_player_ids = {pid for pid, _ in top_players}

        logger.info(f"Top {n_players} players by stints: {[(pid, count) for pid, count in top_players[:5]]}")

        # Filter stints to only include top players
        def has_top_players(row):
            players = []
            for i in range(1, 6):
                players.extend([row[f'home_player_{i}'], row[f'away_player_{i}']])
            return any(pid in top_player_ids for pid in players if not pd.isna(pid))

        stint_data_filtered = stint_data[stint_data.apply(has_top_players, axis=1)]

        # Build design matrix
        design_matrix, player_ids = self.build_design_matrix(stint_data_filtered)
        self.player_ids = player_ids

        # Train models for each domain
        domains = ['shot_influence', 'shot_suppression']
        results = {}

        for domain in domains:
            target = self.calculate_target_variable(stint_data_filtered, domain)
            domain_results = self.train_rapm_model(design_matrix, target, domain)
            results[domain] = domain_results

        pipeline_results = {
            'total_stints': len(stint_data_filtered),
            'total_players': len(player_ids),
            'top_n_players': n_players,
            'domains_trained': domains,
            'results': results,
            'player_ids': player_ids.tolist()
        }

        logger.info("Top players pipeline completed successfully")
        return pipeline_results

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run complete RAPM pipeline on all available data.

        Returns:
            Dictionary with full pipeline results
        """
        logger.info("Running full RAPM pipeline...")

        # Extract all data
        stint_data = self.extract_stint_data()

        # Build design matrix
        design_matrix, player_ids = self.build_design_matrix(stint_data)
        self.player_ids = player_ids

        # Train models for each domain
        domains = ['shot_influence', 'shot_suppression']
        results = {}

        for domain in domains:
            target = self.calculate_target_variable(stint_data, domain)
            domain_results = self.train_rapm_model(design_matrix, target, domain)
            results[domain] = domain_results

        pipeline_results = {
            'total_stints': len(stint_data),
            'total_players': len(player_ids),
            'domains_trained': domains,
            'results': results,
            'player_ids': player_ids.tolist()
        }

        logger.info("Full pipeline completed successfully")
        return pipeline_results

    def get_player_coefficients(self, domain: str, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get player coefficients for a specific domain.

        Args:
            domain: Defensive domain ('shot_influence' or 'shot_suppression')
            top_n: Number of top players to return (optional)

        Returns:
            DataFrame with player IDs, names, and coefficients
        """
        if domain not in self.coefficients:
            raise ValueError(f"No coefficients available for domain: {domain}")

        coeffs = self.coefficients[domain]

        # Get player names from database
        player_names = {}
        for player_id in self.player_ids:
            try:
                result = self.db.fetchone("SELECT player_name FROM players WHERE player_id = ?", (int(player_id),))
                player_names[player_id] = result['player_name'] if result else f"Player {player_id}"
            except:
                player_names[player_id] = f"Player {player_id}"

        # Create results DataFrame
        results_df = pd.DataFrame({
            'player_id': self.player_ids,
            'player_name': [player_names[pid] for pid in self.player_ids],
            'coefficient': coeffs
        })

        # Sort by coefficient (most defensive to least defensive)
        results_df = results_df.sort_values('coefficient', ascending=True)

        if top_n:
            results_df = results_df.head(top_n)

        return results_df

    def plot_coefficient_distribution(self, domain: str, save_path: Optional[str] = None):
        """
        Plot distribution of player coefficients.

        Args:
            domain: Defensive domain to plot
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if domain not in self.coefficients:
                logger.warning(f"No coefficients available for domain: {domain}")
                return

            coeffs = self.coefficients[domain]

            plt.figure(figsize=(10, 6))
            sns.histplot(coeffs, bins=50, kde=True)
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            plt.xlabel(f'{domain.replace("_", " ").title()} Coefficient')
            plt.ylabel('Number of Players')
            plt.title(f'Distribution of {domain.replace("_", " ").title()} Coefficients')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")

    def generate_defensive_rankings(self, include_percentiles: bool = True) -> pd.DataFrame:
        """
        Generate comprehensive defensive player rankings across all domains.

        Args:
            include_percentiles: Whether to include percentile rankings

        Returns:
            DataFrame with complete defensive rankings
        """
        if not self.coefficients:
            raise ValueError("No trained models available. Run a pipeline first.")

        # Start with player IDs and names
        rankings_df = pd.DataFrame({
            'player_id': self.player_ids,
            'player_name': [self._get_player_name(pid) for pid in self.player_ids]
        })

        # Add coefficients for each domain
        for domain in self.coefficients.keys():
            coeffs = self.coefficients[domain]
            # More negative coefficients = better defense
            rankings_df[f'{domain}_coefficient'] = coeffs
            rankings_df[f'{domain}_rank'] = rankings_df[f'{domain}_coefficient'].rank(ascending=True)  # Lower (more negative) = better rank

            if include_percentiles:
                # Convert to percentile (lower coefficient = higher percentile for defense)
                rankings_df[f'{domain}_percentile'] = (1 - rankings_df[f'{domain}_rank'] / len(rankings_df)) * 100

        # Create combined defensive score (average of domains, weighted by domain importance)
        if len(self.coefficients) > 1:
            # For now, equal weighting between domains
            defensive_scores = []
            for domain in self.coefficients.keys():
                # Normalize coefficients to 0-100 scale (100 = best defense)
                coeffs = rankings_df[f'{domain}_coefficient']
                # Invert and scale: most negative (best defense) -> 100, least negative (worst) -> 0
                normalized = 100 * (coeffs.max() - coeffs) / (coeffs.max() - coeffs.min())
                defensive_scores.append(normalized)

            rankings_df['combined_defensive_score'] = np.mean(defensive_scores, axis=0)
            rankings_df['combined_defensive_rank'] = rankings_df['combined_defensive_score'].rank(ascending=False)
            rankings_df['combined_defensive_percentile'] = (rankings_df['combined_defensive_rank'] - 1) / (len(rankings_df) - 1) * 100

        # Sort by combined score if available, otherwise by first domain
        sort_col = 'combined_defensive_score' if 'combined_defensive_score' in rankings_df.columns else f"{list(self.coefficients.keys())[0]}_coefficient"
        rankings_df = rankings_df.sort_values(sort_col, ascending=False)  # Higher scores first

        return rankings_df

    def _get_player_name(self, player_id: int) -> str:
        """Get player name from database or return placeholder."""
        try:
            result = self.db.fetchone("SELECT player_name FROM players WHERE player_id = ?", (int(player_id),))
            return result['player_name'] if result else f"Player {player_id}"
        except:
            return f"Player {player_id}"

    def export_rankings_to_csv(self, filename: str, include_percentiles: bool = True):
        """
        Export defensive rankings to CSV file.

        Args:
            filename: Output filename
            include_percentiles: Whether to include percentile rankings
        """
        rankings_df = self.generate_defensive_rankings(include_percentiles)
        rankings_df.to_csv(filename, index=False)
        logger.info(f"Rankings exported to {filename}")

    def get_top_defenders(self, domain: str, n: int = 10) -> pd.DataFrame:
        """
        Get top N defensive players for a specific domain.

        Args:
            domain: Defensive domain ('shot_influence', 'shot_suppression', etc.)
            n: Number of players to return

        Returns:
            DataFrame with top defensive players
        """
        rankings_df = self.generate_defensive_rankings(include_percentiles=True)

        # Sort by domain coefficient (ascending = better defense)
        domain_coeff_col = f'{domain}_coefficient'
        domain_rank_col = f'{domain}_rank'

        if domain_coeff_col not in rankings_df.columns:
            raise ValueError(f"Domain '{domain}' not found in trained models")

        # Get top N (lowest coefficients = best defense)
        top_defenders = rankings_df.nsmallest(n, domain_coeff_col).copy()

        # Add interpretation
        top_defenders['defensive_interpretation'] = top_defenders[domain_coeff_col].apply(self._interpret_defensive_coefficient)

        return top_defenders[[
            'player_name', 'player_id', domain_coeff_col, domain_rank_col,
            f'{domain}_percentile', 'defensive_interpretation'
        ]]

    def _interpret_defensive_coefficient(self, coefficient: float) -> str:
        """Provide qualitative interpretation of defensive coefficient."""
        abs_coeff = abs(coefficient)

        if coefficient < -3:
            return "Elite Defender (consistently forces low opponent eFG%)"
        elif coefficient < -2:
            return "Strong Defender (regularly suppresses opponent scoring)"
        elif coefficient < -1:
            return "Good Defender (moderately impacts opponent efficiency)"
        elif coefficient < 0:
            return "Average Defender (slight positive defensive impact)"
        elif coefficient < 1:
            return "Below Average Defender (slight negative defensive impact)"
        elif coefficient < 2:
            return "Poor Defender (noticeably hurts team defense)"
        else:
            return "Very Poor Defender (significantly weakens team defense)"

    def compare_domains_correlation(self) -> pd.DataFrame:
        """
        Analyze correlation between different defensive domains.

        Returns:
            DataFrame with domain correlation analysis
        """
        if len(self.coefficients) < 2:
            logger.warning("Need at least 2 domains to analyze correlations")
            return pd.DataFrame()

        # Create correlation matrix
        domain_data = {}
        for domain, coeffs in self.coefficients.items():
            domain_data[domain] = coeffs

        corr_df = pd.DataFrame(domain_data).corr()

        # Add interpretation
        interpretations = {}
        for i in corr_df.index:
            for j in corr_df.columns:
                if i != j:
                    corr_val = corr_df.loc[i, j]
                    key = f"{i}_vs_{j}"
                    if corr_val > 0.7:
                        interpretations[key] = "Strong positive correlation (similar skills)"
                    elif corr_val > 0.3:
                        interpretations[key] = "Moderate positive correlation (related skills)"
                    elif corr_val > -0.3:
                        interpretations[key] = "Weak/no correlation (distinct skills)"
                    elif corr_val > -0.7:
                        interpretations[key] = "Moderate negative correlation (trade-off skills)"
                    else:
                        interpretations[key] = "Strong negative correlation (opposing skills)"

        return corr_df

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()