#!/usr/bin/env python3
"""
Test script for RAPM model implementation.

This script runs the complete RAPM pipeline and validates the results.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from padim.rapm_model import RAPMModel

def test_rapm_pipeline():
    """Test the complete RAPM pipeline."""
    print("🧪 Testing RAPM Pipeline")
    print("=" * 50)

    # Initialize RAPM model
    rapm = RAPMModel(alpha=1.0, cv_folds=3)  # Use fewer folds for speed

    try:
        # Run full pipeline
        print("🚀 Running RAPM pipeline...")
        results = rapm.run_full_pipeline()

        print("✅ Pipeline completed successfully!")
        print(f"📊 Processed {results['total_stints']:,} stints")
        print(f"👥 Found {results['total_players']:,} players")
        print(f"🎯 Trained {len(results['domains_trained'])} defensive domains")

        # Print domain results
        print("\n📈 Domain Results:")
        for domain, domain_results in results['results'].items():
            print(f"\n🏀 {domain.replace('_', ' ').title()}:")
            print(".4f")
            print(".4f")
            print(".4f")
            print(f"   Players with positive impact: {domain_results['stability']['n_players_above_zero']}")
            print(f"   Players with negative impact: {domain_results['stability']['n_players_below_zero']}")

        # Get top players for first domain
        if results['domains_trained']:
            first_domain = results['domains_trained'][0]
            print(f"\n🏆 Top 10 Players - {first_domain.replace('_', ' ').title()}:")
            top_players = rapm.get_player_coefficients(first_domain, top_n=10)
            for _, row in top_players.iterrows():
                print("+.4f")

            # Plot coefficient distribution
            print(f"\n📊 Plotting coefficient distribution for {first_domain}...")
            rapm.plot_coefficient_distribution(first_domain)

        return True

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_extraction():
    """Test data extraction functionality."""
    print("\n🧪 Testing Data Extraction")
    print("=" * 30)

    rapm = RAPMModel()

    try:
        df = rapm.extract_stint_data()
        print("✅ Data extraction successful!")
        print(f"📊 Shape: {df.shape}")
        print(f"🎮 Games: {df['game_id'].nunique()}")
        print(f"⏱️  Avg stint duration: {df['duration'].mean():.1f} seconds")

        # Check for defensive metrics
        defensive_cols = [col for col in df.columns if 'opp_' in col]
        print(f"🏀 Defensive metrics: {len(defensive_cols)}")

        return True

    except Exception as e:
        print(f"❌ Data extraction failed: {e}")
        return False

def test_design_matrix():
    """Test design matrix construction."""
    print("\n🧪 Testing Design Matrix Construction")
    print("=" * 40)

    rapm = RAPMModel()

    try:
        # Get stint data
        df = rapm.extract_stint_data()

        # Build design matrix
        X, player_ids = rapm.build_design_matrix(df)

        print("✅ Design matrix built successfully!")
        print(f"📊 Shape: {X.shape} (stints x players)")
        print(f"👥 Players: {len(player_ids)}")
        print(".1%")
        print(f"📈 Non-zero elements: {np.count_nonzero(X):,}")

        # Check matrix properties
        print(f"🏠 Home player values: {np.unique(X[X > 0])}")
        print(f"✈️  Away player values: {np.unique(X[X < 0])}")

        return True

    except Exception as e:
        print(f"❌ Design matrix construction failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 RAPM MVP Test Suite")
    print("=" * 50)

    # Run individual tests
    tests = [
        ("Data Extraction", test_data_extraction),
        ("Design Matrix", test_design_matrix),
        ("Full Pipeline", test_rapm_pipeline)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))

    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! RAPM MVP is ready for analysis.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
