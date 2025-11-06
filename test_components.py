#!/usr/bin/env python3
"""
Test script to verify PADIM infrastructure components work correctly.
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test all component imports."""
    print("Testing component imports...")

    try:
        # Test configuration
        from padim.config.settings import DB_PATH, RAPM_TRAIN_SEASONS
        print(f"✓ Config import successful - DB_PATH: {DB_PATH}")
        print(f"✓ RAPM seasons: {RAPM_TRAIN_SEASONS}")

        # Test database
        from padim.db.database import DatabaseConnection
        print("✓ Database import successful")

        # Test API client
        from padim.api.client import NBAStatsClient
        print("✓ API client import successful")

        # Test utilities
        from padim.utils.common_utils import get_db_connection
        print("✓ Utilities import successful")

        # Test main config
        from config import get_config
        config = get_config()
        print(f"✓ Main config import successful - App: {config.APP_NAME}")

        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_api_client():
    """Test API client initialization."""
    print("\nTesting API client initialization...")

    try:
        from padim.api.client import NBAStatsClient
        client = NBAStatsClient()
        print("✓ API client initialization successful")
        return True
    except Exception as e:
        print(f"✗ API client test failed: {e}")
        return False

def test_database_connection():
    """Test database connection (without actual database file)."""
    print("\nTesting database connection setup...")

    try:
        from padim.db.database import DatabaseConnection
        # This will fail if database doesn't exist, but should not crash
        try:
            db = DatabaseConnection()
            print("✓ Database connection successful")
            db.close()
        except Exception as e:
            if "no such file" in str(e).lower():
                print("✓ Database connection setup correct (file doesn't exist yet)")
            else:
                raise e
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("PADIM Infrastructure Component Tests")
    print("=" * 40)

    tests = [
        test_imports,
        test_api_client,
        test_database_connection
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All infrastructure components working correctly!")
        return 0
    else:
        print("❌ Some components need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
