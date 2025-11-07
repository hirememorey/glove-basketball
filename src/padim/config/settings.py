"""Configuration settings for the PADIM application."""

import os
from typing import List

# Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Database Configuration
DB_PATH = os.path.join(PROJECT_ROOT, "data", "padim.db")
SEASON_ID = "2024-25"
MIN_MINUTES_THRESHOLD = 1000

# HTTP Request Configuration
MAX_WORKERS = 2
MAX_RETRIES = 12
API_TIMEOUT = 300  # Increased from 120 to 300 seconds for NBA API timeouts

# Delay/Backoff Configuration
MIN_SLEEP = 3.0
MAX_SLEEP = 6.0
RETRY_BACKOFF_BASE = 2
RETRY_BACKOFF_FACTOR = 1.2
MAX_BACKOFF_SLEEP = 300

# RAPM Configuration
RAPM_ALPHA_RANGE = [0.01, 0.1, 1.0, 10.0, 100.0]  # Regularization parameters for Ridge regression
MIN_STINT_LENGTH = 30  # Minimum seconds for stint inclusion in RAPM
MAX_STINT_LENGTH = 1200  # Maximum seconds for stint inclusion in RAPM

# Stability Testing Configuration
STABILITY_CORRELATION_THRESHOLD = 0.3  # Minimum year-over-year correlation for stable metrics
RAPM_TRAIN_SEASONS = ["2021-22", "2022-23"]  # Seasons for training RAPM models
RAPM_VALIDATION_SEASON = "2023-24"  # Season for validation

# User Agents for API requests
USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36"
]

# Base headers for API requests
BASE_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
}

# Logging Configuration (imported from main config)
import os
from pathlib import Path
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "padim.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"

# API Endpoints (PADIM-focused - defensive metrics and stint data)
API_BASE_URL = "https://stats.nba.com/stats"
ENDPOINTS = {
    "play_by_play": "/playbyplayv3",  # For stint aggregation
    "shot_chart": "/shotchartdetail",  # For shot location data
    "player_hustle": "/playerhustlestats",  # Steals, blocks, charges drawn
    "player_stats": "/leaguedashplayerstats",  # Basic player stats
    "common_player_info": "/commonplayerinfo",  # Player metadata
    "player_tracking": "/leaguedashptstats",  # Tracking stats including hustle
    "shooting_splits": "/playerdashboardbyshootingsplits"  # Rim frequency data
}
