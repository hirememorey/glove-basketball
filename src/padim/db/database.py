"""
Database connection and management module for PADIM.
"""
import sqlite3
import pandas as pd
import logging
from typing import Optional, List, Dict, Any
from src.padim.config.settings import DB_PATH

# Configure logging
from src.padim.config.logging_config import get_logger
logger = get_logger(__name__)

class DatabaseConnection:
    def __init__(self, db_path: str = DB_PATH):
        """Initialize database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._connection = sqlite3.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        self._cursor = self._connection.cursor()

        # Enable WAL mode for better concurrency
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")

        logging.info("Database connection established successfully.")

    def execute(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a SQL query with optional parameters."""
        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
        except sqlite3.Error as e:
            logging.error(f"Database error: {str(e)}")
            raise

    def executemany(self, query: str, params: List[tuple]) -> None:
        """Execute a SQL query with multiple parameter sets."""
        try:
            self._cursor.executemany(query, params)
        except sqlite3.Error as e:
            logging.error(f"Database error: {str(e)}")
            raise

    def fetchall(self) -> List[Dict[str, Any]]:
        """Fetch all results from the last query."""
        return [dict(row) for row in self._cursor.fetchall()]

    def fetchone(self) -> Optional[Dict[str, Any]]:
        """Fetch one result from the last query."""
        row = self._cursor.fetchone()
        return dict(row) if row else None

    def commit(self) -> None:
        """Commit the current transaction."""
        self._connection.commit()

    def create_tables(self) -> None:
        """Create initial database tables for PADIM data collection."""
        tables = {
            'players': '''
                CREATE TABLE IF NOT EXISTS players (
                    player_id INTEGER PRIMARY KEY,
                    player_name TEXT NOT NULL,
                    nickname TEXT,
                    team_id INTEGER,
                    team_abbreviation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'games': '''
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    season_id TEXT NOT NULL,
                    game_date DATE NOT NULL,
                    matchup TEXT NOT NULL,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    home_team_abbrev TEXT,
                    away_team_abbrev TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'player_game_stats': '''
                CREATE TABLE IF NOT EXISTS player_game_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    game_id TEXT NOT NULL,
                    team_id INTEGER,
                    team_abbreviation TEXT,
                    matchup TEXT,
                    game_date DATE,
                    wl TEXT,
                    min REAL,
                    pts INTEGER,
                    reb INTEGER,
                    ast INTEGER,
                    stl INTEGER,
                    blk INTEGER,
                    fg_pct REAL,
                    fg3_pct REAL,
                    ft_pct REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, game_id)
                )
            ''',
            'player_season_stats': '''
                CREATE TABLE IF NOT EXISTS player_season_stats (
                    player_id INTEGER PRIMARY KEY,
                    season_id TEXT NOT NULL,
                    player_name TEXT,
                    team_abbreviation TEXT,
                    age REAL,
                    gp INTEGER,
                    w INTEGER,
                    l INTEGER,
                    w_pct REAL,
                    min REAL,
                    pts REAL,
                    reb REAL,
                    ast REAL,
                    stl REAL,
                    blk REAL,
                    fg_pct REAL,
                    fg3_pct REAL,
                    ft_pct REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'shots': '''
                CREATE TABLE IF NOT EXISTS shots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    grid_type TEXT,
                    game_id TEXT NOT NULL,
                    game_event_id INTEGER,
                    player_id INTEGER NOT NULL,
                    player_name TEXT,
                    team_id INTEGER,
                    team_name TEXT,
                    period INTEGER,
                    minutes_remaining INTEGER,
                    seconds_remaining INTEGER,
                    action_type TEXT,
                    shot_type TEXT,
                    shot_zone_basic TEXT,
                    shot_zone_area TEXT,
                    shot_zone_range TEXT,
                    shot_distance REAL,
                    loc_x INTEGER,
                    loc_y INTEGER,
                    shot_attempted_flag INTEGER,
                    shot_made_flag INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'player_hustle_stats': '''
                CREATE TABLE IF NOT EXISTS player_hustle_stats (
                    player_id INTEGER PRIMARY KEY,
                    season_id TEXT NOT NULL,
                    player_name TEXT,
                    team_abbreviation TEXT,
                    contested_shots INTEGER,
                    contested_shots_2pt INTEGER,
                    contested_shots_3pt INTEGER,
                    deflections INTEGER,
                    charges_drawn INTEGER,
                    screen_assists INTEGER,
                    screen_ast_pts REAL,
                    loose_balls_recovered_off INTEGER,
                    loose_balls_recovered_def INTEGER,
                    loose_balls_recovered INTEGER,
                    pct_loose_balls_recovered_off REAL,
                    pct_loose_balls_recovered_def REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'teams': '''
                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY,
                    abbreviation TEXT UNIQUE NOT NULL,
                    nickname TEXT,
                    city TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }

        try:
            for table_name, create_sql in tables.items():
                self.execute(create_sql)
                logging.info(f"Created table: {table_name}")
            self.commit()
        except sqlite3.Error as e:
            logging.error(f"Error creating tables: {e}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            logging.info("Database connection closed successfully.")

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        logging.info(f"Successfully connected to the database at {DB_PATH}.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to the database: {str(e)}")
        raise
