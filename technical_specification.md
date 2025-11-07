# Technical Specification: Data Infrastructure

**Version:** 2.0 (Systemic Efficiencies Initiative)

**Status:** The low-level data collection and database infrastructure from V1 remains operational and serves as the foundation for the new V2 analyses. This document preserves the technical details of that stable infrastructure.

---

### 1. Data Collection Infrastructure

This section details the components responsible for acquiring and storing data from `stats.nba.com`. This infrastructure is considered feature-complete and stable.

- **NBA API Integration**: A complete client for `stats.nba.com` endpoints, located in `src/padim/api/client.py`. It includes handling for request parameters, headers, and rate limiting.
- **Database Schema**: A robust SQLite schema for storing all collected data types. The database connection and session management logic is in `src/padim/db/database.py`.
- **Resumable Batch Processing**: An enterprise-grade, resumable batch processor (`src/padim/resumable_processor.py`) ensures that large-scale data collection can survive interruptions with zero data loss. It maintains its state in `data/processing_state.json`.

### 2. Database Schema

The core schema remains unchanged from V1. The primary tables used for V2 analysis will be `games` and the play-by-play data (which is not stored in a dedicated table but fetched on-demand).

- **`games`**: Core information about each game.
- **`players`**: Basic player information.
- **`teams`**: Team mappings and abbreviations.
- **`player_season_stats`**: Aggregated season performance.
- **`player_game_stats`**: Individual game performance.
- **`shots`**: Shot chart data with location.
- **`player_hustle_stats`**: Advanced defensive metrics.
- **`stints`**: (Legacy) Aggregated lineup data for the V1 RAPM model.

### 3. Data Sources

The system is connected to a variety of `stats.nba.com` endpoints. The most relevant for the V2 analyses will be:

- **`PlayByPlayV2`**: The foundational data source for all three research pods.
- **`LeagueDashPlayerStats`**: Season aggregations for all players.
- **`PlayerGameLogs`**: Individual game performance.
- **`CommonTeamYears`**: Team ID mappings.