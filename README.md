# PADIM (Publicly-Achievable Defensive Impact Model)

A data collection and analysis platform for basketball defensive impact evaluation using only publicly available NBA data. Currently focused on building the foundation for RAPM (Regularized Adjusted Plus-Minus) methodology.

## Current Status

**✅ COMPLETED INFRASTRUCTURE:**
- Data collection infrastructure for NBA stats, hustle metrics, and shot charts
- Stint aggregation from GameRotation data for lineup tracking
- Multi-player batch processing with rate limiting
- Database schema for comprehensive defensive analytics
- **Domain Understanding Phase**: Comprehensive analysis of data quality, timing precision, and possession logic
- **CRITICAL BLOCKER RESOLVED**: Fixed stint aggregation logic - overlaps now properly resolve to 5v5 lineups
- Substitution timing analysis and overlap resolution algorithm
- Batch processing infrastructure with progress tracking and error recovery

**✅ POSSESSION-BASED ATTRIBUTION VALIDATED:**
- **PBP Data Granularity**: Confirmed sufficient for possession-level attribution (100% team extraction success)
- **Domain Logic**: Possessions cannot span lineups (dead ball substitution requirement)
- **Technical Feasibility**: Team extraction from descriptions works reliably
- **Statistical Foundation**: Each possession can be attributed to exactly one defensive lineup

**🔍 KEY DISCOVERIES:**
- **Data Quality**: Excellent (100/100 scores across all metrics)
- **PBP Granularity**: Complete possession-level attribution capability
- **Domain Rules**: Basketball substitutions require dead balls, ensuring clean possession boundaries
- **Team Extraction**: 100% success rate for possession-relevant events via description parsing
- **Current Dataset**: 174 games processed, 5,671 stints, 4.44% defensive outcome rate

**📋 NEXT PHASE: IMPLEMENTATION READY**
Critical assumptions derisked - ready for RAPM implementation:

**✅ COMPLETED DERISKING:**
1. **Extraction Accuracy**: Team extraction patterns validated (100% success rate)
2. **Free Throw Logic**: Possession attribution logic fully defined - free throws belong to foul possession, final FT outcome determines continuation vs. ending
3. **Possession Definition**: Aligned with official NBA methodology - possessions are opportunities to score
4. **Defensive Attribution**: Shooting fouls create extended defensive sequences, technical fouls create separate possessions

**📖 KEY DOCUMENTATION:**
- **[Free Throw Possession Guide](free_throw_possession_guide.md)**: Complete attribution logic for all foul types and outcomes

**🔄 REMAINING VALIDATION:**
1. **RAPM Stability**: Compare possession-based vs. time-based aggregation
2. **Defensive Variance**: Confirm attribution reveals meaningful lineup differences
3. **Data Consistency**: Validate team naming and quality across seasons

## Overview

PADIM will create multi-faceted defensive player fingerprints by analyzing a player's impact on team-level defensive outcomes through regression analysis. The system focuses on three core defensive domains:

1. **Shot Influence**: Impact on opponent shooting efficiency and blocking ability
2. **Shot Suppression**: Ability to deter high-value rim attempts
3. **Possession Creation**: Rate of creating turnovers through steals and charges

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd padim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.padim.data_collector import NBADataCollector

# Initialize data collector
collector = NBADataCollector()

# Collect season stats for all players
collector.collect_player_season_stats('2022-23')

# Collect hustle stats for all players
collector.collect_player_hustle_stats('2022-23')

# Collect data for a specific player (LeBron James)
collector.collect_single_player_data(2544, '2022-23')

# Aggregate stints for a game
from src.padim.stint_aggregator import StintAggregator
stint_agg = StintAggregator()
result = stint_agg.aggregate_game_stints('0022200001')

collector.close()
```

## Current Capabilities

- **Player Data Collection**: Season stats, game logs, shot charts, and hustle metrics
- **Team Processing**: Batch collection for entire rosters with rate limiting
- **Stint Aggregation**: Lineup tracking using GameRotation data with overlap resolution
- **Defensive Outcome Calculation**: Stint-level opponent shooting stats (eFG%, rim attempts, FGM/FGA) - currently limited to 4.5% of stints
- **Database Storage**: SQLite schema for all defensive analytics data
- **Batch Processing**: Large-scale game processing with progress tracking, error recovery, and rate limiting
- **Domain Analysis**: Comprehensive data exploration and quality assessment tools
- **Timing Analysis**: Precision time alignment between rotation and PBP data sources
- **Data Quality Assessment**: Multi-game validation framework with 100/100 quality scores
- **Statistical Analysis**: Player coverage analysis and data sparsity diagnostics

## Project Structure

```
padim/
├── src/padim/
│   ├── api/           # NBA Stats API client ✅
│   ├── db/            # Database utilities & schema ✅
│   ├── config/        # Configuration management ✅
│   ├── utils/         # Common utilities ✅
│   ├── data_collector.py  # Multi-player data collection ✅
│   └── stint_aggregator.py # Stint aggregation from GameRotation ✅
├── data/              # SQLite database with collected data ✅
├── models/            # Empty - RAPM models not yet trained
├── logs/              # Application logs
├── config.py          # Main configuration ✅
├── test_components.py # Infrastructure testing ✅
├── batch_game_processor.py # Large-scale game processing 🆕
├── data_exploration.py    # PBP and rotation data analysis 🆕
├── timing_analysis.py     # Precision timing analysis 🆕
├── possession_logic_study.py # Basketball possession rules 🆕
├── data_quality_assessment.py # Multi-game quality validation 🆕
├── free_throw_analysis.py # Free throw pattern analysis 🆕
├── free_throw_possession_guide.md # Complete FT attribution logic 📖
└── requirements.txt   # Python dependencies ✅
```

## Getting Started for New Developers

### Quick Status Check
```bash
# Check current processing status
python batch_game_processor.py --status

# View data quality assessment
python data_quality_assessment.py

# Examine current database contents
sqlite3 data/padim.db "SELECT COUNT(*) as games FROM (SELECT DISTINCT game_id FROM stints);"
```

### Key Challenges to Address
1. **Assumption Validation**: Derisk critical assumptions before large-scale expansion
2. **Extraction Accuracy**: Ensure 99.5%+ accuracy in team extraction from PBP descriptions
3. **Free Throw Logic**: Implement and validate complex free throw possession attribution
4. **RAPM Validation**: Confirm possession-based approach improves statistical stability

### Immediate Next Steps (Derisking Phase)
1. **Test extraction accuracy** across multiple games and team combinations
2. **Implement and validate free throw possession logic**
3. **Compare RAPM stability** between time-based and possession-based approaches
4. **Validate defensive variance** reveals meaningful lineup differences
5. **Expand dataset** only after assumptions are confirmed

### Critical Files to Understand
- `free_throw_possession_guide.md`: Complete free throw attribution logic (START HERE)
- `batch_game_processor.py`: Large-scale data processing
- `src/padim/stint_aggregator.py`: Current aggregation logic
- `free_throw_analysis.py`: Free throw pattern analysis and validation
- `data_quality_assessment.py`: Data validation framework
- `technical_specification.md`: Detailed technical requirements

## Development

This project was adapted from the basketball lineup optimizer infrastructure, focusing on defensive analytics while maintaining robust data collection and processing capabilities.

## License

[Add license information]
