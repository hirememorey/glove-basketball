# A Framework for Quantifying Strategic Inefficiencies in the NBA

**For the full project history and strategic vision, please read `VISION_AND_STRATEGY.md` first.**

---

## Core Thesis

This project's goal is to discover and quantify systemic, second-order inefficiencies in NBA team strategy, coaching, and psychology. We do not measure player value; we measure the properties of the *system* in which players operate.

Our work is organized into three core research initiatives:

1.  **Project "Jenga"**: Quantifying a team's strategic brittleness and its dependency on a single superstar.
2.  **Project "Counter-Punch"**: Quantifying a coach's in-game tactical adaptability.
3.  **Project "Cognitive Tax"**: Quantifying the mental friction and decision-making chaos a defense imposes.

Each of these initiatives is designed to ask a non-consensus question and produce actionable insights for front-office decision-making, from playoff preparation to player acquisition.

## Quick Start

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

### Running an Analysis (Example from Project "Jenga")

```python
# (Note: This is a conceptual example of future scripts)
from sei.jenga import SystemDependencyCalculator

# Initialize the calculator for the 2023-24 season
calculator = SystemDependencyCalculator("2023-24")

# Calculate the System Dependency Index for the Denver Nuggets
nuggets_sdi = calculator.calculate_sdi(team_abbr="DEN", star_player_id=203999)

print(f"Denver Nuggets System Dependency Index: {nuggets_sdi:.3f}")
```

## Project Structure

Our project is organized around our three core research pods.

```
padim/
├── VISION_AND_STRATEGY.md   # ⭐️ START HERE: The project's full history and vision
├── METHODOLOGY.md           # The intellectual core of the new vision
├── README.md                # This file: a high-level summary
├── src/
│   ├── sei/                 # Systemic Efficiencies Initiative (new V2 code)
│   │   ├── jenga.py
│   │   ├── counter_punch.py
│   │   └── cognitive_tax.py
│   └── padim/               # Legacy V1 data infrastructure (API, DB)
├── archive/                 # ✅ All V1 code and docs (RAPM model) are here
├── data/
├── logs/
└── requirements.txt
```

For a full breakdown of each research pod's methodology and goals, please see `METHODOLOGY.md`.
