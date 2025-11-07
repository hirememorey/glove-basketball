# Free Throw Possession Attribution Guide

## Overview
This guide documents the complete free throw possession attribution logic developed for PADIM's RAPM implementation. Free throws represent a critical component of defensive attribution since they are points allowed due to defensive breakdowns (fouls).

## Official NBA Possession Definition Alignment
**Official Definition:** "A possession is the period of time a team has the basketball and the opportunity to score."

**PADIM Principle:** Free throws are part of the same possession as the foul that caused them. The final free throw outcome determines if the possession ends or continues.

## Attribution Logic by Foul Type

### 1. Shooting Fouls (Most Common - ~96% of Cases)

Shooting fouls create extended defensive sequences where the same lineup is responsible for the entire foul → free throws → outcome sequence.

#### Attribution Rules:

**Made Final Free Throw:**
```
Foul occurs → FT sequence → Final FT made → Score ends possession
Result: Entire sequence attributed to defensive lineup present during foul
New possession begins for opposing team
```

**Missed Final FT + Defensive Rebound:**
```
Foul occurs → FT sequence → Final FT missed → Defensive rebound → Ball changes teams
Result: Entire sequence attributed to defensive lineup present during foul
New possession begins for opposing team
```

**Missed Final FT + Offensive Rebound:**
```
Foul occurs → FT sequence → Final FT missed → Offensive rebound → Ball stays with team
Result: Possession continues - same defensive lineup remains responsible
Offensive team extends their possession
```

### 2. Technical/Flagrant Fouls (~3-4% of Cases)

Technical fouls break the possession continuity and create separate defensive possessions.

```
Foul occurs → Possession ends immediately
Technical FT sequence occurs → Creates new defensive possession
Result: Two separate possessions - one ending with technical foul, one beginning after FTs
```

### 3. Three-Second Violations

Similar to technical fouls, these create separate defensive possessions.

```
Defensive three-second call → Possession ends
FT sequence occurs → Creates new defensive possession
Result: Separate defensive possession created
```

### 4. And-One Free Throws

And-one situations belong to the same possession as the basket that triggered them.

```
Basket made → Foul called → Bonus FT → Same possession attribution
Result: Entire basket + FT sequence attributed to same defensive lineup
```

## Implementation Requirements

### Data Required for Attribution:
1. **Foul Type Classification:** shooting, technical, flagrant, three-second, and-one
2. **Free Throw Sequence Tracking:** All FTs in sequence, final FT outcome
3. **Rebound Classification:** Offensive vs defensive rebounds after missed FTs
4. **Team Identification:** Which team committed foul vs. which team shooting FTs

### Edge Cases to Handle:
1. **Substitution Timing:** Players may substitute during FT sequences
2. **Multiple Foul Types:** Complex sequences with multiple foul types
3. **Turnovers During FTs:** Rare but possible turnovers during free throw sequences
4. **Team Rebounds:** When rebound descriptions don't specify offensive/defensive

## RAPM Implementation Impact

### Shooting Fouls:
- Create extended defensive possessions covering foul + FT sequence + outcome
- Increase defensive observation count per lineup
- More comprehensive defensive attribution (beyond just shots)

### Technical Fouls:
- Create separate defensive possessions
- Defensive lineups get "free" possessions to defend
- Different attribution logic than shooting fouls

### Statistical Considerations:
- Shooting fouls provide richer defensive data (multi-event sequences)
- Technical fouls provide additional defensive possessions
- Proper attribution ensures defensive RAPM coefficients reflect true responsibility

## Validation Approach

### Test Cases Required:
1. **Shooting foul + made FTs:** Verify possession ends, new defensive possession created
2. **Shooting foul + missed FT + defensive rebound:** Verify possession ends, new defensive possession
3. **Shooting foul + missed FT + offensive rebound:** Verify possession continues, same lineup
4. **Technical foul:** Verify separate possession created
5. **And-one:** Verify same possession as basket

### Accuracy Metrics:
- **Foul Type Classification:** >99% accuracy required
- **Final FT Outcome Detection:** 100% accuracy required
- **Rebound Classification:** >95% accuracy required
- **Possession Attribution:** 100% accuracy required

## Code Implementation Structure

```python
def get_ft_possession_attribution(foul_type, final_ft_made, next_event_type, rebound_type=None):
    """
    Determine possession attribution for free throw sequences.

    Args:
        foul_type: 'shooting', 'technical', 'flagrant', 'three_second', 'and_one'
        final_ft_made: Boolean - was the final FT made?
        next_event_type: 'Rebound', 'Made Shot', 'Missed Shot', etc.
        rebound_type: 'offensive', 'defensive', or None

    Returns:
        Attribution result: 'possession_ends', 'possession_continues', 'new_possession'
    """
    # Implementation logic based on rules above
    pass
```

## References
- Official NBA possession definition: "period of time a team has the basketball and opportunity to score"
- Dean Oliver (Basketball on Paper): Possessions end with score, turnover, or opponent possession
- NBA rule interpretations for foul and free throw sequences
