# Project Methodology

This document details the core methodologies for the **Systemic Efficiencies Initiative (SEI)**. Our work is organized into three primary research pods, each dedicated to answering a non-consensus question about the systemic properties of basketball.

For the strategic genesis of this approach, please see `VISION_AND_STRATEGY.md`.

---

## Pod 1: Project "Jenga" (Quantifying System Brittleness)

### The Core Question
How fragile is a team's strategic identity, and how dependent is it on a single player? We are not measuring a star's on-court impact; we are measuring the system's dependency on them as its central pillar.

### Methodology
1.  **Fingerprint the Offense:** Use play-by-play data to create a multi-variate "fingerprint" for every team's offense when their primary star is ON the court. The fingerprint will include metrics like:
    *   Pace (possessions per 48 minutes).
    *   Shot Profile (rim vs. mid-range vs. 3PA rate).
    *   Ball Movement Proxies (assist rate, "hockey-assist" estimates from PBP text).
    *   Isolation Frequency (time of possession per touch).
2.  **Measure the Collapse:** Generate the same fingerprint for when the star player is OFF the court.
3.  **Calculate the "System Dependency Index" (SDI):** Use a statistical measure (like Kullback-Leibler divergence or Mahalanobis distance) to quantify the "distance" between the ON and OFF fingerprints. A larger distance means a higher SDI and a more brittle, player-dependent system.

### Actionable Insight ("So What?")
*   **Playoff Risk:** Identify top-seeded teams with a high SDI as prime targets for upset. A focused defensive scheme against their star could shatter their entire system.
*   **Asset Valuation:** Identify "system-agnostic" players who maintain their performance even when moving between high- and low-SDI teams. They are more valuable and adaptable trade/free agency targets.

---

## Pod 2: Project "Counter-Punch" (Quantifying Coaching Adaptability)

### The Core Question
How quickly and effectively do coaches make tactical adjustments to stop an opponent's momentum in real-time?

### Methodology
1.  **Isolate "Run" Events:** Scrape play-by-play data to identify all opponent scoring runs of 8-0 or greater.
2.  **Analyze the "Problem":** For each run, create a "shot profile" of how the opponent was scoring (e.g., 3/3 on shots at the rim, 2/2 on corner threes).
3.  **Identify the "Intervention":** Log the timestamp of the coach's first intervention (e.g., timeout or a defensive-minded substitution).
4.  **Measure the "Solution":** Analyze the opponent's shot profile in the 5-10 possessions *after* the intervention. How quickly did the rate of those "problem" shots decrease? Did their eFG% on those specific plays regress faster than their season average would predict?

### Actionable Insight ("So What?")
*   **Hiring Decisions:** Use the resulting "Coach Adaptability Score" (CAS) as the first data-driven metric to distinguish great game-planners from great in-game problem solvers.
*   **Playoff Preparation:** Analyze an opposing coach's CAS to determine if they are adaptable or rigid. If they are rigid, we can be confident our initial game plan will be effective for longer stretches.

---

## Pod 3: Project "Cognitive Tax" (Quantifying Mental Friction)

### The Core Question
Can we measure the mental friction and decision-making chaos a defense imposes on an offense, in ways that don't show up in the traditional box score?

### Methodology
1.  **Establish Offensive Fingerprint Baseline:** As in Project Jenga, create a baseline offensive fingerprint for each team (time-to-shot, shot profile, ball movement proxies).
2.  **Measure Deviation Under Pressure:** For each game, measure how much a specific opponent's defense *forces* that team to deviate from its ideal fingerprint.
3.  **Create "Psychological Tax" Metrics:**
    *   **Hesitation Index:** Average time-to-shot. A great defense increases this.
    *   **Flow Disruption Score:** Decrease in assist/hockey-assist rate compared to the team's average.
    *   **Plan B Rate:** Percentage of possessions that end in a late-shot-clock (e.g., last 4 seconds) attempt, indicating the primary options failed.

### Actionable Insight ("So What?")
*   **Player Acquisition:** Identify players who contribute to a high "Cognitive Tax" defense even if their personal stats (steals, blocks) aren't elite. They are undervalued defensive assets.
*   **Strategic Design:** Adopt defensive schemes that rank highly on these metrics, creating a chaotic environment that is more likely to be effective in high-pressure playoff games.

---

## Where This Model Fails (Intellectual Honesty)

Our approach relies on using public data as a proxy for complex, unobservable phenomena. This has limitations:
*   **Correlation vs. Causation:** We can show that a star sitting correlates with a system change, but we can't perfectly isolate the star as the sole cause. Other lineup changes are a confounding variable.
*   **Proxy Fidelity:** Our proxies (e.g., assist rate for "ball movement") are imperfect. A low assist rate could mean great one-on-one play, not poor ball movement. We must always be critical of the gap between the proxy and the reality it represents.
*   **Data Granularity:** PBP data is not as clean or detailed as spatio-temporal data. Ambiguities in event logging can introduce noise into our analysis.
