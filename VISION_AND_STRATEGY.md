# Vision & Strategy: The Evolution of a Project

This document is the mandatory first read for any new developer. It explains the project's history, its current strategic vision, and the "why" behind our work. Understanding this evolution is critical to contributing effectively.

## Section 1: The Original Thesis (PADIM V1)

This project began as PADIM (Publicly-Achievable Defensive Impact Model). The original goal was to build a best-in-class, production-ready Regularized Adjusted Plus-Minus (RAPM) model to measure the defensive impact of individual NBA players using only public data.

This effort was a success. We built a sophisticated and robust data pipeline and a validated RAPM system that produced comprehensive player rankings. This entire V1 system, including its documentation and code, is now preserved in the `archive/` directory for historical reference. It is a complete and successful project, but it is no longer in active development.

## Section 2: The Pivot - The "Morey Critique"

As we evaluated the completed V1, we sought feedback from a leading industry expert on how to push the work to the true frontier of basketball analytics. The feedback we received fundamentally challenged our core assumptions and redefined our vision.

Here is that feedback, which serves as the strategic genesis for the project's current direction:

> ### Core Flaw in Current Public Analysis
>
> The consensus approach with public data is to measure what happened. We count points, shots, and rebounds. We aggregate them into rates and efficiencies. Even RAPM is just a sophisticated way of measuring the end result of a player's time on the court.
>
> The person who gets my attention ignores this. They understand that trying to out-model us on outcomes is a losing game. Instead, they use public data to model the second-order effects and systemic properties of the game. They're not analyzing the players; they're analyzing the system the players operate in.
>
> They come to me with a thesis that sounds something like this: "Everyone is focused on measuring player value. I believe there's a massive, unquantified inefficiency in measuring **System Resilience and Strategic Brittleness**."
>
> ### The Non-Consensus Questions They Ask and Answer
>
> This person doesn't show me another player ranking. They show me a new lens through which to view the entire league. They ask questions like:
>
> **1. The "Jenga" Question: How fragile is a team's offensive system?**
>
> Instead of asking "how good is Nikola Jokić?", they ask, "By how much does the Nuggets' entire strategic identity collapse when Jokić sits down?" They create a "System Dependency Index." A team with a high index might have a great record, but their success is brittle. This provides a new way to evaluate playoff risk and identify "system-agnostic" role players.
>
> **2. The "Counter-Punch" Question: Can we quantify a coach's in-game adaptability?**
>
> Instead of just looking at aggregate defensive stats, they ask, "How quickly and effectively does a coach's defensive scheme neutralize an opponent's hot streak in real-time?" They create a "Schematic Momentum" score for coaches, a data-driven tool to evaluate tactical adjustments.
>
> **3. The "Psychological Tax" Question: Can we measure the mental friction a defense imposes?**
>
> Instead of just measuring turnovers, they ask, "Can we quantify the 'Psychological Tax' a defensive scheme imposes on an offense, measured by its ability to induce hesitation and disrupt offensive flow?" They create a metric that quantifies how much a defense forces an opponent away from their ideal offensive "fingerprint."
>
> ### The Profile of the Work
>
> The way this person presents their work is as important as the work itself.
>
> *   **The Title is a Thesis:** "A New Framework for Quantifying Strategic Brittleness in the NBA."
> *   **The Executive Summary is a Hedge Fund Memo:** A single, dense page with a non-consensus question, novel methodology, and a surprising, actionable insight.
> *   **Intellectual Honesty is Paramount:** A sub-section titled "Where This Model Fails."
> *   **The "So What?" is Crystal Clear:** Each finding is tied to a specific, strategic business decision (a trade target, a draft strategy, a playoff matchup to exploit).

## Section 3: The New Thesis (SEI V2)

In response to this critique, the project has a new vision. We are no longer building a player valuation model. We are now the **Systemic Efficiencies Initiative (SEI)**.

**Our Mission:** To use public data to discover and quantify systemic, second-order properties of team strategy, coaching, and psychology. We build strategic risk and opportunity models.

Our work is now organized into three core research pods, detailed in `METHODOLOGY.md`:
1.  **Project "Jenga"**: Quantifying System Brittleness.
2.  **Project "Counter-Punch"**: Quantifying Coaching Adaptability.
3.  **Project "Cognitive Tax"**: Quantifying Mental Friction.

This is the new frontier for our project.
