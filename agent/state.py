"""
PersonaState: tracks the evolving psychological and behavioral state
of a persona as it navigates a target website.
"""

from __future__ import annotations

import time
from typing import Any


# Friction types that are absolute deal-breakers regardless of patience level.
DEAL_BREAKER_FRICTIONS = frozenset({
    "paywall_no_trial",
    "mandatory_credit_card",
    "broken_page",
    "infinite_loop",
    "aggressive_popup_spam",
    "data_harvesting_warning",
})

# Weights for each goal category when computing conversion score.
DEFAULT_GOAL_WEIGHTS = {
    "pricing_clarity": 0.25,
    "feature_match": 0.30,
    "trust_signals": 0.15,
    "ease_of_use": 0.15,
    "value_proposition": 0.15,
}


class PersonaState:
    """Mutable state object carried through an entire persona journey."""

    def __init__(self, persona: dict) -> None:
        """
        Initialize state from a persona configuration dict.

        Expected persona keys:
            - persona_id: str
            - name: str
            - patience_baseline: float (0-1, how patient this persona is)
            - trust_baseline: float (0-1, initial trust disposition)
            - goals: list[str] (what the persona wants to accomplish)
            - deal_breakers: list[str] (friction types that cause immediate exit)
            - goal_weights: dict[str, float] (optional overrides)
        """
        self.persona = persona
        self.persona_id: str = persona.get("persona_id", "unknown")

        # Core psychological meters (0.0 = depleted, 1.0 = full)
        self.patience: float = float(persona.get("patience_baseline", 0.7))
        self.trust: float = float(persona.get("trust_baseline", 0.4))

        # Goal tracking: goal_name -> float score (0.0 to 1.0)
        self.goals_met: dict[str, float] = {
            goal: 0.0 for goal in persona.get("goals", [])
        }

        # Event logs
        self.friction_log: list[dict[str, Any]] = []
        self.positive_log: list[dict[str, Any]] = []

        # Navigation counters
        self.pages_visited: int = 0
        self.time_elapsed_seconds: float = 0.0
        self._journey_start: float = time.time()

        # Whether the persona is still willing to continue browsing
        self.will_continue: bool = True
        self._exit_reason: str | None = None

        # Deal-breakers: combine persona-specific with global defaults
        persona_breakers = set(persona.get("deal_breakers", []))
        self._deal_breakers: set[str] = DEAL_BREAKER_FRICTIONS | persona_breakers

        # Goal weights for conversion scoring
        self._goal_weights: dict[str, float] = {
            **DEFAULT_GOAL_WEIGHTS,
            **persona.get("goal_weights", {}),
        }

    # ------------------------------------------------------------------
    # State mutation methods
    # ------------------------------------------------------------------

    def encounter_friction(self, friction_type: str, severity: float) -> None:
        """
        Record a friction event. Patience decays multiplicatively so that
        repeated friction compounds (e.g., 0.8 * 0.8 * 0.8 drains fast).

        Args:
            friction_type: categorical label (e.g. "slow_load", "confusing_nav")
            severity: 0.0 (minor annoyance) to 1.0 (infuriating)
        """
        severity = max(0.0, min(1.0, severity))

        self.friction_log.append({
            "type": friction_type,
            "severity": severity,
            "patience_before": self.patience,
            "trust_before": self.trust,
            "page": self.pages_visited,
            "elapsed": self._elapsed(),
        })

        # Multiplicative patience decay: higher severity -> bigger multiplier
        decay_factor = 1.0 - (severity * 0.4)  # severity=1 -> 0.6x multiplier
        self.patience *= decay_factor

        # Trust also takes a hit, but smaller
        self.trust = max(0.0, self.trust - severity * 0.1)

        # Check deal-breakers
        if friction_type in self._deal_breakers:
            self.will_continue = False
            self._exit_reason = f"deal_breaker:{friction_type}"
            return

        # Check patience threshold
        if self.patience < 0.1:
            self.will_continue = False
            self._exit_reason = "patience_depleted"

    def encounter_positive(self, signal_type: str, strength: float) -> None:
        """
        Record a positive signal. Trust rebuilds slowly (additive but
        diminishing). Patience recovers only a tiny amount -- once annoyed,
        a real user doesn't fully reset.

        Args:
            signal_type: categorical label (e.g. "clear_pricing", "social_proof")
            strength: 0.0 (mildly nice) to 1.0 (compelling)
        """
        strength = max(0.0, min(1.0, strength))

        self.positive_log.append({
            "type": signal_type,
            "strength": strength,
            "trust_before": self.trust,
            "patience_before": self.patience,
            "page": self.pages_visited,
            "elapsed": self._elapsed(),
        })

        # Trust grows slowly -- diminishing returns as trust gets higher
        trust_room = 1.0 - self.trust
        self.trust = min(1.0, self.trust + strength * 0.15 * trust_room)

        # Patience recovers only minimally (people remember friction)
        self.patience = min(1.0, self.patience + strength * 0.03)

    def update_goal(self, goal_name: str, score: float) -> None:
        """Set or update the satisfaction score for a tracked goal."""
        if goal_name in self.goals_met:
            self.goals_met[goal_name] = max(0.0, min(1.0, score))

    def record_page_visit(self) -> None:
        """Increment page counter and update elapsed time."""
        self.pages_visited += 1
        self.time_elapsed_seconds = self._elapsed()

    # ------------------------------------------------------------------
    # Decision methods
    # ------------------------------------------------------------------

    def should_convert(self) -> tuple[bool, str]:
        """
        Final conversion decision based on a weighted goal-satisfaction
        score combined with trust level.

        Returns:
            (converted: bool, reason: str)
        """
        if not self.will_continue and self._exit_reason:
            return False, f"abandoned: {self._exit_reason}"

        # Compute weighted goal score
        total_weight = 0.0
        weighted_score = 0.0
        for goal, score in self.goals_met.items():
            weight = self._goal_weights.get(goal, 0.1)
            weighted_score += score * weight
            total_weight += weight

        goal_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Combined score: 60% goal satisfaction, 40% trust
        combined = goal_score * 0.6 + self.trust * 0.4

        if combined >= 0.65 and self.trust >= 0.5:
            return True, (
                f"converted: combined_score={combined:.2f} "
                f"(goal={goal_score:.2f}, trust={self.trust:.2f})"
            )

        # Build a reason for not converting
        reasons = []
        if goal_score < 0.5:
            unmet = [g for g, s in self.goals_met.items() if s < 0.5]
            reasons.append(f"unmet_goals={unmet}")
        if self.trust < 0.5:
            reasons.append(f"low_trust={self.trust:.2f}")
        if self.patience < 0.2:
            reasons.append(f"low_patience={self.patience:.2f}")

        reason_str = "did_not_convert: " + "; ".join(reasons) if reasons else "did_not_convert: insufficient_combined_score"
        return False, reason_str

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the full state to a JSON-compatible dict."""
        return {
            "persona_id": self.persona_id,
            "patience": round(self.patience, 4),
            "trust": round(self.trust, 4),
            "goals_met": {k: round(v, 4) for k, v in self.goals_met.items()},
            "friction_log": self.friction_log,
            "positive_log": self.positive_log,
            "pages_visited": self.pages_visited,
            "time_elapsed_seconds": round(self.time_elapsed_seconds, 2),
            "will_continue": self.will_continue,
            "exit_reason": self._exit_reason,
        }

    def get_summary(self) -> str:
        """Human-readable summary of the persona's current state."""
        converted, reason = self.should_convert()
        goal_lines = "\n".join(
            f"    - {goal}: {score:.0%}" for goal, score in self.goals_met.items()
        )
        return (
            f"Persona: {self.persona.get('name', self.persona_id)}\n"
            f"  Patience: {self.patience:.0%}  |  Trust: {self.trust:.0%}\n"
            f"  Pages visited: {self.pages_visited}  |  Time: {self.time_elapsed_seconds:.1f}s\n"
            f"  Will continue: {self.will_continue}\n"
            f"  Goals:\n{goal_lines}\n"
            f"  Friction events: {len(self.friction_log)}\n"
            f"  Positive signals: {len(self.positive_log)}\n"
            f"  Conversion: {'YES' if converted else 'NO'} -- {reason}\n"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        return time.time() - self._journey_start
