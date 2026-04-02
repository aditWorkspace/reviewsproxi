"""Persona validation utilities.

Checks that generated personas are specific, testable, and internally
consistent enough to drive realistic simulations.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Action verbs used to judge whether a behavioural rule is testable.
# ---------------------------------------------------------------------------

_ACTION_VERBS: set[str] = {
    "abandon", "accept", "ask", "avoid", "buy", "cancel", "check",
    "choose", "click", "close", "compare", "complain", "contact",
    "decline", "delay", "demand", "dismiss", "download", "drop",
    "email", "escalate", "evaluate", "exit", "expect", "explore",
    "file", "focus", "follow", "ignore", "increase", "install",
    "leave", "negotiate", "open", "opt", "pause", "pay", "prefer",
    "prioritize", "purchase", "quit", "read", "recommend", "reduce",
    "refuse", "reject", "rely", "remove", "renew", "replace",
    "report", "request", "require", "research", "respond", "return",
    "review", "schedule", "search", "select", "send", "share",
    "sign", "skip", "spend", "stop", "subscribe", "switch", "test",
    "try", "uninstall", "unsubscribe", "upgrade", "use", "verify",
    "visit", "wait", "walk", "warn", "write",
}

# Patterns that indicate a concrete, binary deal-breaker.
_BINARY_INDICATORS: list[str] = [
    "no ", "never", "must", "require", "cannot", "won't", "will not",
    "if ", "unless", "without", "lack of", "absence of", "zero",
    "any ", "always",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_persona_specificity(persona: dict) -> list[str]:
    """Validate that a persona is concrete and simulation-ready.

    Parameters
    ----------
    persona:
        A structured persona dict as produced by
        :func:`engine.persona_builder.synthesize_persona`.

    Returns
    -------
    list[str]
        A list of human-readable issue descriptions.  An empty list means
        the persona passed all checks.
    """
    issues: list[str] = []

    # ------------------------------------------------------------------
    # 1. Behavioural rules must be testable (action verb or number).
    # ------------------------------------------------------------------
    behavioral_rules: list[str] = persona.get("behavioral_rules", [])
    for i, rule in enumerate(behavioral_rules):
        rule_lower = rule.lower()
        words = set(re.findall(r"[a-z]+", rule_lower))
        has_action_verb = bool(words & _ACTION_VERBS)
        has_number = bool(re.search(r"\d", rule))
        if not has_action_verb and not has_number:
            issues.append(
                f"behavioral_rules[{i}] is not testable — missing action verb "
                f"or numeric threshold: {rule!r}"
            )

    # ------------------------------------------------------------------
    # 2. Decision-weight spread: max - min >= 0.15.
    # ------------------------------------------------------------------
    decision_weights: dict[str, float] = persona.get("decision_weights", {})
    if decision_weights:
        weights = list(decision_weights.values())
        weight_spread = max(weights) - min(weights)
        if weight_spread < 0.15:
            issues.append(
                f"decision_weights spread too narrow ({weight_spread:.2f}). "
                f"Max–min gap must be at least 0.15 to differentiate the persona."
            )

    # ------------------------------------------------------------------
    # 3. Deal-breakers must be concrete and binary.
    # ------------------------------------------------------------------
    deal_breakers: list[str] = persona.get("deal_breakers", [])

    unique_deal_breakers = set(db.strip().lower() for db in deal_breakers)
    if len(unique_deal_breakers) < 2:
        issues.append(
            f"Fewer than 2 unique deal_breakers ({len(unique_deal_breakers)} found). "
            f"Personas need at least 2 distinct deal-breakers."
        )

    for i, db in enumerate(deal_breakers):
        db_lower = db.lower()
        is_binary = any(indicator in db_lower for indicator in _BINARY_INDICATORS)
        if not is_binary:
            issues.append(
                f"deal_breakers[{i}] does not appear concrete/binary — "
                f"missing indicator words (e.g. 'must', 'no', 'if'): {db!r}"
            )

    # ------------------------------------------------------------------
    # 4. Voice sample must contain specific details.
    # ------------------------------------------------------------------
    voice_sample: str = persona.get("voice_sample", "")
    if not voice_sample:
        issues.append("voice_sample is empty.")
    else:
        has_number = bool(re.search(r"\d", voice_sample))
        has_proper_noun = bool(re.search(r"\b[A-Z][a-z]{2,}", voice_sample))
        # Specific details proxy: contains at least a number or a proper noun.
        if not has_number and not has_proper_noun:
            issues.append(
                "voice_sample lacks specific details — should include numbers, "
                "product names, or concrete scenarios."
            )

    return issues
