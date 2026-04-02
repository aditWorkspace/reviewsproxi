"""Cross-persona journey comparison.

Compares multiple persona runs on the same site to surface universal
friction, segment-specific issues, conversion path divergence,
missed opportunities, and a priority matrix.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_COMPARE_SYSTEM = """\
You are an expert UX researcher specializing in cross-segment analysis.
You are given journey logs from multiple simulated personas navigating the
same website, along with each persona's profile.

Produce a structured comparison in JSON with exactly these top-level keys:

1. "universal_friction" – list of dicts, each with "issue" (str),
   "description" (str), "affected_personas" (list of persona names).
   These are problems that ALL personas encountered.

2. "segment_specific_friction" – list of dicts, each with "issue" (str),
   "description" (str), "affected_segment" (str), "unaffected_segments"
   (list of str). These are problems unique to certain persona types.

3. "conversion_path_divergence" – list of dicts, each with
   "persona" (str), "path_taken" (str, brief description),
   "outcome" (str), "divergence_point" (str, where this persona's path
   differed from others).

4. "missed_opportunities" – list of dicts, each with
   "opportunity" (str), "description" (str), "potential_impact" (str).

5. "priority_matrix" – list of dicts, each with "issue" (str),
   "impact" (one of "high", "medium", "low"),
   "severity" (one of "critical", "high", "medium", "low"),
   "recommendation" (str), sorted by impact * severity descending.

Return ONLY valid JSON. No markdown fences or commentary."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_journeys(
    journey_logs: list[dict[str, Any]],
    personas: list[dict[str, Any]],
    client: anthropic.Anthropic,
    *,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Compare journey logs from multiple personas on the same site.

    Parameters
    ----------
    journey_logs:
        List of journey log dicts, one per persona run. Each should
        contain keys like ``"steps"``, ``"outcome"``, ``"target_url"``.
    personas:
        List of persona dicts aligned with *journey_logs* (same order
        and length). Each should have ``"name"``, ``"goals"``, etc.
    client:
        An initialized ``anthropic.Anthropic`` client.
    model:
        The Claude model to use for analysis.

    Returns
    -------
    dict
        Structured comparison with keys: ``universal_friction``,
        ``segment_specific_friction``, ``conversion_path_divergence``,
        ``missed_opportunities``, ``priority_matrix``.

    Raises
    ------
    ValueError
        If the number of journey logs and personas don't match.
    """
    if len(journey_logs) != len(personas):
        raise ValueError(
            f"Mismatch: {len(journey_logs)} journey logs vs "
            f"{len(personas)} personas."
        )

    # Build a combined payload with each persona + journey paired together.
    paired_entries: list[str] = []
    for i, (log, persona) in enumerate(zip(journey_logs, personas)):
        entry = (
            f"### Persona {i + 1}: {persona.get('name', f'Persona {i + 1}')}\n"
            f"**Profile:**\n{json.dumps(persona, indent=2)}\n\n"
            f"**Journey Log:**\n{json.dumps(log, indent=2)}"
        )
        paired_entries.append(entry)

    user_message = "\n\n---\n\n".join(paired_entries)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_COMPARE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if present.
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    try:
        comparison: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError:
        comparison = {
            "raw_response": raw_text,
            "parse_error": "Model response was not valid JSON.",
        }

    return comparison
