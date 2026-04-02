"""Post-run insight generation.

Analyzes a single persona journey to produce root cause analysis,
conversion blockers, quick wins, competitive vulnerabilities,
and persona-specific insights.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_INSIGHT_SYSTEM = """\
You are an expert UX researcher and conversion rate optimization analyst.
You are given a detailed journey log from a simulated user persona navigating
a website, along with the persona profile that drove the simulation.

Produce a structured analysis in JSON with exactly these top-level keys:

1. "root_cause_analysis" – list of dicts, each with "friction_point" (str),
   "root_cause" (str), "evidence" (str from the journey log).
2. "conversion_blockers" – list of dicts, each with "blocker" (str),
   "severity" (one of "critical", "high", "medium", "low"),
   "description" (str), ranked from most to least severe.
3. "quick_wins" – list of dicts, each with "fix" (str),
   "effort" (str, e.g. "< 1 hour", "< 4 hours", "< 1 day"),
   "expected_impact" (str).
4. "competitive_vulnerabilities" – list of dicts, each with
   "vulnerability" (str), "risk" (str), "recommendation" (str).
5. "persona_specific_insights" – list of dicts, each with
   "insight" (str), "relevance_to_persona" (str).

Return ONLY valid JSON. No markdown fences or commentary."""

_SUMMARY_SYSTEM = """\
You are an expert UX researcher. Given a journey log from a simulated user
navigating a website, produce a concise executive summary (3-5 sentences).
Focus on: what the persona tried to accomplish, the main friction points
encountered, whether the journey ended in success or failure, and the single
most impactful improvement the site could make.

Return ONLY the summary text, no JSON or markdown."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_insights(
    journey_log: dict[str, Any],
    persona: dict[str, Any],
    client: anthropic.Anthropic,
    *,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Analyze a single journey run and return structured insights.

    Parameters
    ----------
    journey_log:
        The full journey log dict from a completed agent run, expected to
        contain keys like ``"steps"``, ``"outcome"``, ``"target_url"``, etc.
    persona:
        The persona dict that drove the journey, with keys like ``"name"``,
        ``"goals"``, ``"frustrations"``, ``"tech_savviness"``, etc.
    client:
        An initialized ``anthropic.Anthropic`` client.
    model:
        The Claude model to use for analysis.

    Returns
    -------
    dict
        Structured insights with keys: ``root_cause_analysis``,
        ``conversion_blockers``, ``quick_wins``,
        ``competitive_vulnerabilities``, ``persona_specific_insights``.
    """
    user_message = (
        f"## Persona\n{json.dumps(persona, indent=2)}\n\n"
        f"## Journey Log\n{json.dumps(journey_log, indent=2)}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_INSIGHT_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if the model included them.
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    try:
        insights: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError:
        insights = {
            "raw_response": raw_text,
            "parse_error": "Model response was not valid JSON.",
        }

    return insights


def generate_summary(
    journey_log: dict[str, Any],
    client: anthropic.Anthropic,
    *,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Produce a short executive summary of a journey run.

    Parameters
    ----------
    journey_log:
        The full journey log dict from a completed agent run.
    client:
        An initialized ``anthropic.Anthropic`` client.
    model:
        The Claude model to use.

    Returns
    -------
    str
        A 3-5 sentence executive summary.
    """
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SUMMARY_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"## Journey Log\n{json.dumps(journey_log, indent=2)}",
            }
        ],
    )

    return response.content[0].text.strip()
