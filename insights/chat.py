"""Chat interface over journey data.

Provides a conversational interface for asking questions about a
completed persona journey, with full conversation history support
for follow-up questions.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_CHAT_SYSTEM_TEMPLATE = """\
You are an expert UX research assistant. You have access to a detailed
journey log from a simulated user persona navigating a website.

## Persona Profile
{persona_json}

## Journey Log
{journey_json}

Answer the user's questions about this journey conversationally and
accurately. Reference specific steps, screenshots, or observations from
the journey log when relevant. If you don't have enough information to
answer a question, say so clearly.

Keep answers concise but thorough. Use bullet points for lists."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class JourneyChat:
    """Conversational interface for querying journey data.

    Maintains a message history so users can ask follow-up questions
    that reference prior answers.

    Parameters
    ----------
    journey_log:
        The full journey log dict from a completed agent run.
    persona:
        The persona dict that drove the journey.
    client:
        An initialized ``anthropic.Anthropic`` client.
    model:
        The Claude model to use for chat responses.
    """

    def __init__(
        self,
        journey_log: dict[str, Any],
        persona: dict[str, Any],
        client: anthropic.Anthropic,
        *,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.journey_log = journey_log
        self.persona = persona
        self.client = client
        self.model = model

        self._system_prompt = _CHAT_SYSTEM_TEMPLATE.format(
            persona_json=json.dumps(persona, indent=2),
            journey_json=json.dumps(journey_log, indent=2),
        )
        self._messages: list[dict[str, str]] = []

    def ask(self, question: str) -> str:
        """Ask a question about the journey and get a conversational answer.

        Parameters
        ----------
        question:
            The user's question about the journey.

        Returns
        -------
        str
            The assistant's response.
        """
        self._messages.append({"role": "user", "content": question})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self._system_prompt,
            messages=self._messages,
        )

        answer = response.content[0].text.strip()
        self._messages.append({"role": "assistant", "content": answer})

        return answer

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a copy of the conversation history."""
        return list(self._messages)

    def reset(self) -> None:
        """Clear conversation history to start fresh."""
        self._messages.clear()
