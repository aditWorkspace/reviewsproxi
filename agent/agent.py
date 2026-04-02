"""
PersonaAgent: drives a single persona through a website, producing a
structured journey log with full observation/decision/action traces.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from agent.browser import BrowserManager
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.state import PersonaState
from engine.knowledge import get_full_persona_context

logger = logging.getLogger(__name__)

# Default configuration knobs
_DEFAULTS = {
    "max_steps": 25,
    "human_delay_seconds": 1.5,
    "screenshot": True,
    "video": True,
    "timeout_ms": 10_000,
}


class PersonaAgent:
    """
    Runs one persona journey end-to-end: launch browser, navigate the
    target site, observe pages, ask the LLM for decisions, execute
    actions, and collect a full journey log.
    """

    def __init__(
        self,
        persona: dict,
        target_url: str,
        anthropic_client: Any,
        config: dict | None = None,
    ) -> None:
        # Load full context: base config + trained knowledge
        persona_id = persona.get("id", "")
        try:
            full_context = get_full_persona_context(persona_id)
            self.persona = full_context["config"]
            self.trained_knowledge = full_context.get("trained_knowledge")
            self.is_trained = full_context.get("is_trained", False)
        except (FileNotFoundError, Exception):
            # Fallback to raw persona dict if knowledge store not set up
            self.persona = persona
            self.trained_knowledge = None
            self.is_trained = False

        self.target_url = target_url
        self.client = anthropic_client

        cfg = {**_DEFAULTS, **(config or {})}
        self.max_steps: int = cfg["max_steps"]
        self.human_delay: float = cfg["human_delay_seconds"]
        self.take_screenshots: bool = cfg["screenshot"]
        self.record_video: bool = cfg["video"]
        self.timeout_ms: int = cfg.get("timeout_ms", 10_000)

        self.state = PersonaState(self.persona)
        self.run_id: str = str(uuid.uuid4())
        self.steps: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> dict:
        """
        Execute the full persona journey. Returns the complete journey
        log dict.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        video_dir = f"runs/{self.run_id}/video" if self.record_video else None
        screenshot_dir = f"runs/{self.run_id}/screenshots" if self.take_screenshots else None

        async with BrowserManager() as bm:
            page = await bm.launch(headless=True, video_dir=video_dir)

            # Initial navigation
            try:
                await page.goto(self.target_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            except Exception as e:
                logger.error("Failed to load target URL %s: %s", self.target_url, e)
                return self._build_journey_log(
                    started_at=started_at,
                    outcome="error",
                    outcome_reason=f"initial_load_failed: {e}",
                )

            self.state.record_page_visit()

            for step_num in range(1, self.max_steps + 1):
                if not self.state.will_continue:
                    break

                # --- Observe ---
                try:
                    page_context = await bm.get_page_context(page)
                except Exception as e:
                    logger.warning("Failed to get page context at step %d: %s", step_num, e)
                    page_context = {
                        "title": "",
                        "url": page.url,
                        "headings": [],
                        "body_preview": "",
                        "interactive_elements": [],
                        "has_pricing": False,
                        "has_signup": False,
                        "has_testimonials": False,
                    }

                # Screenshot
                if self.take_screenshots and screenshot_dir:
                    try:
                        await bm.take_screenshot(
                            page,
                            f"{screenshot_dir}/step_{step_num:03d}.png",
                        )
                    except Exception:
                        logger.debug("Screenshot failed at step %d", step_num, exc_info=True)

                # --- Decide ---
                decision = await self._decide(page_context, step_num)

                # --- Record step ---
                step_record = {
                    "step": step_num,
                    "url": page_context["url"],
                    "observation": {
                        "title": page_context["title"],
                        "headings_count": len(page_context["headings"]),
                        "interactive_count": len(page_context["interactive_elements"]),
                        "has_pricing": page_context["has_pricing"],
                        "has_signup": page_context["has_signup"],
                        "has_testimonials": page_context["has_testimonials"],
                    },
                    "inner_monologue": decision.get("inner_monologue", ""),
                    "goal_relevance": decision.get("goal_relevance", {}),
                    "emotional_state": decision.get("emotional_state", "neutral"),
                    "action": decision.get("action", {"type": "leave", "reason": "decision_parse_failure"}),
                    "action_reasoning": decision.get("action_reasoning", ""),
                }
                self.steps.append(step_record)

                # --- Process friction / positive events ---
                for fe in decision.get("friction_events", []):
                    self.state.encounter_friction(
                        fe.get("type", "unknown"),
                        fe.get("severity", 0.3),
                    )

                for pe in decision.get("positive_events", []):
                    self.state.encounter_positive(
                        pe.get("type", "unknown"),
                        pe.get("strength", 0.3),
                    )

                # Update goal scores from relevance ratings
                for goal_name, rating in decision.get("goal_relevance", {}).items():
                    if isinstance(rating, dict):
                        score = rating.get("score", 0.0)
                    else:
                        score = float(rating)
                    # Keep the maximum score seen for each goal
                    current = self.state.goals_met.get(goal_name, 0.0)
                    if score > current:
                        self.state.update_goal(goal_name, score)

                # --- Act ---
                action = decision.get("action", {})
                action_type = action.get("type", "leave")

                if action_type == "leave":
                    self.state.will_continue = False
                    self.state._exit_reason = action.get("reason", "persona_chose_to_leave")
                    break

                if action_type == "convert":
                    self.state.will_continue = False
                    break

                try:
                    await self._execute_action(page, action)
                except Exception as e:
                    logger.warning("Action failed at step %d: %s", step_num, e)
                    self.state.encounter_friction("broken_element", 0.4)

                self.state.record_page_visit()

                # Human-like delay
                await asyncio.sleep(self.human_delay)

        # --- Final outcome ---
        last_action = self.steps[-1]["action"] if self.steps else {}
        if last_action.get("type") == "convert":
            outcome = "converted"
            outcome_reason = last_action.get("reason", "persona_chose_to_convert")
        else:
            converted, reason = self.state.should_convert()
            outcome = "converted" if converted else "abandoned"
            outcome_reason = reason

        return self._build_journey_log(
            started_at=started_at,
            outcome=outcome,
            outcome_reason=outcome_reason,
        )

    # ------------------------------------------------------------------
    # LLM decision
    # ------------------------------------------------------------------

    async def _decide(self, page_context: dict[str, Any], step: int) -> dict:
        """
        Build the prompt from current state + page context, call the LLM,
        and return a parsed decision dict.
        """
        # Format interactive elements for the prompt
        ie_lines = []
        for i, el in enumerate(page_context.get("interactive_elements", []), 1):
            tag = el.get("tag", "?")
            text = el.get("text", "")
            href = el.get("href", "")
            desc = f"[{i}] <{tag}> {text}"
            if href:
                desc += f"  (href={href})"
            ie_lines.append(desc)

        goals_summary = ", ".join(
            f"{g}: {s:.0%}" for g, s in self.state.goals_met.items()
        )

        prompt = AGENT_SYSTEM_PROMPT.format(
            persona_description=self.persona.get("background", ""),
            persona_goals="\n".join(f"- {g}" for g in self.persona.get("goals", [])),
            persona_deal_breakers="\n".join(f"- {d}" for d in self.persona.get("deal_breakers", [])),
            patience_pct=f"{self.state.patience * 100:.0f}",
            trust_pct=f"{self.state.trust * 100:.0f}",
            pages_visited=self.state.pages_visited,
            time_elapsed=f"{self.state.time_elapsed_seconds:.1f}",
            goals_met_summary=goals_summary or "none yet",
            step_number=step,
            max_steps=self.max_steps,
            current_url=page_context.get("url", ""),
            page_title=page_context.get("title", ""),
            body_preview=page_context.get("body_preview", "")[:2000],
            headings="\n".join(page_context.get("headings", [])) or "(none)",
            interactive_elements="\n".join(ie_lines) or "(none)",
            has_pricing=page_context.get("has_pricing", False),
            has_signup=page_context.get("has_signup", False),
            has_testimonials=page_context.get("has_testimonials", False),
        )

        # Inject trained knowledge as system context if available
        system_context = ""
        if self.trained_knowledge:
            knowledge_summary = json.dumps({
                k: v for k, v in self.trained_knowledge.items()
                if k in (
                    "core_pain_points", "purchase_triggers", "objections",
                    "deal_breakers_confirmed", "behavioral_patterns",
                    "price_sensitivity_details", "friction_patterns",
                    "competitive_context", "emotional_vocabulary",
                )
            }, indent=2)
            system_context = (
                "\n\n## TRAINED KNOWLEDGE (learned from real review data)\n"
                "This persona has been trained on real user reviews. Use this knowledge "
                "to make more realistic and specific decisions:\n"
                f"{knowledge_summary}"
            )

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.persona.get("_model", "claude-sonnet-4-6"),
                max_tokens=1500,
                system=system_context if system_context else None,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
        except Exception as e:
            logger.error("LLM call failed at step %d: %s", step, e)
            return {
                "inner_monologue": f"[LLM error: {e}]",
                "emotional_state": "confused",
                "friction_events": [],
                "positive_events": [],
                "goal_relevance": {},
                "action": {"type": "leave", "reason": f"llm_error: {e}"},
                "action_reasoning": "LLM call failed, ending journey.",
            }

        return self._parse_decision(raw_text)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_decision(response: str) -> dict:
        """
        Robustly parse the LLM JSON response. Handles markdown fences,
        trailing commas, and partial responses.
        """
        text = response.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove opening fence (with optional language tag)
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            candidate = match.group(0)
            # Remove trailing commas before closing braces/brackets
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Last resort: return a leave action
        logger.warning("Could not parse LLM decision, returning leave action")
        return {
            "inner_monologue": f"[Parse failure] Raw: {text[:200]}",
            "emotional_state": "confused",
            "friction_events": [],
            "positive_events": [],
            "goal_relevance": {},
            "action": {"type": "leave", "reason": "response_parse_failure"},
            "action_reasoning": "Could not parse LLM response.",
        }

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def _execute_action(self, page: Any, action: dict) -> None:
        """Execute a browser action based on the LLM decision."""
        action_type = action.get("type", "")
        timeout = self.timeout_ms

        if action_type == "click":
            element_ref = action.get("element", "")
            # Try text-based selector first, then CSS selector
            try:
                await page.get_by_text(element_ref, exact=False).first.click(timeout=timeout)
            except Exception:
                try:
                    await page.click(element_ref, timeout=timeout)
                except Exception:
                    # Try as a link by partial text
                    await page.locator(f"a:has-text('{element_ref}'), button:has-text('{element_ref}')").first.click(timeout=timeout)

        elif action_type == "scroll":
            direction = action.get("direction", "down")
            delta = 600 if direction == "down" else -600
            await page.mouse.wheel(0, delta)
            await asyncio.sleep(0.5)

        elif action_type == "type":
            element_ref = action.get("element", "input")
            text = action.get("text", "")
            try:
                await page.fill(element_ref, text, timeout=timeout)
            except Exception:
                # Try focusing and typing character by character
                await page.click(element_ref, timeout=timeout)
                await page.keyboard.type(text, delay=50)

        elif action_type == "navigate":
            url = action.get("url", "")
            if url:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout)

        elif action_type == "wait":
            seconds = min(action.get("seconds", 2), 5)
            await asyncio.sleep(seconds)

        # "leave" and "convert" are handled in the main loop, not here

    # ------------------------------------------------------------------
    # Journey log construction
    # ------------------------------------------------------------------

    def _build_journey_log(
        self,
        started_at: str,
        outcome: str,
        outcome_reason: str,
    ) -> dict:
        """Assemble the complete journey log."""
        return {
            "run_id": self.run_id,
            "persona_id": self.persona.get("persona_id", "unknown"),
            "persona_name": self.persona.get("name", "unknown"),
            "target_url": self.target_url,
            "timestamp": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "max_steps": self.max_steps,
                "human_delay_seconds": self.human_delay,
            },
            "steps": self.steps,
            "friction_events": self.state.friction_log,
            "positive_events": self.state.positive_log,
            "outcome": outcome,
            "outcome_reason": outcome_reason,
            "final_state": self.state.to_dict(),
            "summary": self.state.get_summary(),
        }
