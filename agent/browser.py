"""
Playwright browser wrapper for Proxi AI agent navigation.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger(__name__)


class BrowserManager:
    """
    Async context manager wrapping Playwright for headless or headed
    browser automation with optional video recording.

    Usage::

        async with BrowserManager() as bm:
            page = await bm.launch(headless=True, video_dir="./videos")
            await page.goto("https://example.com")
            ctx = await bm.get_page_context(page)
            await bm.take_screenshot(page, "shot.png")
    """

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BrowserManager":
        self._playwright = await async_playwright().start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def launch(
        self,
        headless: bool = True,
        video_dir: str | None = None,
    ) -> Page:
        """
        Launch a Chromium browser, create a context (with optional video
        recording), and return a fresh page.
        """
        if self._playwright is None:
            raise RuntimeError("BrowserManager must be used as an async context manager")

        self._browser = await self._playwright.chromium.launch(headless=headless)

        context_kwargs: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 900},
            "user_agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        }
        if video_dir:
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            context_kwargs["record_video_dir"] = video_dir
            context_kwargs["record_video_size"] = {"width": 1280, "height": 900}

        self._context = await self._browser.new_context(**context_kwargs)
        self._page = await self._context.new_page()
        return self._page

    async def close(self) -> None:
        """Tear down context, browser, and playwright in order."""
        try:
            if self._context:
                await self._context.close()
                self._context = None
        except Exception:
            logger.debug("Error closing browser context", exc_info=True)

        try:
            if self._browser:
                await self._browser.close()
                self._browser = None
        except Exception:
            logger.debug("Error closing browser", exc_info=True)

        try:
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
        except Exception:
            logger.debug("Error stopping playwright", exc_info=True)

    # ------------------------------------------------------------------
    # Page introspection
    # ------------------------------------------------------------------

    async def get_page_context(self, page: Page) -> dict[str, Any]:
        """
        Extract a structured snapshot of the current page suitable for
        feeding to the LLM agent.

        Returns dict with:
            title, url, headings, body_preview (first 2000 chars),
            interactive_elements (up to 30 visible a/button/input),
            has_pricing, has_signup, has_testimonials
        """
        title = await page.title()
        url = page.url

        # Headings
        headings: list[str] = await page.evaluate("""
            () => {
                const hs = document.querySelectorAll('h1, h2, h3');
                return Array.from(hs).slice(0, 20).map(h =>
                    `${h.tagName}: ${h.innerText.trim().substring(0, 120)}`
                );
            }
        """)

        # Body text preview
        body_preview: str = await page.evaluate("""
            () => {
                const body = document.body;
                if (!body) return '';
                return body.innerText.substring(0, 2000);
            }
        """)

        # Interactive elements (visible links, buttons, inputs -- up to 30)
        interactive_elements: list[dict[str, str]] = await page.evaluate("""
            () => {
                const elems = document.querySelectorAll('a, button, input, [role="button"], [role="link"]');
                const results = [];
                for (const el of elems) {
                    if (results.length >= 30) break;
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 && rect.height === 0) continue;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;

                    const tag = el.tagName.toLowerCase();
                    let text = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '').trim().substring(0, 80);
                    const href = el.getAttribute('href') || '';
                    const type = el.getAttribute('type') || '';
                    const name = el.getAttribute('name') || '';

                    results.push({tag, text, href, type, name});
                }
                return results;
            }
        """)

        # Boolean page signals
        body_lower = body_preview.lower()
        has_pricing = await page.evaluate("""
            () => {
                const text = document.body ? document.body.innerText.toLowerCase() : '';
                return /\\$\\d|price|pricing|per month|\\/mo|free tier|free plan|cost/i.test(text);
            }
        """)
        has_signup = await page.evaluate("""
            () => {
                const text = document.body ? document.body.innerHTML.toLowerCase() : '';
                return /sign.?up|get started|create account|start free|register|join now/i.test(text);
            }
        """)
        has_testimonials = await page.evaluate("""
            () => {
                const text = document.body ? document.body.innerText.toLowerCase() : '';
                return /testimonial|customer.?stor|review|".*said|trusted by|used by \\d/i.test(text);
            }
        """)

        return {
            "title": title,
            "url": url,
            "headings": headings,
            "body_preview": body_preview,
            "interactive_elements": interactive_elements,
            "has_pricing": bool(has_pricing),
            "has_signup": bool(has_signup),
            "has_testimonials": bool(has_testimonials),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def take_screenshot(self, page: Page, path: str) -> None:
        """Save a full-page screenshot to the given path."""
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=path, full_page=True)
