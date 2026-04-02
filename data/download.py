"""Download and filter Amazon reviews from HuggingFace datasets.

Targets categories relevant to college tech entrepreneurs:
Electronics, Software, Office_Products, Computers.

Source: McAuley-Lab/Amazon-Reviews-2023
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
TARGET_CATEGORIES: list[str] = [
    "Electronics",
    "Software",
    "Office_Products",
    "Computers",
]

RAW_DIR = Path(__file__).resolve().parent / "raw"

# Fields to extract from each review record.
REVIEW_FIELDS: list[str] = [
    "rating",
    "text",
    "title",
    "asin",
    "parent_asin",
    "verified_purchase",
    "timestamp",
    "helpful_vote",
    "images",
]


def _output_path(category: str) -> Path:
    """Return the JSON-lines output path for a given category."""
    return RAW_DIR / f"{category}.jsonl"


def _is_downloaded(category: str) -> bool:
    """Check whether a category has already been downloaded."""
    path = _output_path(category)
    return path.exists() and path.stat().st_size > 0


def _passes_filter(review: dict[str, Any]) -> bool:
    """Return True if a review meets our quality criteria.

    Criteria:
      - Verified purchase
      - Review body longer than 50 characters
    """
    if not review.get("verified_purchase", False):
        return False
    text = review.get("text") or ""
    if len(text) <= 50:
        return False
    return True


def _extract_fields(review: dict[str, Any]) -> dict[str, Any]:
    """Pull only the fields we care about from a raw review record."""
    extracted: dict[str, Any] = {}
    for field in REVIEW_FIELDS:
        extracted[field] = review.get(field)
    return extracted


def _sort_key(review: dict[str, Any]) -> tuple[int, float]:
    """Sort reviews so those with helpful votes come first, then by rating."""
    helpful = review.get("helpful_vote") or 0
    rating = review.get("rating") or 0.0
    # Negate helpful so higher values sort first.
    return (-helpful, -rating)


def download_reviews(
    categories: list[str] | None = None,
    max_per_category: int = 5000,
) -> dict[str, Path]:
    """Download Amazon reviews for the specified categories.

    Already-downloaded categories are skipped automatically.

    Args:
        categories: List of Amazon category names. Defaults to
            ``TARGET_CATEGORIES``.
        max_per_category: Maximum number of filtered reviews to keep per
            category.

    Returns:
        Mapping of category name to the path of the saved JSONL file.
    """
    if categories is None:
        categories = list(TARGET_CATEGORIES)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for category in categories:
        out_path = _output_path(category)

        if _is_downloaded(category):
            console.print(
                f"[green]Skipping[/green] {category} — already downloaded "
                f"at {out_path}"
            )
            results[category] = out_path
            continue

        console.print(f"\n[bold cyan]Downloading[/bold cyan] {category} …")

        # The Amazon-Reviews-2023 dataset uses category-specific configs.
        # Each config is named "raw_review_{Category}".
        config_name = f"raw_review_{category}"
        try:
            ds = load_dataset(
                REPO_ID,
                config_name,
                split="full",
                trust_remote_code=True,
            )
        except Exception as exc:
            console.print(f"[red]Failed to load {category}:[/red] {exc}")
            continue

        total_rows = len(ds)
        console.print(f"  Loaded {total_rows:,} raw reviews. Filtering …")

        filtered: list[dict[str, Any]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Filtering {category}", total=total_rows
            )
            for review in ds:
                progress.advance(task)
                if _passes_filter(review):
                    filtered.append(_extract_fields(review))
                    # Early exit once we have enough candidates (with margin
                    # for sorting).
                    if len(filtered) >= max_per_category * 3:
                        break

        # Prefer reviews that received helpful votes.
        filtered.sort(key=_sort_key)
        filtered = filtered[:max_per_category]

        # Write to JSONL.
        with open(out_path, "w", encoding="utf-8") as fh:
            for record in filtered:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        console.print(
            f"  [green]Saved[/green] {len(filtered):,} reviews → {out_path}"
        )
        results[category] = out_path

    return results


def load_reviews(category: str) -> list[dict[str, Any]]:
    """Load previously downloaded reviews for a single category.

    Args:
        category: Amazon product category name.

    Returns:
        List of review dicts.

    Raises:
        FileNotFoundError: If the category has not been downloaded yet.
    """
    path = _output_path(category)
    if not path.exists():
        raise FileNotFoundError(
            f"No downloaded data for '{category}'. "
            f"Expected file at {path}. Run download_reviews() first."
        )

    reviews: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    return reviews


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    console.rule("[bold]Proxi AI — Amazon Review Downloader[/bold]")

    saved = download_reviews()

    console.rule("[bold]Summary[/bold]")
    for cat, path in saved.items():
        reviews = load_reviews(cat)
        console.print(f"  {cat:20s}  {len(reviews):>6,} reviews  ({path})")

    console.print("\n[bold green]Done.[/bold green]")
