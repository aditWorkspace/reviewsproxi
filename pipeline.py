#!/usr/bin/env python3
"""Proxi AI — Persona Training Pipeline CLI

Commands:
    collect     Stage 1: collect reviews (HuggingFace + Playwright)
    extract     Stage 2: extract behavioral signals
    cluster     Stage 3: embed and cluster signals
    export      Stage 4: synthesize persona, write JSON + Markdown
    run-all     Run all 4 stages in sequence
"""
from __future__ import annotations

import asyncio

import click
import yaml
from rich.console import Console

console = Console()


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """Proxi AI — Persona Training Pipeline"""


@cli.command()
@click.option("--force", is_flag=True, help="Re-collect even if reviews.jsonl exists")
@click.option("--hf-only", is_flag=True, help="HuggingFace only, skip Playwright")
@click.option("--scrape-only", is_flag=True, help="Playwright only, skip HuggingFace")
def collect(force: bool, hf_only: bool, scrape_only: bool) -> None:
    """Stage 1: Collect reviews from HuggingFace + Playwright."""
    from pathlib import Path
    from pipeline.collect import collect_huggingface, load_existing_hashes, scrape_amazon

    reviews_path = Path("data/raw/reviews.jsonl")
    if reviews_path.exists() and not force:
        console.print("[yellow]reviews.jsonl already exists. Use --force to re-collect.[/yellow]")
        return

    config = _load_config()
    existing_hashes = load_existing_hashes()

    if not scrape_only:
        console.print("[bold]Pulling from HuggingFace...[/bold]")
        n = collect_huggingface(config, existing_hashes)
        console.print(f"[green]HuggingFace: {n} reviews collected[/green]")

    if not hf_only:
        console.print("[bold]Running Playwright scraper...[/bold]")
        n = asyncio.run(scrape_amazon(config, existing_hashes))
        console.print(f"[green]Playwright: {n} reviews scraped[/green]")


@cli.command()
@click.option("--force", is_flag=True, help="Re-extract even if signals.jsonl exists")
@click.option("--batch-size", default=30, show_default=True, help="Reviews per extraction batch")
def extract(force: bool, batch_size: int) -> None:
    """Stage 2: Extract behavioral signals from reviews."""
    from pipeline.extract import run_extract
    console.print("[bold]Extracting signals...[/bold]")
    n = run_extract(batch_size=batch_size, force=force)
    console.print(f"[green]Processed {n} new batches[/green]")


@cli.command()
@click.option("--force", is_flag=True, help="Re-cluster even if clusters.json exists")
@click.option("--n-clusters", default=None, type=int, help="Override auto-optimization")
def cluster(force: bool, n_clusters: int | None) -> None:
    """Stage 3: Embed signals and auto-optimize clusters."""
    from pipeline.cluster import run_cluster
    config = _load_config()
    console.print("[bold]Clustering signals...[/bold]")
    result = run_cluster(config, n_clusters_override=n_clusters, force=force)
    chosen = result["chosen_n_clusters"]
    console.print(f"[green]Clustered into {chosen} groups[/green]")


@cli.command()
@click.option("--force", is_flag=True, help="Re-export even if persona.json exists")
@click.option("--persona-id", default="college_student", show_default=True, help="Output persona ID")
def export(force: bool, persona_id: str) -> None:
    """Stage 4: Synthesize persona and export JSON + Markdown."""
    from pipeline.export import run_export
    console.print("[bold]Synthesizing persona...[/bold]")
    run_export(persona_id=persona_id, force=force)
    console.print(f"[green]Persona exported to data/personas/{persona_id}/[/green]")


@cli.command("run-all")
@click.option("--force", is_flag=True, help="Force re-run all stages")
@click.option("--persona-id", default="college_student", show_default=True)
@click.pass_context
def run_all(ctx: click.Context, force: bool, persona_id: str) -> None:
    """Run all 4 pipeline stages in sequence."""
    ctx.invoke(collect, force=force, hf_only=False, scrape_only=False)
    ctx.invoke(extract, force=force, batch_size=30)
    ctx.invoke(cluster, force=force, n_clusters=None)
    ctx.invoke(export, force=force, persona_id=persona_id)


if __name__ == "__main__":
    cli()
