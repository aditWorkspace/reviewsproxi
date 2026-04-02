#!/usr/bin/env python3
"""
Proxi AI — CLI Entry Point

Commands:
    train       Train a persona on a CSV of reviews
    run         Run a persona agent against a URL
    analyze     Generate insights from a run
    compare     Compare multiple runs
    personas    List all personas and training status
    serve       Launch the Streamlit dashboard
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli():
    """Proxi AI — Synthetic User Persona Engine"""
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--persona", "-p", required=True, help="Persona ID to train")
@click.option("--source-name", "-s", default=None, help="Name for this training source")
def train(csv_path: str, persona: str, source_name: str | None):
    """Train a persona on a CSV of reviews."""
    import anthropic
    from engine.knowledge import train_persona_on_reviews, parse_csv_reviews

    console.print(f"[bold]Loading reviews from {csv_path}...[/bold]")

    with open(csv_path) as f:
        csv_content = f.read()

    reviews = parse_csv_reviews(csv_content)
    console.print(f"[green]Parsed {len(reviews)} valid reviews[/green]")

    if not reviews:
        console.print("[red]No valid reviews found in CSV. Check column names.[/red]")
        sys.exit(1)

    client = anthropic.Anthropic()
    source = source_name or Path(csv_path).name

    def progress(msg: str, pct: float):
        console.print(f"  [{pct:.0%}] {msg}")

    console.print(f"\n[bold]Training persona '{persona}'...[/bold]\n")
    knowledge = train_persona_on_reviews(
        persona_id=persona,
        reviews=reviews,
        source_name=source,
        client=client,
        progress_callback=progress,
    )

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Pain points learned: {len(knowledge.get('core_pain_points', []))}")
    console.print(f"  Behavioral patterns: {len(knowledge.get('behavioral_patterns', []))}")
    console.print(f"  Deal breakers confirmed: {len(knowledge.get('deal_breakers_confirmed', []))}")
    console.print(f"  Total reviews trained on: {knowledge.get('total_reviews_trained_on', '?')}")


@cli.command()
@click.argument("url")
@click.option("--persona", "-p", required=True, help="Persona ID to use")
@click.option("--max-steps", "-n", default=25, help="Maximum steps")
@click.option("--no-video", is_flag=True, help="Disable video recording")
@click.option("--headed", is_flag=True, help="Show browser window")
def run(url: str, persona: str, max_steps: int, no_video: bool, headed: bool):
    """Run a persona agent against a target URL."""
    import anthropic
    from engine.knowledge import load_persona_config
    from agent.agent import PersonaAgent
    from storage.db import init_db, save_run

    console.print(f"[bold]Loading persona '{persona}'...[/bold]")
    try:
        persona_config = load_persona_config(persona)
    except FileNotFoundError:
        console.print(f"[red]Persona '{persona}' not found[/red]")
        sys.exit(1)

    client = anthropic.Anthropic()
    config = {
        "max_steps": max_steps,
        "headless": not headed,
        "video": not no_video,
        "screenshot": True,
    }

    console.print(f"[bold]Launching {persona_config['label']} against {url}...[/bold]\n")

    agent = PersonaAgent(
        persona=persona_config,
        target_url=url,
        anthropic_client=client,
        config=config,
    )

    journey = asyncio.run(agent.run())

    # Save
    db = init_db()
    save_run(
        db,
        run_id=journey["run_id"],
        persona_id=persona,
        target_url=url,
        outcome=journey.get("outcome", "unknown"),
        outcome_reason=journey.get("outcome_reason", ""),
        journey_json=json.dumps(journey, default=str),
        summary=journey.get("outcome_reason", ""),
    )

    # Save to disk
    run_dir = Path("runs") / journey["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "journey.json", "w") as f:
        json.dump(journey, f, indent=2, default=str)

    # Print results
    outcome = journey.get("outcome", "unknown")
    color = "green" if outcome == "converted" else "red"
    console.print(f"\n[bold {color}]Outcome: {outcome}[/bold {color}]")
    console.print(f"Reason: {journey.get('outcome_reason', 'N/A')}")
    console.print(f"Steps: {len(journey.get('steps', []))}")
    console.print(f"Run ID: {journey['run_id']}")
    console.print(f"Journey saved to: runs/{journey['run_id']}/journey.json")

    # Print step summary
    console.print("\n[bold]Journey Summary:[/bold]")
    for step in journey.get("steps", []):
        action = step.get("action", {})
        monologue = step.get("inner_monologue", "")[:80]
        console.print(
            f"  Step {step.get('step', '?'):2d}: [{action.get('type', '?'):8s}] {monologue}..."
        )


@cli.command()
@click.argument("run_id")
def analyze(run_id: str):
    """Generate insights from a specific run."""
    import anthropic
    from storage.db import init_db, get_run
    from engine.knowledge import load_persona_config
    from insights.analyze import generate_insights

    db = init_db()
    run_data = get_run(db, run_id)
    if not run_data:
        console.print(f"[red]Run '{run_id}' not found[/red]")
        sys.exit(1)

    journey = json.loads(run_data["journey_json"]) if isinstance(run_data.get("journey_json"), str) else {}
    persona_id = run_data.get("persona_id", "")

    try:
        persona = load_persona_config(persona_id)
    except FileNotFoundError:
        persona = {}

    client = anthropic.Anthropic()
    console.print("[bold]Generating insights...[/bold]")
    insights = generate_insights(journey, persona, client)

    console.print_json(json.dumps(insights, indent=2, default=str))

    # Save insights
    from storage.db import save_insights
    save_insights(db, run_id, json.dumps(insights, default=str))
    console.print(f"\n[green]Insights saved for run {run_id}[/green]")


@cli.command()
def personas():
    """List all personas and their training status."""
    from engine.knowledge import list_all_personas

    all_personas = list_all_personas()

    if not all_personas:
        console.print("[yellow]No personas found. Add JSON files to data/personas/[/yellow]")
        return

    table = Table(title="Proxi AI — Personas")
    table.add_column("ID", style="cyan")
    table.add_column("Label", style="bold")
    table.add_column("Role")
    table.add_column("Trained", justify="center")
    table.add_column("Sessions", justify="right")
    table.add_column("Reviews", justify="right")

    for p in all_personas:
        config = p["config"]
        trained = "✅" if p["is_trained"] else "—"
        table.add_row(
            config.get("id", "?"),
            config.get("label", "?"),
            config.get("segment", {}).get("role", "?"),
            trained,
            str(p["training_sessions"]),
            str(p["total_reviews_trained"]),
        )

    console.print(table)


@cli.command()
@click.option("--port", "-p", default=8501, help="Port for Streamlit")
def serve(port: int):
    """Launch the Streamlit dashboard."""
    import subprocess
    console.print(f"[bold]Starting Proxi AI dashboard on port {port}...[/bold]")
    subprocess.run(["streamlit", "run", "app.py", "--server.port", str(port)])


if __name__ == "__main__":
    cli()
