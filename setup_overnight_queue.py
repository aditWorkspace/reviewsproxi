#!/usr/bin/env python3
"""
Overnight batch queue setup.

Jobs are structured as:
  project_name = company name   (e.g. "Disgo")
  label        = demographic title — first line of the description
  description  = full demographic paragraph

Output files will be named:
  {company_slug}_{demographic_label_slug}_persona.json
  {company_slug}_{demographic_label_slug}_persona.md
  {company_slug}_{demographic_label_slug}_rag_index.jsonl

Usage:
    python setup_overnight_queue.py          # preview + confirm
    python setup_overnight_queue.py --yes    # skip confirmation

After confirming, start the worker:
    caffeinate -i python worker.py
  or via Streamlit UI → ⚡ Batch Queue → ▶ Start Worker
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)

from dotenv import load_dotenv
load_dotenv(".env")

from data.queue_manager import add_job, read_queue, is_worker_alive

# ---------------------------------------------------------------------------
# Job definitions — (project_name, label, description)
# ---------------------------------------------------------------------------

JOBS: list[tuple[str, str, str]] = [

    # ── Disgo ────────────────────────────────────────────────────────────────
    (
        "Disgo",
        "Consumer App User (Hospitality / Food Discovery Context)",
        "A user of a consumer app focused on food and bar recommendations, interacting "
        "with features related to check-ins and personalized suggestions. Provides "
        "feedback through surveys (≈50 users) on feature design, naming, and functionality. "
        "Engages with the product by comparing it to platforms like TikTok or Yelp.",
    ),
    (
        "Disgo",
        "Active User Providing Organic Feedback",
        "Uses the app and gives feedback through messages or conversations rather than "
        "structured systems. Interacts with features like check-ins and content posting, "
        "sharing issues encountered during real usage (e.g., posting while dining). "
        "Feedback is collected informally through ongoing user interaction.",
    ),
    (
        "Disgo",
        "Internal Team Member (Startup Team of ~3)",
        "Part of a small founding team building the product and actively using it "
        "themselves. Participates in feature decisions based on internal discussions and "
        "personal usage of the app. Combines internal preferences with external user "
        "feedback to guide product development.",
    ),

    # ── EyeCanKnow ───────────────────────────────────────────────────────────
    (
        "EyeCanKnow",
        "Direct Outreach User (Multi-Channel Contact)",
        "A user contacted via phone, email, LinkedIn, Trustpilot, or Google reviews. "
        "Provides feedback inconsistently depending on response rates. Interaction is "
        "dependent on engagement and often limited.",
    ),
    (
        "EyeCanKnow",
        "Trusted Early Feedback Partner",
        "A user or partner selected for early-stage validation who provides feedback "
        "during testing. Participates in iterative product development cycles. "
        "Interaction is more consistent than general outreach users.",
    ),
    (
        "EyeCanKnow",
        "B2C Platform User (≈7,700 Users)",
        "One of thousands of users interacting with the product at scale. Feedback is "
        "primarily indirect and inferred from usage patterns, funnels, and conversion "
        "behavior. Interaction is analyzed through data rather than direct communication.",
    ),

    # ── SailGTX ──────────────────────────────────────────────────────────────
    (
        "SailGTX",
        "Enterprise Trade Compliance User (B2B SaaS)",
        "Works in trade compliance within enterprise environments and interacts with "
        "structured workflows and outputs. Provides feedback based on how current "
        "processes operate and how outputs can be improved. Feedback is gathered through "
        "ongoing discovery conversations rather than high-frequency input.",
    ),
    (
        "SailGTX",
        "Early Adopter Customer (Segmented User Group)",
        "A subset of enterprise users identified internally as more willing to test new "
        "features. Interacts with new functionality earlier and provides feedback that is "
        "weighted more heavily in decision-making. Participates in validating new ideas "
        "before broader rollout.",
    ),
    (
        "SailGTX",
        "Long-Tenured Industry User (15+ Years Experience)",
        "Works in the same role for extended periods (explicitly referenced as ~15 years) "
        "within compliance or related workflows. Provides feedback based on deep familiarity "
        "with existing systems and processes. Interaction reflects established habits and "
        "long-term workflow experience.",
    ),

    # ── NROC Security ────────────────────────────────────────────────────────
    (
        "NROC Security",
        "Existing Enterprise Customer (AI Security Context)",
        "A current customer using enterprise security tools related to AI usage visibility. "
        "Provides feedback based on real usage and internal workflows. Interaction occurs "
        "through direct conversations about product improvements.",
    ),
    (
        "NROC Security",
        "Existing Customer Requesting Features",
        "A user within an organization who suggests new features based on their specific "
        "operational needs. Feedback reflects their internal setup and may vary across "
        "organizations. Interaction is driven by feature requests during engagement.",
    ),
    (
        "NROC Security",
        "External Market User (Non-Customer Segment)",
        "A user outside the existing customer base whose needs are considered for expansion. "
        "Interaction involves understanding broader market requirements beyond current users. "
        "Feedback is gathered through outreach beyond existing accounts.",
    ),

    # ── DecycleBio ───────────────────────────────────────────────────────────
    (
        "DecycleBio",
        "Industry Expert (Biotech / MIT Network)",
        "A professional connected through academic or industry networks (e.g., MIT, "
        "Stanford) who participates in interviews. Provides feedback through scheduled "
        "conversations about workflows and problems. Interaction is based on expertise "
        "rather than product usage.",
    ),

    # ── Amoofy ───────────────────────────────────────────────────────────────
    (
        "Amoofy",
        "Interview Participant (Global Sample, ~1200 Interviews)",
        "A participant in a large-scale global interview set conducted post-COVID. "
        "Provides qualitative responses during conversations. Interaction is part of "
        "broad data collection efforts.",
    ),
    (
        "Amoofy",
        "Pilot Customer (Early Product Testing)",
        "A user or organization testing the product in pilot deployments. Provides "
        "feedback during early-stage usage. Interaction focuses on validating "
        "functionality.",
    ),
    (
        "Amoofy",
        "Pattern-Contributing Participant",
        "An individual whose responses contribute to identifying recurring patterns "
        "across interviews. Feedback is aggregated with others rather than evaluated "
        "individually. Interaction is analyzed collectively.",
    ),

    # ── Notion ───────────────────────────────────────────────────────────────
    (
        "Notion",
        "Product Manager (Large Tech Company)",
        "Works as a PM using tools like Hex dashboards and Zendesk to review user "
        "feedback and metrics. Interacts with multiple data sources and synthesizes "
        "patterns from them. Behavior involves structured analysis of product data.",
    ),
    (
        "Notion",
        "PM Using AI Tools (Claude / ChatGPT)",
        "Uses AI tools to summarize feedback and identify patterns across datasets. "
        "Still reviews raw interviews or source material for validation. Interaction "
        "combines automated and manual analysis.",
    ),
    (
        "Notion",
        "Cross-Functional PM (Design + Engineering Collaboration)",
        "Works closely with design and engineering teams using tools like Notion, Slack, "
        "and whiteboarding platforms. Participates in collaborative sessions to define "
        "solutions and execute features. Interaction is team-based and iterative.",
    ),

    # ── Altude ───────────────────────────────────────────────────────────────
    (
        "Altude",
        "Solana Developer (Blockchain Ecosystem)",
        "A developer working within the Solana ecosystem building products involving "
        "embedded wallets or gasless transactions. Interaction with the product is "
        "primarily through usage rather than interviews, and feedback is difficult to "
        "extract. Communication is limited and often indirect.",
    ),
    (
        "Altude",
        "Funnel User (Sign-up Stage, No Usage)",
        "A user reached through outreach (e.g., Telegram groups) who expresses interest "
        "and signs up for the product. Does not proceed to actual usage or integration "
        "after sign-up. Behavior is observed through funnel drop-off rather than direct "
        "feedback.",
    ),
    (
        "Altude",
        "Outreach Contact (Telegram Group Member)",
        "A user contacted in blockchain-related Telegram groups who shows initial interest "
        "when approached. Interaction is brief and often does not progress beyond initial "
        "conversations. Provides minimal or no actionable feedback.",
    ),

    # ── Pantri App ───────────────────────────────────────────────────────────
    (
        "Pantri App",
        "Survey Respondent (150+ Sample Size)",
        "A user participating in surveys used to validate problem relevance. Provides "
        "structured responses during early-stage research. Interaction occurs before "
        "product development.",
    ),
    (
        "Pantri App",
        "Service Platform User (Integrated Tools like Instacart, Uber)",
        "Uses services such as grocery ordering, meal preparation, and household "
        "coordination within an integrated system. Interacts through booking flows, "
        "shopping tools, and service matching. Engagement is through product usage "
        "rather than interviews.",
    ),
    (
        "Pantri App",
        "Behavioral Data User (Tracked Usage Patterns)",
        "A user whose activity is tracked through system data such as bookings and "
        "interactions. Feedback is inferred from observed behavior rather than direct "
        "input. Interaction is passive and data-driven.",
    ),

    # ── Green Corridors ──────────────────────────────────────────────────────
    (
        "Green Corridors",
        "Third-Party Logistics Operator (US / Global Firms)",
        "Works at large logistics companies such as FedEx, Uber Freight, Echo, or "
        "Redwood, which operate across the U.S. and globally. Interacts with simulation "
        "tools like digital twins, animations, and operational models during meetings to "
        "evaluate infrastructure concepts. Provides feedback during structured sessions "
        "focused on operations and data performance.",
    ),
    (
        "Green Corridors",
        "Oil & Gas Field Operator (Canada / Texas / Costa Rica context)",
        "Works in oil and gas field operations across locations mentioned such as Texas, "
        "British Columbia (Canada), and Costa Rica. Interacts directly with equipment in "
        "the field and provides feedback based on real operational issues observed during "
        "workflows. Participates in iterative product improvement by describing specific "
        "problems encountered on-site.",
    ),
    (
        "Green Corridors",
        "Oil & Gas Engineer (Exploration & Production Offices)",
        "Holds roles like drilling engineer or completions engineer within exploration "
        "and production companies. Works in office environments tied to field operations "
        "and reviews product outputs after demonstrations or deployments. Provides "
        "feedback by suggesting additional use cases and improvements after seeing "
        "product capabilities.",
    ),

    # ── Lamar Health ─────────────────────────────────────────────────────────
    (
        "Lamar Health",
        "Workflow Observation Customer (Healthcare Context)",
        "A customer whose workflows are recorded and analyzed to understand operational "
        "processes. Interacts by demonstrating real processes rather than describing them "
        "abstractly. Feedback is validated through confirmation of observed workflows.",
    ),
    (
        "Lamar Health",
        "UAT Testing Customer",
        "A customer testing features in a user acceptance testing environment before "
        "full rollout. Provides feedback based on hands-on interaction with test "
        "versions. Interaction occurs during controlled testing phases.",
    ),
    (
        "Lamar Health",
        "Repeated Feedback Customer",
        "A user whose repeated requests signal consistent issues or needs. Interaction "
        "is tracked across multiple instances of similar feedback. Behavior is identified "
        "through frequency rather than single interactions.",
    ),
]

# Conservative config for overnight batch run
OVERNIGHT_CONFIG = {
    "target_total":     3_000,
    "max_threads":      6,
    "max_comments":     100,
    "hn_max_per_query": 150,
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _preview() -> None:
    print("\n" + "─" * 78)
    print(f"  {'#':>2}  {'Company':<20}  {'Demographic label (first line)'}")
    print("─" * 78)
    for i, (company, label, _) in enumerate(JOBS, 1):
        stem = f"{_slug(company)}_{_slug(label)}"
        print(f"  {i:>2}  {company:<20}  {label}")
        print(f"       {'':20}  → {stem}_persona.json")
    print("─" * 78)

    by_company: dict[str, int] = {}
    for company, _, _ in JOBS:
        by_company[company] = by_company.get(company, 0) + 1
    print(f"\n  Companies : {', '.join(f'{c}({n})' for c, n in by_company.items())}")
    print(f"  Total jobs: {len(JOBS)}")
    cfg = OVERNIGHT_CONFIG
    lo = len(JOBS) * 12
    hi = len(JOBS) * 18
    print(f"  Config    : {cfg['target_total']:,} reviews/job, "
          f"{cfg['max_threads']} threads, {cfg['max_comments']} comments/thread")
    print(f"  Est. time : {lo // 60}h{lo % 60}m – {hi // 60}h{hi % 60}m\n")


def _already_queued() -> set[str]:
    return {(j["project_name"], j["label"]) for j in read_queue()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    _preview()

    existing = _already_queued()
    dupes = [(c, l) for c, l, _ in JOBS if (c, l) in existing]
    if dupes:
        print(f"  ⚠  Already queued (will skip): {len(dupes)} job(s)")

    if not args.yes:
        ans = input("  Add all to queue? [y/N] ").strip().lower()
        if ans != "y":
            print("  Cancelled.")
            sys.exit(0)

    added = skipped = 0
    for company, label, desc in JOBS:
        if (company, label) in existing:
            print(f"  skip  {company} — {label[:50]}")
            skipped += 1
            continue
        add_job(label, desc, project_name=company, config=OVERNIGHT_CONFIG)
        stem = f"{_slug(company)}_{_slug(label)}"
        print(f"  ✓  {company:<20} {label[:45]}")
        print(f"     → {stem}_persona.json")
        added += 1

    print(f"\n  Done. {added} added, {skipped} skipped.")

    if is_worker_alive():
        print("  Worker already running — jobs will start shortly.")
    else:
        print("\n  Worker NOT running. Start it with:")
        print("    caffeinate -i python worker.py")
        print("  or Streamlit UI → ⚡ Batch Queue → ▶ Start Worker")


if __name__ == "__main__":
    main()
