"""Proxi AI — Streamlit Dashboard"""

import json
import os
import streamlit as st
from pathlib import Path
from datetime import datetime

from storage.db import init_db, get_all_runs, get_run, get_all_personas, get_persona, save_persona
from insights.chat import JourneyChat
from engine.knowledge import (
    list_all_personas,
    load_trained_knowledge,
    load_training_log,
    train_persona_on_reviews,
    parse_csv_reviews,
    get_full_persona_context,
)

# --- Config ---
st.set_page_config(
    page_title="Proxi AI — Synthetic User Personas",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "proxi_runs.db"
PERSONAS_DIR = Path("data/personas")
RUNS_DIR = Path("runs")


@st.cache_resource
def get_db():
    return init_db(DB_PATH)


def load_persona_files() -> list[dict]:
    """Load all persona configs (supports both flat files and directory format)."""
    all_personas = list_all_personas()
    return [p["config"] for p in all_personas]


def load_personas_with_status() -> list[dict]:
    """Load personas with their training status."""
    return list_all_personas()


# --- Sidebar ---
st.sidebar.title("Proxi AI")
st.sidebar.markdown("Synthetic User Persona Engine")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Train Persona", "Personas", "Run Agent", "Journey Viewer", "Insights", "Chat"],
)


# --- Dashboard ---
if page == "Dashboard":
    st.title("Proxi AI Dashboard")
    st.markdown("### Synthetic User Persona Engine")

    col1, col2, col3, col4 = st.columns(4)

    personas_status = load_personas_with_status()
    db = get_db()
    runs = get_all_runs(db)
    trained_count = sum(1 for p in personas_status if p["is_trained"])

    col1.metric("Personas", len(personas_status))
    col2.metric("Trained", f"{trained_count}/{len(personas_status)}")
    col3.metric("Agent Runs", len(runs))
    col4.metric(
        "Conversion Rate",
        f"{sum(1 for r in runs if r.get('outcome') == 'converted') / max(len(runs), 1) * 100:.0f}%"
        if runs
        else "N/A",
    )

    if runs:
        st.markdown("### Recent Runs")
        for run in runs[:10]:
            outcome_color = "🟢" if run.get("outcome") == "converted" else "🔴"
            st.markdown(
                f"{outcome_color} **{run.get('persona_id', 'unknown')}** → "
                f"`{run.get('target_url', '')}` — {run.get('outcome', 'unknown')} "
                f"({run.get('created_at', '')})"
            )
    else:
        st.info("No agent runs yet. Train a persona first, then go to 'Run Agent'.")

    st.markdown("### Personas")
    for p in personas_status:
        config = p["config"]
        trained_badge = "✅ Trained" if p["is_trained"] else "⚪ Untrained"
        review_count = p.get("total_reviews_trained", 0)
        sessions = p.get("training_sessions", 0)

        with st.expander(f"**{config['label']}** — {trained_badge} ({review_count} reviews, {sessions} sessions)"):
            st.json(config["segment"])
            st.markdown(f"*\"{config['voice_sample']}\"*")


# --- Train Persona ---
elif page == "Train Persona":
    st.title("Train Persona on Review Data")
    st.markdown(
        "Upload a CSV of product reviews to train a persona. "
        "The persona learns behavioral patterns from real reviews and remembers them permanently. "
        "You can train the same persona multiple times — knowledge accumulates."
    )

    personas = load_persona_files()
    if not personas:
        st.error("No personas available. Add persona JSON files to data/personas/ first.")
        st.stop()

    # Persona selection
    selected_idx = st.selectbox(
        "Select Persona to Train",
        options=range(len(personas)),
        format_func=lambda i: f"{personas[i]['label']} ({personas[i]['id']})",
    )
    persona = personas[selected_idx]
    persona_id = persona["id"]

    # Show current training status
    knowledge = load_trained_knowledge(persona_id)
    training_log = load_training_log(persona_id)

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Sessions", len(training_log))
    col2.metric("Total Reviews Trained", sum(s.get("n_reviews", 0) for s in training_log))
    col3.metric("Status", "Trained" if knowledge else "Untrained")

    if knowledge:
        with st.expander("View Current Trained Knowledge"):
            st.json(knowledge)

    if training_log:
        with st.expander("Training History"):
            for i, session in enumerate(reversed(training_log), 1):
                st.markdown(
                    f"**Session {len(training_log) - i + 1}** — "
                    f"{session.get('timestamp', 'unknown')} — "
                    f"{session.get('n_reviews', 0)} reviews from `{session.get('source', 'unknown')}`"
                )
                if session.get("signals_extracted"):
                    st.json(session["signals_extracted"])

    st.markdown("---")
    st.markdown("### Upload Review Data")
    st.markdown(
        "**Supported CSV columns:** `text`/`review`/`body`/`review_text`, "
        "`rating`/`stars`, `title`/`summary`, `product_name`/`product`, "
        "`category`, `date`, `helpful_votes`, `verified_purchase`"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV of reviews",
        type=["csv"],
        help="CSV with at least a text/review column. More columns = richer training.",
    )

    if uploaded_file is not None:
        # Parse and preview
        csv_content = uploaded_file.read().decode("utf-8", errors="replace")
        reviews = parse_csv_reviews(csv_content)

        st.success(f"Parsed **{len(reviews)}** valid reviews from `{uploaded_file.name}`")

        # Preview
        with st.expander(f"Preview Reviews ({min(5, len(reviews))} of {len(reviews)})"):
            for r in reviews[:5]:
                rating = r.get("rating", "?")
                text = (r.get("text", "") or "")[:300]
                product = r.get("product_name", r.get("asin", "unknown"))
                st.markdown(f"**{'⭐' * int(float(rating))} ({rating})** — {product}")
                st.markdown(f"> {text}{'...' if len(r.get('text', '')) > 300 else ''}")
                st.markdown("---")

        # Column detection summary
        detected_cols = set()
        if reviews:
            sample = reviews[0]
            for k in sample:
                if sample[k]:
                    detected_cols.add(k)
        st.markdown(f"**Detected fields:** {', '.join(sorted(detected_cols))}")

        # Training button
        source_name = uploaded_file.name

        if st.button("🚀 Train Persona on This Data", type="primary"):
            import anthropic
            client = anthropic.Anthropic()

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(message: str, progress: float):
                progress_bar.progress(progress)
                status_text.markdown(f"**{message}**")

            try:
                updated_knowledge = train_persona_on_reviews(
                    persona_id=persona_id,
                    reviews=reviews,
                    source_name=source_name,
                    client=client,
                    progress_callback=progress_callback,
                )

                st.success(f"Training complete! Persona `{persona['label']}` has been updated.")

                # Show what was learned
                st.markdown("### What Was Learned")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Core Pain Points")
                    for pp in updated_knowledge.get("core_pain_points", [])[:8]:
                        confidence = pp.get("confidence", 0)
                        st.markdown(f"- **{pp.get('signal', '?')}** (confidence: {confidence:.0%})")

                    st.markdown("#### Deal Breakers Confirmed")
                    for db_item in updated_knowledge.get("deal_breakers_confirmed", [])[:8]:
                        st.error(f"{db_item.get('deal_breaker', '?')} (seen {db_item.get('times_seen', '?')}x)")

                with col2:
                    st.markdown("#### Purchase Triggers")
                    for pt in updated_knowledge.get("purchase_triggers", [])[:8]:
                        st.markdown(f"- {pt.get('trigger', '?')} ({pt.get('context', '')})")

                    st.markdown("#### Behavioral Patterns")
                    for bp in updated_knowledge.get("behavioral_patterns", [])[:8]:
                        st.markdown(f"- {bp.get('pattern', '?')} (confidence: {bp.get('confidence', 0):.0%})")

                if updated_knowledge.get("evolved_voice_sample"):
                    st.markdown("#### Evolved Voice")
                    st.info(updated_knowledge["evolved_voice_sample"], icon="🗣️")

                if updated_knowledge.get("updated_behavioral_rules"):
                    st.markdown("#### Updated Behavioral Rules")
                    for rule in updated_knowledge["updated_behavioral_rules"]:
                        st.markdown(f"- {rule}")

                st.markdown("### Full Trained Knowledge")
                st.json(updated_knowledge)

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)


# --- Personas ---
elif page == "Personas":
    st.title("Persona Library")

    personas_status = load_personas_with_status()
    if not personas_status:
        st.warning("No personas found. Add persona JSON files to data/personas/")
        st.stop()

    tabs = st.tabs([p["config"]["label"] for p in personas_status])
    for tab, persona_info in zip(tabs, personas_status):
        persona = persona_info["config"]
        with tab:
            # Training badge
            if persona_info["is_trained"]:
                st.success(
                    f"Trained on {persona_info['total_reviews_trained']} reviews "
                    f"across {persona_info['training_sessions']} sessions",
                    icon="✅",
                )
            else:
                st.warning("Not yet trained on review data. Go to 'Train Persona' to upload reviews.", icon="⚪")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### Profile")
                st.markdown(f"**ID:** `{persona['id']}`")
                for k, v in persona["segment"].items():
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

                st.markdown("#### Goals")
                for g in persona["goals"]:
                    st.markdown(f"{g['priority']}. {g['goal']}")

                st.markdown("#### Constraints")
                for c in persona["constraints"]:
                    st.markdown(f"- {c}")

            with col2:
                st.markdown("#### Decision Weights")
                weights = persona["decision_weights"]
                for factor, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    st.progress(weight, text=f"{factor.replace('_', ' ').title()}: {weight:.0%}")

                st.markdown("#### Emotional Profile")
                ep = persona["emotional_profile"]
                st.progress(ep["baseline_patience"], text=f"Patience: {ep['baseline_patience']:.0%}")
                st.progress(ep["trust_starting_point"], text=f"Trust: {ep['trust_starting_point']:.0%}")
                st.metric("Frustration Decay", f"{ep['frustration_decay']:.2f}")

                st.markdown("#### Deal Breakers")
                for db_item in persona["deal_breakers"]:
                    st.error(db_item, icon="🚫")

            st.markdown("#### Behavioral Rules")
            for rule in persona["behavioral_rules"]:
                st.markdown(f"- {rule}")

            st.markdown("#### Voice")
            st.info(persona["voice_sample"], icon="🗣️")

            # Show trained knowledge if available
            if persona_info["is_trained"]:
                knowledge = load_trained_knowledge(persona["id"])
                if knowledge:
                    st.markdown("#### Trained Knowledge (from reviews)")
                    with st.expander("Core Pain Points"):
                        for pp in knowledge.get("core_pain_points", []):
                            st.markdown(f"- **{pp.get('signal', '?')}** (confidence: {pp.get('confidence', 0):.0%}, seen {pp.get('source_count', '?')}x)")
                    with st.expander("Behavioral Patterns Learned"):
                        for bp in knowledge.get("behavioral_patterns", []):
                            st.markdown(f"- {bp.get('pattern', '?')} (confidence: {bp.get('confidence', 0):.0%})")
                    with st.expander("Full Trained Knowledge JSON"):
                        st.json(knowledge)

            st.markdown("#### Raw Config JSON")
            with st.expander("View JSON"):
                st.json(persona)


# --- Run Agent ---
elif page == "Run Agent":
    st.title("Run Persona Agent")

    personas_status = load_personas_with_status()
    personas = [p["config"] for p in personas_status]
    if not personas:
        st.error("No personas available. Add persona files to data/personas/")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        selected_persona = st.selectbox(
            "Select Persona",
            options=range(len(personas)),
            format_func=lambda i: (
                f"{personas[i]['label']} {'✅' if personas_status[i]['is_trained'] else '⚪'}"
            ),
        )
    with col2:
        target_url = st.text_input("Target URL", placeholder="https://example.com")

    # Show training status
    if personas_status[selected_persona]["is_trained"]:
        st.success(
            f"This persona is trained on {personas_status[selected_persona]['total_reviews_trained']} reviews. "
            "Agent will use learned behavioral patterns.",
            icon="✅",
        )
    else:
        st.warning(
            "This persona has not been trained on review data. "
            "It will use base config only. Consider training it first.",
            icon="⚠️",
        )

    with st.expander("Advanced Settings"):
        max_steps = st.slider("Max Steps", 5, 50, 25)
        headless = st.checkbox("Headless Mode", value=True)
        record_video = st.checkbox("Record Video", value=True)

    if st.button("Launch Agent", type="primary", disabled=not target_url):
        persona = personas[selected_persona]
        st.markdown(f"**Launching** `{persona['label']}` against `{target_url}`...")

        progress_bar = st.progress(0)
        status_area = st.empty()

        import asyncio
        from agent.agent import PersonaAgent
        import anthropic

        async def run_agent():
            client = anthropic.Anthropic()
            config = {
                "max_steps": max_steps,
                "headless": headless,
                "video": record_video,
                "screenshot": True,
            }
            agent = PersonaAgent(
                persona=persona,
                target_url=target_url,
                anthropic_client=client,
                config=config,
            )
            journey = await agent.run()
            return journey

        try:
            journey = asyncio.run(run_agent())
            st.success(f"Agent completed! Outcome: **{journey.get('outcome', 'unknown')}**")

            if journey.get("outcome_reason"):
                st.info(journey["outcome_reason"])

            # Save to DB
            db = get_db()
            from storage.db import save_run
            save_run(
                db,
                run_id=journey["run_id"],
                persona_id=persona["id"],
                target_url=target_url,
                outcome=journey.get("outcome", "unknown"),
                outcome_reason=journey.get("outcome_reason", ""),
                journey_json=json.dumps(journey),
                summary=journey.get("outcome_reason", ""),
            )

            # Save journey to disk
            run_dir = RUNS_DIR / journey["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "journey.json", "w") as f:
                json.dump(journey, f, indent=2, default=str)

            st.markdown("### Journey Steps")
            for step in journey.get("steps", []):
                with st.expander(f"Step {step.get('step', '?')}: {step.get('action', {}).get('type', 'unknown')}"):
                    st.markdown(f"**URL:** {step.get('url', 'N/A')}")
                    st.markdown(f"**Observation:** {step.get('observation', 'N/A')}")
                    st.markdown(f"**Inner Monologue:** *{step.get('inner_monologue', 'N/A')}*")
                    if step.get("emotional_state"):
                        es = step["emotional_state"]
                        if isinstance(es, dict):
                            st.markdown(f"Patience: {es.get('patience_delta', 0):+.2f} | Trust: {es.get('trust_delta', 0):+.2f}")
                    st.json(step.get("action", {}))

        except Exception as e:
            st.error(f"Agent run failed: {e}")
            st.exception(e)


# --- Journey Viewer ---
elif page == "Journey Viewer":
    st.title("Journey Viewer")

    db = get_db()
    runs = get_all_runs(db)

    if not runs:
        st.info("No runs to display. Run an agent first.")
        st.stop()

    run_options = {
        r["run_id"]: f"{r.get('persona_id', '?')} → {r.get('target_url', '?')} ({r.get('outcome', '?')})"
        for r in runs
    }
    selected_run_id = st.selectbox("Select Run", options=list(run_options.keys()), format_func=lambda x: run_options[x])

    run_data = get_run(db, selected_run_id)
    if not run_data:
        st.error("Run not found")
        st.stop()

    journey = json.loads(run_data["journey_json"]) if isinstance(run_data.get("journey_json"), str) else run_data.get("journey_json", {})

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Outcome", journey.get("outcome", "unknown"))
    col2.metric("Steps", journey.get("total_steps", len(journey.get("steps", []))))
    final_state = journey.get("final_state", {})
    col3.metric("Final Patience", f"{final_state.get('patience', 0):.2f}")
    col4.metric("Final Trust", f"{final_state.get('trust', 0):.2f}")

    if journey.get("outcome_reason"):
        st.info(journey["outcome_reason"])

    # Emotional journey chart
    steps = journey.get("steps", [])
    if steps:
        st.markdown("### Emotional Journey")
        patience_data = []
        trust_data = []
        for s in steps:
            es = s.get("emotional_state", {})
            if isinstance(es, dict):
                patience_data.append(es.get("patience_after", es.get("patience_delta", 0)))
                trust_data.append(es.get("trust_after", es.get("trust_delta", 0)))
            else:
                patience_data.append(0)
                trust_data.append(0)

        import pandas as pd
        chart_df = pd.DataFrame({"Patience": patience_data, "Trust": trust_data})
        st.line_chart(chart_df)

    # Friction events
    friction = journey.get("friction_events", [])
    if friction:
        st.markdown("### Friction Events")
        for f_event in friction:
            st.warning(f"Step {f_event.get('step', '?')}: {f_event.get('type', 'unknown')} (severity: {f_event.get('severity', '?')})")

    # Step-by-step
    st.markdown("### Step-by-Step Journey")
    for step in steps:
        with st.expander(f"Step {step.get('step', '?')}: {step.get('action', {}).get('type', '?')} — {step.get('url', '')}"):
            st.markdown(f"**Observation:** {step.get('observation', '')}")
            st.markdown(f"**Inner Monologue:** *{step.get('inner_monologue', '')}*")
            st.markdown(f"**Goal Relevance:** {step.get('goal_relevance', '')}")
            st.json(step.get("action", {}))

            screenshot_path = step.get("screenshot_path")
            if screenshot_path and os.path.exists(screenshot_path):
                st.image(screenshot_path, caption=f"Step {step.get('step', '?')}")


# --- Insights ---
elif page == "Insights":
    st.title("Insights")

    db = get_db()
    runs = get_all_runs(db)

    if not runs:
        st.info("No runs to analyze. Run an agent first.")
        st.stop()

    st.markdown("### Generate Insights")

    run_options = {
        r["run_id"]: f"{r.get('persona_id', '?')} → {r.get('target_url', '?')} ({r.get('outcome', '?')})"
        for r in runs
    }
    selected_runs = st.multiselect(
        "Select runs to analyze",
        options=list(run_options.keys()),
        format_func=lambda x: run_options[x],
    )

    if st.button("Generate Insights", type="primary", disabled=not selected_runs):
        import anthropic
        from insights.analyze import generate_insights
        from insights.compare import compare_journeys

        client = anthropic.Anthropic()
        personas_loaded = load_persona_files()
        persona_map = {p["id"]: p for p in personas_loaded}

        journeys = []
        personas_for_compare = []
        for rid in selected_runs:
            run_data = get_run(db, rid)
            journey = json.loads(run_data["journey_json"]) if isinstance(run_data.get("journey_json"), str) else {}
            journeys.append(journey)
            pid = run_data.get("persona_id", "")
            personas_for_compare.append(persona_map.get(pid, {}))

        if len(selected_runs) == 1:
            with st.spinner("Generating insights..."):
                insights = generate_insights(journeys[0], personas_for_compare[0], client)
            st.json(insights)
        else:
            with st.spinner("Comparing journeys..."):
                comparison = compare_journeys(journeys, personas_for_compare, client)
            st.json(comparison)


# --- Chat ---
elif page == "Chat":
    st.title("Chat with Journey Data")

    db = get_db()
    runs = get_all_runs(db)

    if not runs:
        st.info("No runs available for chat. Run an agent first.")
        st.stop()

    run_options = {
        r["run_id"]: f"{r.get('persona_id', '?')} → {r.get('target_url', '?')}"
        for r in runs
    }
    selected_run_id = st.selectbox(
        "Select a run to chat about",
        options=list(run_options.keys()),
        format_func=lambda x: run_options[x],
        key="chat_run_select",
    )

    # Initialize chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_run_id" not in st.session_state or st.session_state.chat_run_id != selected_run_id:
        st.session_state.chat_messages = []
        st.session_state.chat_run_id = selected_run_id

    run_data = get_run(db, selected_run_id)
    journey = json.loads(run_data["journey_json"]) if isinstance(run_data.get("journey_json"), str) else {}
    personas_loaded = load_persona_files()
    persona_map = {p["id"]: p for p in personas_loaded}
    persona = persona_map.get(run_data.get("persona_id", ""), {})

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about this journey..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        import anthropic
        client = anthropic.Anthropic()
        chat = JourneyChat(journey, persona, client)
        # Replay history
        for msg in st.session_state.chat_messages[:-1]:
            chat._messages.append({"role": msg["role"], "content": msg["content"]})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat.ask(prompt)
            st.markdown(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})
