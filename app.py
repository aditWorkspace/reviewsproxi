"""Proxi AI — Streamlit Dashboard

Thin wrapper that runs the train.py UI (single-page project workflow).
Run:  streamlit run app.py
"""
import importlib
import importlib.util
import sys
from pathlib import Path

# Load train.py as a module and run it — it contains the full Streamlit UI
spec = importlib.util.spec_from_file_location("train_ui", Path(__file__).parent / "train.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["train_ui"] = mod
spec.loader.exec_module(mod)
