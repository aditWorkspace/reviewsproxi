"""Stage 3: Embed signals and auto-optimize KMeans clustering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

SIGNALS_PATH = Path("data/extracted/signals.jsonl")
CLUSTERS_DIR = Path("data/clusters")
EMBEDDINGS_PATH = CLUSTERS_DIR / "embeddings.npy"
SWEEP_PATH = CLUSTERS_DIR / "sweep.json"
CLUSTERS_PATH = CLUSTERS_DIR / "clusters.json"


def _signals_to_text(signals: dict) -> str:
    """Flatten a signal dict into a single text blob for embedding."""
    parts: list[str] = []
    for pp in signals.get("pain_points", []):
        parts.append(pp.get("signal", ""))
    for do in signals.get("desired_outcomes", []):
        parts.append(do.get("outcome", ""))
    for db in signals.get("deal_breakers", []):
        if isinstance(db, str):
            parts.append(db)
    for df in signals.get("decision_factors_ranked", []):
        parts.append(df)
    return " ".join(filter(None, parts))


def embed_signals(signals_list: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [_signals_to_text(s) for s in signals_list]
    return model.encode(texts, show_progress_bar=True)


def score_clustering(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute silhouette, intra-cluster cosine similarity, inter-cluster distance."""
    n_clusters = len(set(labels))

    sil = float(silhouette_score(embeddings, labels)) if n_clusters > 1 else 0.0

    intra_sims: list[float] = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_embs = embeddings[mask]
        if len(cluster_embs) > 1:
            sim_matrix = cosine_similarity(cluster_embs)
            n = len(cluster_embs)
            upper = sim_matrix[np.triu_indices(n, k=1)]
            intra_sims.append(float(upper.mean()))
    intra = float(np.mean(intra_sims)) if intra_sims else 0.0

    centroids = np.array([embeddings[labels == c].mean(axis=0) for c in range(n_clusters)])
    if len(centroids) > 1:
        centroid_sims = cosine_similarity(centroids)
        n = len(centroids)
        upper = centroid_sims[np.triu_indices(n, k=1)]
        inter = float(1.0 - upper.mean())
    else:
        inter = 0.0

    return {"silhouette": sil, "intra_similarity": intra, "inter_distance": inter}


def combined_score(metrics: dict, weights: dict) -> float:
    return (
        metrics["silhouette"] * weights["silhouette"]
        + metrics["intra_similarity"] * weights["intra_similarity"]
        + metrics["inter_distance"] * weights["inter_distance"]
    )


def run_cluster(config: dict, n_clusters_override: int | None = None, force: bool = False) -> dict:
    """Cluster signals with auto-optimized or fixed n_clusters."""
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    signals_list: list[dict] = []
    with open(SIGNALS_PATH) as f:
        for line in f:
            signals_list.append(json.loads(line))

    # Embed (reuse checkpoint unless forced)
    if EMBEDDINGS_PATH.exists() and not force:
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        embeddings = embed_signals(signals_list)
        np.save(EMBEDDINGS_PATH, embeddings)

    weights = config["clustering"]["scoring_weights"]

    if n_clusters_override:
        best_k = n_clusters_override
        sweep_results: list[dict] = []
    else:
        sweep_range = config["clustering"]["sweep_range"]
        sweep_results = []
        best_k = sweep_range[0]
        best_score = -1.0

        for k in range(sweep_range[0], min(sweep_range[1] + 1, len(signals_list))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            metrics = score_clustering(embeddings, labels)
            cs = combined_score(metrics, weights)
            sweep_results.append({"n_clusters": k, **metrics, "combined_score": cs})
            if cs > best_score:
                best_score = cs
                best_k = k

        with open(SWEEP_PATH, "w") as f:
            json.dump(sweep_results, f, indent=2)

    # Final clustering with chosen k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = km.fit_predict(embeddings)
    final_metrics = score_clustering(embeddings, final_labels)

    result: dict[str, Any] = {
        "chosen_n_clusters": best_k,
        "sweep_winner": {
            **final_metrics,
            "combined_score": combined_score(final_metrics, weights),
        },
    }

    for c in range(best_k):
        mask = final_labels == c
        indices = np.where(mask)[0].tolist()
        centroid = embeddings[mask].mean(axis=0)

        dists = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        rep_local_idx = np.argsort(dists)[:5].tolist()
        rep_global_idx = [indices[i] for i in rep_local_idx]
        rep_batches = [signals_list[i] for i in rep_global_idx]

        all_pain: list[str] = []
        all_outcomes: list[str] = []
        all_deal_breakers: list[str] = []
        friction_counts: dict[str, int] = {}

        for idx in indices:
            s = signals_list[idx]
            all_pain.extend(
                pp["signal"] for pp in s.get("pain_points", [])
                if isinstance(pp, dict) and "signal" in pp
            )
            all_outcomes.extend(
                do["outcome"] for do in s.get("desired_outcomes", [])
                if isinstance(do, dict) and "outcome" in do
            )
            all_deal_breakers.extend(
                db for db in s.get("deal_breakers", []) if isinstance(db, str)
            )
            ft = s.get("friction_tolerance", "medium")
            friction_counts[ft] = friction_counts.get(ft, 0) + 1

        dominant_friction = max(friction_counts, key=friction_counts.get) if friction_counts else "medium"

        result[f"cluster_{c}"] = {
            "label_hint": None,
            "member_count": int(mask.sum()),
            "representative_batches": rep_batches,
            "aggregate_signals": {
                "top_pain_points": list(dict.fromkeys(all_pain))[:10],
                "top_desired_outcomes": list(dict.fromkeys(all_outcomes))[:10],
                "top_deal_breakers": list(dict.fromkeys(all_deal_breakers))[:10],
                "dominant_friction_tolerance": dominant_friction,
            },
        }

    with open(CLUSTERS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return result
