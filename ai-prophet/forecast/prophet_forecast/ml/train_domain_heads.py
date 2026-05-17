"""Step 5/8/9: Train domain-specific ML heads.

Trains:
- sports_logistic.pkl  — logistic regression on Elo features
- culture_logistic.pkl — logistic regression on market features for culture/entertainment
- faiss_index.bin + faiss_metadata.pkl — k-NN embeddings for science/tech

Run:
    python -m prophet_forecast.ml.train_domain_heads

Reads:  artifacts/training_data.csv
Writes: artifacts/sports_logistic.pkl
        artifacts/culture_logistic.pkl
        artifacts/faiss_index.bin
        artifacts/faiss_metadata.pkl
"""

from __future__ import annotations

import csv
import logging
import math
import os
import sys

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TRAINING_CSV = os.path.join(ARTIFACTS_DIR, "training_data.csv")


def _load_domain_rows(domain_keywords: list[str]) -> tuple[list, list, list]:
    """Load rows matching domain, returning (questions, feature_lists, outcomes)."""
    questions, features, outcomes = [], [], []
    with open(TRAINING_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row.get("category", "").lower()
            if not any(kw in cat for kw in domain_keywords):
                continue
            try:
                p = float(row["p_market"])
                h = float(row["hours_to_close"])
                o = int(row["outcome"])
                if not (0.0 < p < 1.0):
                    continue
                questions.append(row.get("title", ""))
                features.append([p, h])
                outcomes.append(o)
            except (ValueError, KeyError):
                continue
    return questions, features, outcomes


def train_sports() -> None:
    """Sports logistic regression on market features (Elo injected at inference time)."""
    try:
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        log.error("Missing deps")
        return

    _, features, outcomes = _load_domain_rows(["sports"])
    if len(features) < 30:
        log.warning("Only %d sports rows — skipping sports model", len(features))
        return

    log.info("Training sports logistic on %d rows", len(features))

    # Features available at training time: p_market + hours_to_close
    # Elo features are available at inference via external API — not in training CSV
    # We train a "market correction" model for sports; Elo is added at inference
    X = np.array([[math.log(p / (1 - p)), math.log1p(h)] for p, h in features])
    y = np.array(outcomes)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=1.0, max_iter=300)
    model.fit(X_tr, y_tr)

    brier_cal = sum((model.predict_proba(X_val)[:, 1] - y_val) ** 2) / len(y_val)
    brier_raw = sum(((1 / (1 + math.exp(-X_val[i, 0]))) - y_val[i]) ** 2
                    for i in range(len(y_val))) / len(y_val)
    log.info("Sports  Brier(model)=%.4f  Brier(market)=%.4f", brier_cal, brier_raw)

    path = os.path.join(ARTIFACTS_DIR, "sports_logistic.pkl")
    joblib.dump(model, path)
    log.info("Saved %s", path)


def train_culture() -> None:
    """Culture/entertainment logistic regression."""
    try:
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        log.error("Missing deps")
        return

    _, features, outcomes = _load_domain_rows(["culture", "entertainment"])
    if len(features) < 20:
        log.warning("Only %d culture rows — skipping culture model", len(features))
        return

    log.info("Training culture logistic on %d rows", len(features))
    X = np.array([[math.log(p / (1 - p)), math.log1p(h)] for p, h in features])
    y = np.array(outcomes)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=1.0, max_iter=300)
    model.fit(X_tr, y_tr)

    brier_cal = sum((model.predict_proba(X_val)[:, 1] - y_val) ** 2) / len(y_val)
    log.info("Culture  Brier(model)=%.4f", brier_cal)

    path = os.path.join(ARTIFACTS_DIR, "culture_logistic.pkl")
    joblib.dump(model, path)
    log.info("Saved %s", path)


def train_science_knn() -> None:
    """Science/tech k-NN on sentence embeddings."""
    try:
        import joblib
        import numpy as np
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        log.error("Missing deps: pip install sentence-transformers faiss-cpu")
        return

    questions, _, outcomes = _load_domain_rows(["science", "tech", "technology"])
    if len(questions) < 5:
        log.warning("Only %d science/tech rows — skipping k-NN", len(questions))
        return

    log.info("Building FAISS index for %d science/tech questions", len(questions))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(questions, normalize_embeddings=True).astype("float32")

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product = cosine on normalized vectors
    index.add(embeddings)

    idx_path = os.path.join(ARTIFACTS_DIR, "faiss_index.bin")
    meta_path = os.path.join(ARTIFACTS_DIR, "faiss_metadata.pkl")
    faiss.write_index(index, idx_path)
    joblib.dump({"questions": questions, "outcomes": outcomes}, meta_path)
    log.info("Saved FAISS index (%d vectors) → %s", index.ntotal, idx_path)


def train_all() -> None:
    if not os.path.exists(TRAINING_CSV):
        log.error("Run build_training_data.py first")
        sys.exit(1)

    train_sports()
    train_culture()
    train_science_knn()
    log.info("Domain head training complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_all()
