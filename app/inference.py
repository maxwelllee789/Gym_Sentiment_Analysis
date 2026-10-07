"""
LSTM inference for sentiment + rule-based aspect categorization.

Pipeline:
- LSTM (Deep Learning)  → sentiment prediction per sentence
- Keyword matching      → aspect categorization per sentence
"""

import os
import re
import pickle
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import clean_text, split_sentences


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_lstm_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

MAX_LEN = 120
MIN_CONFIDENCE = 0.60
MIN_WORDS = 3


# ============================================================
# ASPECT CATEGORIES (12 categories, English-first keywords)
# ============================================================
ASPECT_CATEGORIES = {
    "Equipment & Machines": [
        "equipment", "machine", "dumbbell", "barbell", "treadmill", "rack",
        "weight", "bench", "cardio", "cable", "smith", "plate", "kettlebell",
        "squat", "deadlift", "lifting", "gear", "apparatus",
    ],
    "Cleanliness": [
        "clean", "dirty", "hygiene", "dust", "smell", "stink", "filthy",
        "spotless", "sanitized", "grimy", "sticky", "odor",
    ],
    "Showers & Lockers": [
        "shower", "locker", "toilet", "bathroom", "changing room", "restroom",
        "sink", "towel", "soap", "hot water", "cold water",
    ],
    "Price & Membership": [
        "price", "membership", "cost", "fee", "expensive", "cheap", "worth",
        "affordable", "overpriced", "value", "subscription", "contract",
        "cancel", "refund", "payment",
    ],
    "Location & Parking": [
        "location", "parking", "park", "access", "near", "far", "area",
        "neighborhood", "transit", "commute", "convenient", "central",
    ],
    "Staff & Service": [
        "staff", "service", "helpful", "friendly", "receptionist", "manager",
        "rude", "unprofessional", "welcoming", "attentive", "crew", "employee",
    ],
    "Atmosphere & Crowd": [
        "crowd", "crowded", "atmosphere", "quiet", "busy", "vibe", "noise",
        "music", "energy", "empty", "packed", "cozy", "spacious", "cramped",
    ],
    "Operating Hours": [
        "hour", "open", "close", "24 hour", "weekend", "schedule", "midnight",
        "early", "late", "always open",
    ],
    "Classes & Personal Training": [
        "class", "personal trainer", "yoga", "zumba", "instructor", "coach",
        "spin", "hiit", "bootcamp", "pilates", "crossfit", "trainer",
    ],
    "AC & Ventilation": [
        "ac", "air conditioner", "ventilation", "aircon", "temperature",
        "hot", "cold", "humid", "stuffy", "airflow", "fan",
    ],
    "Safety & Security": [
        "safe", "safety", "security", "cctv", "guard", "emergency", "locker room security",
        "theft", "stolen", "secure",
    ],
    "Amenities & Cafe": [
        "supplement", "cafe", "protein", "bar", "drink", "vitamin", "smoothie",
        "juice", "snack", "vending", "shop", "store", "merch",
    ],
}


# ============================================================
# MODEL LOADING (lazy, once)
# ============================================================
_model = None
_tokenizer = None
_label_encoder = None


def _load_artifacts():
    """Load model + tokenizer + label encoder (once)."""
    global _model, _tokenizer, _label_encoder

    if _model is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        print(f"[LSTM] Model not found: {MODEL_PATH}")
        return False

    try:
        _model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)
        print(f"[LSTM] Model loaded: {MODEL_PATH}")
        print(f"[LSTM] Labels: {list(_label_encoder.classes_)}")
        return True
    except Exception as e:
        print(f"[LSTM] Failed to load model: {e}")
        return False


# ============================================================
# SENTIMENT PREDICTION
# ============================================================
def predict_sentiment(text: str) -> tuple:
    """
    Predict sentiment for a single text.
    Returns: (label_str, confidence_float)
    """
    if not _load_artifacts():
        return ("netral", 0.0)

    cleaned = clean_text(text)
    if not cleaned.strip():
        return ("netral", 0.0)

    seq = _tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = _model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = _label_encoder.classes_[idx]
    confidence = float(probs[idx])

    return (str(label), confidence)


def predict_sentiment_batch(texts: list) -> list:
    """Batch prediction. Returns list of (label, confidence)."""
    if not texts:
        return []

    if not _load_artifacts():
        return [("netral", 0.0)] * len(texts)

    cleaned = [clean_text(t) or "" for t in texts]
    seqs = _tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = _model.predict(padded, verbose=0)
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        results.append((str(_label_encoder.classes_[idx]), float(p[idx])))
    return results


# ============================================================
# ASPECT CATEGORIZATION (rule-based)
# ============================================================
def categorize_sentence(sentence: str) -> list:
    """
    Return list of categories matched by the sentence.
    Can return multiple categories if multiple aspects appear.
    Returns ["General"] if no match.
    """
    s = sentence.lower()
    matched = []

    for category, keywords in ASPECT_CATEGORIES.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, s):
                matched.append(category)
                break

    return matched if matched else ["General"]


# ============================================================
# FULL ANALYSIS FOR ONE STORE
# ============================================================
def analyze_reviews(reviews: list) -> dict:
    """
    reviews: list of dict {"text": str, "rating": int}

    Returns: {
        "positif_pct": float,
        "netral_pct": float,
        "negatif_pct": float,
        "pros_by_category": {category: [sentences]},
        "cons_by_category": {category: [sentences]},
        "total_sentences": int,
    }
    """
    # 1. Split all reviews into sentences
    all_sentences = []
    for r in reviews:
        text = r.get("text", "") or ""
        chunks = split_sentences(text) if text.strip() else []
        for chunk in chunks:
            if len(chunk.split()) >= MIN_WORDS:
                all_sentences.append(chunk)

    if not all_sentences:
        return {
            "positif_pct": 0.0,
            "netral_pct": 0.0,
            "negatif_pct": 0.0,
            "pros_by_category": {},
            "cons_by_category": {},
            "total_sentences": 0,
        }

    # 2. Predict sentiment in batch
    predictions = predict_sentiment_batch(all_sentences)

    # 3. Group by category
    pros_by_cat = {}
    cons_by_cat = {}
    counter = {"positif": 0, "netral": 0, "negatif": 0}

    for sentence, (label, conf) in zip(all_sentences, predictions):
        if conf < MIN_CONFIDENCE:
            label = "netral"

        counter[label] = counter.get(label, 0) + 1

        if label == "netral":
            continue

        categories = categorize_sentence(sentence)

        for cat in categories:
            if label == "positif":
                pros_by_cat.setdefault(cat, []).append(sentence.strip())
            elif label == "negatif":
                cons_by_cat.setdefault(cat, []).append(sentence.strip())

    # 4. Compute percentages
    total = sum(counter.values())
    pct = {
        "positif_pct": round(counter["positif"] / total * 100, 2) if total else 0,
        "netral_pct": round(counter["netral"] / total * 100, 2) if total else 0,
        "negatif_pct": round(counter["negatif"] / total * 100, 2) if total else 0,
    }

    return {
        **pct,
        "pros_by_category": pros_by_cat,
        "cons_by_category": cons_by_cat,
        "total_sentences": total,
    }