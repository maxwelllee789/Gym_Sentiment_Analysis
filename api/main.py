"""
FastAPI backend for Gym Comparator AI.
Exposes LSTM sentiment analysis as REST API endpoints.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Path setup
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from inference import predict_sentiment, analyze_reviews
from scrape_reviews import scrape_store_live

app = FastAPI(
    title="Gym Comparator AI API",
    description="LSTM-based sentiment analysis for gym reviews",
    version="1.0.0",
)


# ============================================================
# SCHEMAS
# ============================================================
class ReviewInput(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    confidence: float


class StoreInput(BaseModel):
    gmaps_link: str


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {"message": "Gym Comparator AI API", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=SentimentResponse)
def predict(review: ReviewInput):
    """Predict sentiment for a single review text."""
    if not review.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    label, confidence = predict_sentiment(review.text)
    return SentimentResponse(label=label, confidence=round(confidence, 4))


@app.post("/analyze-store")
def analyze_store_endpoint(store: StoreInput):
    """Scrape + analyze a full store from Google Maps link."""
    if not store.gmaps_link.strip():
        raise HTTPException(status_code=400, detail="Link cannot be empty")

    info, reviews_df = scrape_store_live(store.gmaps_link)
    if info is None or reviews_df.empty:
        raise HTTPException(status_code=404, detail="Failed to fetch store data")

    analysis = analyze_reviews(reviews_df.to_dict("records"))

    return {
        "store": info,
        "analysis": analysis,
    }