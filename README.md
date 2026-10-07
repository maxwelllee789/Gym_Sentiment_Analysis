# 🏋️ Gym Comparator AI

An AI-powered web application that compares two gyms from Google Maps links, using **Bidirectional LSTM** for sentiment analysis and **rule-based aspect categorization** for pros/cons extraction.

![CI](https://github.com/maxwelllee789/Gym_Sentiment_Analysis/actions/workflows/ci.yml/badge.svg)

---

## 📖 Overview

Choosing a gym is hard — reviews are scattered, biased, and hard to compare. **Gym Comparator AI** solves this by:

1. **Scraping** Google Maps reviews automatically (via Apify)
2. **Analyzing** sentiment using a custom-trained Bidirectional LSTM model (87% accuracy)
3. **Categorizing** pros/cons into 12 aspects (Equipment, Cleanliness, Price, Location, etc.)
4. **Comparing** two gyms side-by-side and recommending the better one
5. **Storing** every analysis in a local SQLite database

---

## ✨ Features

- 🔗 **Paste 2 Google Maps links** → get instant comparison
- 🧠 **Custom LSTM model** trained on 26K+ gym reviews (UCSD Google Local dataset)
- 📊 **12-aspect categorization**: Equipment, Cleanliness, Showers, Price, Location, Staff, Atmosphere, Hours, Classes, AC, Safety, Amenities
- 💾 **SQLite persistence**: every analysis is logged and queryable
- ⚡ **Smart caching**: repeated queries load instantly from cache
- 🎨 **Dark neon UI** with Streamlit
- 🚀 **REST API** via FastAPI with Swagger docs
- ✅ **CI/CD pipeline** via GitHub Actions

---

## 🏗️ Architecture
┌─────────────────────┐
│ User Input (URL) │
└──────────┬──────────┘
↓
┌─────────────────────┐
│ Apify Scraper │ → Google Maps reviews
└──────────┬──────────┘
↓
┌─────────────────────┐
│ Cache Layer │ → JSON + CSV
└──────────┬──────────┘
↓
┌─────────────────────┐
│ LSTM Inference │ → Sentiment per sentence
└──────────┬──────────┘
↓
┌─────────────────────┐
│ Aspect Categorizer │ → 12 rule-based categories
└──────────┬──────────┘
↓
┌─────────────────────┐
│ SQLite + Streamlit │ → UI + persistence
└─────────────────────┘


---

## 📁 Project Structure
Gym_Sentiment_Analysis/
├── .github/workflows/ci.yml # CI/CD pipeline
├── api/
│ └── main.py # FastAPI REST API
├── app/
│ ├── app.py # Streamlit UI
│ ├── inference.py # LSTM + aspect categorization
│ ├── recommendation.py # Winner selection logic
│ └── assets/ # Logo & background
├── data/
│ ├── raw/ # Original dataset (not pushed)
│ ├── processed/ # Train/val/test splits (not pushed)
│ └── cache/ # Scraped gym data (not pushed)
├── model/ # Trained LSTM artifacts (not pushed)
├── db.py # SQLite schema & helpers
├── train_model.py # LSTM training script
├── tune_model.py # Hyperparameter tuning
├── preprocessing.py # Text cleaning + sentence splitting
├── scrape_reviews.py # Apify integration
├── eda.py # Exploratory Data Analysis
└── requirements.txt


---

## 🧠 Model Details

| Metric | Value |
|---|---|
| **Architecture** | Bidirectional LSTM + Embedding + Dense |
| **Dataset** | UCSD Google Local Reviews (gym subset, English) |
| **Training samples** | ~19,000 reviews |
| **Test accuracy** | **87%** |
| **F1 (positive)** | 0.93 |
| **F1 (negative)** | 0.83 |
| **F1 (neutral)** | 0.35 |
| **Hyperparameter tuning** | 24 combinations (grid search) |
| **Best config** | embed_dim=64, lstm_units=64, dropout=0.2, lr=0.001 |

> **Note:** Neutral class F1 is low due to class imbalance (7% of data) and the inherent ambiguity of 3-star reviews. This is expected and documented.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/maxwelllee789/Gym_Sentiment_Analysis.git
cd Gym_Sentiment_Analysis
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

🛠️ Tech Stack
Category	Tools
Deep Learning	TensorFlow / Keras
NLP	Custom LSTM, rule-based aspect categorization
Web Framework	Streamlit, FastAPI
Database	SQLite
Scraping	Apify (Google Maps Reviews Scraper)
Visualization	Matplotlib, Seaborn, WordCloud
CI/CD	GitHub Actions
Language	Python 3.11+

📈 Model Performance
Trained model reaches 87% test accuracy across 3 sentiment classes on ~19K gym reviews. Detailed evaluation:
Positive reviews: F1 = 0.93 (excellent)
Negative reviews: F1 = 0.83 (good)
Neutral reviews: F1 = 0.35 (limited by data imbalance)
Hyperparameter tuning was performed with 24 grid-search combinations. See tune_model.py.

🤝 Contributing
This is a final bootcamp project. Pull requests are welcome for educational purposes.

📝 License
For academic use only.

🙏 Acknowledgments
UCSD Google Local Reviews dataset
Apify for scraping infrastructure
Streamlit & FastAPI communities
