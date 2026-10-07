"""
Streamlit app — Gym Comparator AI
LSTM-based sentiment analysis + rule-based aspect categorization + SQLite persistence.
Dark neon purple/pink theme + custom background + logo + cache picker.
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import json
import re
import base64
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


# ============================================================
# PATH SETUP
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
ASSETS_DIR = os.path.join(APP_DIR, "assets")

sys.path.insert(0, APP_DIR)
sys.path.insert(0, ROOT_DIR)

from inference import analyze_reviews
from recommendation import build_summary, pick_winner
from scrape_reviews import (
    scrape_store_live,
    save_store_cache_from_data,
    link_to_key,
    CACHE_DIR,
)
from db import (
    init_db,
    upsert_gym,
    save_analysis,
    save_comparison,
    get_stats,
    get_recent_comparisons,
)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Gym Comparator AI",
    page_icon=os.path.join(ASSETS_DIR, "Logo.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Initialize database (safe to call multiple times)
init_db()


# ============================================================
# ASSET LOADER
# ============================================================
LOGO_CANDIDATES = [
    os.path.join(ASSETS_DIR, "Logo.png"),
    os.path.join(ASSETS_DIR, "logo.png"),
    os.path.join(ASSETS_DIR, "logo_dumbbell.png"),
    os.path.join(ROOT_DIR, "download.png"),
    os.path.join(ROOT_DIR, "logo.png"),
]

BG_CANDIDATES = [
    os.path.join(ASSETS_DIR, "Design_Logo.png"),
    os.path.join(ASSETS_DIR, "Design_Logo.PNG"),
    os.path.join(ASSETS_DIR, "design_logo.png"),
    os.path.join(ASSETS_DIR, "background.png"),
    os.path.join(ASSETS_DIR, "bg.png"),
    os.path.join(ROOT_DIR, "image_d47747.png"),
    os.path.join(ROOT_DIR, "background.png"),
]


def _find_existing(paths):
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 100:
            return p
    return None


def _img_to_base64(path):
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


LOGO_PATH = _find_existing(LOGO_CANDIDATES)
BG_PATH = _find_existing(BG_CANDIDATES)
LOGO_B64 = _img_to_base64(LOGO_PATH)
BG_B64 = _img_to_base64(BG_PATH)

print(f"[ASSET] Logo       -> {LOGO_PATH}")
print(f"[ASSET] Background -> {BG_PATH}")


# ============================================================
# CACHE INDEX LOADER
# ============================================================
STORE_INDEX_PATH = os.path.join(CACHE_DIR, "store_index.json")


def load_cached_stores():
    if not os.path.exists(STORE_INDEX_PATH):
        return []
    try:
        with open(STORE_INDEX_PATH, encoding="utf-8") as f:
            stores = json.load(f)
        return [s for s in stores if s.get("gmaps_link") and s.get("name")]
    except Exception:
        return []


# ============================================================
# PHOTO FETCHER
# ============================================================
@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_photo_b64(url: str):
    if not url:
        return None
    try:
        r = requests.get(
            url, timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
        r.raise_for_status()
        if len(r.content) < 500:
            return None
        return base64.b64encode(r.content).decode()
    except Exception:
        return None


# ============================================================
# CUSTOM CSS
# ============================================================
_CSS_TEMPLATE = """
<style>
    .stApp {
        background:
            linear-gradient(rgba(6,6,13,0.88), rgba(10,5,20,0.94)),
            url('data:image/png;base64,__BG_B64__') center center / cover no-repeat,
            radial-gradient(ellipse at top, #1a1030 0%, #0a0a14 55%, #06060d 100%) !important;
        color: #eaeaea;
        min-height: 100vh;
    }
    [data-testid="stAppViewContainer"] { background: transparent !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stMain"] { background: transparent !important; }
    [data-testid="stToolbar"] { display: none !important; }
    footer { display: none !important; }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="column"] > div {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 !important;
    }

    .logo-header {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 10px;
    }
    .logo-img {
        height: 90px;
        width: auto;
        max-width: 220px;
        object-fit: contain;
        filter: drop-shadow(0 0 20px rgba(177,140,255,0.75));
    }
    .brand-title {
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1;
        background: linear-gradient(135deg, #ffffff 0%, #b18cff 60%, #ff6bcb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .brand-sub {
        color: #999;
        font-size: 0.9rem;
        margin-top: 8px;
    }

    [data-testid="stWidgetLabel"] p {
        color: #b18cff !important;
        font-weight: 800 !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    .stTextInput > div > div > input {
        background: rgba(14,14,26,0.9) !important;
        border: 1px solid #2a2a45 !important;
        color: #eaeaea !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        font-size: 0.88rem !important;
        backdrop-filter: blur(6px) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #555 !important; }
    .stTextInput > div > div > input:focus {
        border-color: #b18cff !important;
        box-shadow: 0 0 0 3px rgba(177,140,255,0.2) !important;
        outline: none !important;
    }

    .stSelectbox > div > div {
        background: rgba(14,14,26,0.9) !important;
        border: 1px solid #2a2a45 !important;
        border-radius: 12px !important;
        color: #eaeaea !important;
        backdrop-filter: blur(6px) !important;
    }
    .stSelectbox > div > div:hover { border-color: #b18cff !important; }
    .stSelectbox [data-baseweb="select"] > div {
        background: transparent !important;
        color: #eaeaea !important;
    }
    .stSelectbox svg { fill: #b18cff !important; }

    .stButton > button {
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        font-size: 0.85rem !important;
        padding: 14px 20px !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 20px rgba(168,85,247,0.5) !important;
        height: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(236,72,153,0.7) !important;
    }

    .gym-card {
        background: linear-gradient(180deg, rgba(20,20,37,0.92) 0%, rgba(15,15,28,0.92) 100%);
        border: 1px solid #2a2a45;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(177,140,255,0.06);
        backdrop-filter: blur(8px);
        min-height: 680px;
        display: flex;
        flex-direction: column;
    }

    .gym-label-a {
        color: #b18cff;
        font-weight: 900;
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 14px;
        flex-shrink: 0;
    }
    .gym-label-b {
        color: #f472b6;
        font-weight: 900;
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 14px;
        flex-shrink: 0;
    }
    .cache-badge {
        color: #d8b4fe;
        font-size: 0.68rem;
        font-weight: 600;
        background: rgba(177,140,255,0.12);
        padding: 2px 8px;
        border-radius: 20px;
        border: 1px solid rgba(177,140,255,0.3);
        margin-left: 8px;
    }

    .rating-label {
        color: #777;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .rating-number {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
        background: linear-gradient(135deg, #b18cff 0%, #ff6bcb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .rating-count {
        color: #999;
        font-size: 0.82rem;
        margin-top: 8px;
    }

    .sentiment-row {
        display: flex;
        gap: 10px;
        margin: 18px 0 6px 0;
        padding: 18px 0;
        border-top: 1px solid #2a2a45;
        border-bottom: 1px solid #2a2a45;
        flex-shrink: 0;
    }
    .sentiment-item { flex: 1; text-align: center; }
    .sentiment-name {
        color: #777;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .sentiment-value {
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 8px;
        line-height: 1;
    }
    .sentiment-bar { height: 3px; border-radius: 3px; margin-top: 4px; }

    .pos-text { color: #b18cff; }
    .net-text { color: #e9d5ff; }
    .neg-text { color: #f472b6; }
    .pos-bar { background: linear-gradient(90deg, #b18cff, #8b5cf6); }
    .net-bar { background: linear-gradient(90deg, #e9d5ff, #c4b5fd); }
    .neg-bar { background: linear-gradient(90deg, #f472b6, #ec4899); }

    .pros-cons-wrapper {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        overflow-y: auto;
        max-height: 380px;
    }
    .pros-cons-wrapper::-webkit-scrollbar { width: 4px; }
    .pros-cons-wrapper::-webkit-scrollbar-track {
        background: rgba(42,42,69,0.3);
        border-radius: 4px;
    }
    .pros-cons-wrapper::-webkit-scrollbar-thumb {
        background: #b18cff;
        border-radius: 4px;
    }

    .section-header {
        font-size: 0.75rem;
        font-weight: 900;
        letter-spacing: 2px;
        margin: 16px 0 8px 0;
        text-transform: uppercase;
        flex-shrink: 0;
    }
    .header-pros { color: #b18cff; }
    .header-cons { color: #f472b6; }
    .header-blue { color: #d8b4fe; }

    .category-block {
        margin: 8px 0 12px 0;
        padding-left: 8px;
        border-left: 2px solid rgba(177,140,255,0.25);
    }
    .category-name {
        font-size: 0.72rem;
        font-weight: 800;
        color: #c4b5fd;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .category-name-cons { color: #f9a8d4; }
    .bullet-list { list-style: none; padding: 0; margin: 0; }
    .bullet-list li {
        padding: 3px 0 3px 18px;
        position: relative;
        font-size: 0.82rem;
        color: #c5c5d5;
        line-height: 1.5;
    }
    .bullet-list li::before {
        content: '•';
        position: absolute;
        left: 4px;
        top: 3px;
        color: #b18cff;
        font-weight: 900;
    }
    .bullet-list.cons li::before { color: #f472b6; }

    .winner-block {
        background: linear-gradient(135deg, rgba(26,18,48,0.94) 0%, rgba(20,22,42,0.94) 100%);
        border: 1px solid #3a2a55;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(177,140,255,0.2);
        height: 100%;
        min-height: 260px;
        backdrop-filter: blur(8px);
    }
    .winner-label {
        color: #999;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .winner-name {
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1.15;
        background: linear-gradient(135deg, #b18cff 0%, #ff6bcb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 8px 0 16px 0;
    }
    .winner-reason {
        color: #c5c5d5;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .kesimpulan-block {
        background: linear-gradient(135deg, rgba(18,22,42,0.94) 0%, rgba(15,15,28,0.94) 100%);
        border: 1px solid #2a2a45;
        border-radius: 18px;
        padding: 24px;
        height: 100%;
        min-height: 260px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        backdrop-filter: blur(8px);
    }
    .kesimpulan-text {
        color: #c5c5d5;
        font-size: 0.88rem;
        line-height: 1.65;
        margin-bottom: 14px;
    }
    .kesimpulan-text:last-child { margin-bottom: 0; }
    .kesimpulan-store { color: #b18cff; font-weight: 700; }

    .cache-hint {
        color: #a78bfa;
        font-size: 0.78rem;
        margin-top: 6px;
        font-style: italic;
    }
    .cache-empty {
        background: rgba(177,140,255,0.08);
        border: 1px solid rgba(177,140,255,0.25);
        border-radius: 12px;
        padding: 12px 16px;
        color: #d8b4fe;
        font-size: 0.85rem;
        margin-top: 8px;
    }

    [data-testid="stExpander"] {
        background: rgba(20,20,37,0.75) !important;
        border: 1px solid #2a2a45 !important;
        border-radius: 14px !important;
        backdrop-filter: blur(6px);
    }
    [data-testid="stExpander"] summary {
        color: #d8b4fe !important;
        font-weight: 700 !important;
    }

    /* ---------- DB Stats ---------- */
    .db-stats {
        background: linear-gradient(135deg, rgba(18,22,42,0.94) 0%, rgba(15,15,28,0.94) 100%);
        border: 1px solid #2a2a45;
        border-radius: 16px;
        padding: 18px 24px;
        margin-top: 16px;
        backdrop-filter: blur(8px);
    }
    .db-stats-label {
        color: #999;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .db-stats-row {
        display: flex;
        gap: 24px;
        justify-content: space-around;
    }
    .db-stat-item { text-align: center; }
    .db-stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #b18cff 0%, #ff6bcb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    .db-stat-name {
        color: #888;
        font-size: 0.72rem;
        letter-spacing: 1px;
        margin-top: 4px;
        text-transform: uppercase;
    }
</style>
"""
st.markdown(_CSS_TEMPLATE.replace("__BG_B64__", BG_B64), unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
if LOGO_B64:
    logo_html = f"<img src='data:image/png;base64,{LOGO_B64}' class='logo-img'/>"
else:
    logo_html = "<div style='font-size:4rem;line-height:1;'>🏋️</div>"

st.markdown(f"""
<div class='logo-header'>
    {logo_html}
    <div>
        <div class='brand-title'>Gym Comparator AI</div>
        <div class='brand-sub'>Compare two gyms from Google Maps links — powered by LSTM + aspect categorization.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def load_store_from_cache(gmaps_link):
    key = link_to_key(gmaps_link)
    info_path = os.path.join(CACHE_DIR, f"{key}_info.json")
    reviews_path = os.path.join(CACHE_DIR, f"{key}_reviews.csv")
    if os.path.exists(info_path) and os.path.exists(reviews_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        reviews_df = pd.read_csv(reviews_path)
        return info, reviews_df
    return None, None


def get_or_scrape(gmaps_link, progress_text):
    info, reviews_df = load_store_from_cache(gmaps_link)
    if info is not None:
        return info, reviews_df, True

    progress_text.markdown(
        "<div style='background:linear-gradient(135deg,rgba(26,18,48,0.9),rgba(18,22,42,0.9));"
        "border:1px solid #3a2a55;border-radius:12px;padding:14px 18px;color:#d8b4fe;"
        "font-size:0.88rem;margin:8px 0;backdrop-filter:blur(6px);'>"
        "⏳ New link detected. Fetching data from Google Maps... (this may take a few seconds)"
        "</div>",
        unsafe_allow_html=True,
    )
    info, reviews_df = scrape_store_live(gmaps_link)
    if info is None or reviews_df.empty:
        return None, None, False

    save_store_cache_from_data(gmaps_link, info, reviews_df)
    return info, reviews_df, False


def extract_latlng_from_link(link):
    if not link:
        return None, None
    m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", link)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)", link)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def render_map(info):
    lat, lng = info.get("lat"), info.get("lng")
    if not lat or not lng:
        lat, lng = extract_latlng_from_link(info.get("gmaps_link", ""))

    if lat and lng:
        embed_url = f"https://www.google.com/maps?q={lat},{lng}&z=15&output=embed"
        components.html(
            f"""
            <iframe width="100%" height="180"
                style="border:1px solid #2a2a45;border-radius:14px;display:block;"
                loading="lazy" allowfullscreen
                referrerpolicy="no-referrer-when-downgrade"
                src="{embed_url}"></iframe>
            """,
            height=192,
        )
    elif info.get("gmaps_link"):
        st.link_button("📍 Open in Google Maps", info["gmaps_link"], use_container_width=True)


def _render_category_blocks(by_cat: dict, variant: str, max_per_cat: int = 3) -> str:
    """Render pros/cons grouped by category as HTML."""
    if not by_cat:
        return "<div style='color:#666;font-size:0.82rem;padding:6px 0;'>— No data —</div>"

    sorted_cats = sorted(by_cat.items(), key=lambda x: -len(x[1]))

    html_parts = []
    for category, sentences in sorted_cats:
        seen = set()
        unique = []
        for s in sentences:
            key = s.lower().strip()[:60]
            if key not in seen:
                seen.add(key)
                unique.append(s.strip())
            if len(unique) >= max_per_cat:
                break

        items_html = "".join(f"<li>{s}</li>" for s in unique)
        category_class = "category-name-cons" if variant == "cons" else ""
        list_class = "bullet-list cons" if variant == "cons" else "bullet-list"

        html_parts.append(
            f"<div class='category-block'>"
            f"<div class='category-name {category_class}'>{category}</div>"
            f"<ul class='{list_class}'>{items_html}</ul>"
            f"</div>"
        )

    return "".join(html_parts)


def render_card(info, analysis, label, variant):
    badge = "<span class='cache-badge'>⚡ CACHED</span>" if analysis.get("from_cache") else ""

    pros_html = _render_category_blocks(analysis.get("pros_by_category", {}), "pros")
    cons_html = _render_category_blocks(analysis.get("cons_by_category", {}), "cons")

    raw_url = info.get("photo_url")
    b64 = _fetch_photo_b64(raw_url) if raw_url else None
    if b64:
        photo_html = (
            f"<img src='data:image/jpeg;base64,{b64}' "
            f"style='width:100%;height:100%;object-fit:cover;border-radius:12px;'/>"
        )
    else:
        photo_html = (
            "<div style='background:#1a1a2e;border-radius:12px;height:100%;"
            "display:flex;align-items:center;justify-content:center;"
            "color:#555;font-size:0.85rem;'>No photo</div>"
        )

    html = (
        f"<div class='gym-card'>"
        f"<div class='gym-label-{variant}'>"
        f"<span>● {label} · {info['name']}</span>"
        f"{badge}"
        f"</div>"
        f"<div style='display:flex;gap:16px;margin-bottom:4px;flex-shrink:0;'>"
        f"<div style='flex:0 0 42%;height:150px;'>{photo_html}</div>"
        f"<div style='flex:1;display:flex;flex-direction:column;justify-content:center;'>"
        f"<div class='rating-label'>Google Rating</div>"
        f"<div class='rating-number'>{info.get('rating', '-')}</div>"
        f"<div class='rating-count'>({info.get('review_count', 0):,} reviews)</div>"
        f"</div>"
        f"</div>"
        f"<div class='sentiment-row'>"
        f"<div class='sentiment-item'>"
        f"<div class='sentiment-name'>Positive</div>"
        f"<div class='sentiment-value pos-text'>{analysis['positif_pct']}%</div>"
        f"<div class='sentiment-bar pos-bar'></div>"
        f"</div>"
        f"<div class='sentiment-item'>"
        f"<div class='sentiment-name'>Neutral</div>"
        f"<div class='sentiment-value net-text'>{analysis['netral_pct']}%</div>"
        f"<div class='sentiment-bar net-bar'></div>"
        f"</div>"
        f"<div class='sentiment-item'>"
        f"<div class='sentiment-name'>Negative</div>"
        f"<div class='sentiment-value neg-text'>{analysis['negatif_pct']}%</div>"
        f"<div class='sentiment-bar neg-bar'></div>"
        f"</div>"
        f"</div>"
        f"<div class='pros-cons-wrapper'>"
        f"<div class='section-header header-pros'>Pros</div>"
        f"{pros_html}"
        f"<div class='section-header header-cons'>Cons</div>"
        f"{cons_html}"
        f"</div>"
        f"</div>"
    )

    st.html(html)
    render_map(info)


# ============================================================
# SESSION STATE
# ============================================================
if "link_a_val" not in st.session_state:
    st.session_state.link_a_val = ""
if "link_b_val" not in st.session_state:
    st.session_state.link_b_val = ""


def _fill_link_a():
    picked = st.session_state.get("pick_a", "")
    if picked:
        st.session_state.link_a_val = picked


def _fill_link_b():
    picked = st.session_state.get("pick_b", "")
    if picked:
        st.session_state.link_b_val = picked


# ============================================================
# INPUT SECTION
# ============================================================
col1, col2, col3 = st.columns([3, 3, 1], gap="small")
with col1:
    st.text_input("Gym A", placeholder="https://maps.app.goo.gl/...", key="link_a_val")
with col2:
    st.text_input("Gym B", placeholder="https://maps.app.goo.gl/...", key="link_b_val")
with col3:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    submit = st.button("COMPARE", type="primary", use_container_width=True)


# ============================================================
# CACHE PICKER ("Battle Ready")
# ============================================================
cached_stores = load_cached_stores()

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

if cached_stores:
    def _label(s):
        rating = s.get("rating", "-")
        count = s.get("review_count", 0)
        return f"{s['name']}  ·  {rating}★  ·  {count} reviews"

    cached_stores_sorted = sorted(cached_stores, key=lambda s: s["name"].lower())

    url_to_label = {"": "— pick from roster —"}
    url_options = [""]
    for s in cached_stores_sorted:
        u = s["gmaps_link"]
        url_options.append(u)
        url_to_label[u] = _label(s)

    with st.expander(f"⚡ Battle Ready — {len(cached_stores)} gyms available", expanded=False):
        st.markdown(
            "<div class='cache-hint'>Pick a gym from the roster below — the link auto-fills. "
            "Cached gyms load instantly without re-scraping.</div>",
            unsafe_allow_html=True,
        )

        pc1, pc2 = st.columns(2, gap="small")
        with pc1:
            st.selectbox(
                "Fill Gym A",
                url_options,
                format_func=lambda u: url_to_label.get(u, u),
                key="pick_a",
                on_change=_fill_link_a,
            )
        with pc2:
            st.selectbox(
                "Fill Gym B",
                url_options,
                format_func=lambda u: url_to_label.get(u, u),
                key="pick_b",
                on_change=_fill_link_b,
            )

        picked_a = st.session_state.get("pick_a", "")
        picked_b = st.session_state.get("pick_b", "")

        if picked_a:
            st.markdown(
                f"<div class='cache-hint'>🔵 Gym A → <b>{url_to_label.get(picked_a, '?')}</b></div>",
                unsafe_allow_html=True,
            )
        if picked_b:
            st.markdown(
                f"<div class='cache-hint'>🟣 Gym B → <b>{url_to_label.get(picked_b, '?')}</b></div>",
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        "<div class='cache-empty'>📭 Roster is empty. "
        "Paste a Google Maps link above and click COMPARE to add a gym to the roster.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN LOGIC
# ============================================================
if submit:
    link_a_final = st.session_state.link_a_val.strip()
    link_b_final = st.session_state.link_b_val.strip()

    if not link_a_final or not link_b_final:
        st.warning("Please fill in both links first.")
        st.stop()

    progress = st.empty()

    with st.spinner("Fetching data..."):
        info_a, rev_a, cached_a = get_or_scrape(link_a_final, progress)
        info_b, rev_b, cached_b = get_or_scrape(link_b_final, progress)

    progress.empty()

    if info_a is None or info_b is None:
        st.error("Failed to fetch data for one of the gyms. Check the link or try again later.")
        st.stop()

    with st.spinner("Running LSTM sentiment analysis..."):
        analysis_a = analyze_reviews(rev_a.to_dict("records"))
        analysis_b = analyze_reviews(rev_b.to_dict("records"))

    analysis_a["kesimpulan"] = build_summary(analysis_a, info_a["name"])
    analysis_b["kesimpulan"] = build_summary(analysis_b, info_b["name"])

    with st.spinner("Building comparison..."):
        rekom = pick_winner(analysis_a, analysis_b, info_a["name"], info_b["name"])

    # ---------- SAVE TO DATABASE ----------
    try:
        gym_a_id = upsert_gym(info_a)
        analysis_a_id = save_analysis(gym_a_id, analysis_a)

        gym_b_id = upsert_gym(info_b)
        analysis_b_id = save_analysis(gym_b_id, analysis_b)

        save_comparison(gym_a_id, gym_b_id, analysis_a_id, analysis_b_id, rekom)
        print(f"[DB] Saved comparison: {info_a['name']} vs {info_b['name']}")
    except Exception as e:
        print(f"[DB] Failed to save: {e}")

    # ---------- RENDER CARDS ----------
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        render_card(info_a, analysis_a, "GYM A", "a")
    with col_b:
        render_card(info_b, analysis_b, "GYM B", "b")

    # ---------- SUMMARY & WINNER ----------
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    col_sum, col_rec = st.columns([3, 2], gap="medium")

    with col_sum:
        st.markdown(f"""
        <div class='kesimpulan-block'>
            <div class='section-header header-blue' style='margin-top:0;'>Strategic Summary</div>
            <div class='kesimpulan-text'>
                <span class='kesimpulan-store'>{info_a['name']}:</span>
                {analysis_a.get('kesimpulan', '-')}
            </div>
            <div class='kesimpulan-text'>
                <span class='kesimpulan-store'>{info_b['name']}:</span>
                {analysis_b.get('kesimpulan', '-')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_rec:
        winner = rekom.get("pemenang", "-")
        st.markdown(f"""
        <div class='winner-block'>
            <div class='winner-label'>Recommended Gym</div>
            <div class='winner-name'>{winner}</div>
            <div class='winner-reason'>{rekom.get('alasan', '-')}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# DATABASE STATS (always visible at bottom)
# ============================================================
try:
    stats = get_stats()
    if stats["total_gyms"] > 0:
        st.markdown(f"""
        <div class='db-stats'>
            <div class='db-stats-label'>📊 Database Overview</div>
            <div class='db-stats-row'>
                <div class='db-stat-item'>
                    <div class='db-stat-value'>{stats['total_gyms']}</div>
                    <div class='db-stat-name'>Gyms</div>
                </div>
                <div class='db-stat-item'>
                    <div class='db-stat-value'>{stats['total_analyses']}</div>
                    <div class='db-stat-name'>Analyses</div>
                </div>
                <div class='db-stat-item'>
                    <div class='db-stat-value'>{stats['total_comparisons']}</div>
                    <div class='db-stat-name'>Comparisons</div>
                </div>
                <div class='db-stat-item'>
                    <div class='db-stat-value'>{stats['avg_gym_rating']}</div>
                    <div class='db-stat-name'>Avg Rating</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    print(f"[DB] Failed to load stats: {e}")