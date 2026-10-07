"""
Scrape review + info toko dari Google Maps pakai Apify.
"""

import argparse
import hashlib
import json
import os

import pandas as pd
from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
MAX_REVIEWS_PER_STORE = 500
MAX_REVIEWS_LIVE = 100
ACTOR_ID = "compass/Google-Maps-Reviews-Scraper"


def link_to_key(link: str) -> str:
    return hashlib.md5(link.strip().lower().encode()).hexdigest()[:10]


def extract_dataset_id(run):
    if isinstance(run, dict):
        return run.get("defaultDatasetId") or run.get("default_dataset_id")
    return getattr(run, "default_dataset_id", getattr(run, "defaultDatasetId", None))


def _get_client() -> ApifyClient:
    token = os.environ.get("APIFY_API_TOKEN") or os.environ.get("APIFY_TOKEN")
    if not token:
        raise ValueError("APIFY_API_TOKEN tidak ditemukan. Isi file .env dulu.")
    return ApifyClient(token)


def _extract_photo_url(first: dict):
    """Coba semua kemungkinan field foto dari Apify actor."""
    PHOTO_KEYS = [
        "imageUrl", "photoUrl", "photo", "image", "mainImage",
        "thumbnailUrl", "thumbnail", "mainPhoto", "heroImage",
        "image_url", "photo_url", "reviewImageUrl", "photoUrl1",
    ]
    for k in PHOTO_KEYS:
        v = first.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v

    # Kalau field-nya list
    for k in ["imageUrls", "photos", "images", "photoUrls"]:
        v = first.get(k)
        if isinstance(v, list) and v:
            for item in v:
                if isinstance(item, str) and item.startswith("http"):
                    return item
                if isinstance(item, dict):
                    for ik in ("url", "imageUrl", "src"):
                        if isinstance(item.get(ik), str):
                            return item[ik]
    return None


def _items_to_store_info(gmaps_link: str, items: list) -> dict:
    first = items[0]

    photo_url = _extract_photo_url(first)

    location = first.get("location") or first.get("coordinates") or {}
    lat = (
        location.get("lat") or location.get("latitude")
        or first.get("latitude") or first.get("lat")
    )
    lng = (
        location.get("lng") or location.get("longitude")
        or first.get("longitude") or first.get("lng")
    )

    name = first.get("title") or first.get("name") or "Nama tidak tersedia"

    print(f"[STORE_INFO] {name}")
    print(f"[STORE_INFO]   photo fields: {[k for k in first.keys() if 'photo' in k.lower() or 'image' in k.lower()]}")
    print(f"[STORE_INFO]   photo_url   : {str(photo_url)[:100]}")

    return {
        "gmaps_link": gmaps_link,
        "name": name,
        "address": first.get("address") or "-",
        "rating": first.get("totalScore") or first.get("rating") or 0,
        "review_count": first.get("reviewsCount") or first.get("numberOfReviews") or 0,
        "photo_url": photo_url,
        "lat": lat,
        "lng": lng,
        "scraped_at": pd.Timestamp.now().isoformat(),
    }


def _items_to_reviews_df(items: list) -> pd.DataFrame:
    return pd.DataFrame([
        {"text": it.get("text") or it.get("reviewText"),
         "rating": it.get("stars") or it.get("rating")}
        for it in items
        if it.get("text") or it.get("reviewText")
    ])


def scrape_store(client: ApifyClient, gmaps_link: str, max_reviews: int = MAX_REVIEWS_PER_STORE) -> list:
    run_input = {
        "startUrls": [{"url": gmaps_link}],
        "maxReviews": max_reviews,
        "language": "id",
    }
    run = client.actor(ACTOR_ID).call(run_input=run_input)
    dataset_id = extract_dataset_id(run)
    return list(client.dataset(dataset_id).iterate_items())


def scrape_store_live(gmaps_link: str, max_reviews: int = MAX_REVIEWS_LIVE):
    client = _get_client()
    items = scrape_store(client, gmaps_link, max_reviews=max_reviews)
    if not items:
        return None, pd.DataFrame()
    return _items_to_store_info(gmaps_link, items), _items_to_reviews_df(items)


def save_store_cache_from_data(gmaps_link: str, store_info: dict, reviews_df: pd.DataFrame):
    key = link_to_key(gmaps_link)
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(os.path.join(CACHE_DIR, f"{key}_info.json"), "w", encoding="utf-8") as f:
        json.dump(store_info, f, indent=2, ensure_ascii=False)
    reviews_df.to_csv(os.path.join(CACHE_DIR, f"{key}_reviews.csv"), index=False)

    _add_to_store_index(store_info)
    print(f"Cache tersimpan: {key} ({store_info['name']}) — {len(reviews_df)} review")


def _add_to_store_index(store_info: dict):
    index_path = os.path.join(CACHE_DIR, "store_index.json")
    existing = []
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            existing = json.load(f)

    existing_links = {e["gmaps_link"] for e in existing}
    if store_info["gmaps_link"] not in existing_links:
        existing.append(store_info)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


def read_links_file(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--links-file", help="Path ke file .txt berisi link gmaps, 1 per baris")
    parser.add_argument("--links", nargs="+", help="Link gmaps langsung, dipisah spasi")
    args = parser.parse_args()

    if not args.links_file and not args.links:
        raise SystemExit("Isi salah satu: --links-file atau --links")

    links = read_links_file(args.links_file) if args.links_file else args.links
    print(f"Total toko yang akan di-scrape: {len(links)}")

    client = _get_client()

    for i, link in enumerate(links, 1):
        print(f"[{i}/{len(links)}] Scraping: {link}")
        try:
            items = scrape_store(client, link, max_reviews=MAX_REVIEWS_PER_STORE)
            if not items:
                print(f"Tidak ada data untuk {link}, skip.")
                continue
            store_info = _items_to_store_info(link, items)
            reviews_df = _items_to_reviews_df(items)
            save_store_cache_from_data(link, store_info, reviews_df)
        except Exception as e:
            print(f"Gagal scrape {link}: {e}")


if __name__ == "__main__":
    main()