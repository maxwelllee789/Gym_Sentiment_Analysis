"""
Load gym_reviews_NY_clean.csv + gym_reviews_VT_clean.csv (sudah difilter manual
khusus gym/fitness center oleh user dari data UCSD Google Local Reviews)
-> cleaning teks -> generate label sentimen -> split train/val/test -> simpan CSV.

Taruh kedua file CSV itu di data/raw/ sebelum menjalankan script ini.
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import clean_text, rating_to_label  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
RAW_FILES = ["gym_reviews_NY_clean.csv", "gym_reviews_VT_clean.csv"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")


def load_raw() -> pd.DataFrame:
    """Gabungkan semua file CSV di RAW_FILES yang ada di data/raw/."""
    parts = []
    for filename in RAW_FILES:
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            print(f"PERINGATAN: {filename} tidak ditemukan di data/raw/, dilewati.")
            continue
        df = pd.read_csv(path)
        parts.append(df)

    if not parts:
        raise SystemExit(
            f"Tidak ada file ditemukan di {RAW_DIR}. "
            f"Taruh {' dan '.join(RAW_FILES)} di sana dulu."
        )
    return pd.concat(parts, ignore_index=True)


def main():
    print("Loading raw data (gym reviews NY + VT)...")
    df = load_raw()
    print(f"Total baris ter-load: {len(df)}")

    print("Cleaning teks & generate label...")
    df["clean_text"] = df["text"].apply(clean_text)
    df["label"] = df["rating"].apply(rating_to_label)

    # buang baris yang teksnya kosong setelah cleaning (mis. cuma emoji/simbol)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    print("Distribusi label:")
    print(df["label"].value_counts())

    # split 70:15:15
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print(f"Selesai. train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Disimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
