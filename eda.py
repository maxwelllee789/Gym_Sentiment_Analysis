"""
EDA (Exploratory Data Analysis) — STANDALONE, bukan bagian dari alur running aplikasi.

Script ini TIDAK dipanggil oleh train_model.py atau app.py. Tujuannya cuma buat
menghasilkan grafik-grafik yang dipakai di laporan/slide presentasi final project,
sebagai bukti pemahaman karakteristik data sebelum masuk ke modeling.

Cara jalanin (sekali aja, manual):
    python eda.py

Input : data/processed/train.csv, val.csv, test.csv (hasil prepare_dataset.py)
Output: folder eda_output/ berisi beberapa file .png siap di-embed ke laporan
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")

# palet warna senada sama tampilan app (muted, elegan) biar konsisten di laporan
COLOR_POSITIF = "#8FAF8A"   # sage hijau muted
COLOR_NEGATIF = "#B37268"   # terracotta muted
COLOR_NETRAL = "#C7A468"    # ochre/amber muted
LABEL_COLORS = {"positif": COLOR_POSITIF, "negatif": COLOR_NEGATIF, "netral": COLOR_NETRAL}

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "nya",
    "saya", "kami", "juga", "ada", "tidak", "the", "and", "was", "is", "very",
    "for", "with", "this", "that", "were", "are", "a", "of", "to", "in", "it",
}

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11


def load_all_data() -> pd.DataFrame:
    parts = []
    for name in ("train.csv", "val.csv", "test.csv"):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            parts.append(pd.read_csv(path))
    df = pd.concat(parts, ignore_index=True)
    df["clean_text"] = df["clean_text"].fillna("")
    return df


def plot_rating_distribution(df: pd.DataFrame, out_dir: str):
    """Distribusi rating asli (1-5 bintang) sebelum jadi label sentimen."""
    plt.figure(figsize=(6, 4))
    counts = df["rating"].value_counts().sort_index()
    ax = sns.barplot(x=counts.index, y=counts.values, color="#c9b896")
    ax.set_title("Distribusi Rating Review (1-5 Bintang)")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Jumlah Review")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_rating_distribution.png"), dpi=150)
    plt.close()


def plot_label_distribution(df: pd.DataFrame, out_dir: str):
    """Distribusi label sentimen (positif/negatif/netral) -> nunjukin class imbalance."""
    plt.figure(figsize=(6, 4))
    counts = df["label"].value_counts()
    colors = [LABEL_COLORS.get(lbl, "#999999") for lbl in counts.index]
    ax = sns.barplot(x=counts.index, y=counts.values, palette=colors)
    ax.set_title("Distribusi Label Sentimen (Class Imbalance)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Jumlah Review")
    total = counts.sum()
    for i, v in enumerate(counts.values):
        pct = v / total * 100
        ax.text(i, v, f"{v:,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_label_distribution.png"), dpi=150)
    plt.close()


def plot_text_length_distribution(df: pd.DataFrame, out_dir: str):
    """Distribusi panjang review (jumlah kata), dipecah per label sentimen."""
    df = df.copy()
    df["word_count"] = df["clean_text"].str.split().apply(len)

    plt.figure(figsize=(7, 4))
    for label, color in LABEL_COLORS.items():
        subset = df[df["label"] == label]["word_count"]
        sns.kdeplot(subset, label=label, color=color, fill=True, alpha=0.15)
    plt.title("Distribusi Panjang Review (Jumlah Kata) per Label")
    plt.xlabel("Jumlah Kata")
    plt.ylabel("Densitas")
    plt.legend()
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_text_length_distribution.png"), dpi=150)
    plt.close()


def get_top_words(texts: pd.Series, top_n: int = 15) -> list:
    words = []
    for t in texts:
        for w in str(t).split():
            if len(w) > 3 and w not in STOPWORDS:
                words.append(w)
    return Counter(words).most_common(top_n)


def plot_top_words(df: pd.DataFrame, label: str, out_dir: str, filename: str):
    """Top-15 kata paling sering muncul untuk 1 kelas label tertentu."""
    texts = df[df["label"] == label]["clean_text"]
    top_words = get_top_words(texts)
    if not top_words:
        return

    words, counts = zip(*top_words)
    plt.figure(figsize=(7, 5))
    color = LABEL_COLORS.get(label, "#999999")
    sns.barplot(x=list(counts), y=list(words), color=color)
    plt.title(f"15 Kata Paling Sering Muncul — Review {label.capitalize()}")
    plt.xlabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()


def plot_wordcloud(df: pd.DataFrame, label: str, out_dir: str, filename: str):
    texts = df[df["label"] == label]["clean_text"]
    combined_text = " ".join(texts)
    if not combined_text.strip():
        return

    wc = WordCloud(
        width=900, height=500, background_color="#14131a",
        colormap="YlOrBr" if label != "negatif" else "OrRd",
        stopwords=STOPWORDS,
    ).generate(combined_text)

    plt.figure(figsize=(9, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — Review {label.capitalize()}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading data...")
    df = load_all_data()
    print(f"Total baris dianalisis: {len(df)}")

    print("Membuat grafik distribusi rating...")
    plot_rating_distribution(df, OUT_DIR)

    print("Membuat grafik distribusi label (class imbalance)...")
    plot_label_distribution(df, OUT_DIR)

    print("Membuat grafik distribusi panjang teks...")
    plot_text_length_distribution(df, OUT_DIR)

    print("Membuat grafik top words per label...")
    plot_top_words(df, "positif", OUT_DIR, "04_top_words_positif.png")
    plot_top_words(df, "negatif", OUT_DIR, "05_top_words_negatif.png")
    plot_top_words(df, "netral", OUT_DIR, "06_top_words_netral.png")

    print("Membuat word cloud per label...")
    plot_wordcloud(df, "positif", OUT_DIR, "07_wordcloud_positif.png")
    plot_wordcloud(df, "negatif", OUT_DIR, "08_wordcloud_negatif.png")

    print(f"\nSelesai. Semua grafik tersimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
