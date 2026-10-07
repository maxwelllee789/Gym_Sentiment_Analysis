"""
Translate dataset (train/val/test.csv hasil prepare_dataset.py) dari bahasa
Inggris ke Indonesia, pakai Gemini API (Google AI Studio) -- GRATIS (free tier,
tanpa kartu kredit), tapi rate-limited (dibatasi jumlah request, bukan karakter).

Dijalankan SETELAH prepare_dataset.py, SEBELUM train_model.py:
    python data/prepare_dataset.py
    python data/translate_dataset.py   <-- script ini
    python train_model.py

Cara kerja:
- Backup file asli (bahasa Inggris) disimpan sekali ke *_en_backup.csv, supaya
  tidak hilang kalau mau dibandingkan/diulang.
- Teks di-translate per BATCH (beberapa baris sekaligus dalam 1 request) biar
  nggak terlalu banyak API call (rate limit Gemini dihitung per request/menit).
- RESUMABLE: progress disimpan tiap beberapa batch ke file sementara. Kalau
  script berhenti di tengah jalan (error/limit/koneksi putus), tinggal jalankan
  ulang -- baris yang sudah diterjemahkan tidak diulang.
- Setelah selesai 1 split, kolom 'clean_text' di train.csv/val.csv/test.csv
  DITIMPA dengan hasil terjemahan (dibersihkan ulang pakai clean_text() yang
  sama), karena ini kolom yang dipakai train_model.py untuk training.
"""

import os
import sys
import time

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import clean_text  # noqa: E402

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "processed")
SPLITS = ["train", "val", "test"]
BATCH_SIZE = 20          # jumlah teks per 1 API call
SAVE_EVERY = 10          # simpan progress tiap 10 batch (= 200 baris)
MAX_RETRIES = 5
MODEL_NAME = "gemini-2.5-flash"  # cepat & murah, cocok buat translate massal


def get_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY tidak ditemukan. Isi file .env dulu (lihat .env.example).")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def translate_batch(model, texts: list) -> list:
    """Translate beberapa teks sekaligus dalam 1 API call. Teks dikirim sebagai
    list bernomor, hasil di-parse balik berdasarkan nomornya."""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = (
        "Terjemahkan setiap baris review gym berikut ke Bahasa Indonesia yang natural "
        "(bukan terjemahan kaku kata-per-kata). Balas HANYA dengan daftar bernomor yang "
        "formatnya sama persis (nomor diikuti titik, lalu hasil terjemahan), tanpa teks "
        "tambahan apapun:\n\n" + numbered
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            lines = [l.strip() for l in response.text.strip().split("\n") if l.strip()]
            results = {}
            for line in lines:
                if ". " in line:
                    num_str, translated = line.split(". ", 1)
                    if num_str.strip().isdigit():
                        results[int(num_str.strip())] = translated.strip()

            return [results.get(i + 1, texts[i]) for i in range(len(texts))]

        except Exception as e:
            wait = 2 ** attempt
            print(f"  Gagal (percobaan {attempt + 1}/{MAX_RETRIES}): {e} -> tunggu {wait}s")
            time.sleep(wait)

    print("  Gagal setelah semua percobaan, pakai teks asli (tidak diterjemahkan).")
    return texts


def translate_split(model, split: str):
    path = os.path.join(DATA_DIR, f"{split}.csv")
    backup_path = os.path.join(DATA_DIR, f"{split}_en_backup.csv")
    progress_path = os.path.join(DATA_DIR, f"{split}_translating_progress.csv")

    df = pd.read_csv(path)

    if not os.path.exists(backup_path):
        df.to_csv(backup_path, index=False)
        print(f"Backup bahasa Inggris disimpan: {backup_path}")

    if os.path.exists(progress_path):
        df = pd.read_csv(progress_path)
        print(f"Melanjutkan progress sebelumnya dari {progress_path}")

    if "translated" not in df.columns:
        df["translated"] = False

    todo_idx = df[~df["translated"]].index.tolist()
    print(f"[{split}] Total baris: {len(df)}, sudah diterjemahkan: {len(df) - len(todo_idx)}, sisa: {len(todo_idx)}")

    for batch_num, start in enumerate(range(0, len(todo_idx), BATCH_SIZE)):
        batch_idx = todo_idx[start:start + BATCH_SIZE]
        texts = df.loc[batch_idx, "text"].astype(str).tolist()

        translated = translate_batch(model, texts)
        df.loc[batch_idx, "clean_text"] = [clean_text(t) for t in translated]
        df.loc[batch_idx, "translated"] = True

        done = start + len(batch_idx)
        print(f"[{split}] {done}/{len(todo_idx)} baris diterjemahkan...")

        if (batch_num + 1) % SAVE_EVERY == 0:
            df.to_csv(progress_path, index=False)
            print(f"[{split}] Progress disimpan ({done}/{len(todo_idx)})")

    df = df.drop(columns=["translated"])
    df.to_csv(path, index=False)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print(f"[{split}] SELESAI. Disimpan ke {path}")


def main():
    model = get_model()
    for split in SPLITS:
        print(f"\n=== Translating {split}.csv ===")
        translate_split(model, split)
    print("\nSemua split selesai diterjemahkan ke Bahasa Indonesia.")


if __name__ == "__main__":
    main()