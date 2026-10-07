"""
Fungsi cleaning teks. Dipakai baik saat training model (train_model.py)
maupun saat inference di aplikasi Streamlit (app/inference.py), supaya
teks training dan teks yang diprediksi diproses dengan cara yang sama persis.
"""

import re


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, hapus URL, hapus karakter non-alfabet, rapikan spasi."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                 # hanya huruf & spasi
    text = re.sub(r"\s+", " ", text).strip()                 # rapikan spasi ganda
    return text


def rating_to_label(rating: float) -> str:
    """Konversi rating bintang (1-5) menjadi label sentimen 3 kelas."""
    if rating >= 4:
        return "positif"
    elif rating == 3:
        return "netral"
    else:
        return "negatif"


# kata sambung yang sering menandai pergantian sentimen DALAM 1 kalimat
# (mis. "the equipment is great but membership is a nightmare") -> tanpa ini,
# split cuma berdasarkan titik/tanda baca nggak akan motong kalimat gabungan ini.
CONTRAST_WORDS = r"\b(but|however|although|though|yet|except|unfortunately)\b"


def split_sentences(text: str) -> list:
    """Pecah 1 review jadi beberapa 'kalimat' — dipisah tanda baca akhir kalimat
    (. ! ?) DAN kata sambung kontras (but/however/dll), supaya review campuran
    seperti "equipment is great but membership is a nightmare" dianalisis sebagai
    2 bagian terpisah, bukan 1 sentimen tunggal."""
    if not isinstance(text, str) or not text.strip():
        return []

    # split di tanda baca akhir kalimat dulu
    rough = re.split(r"(?<=[.!?])\s+", text.strip())

    parts = []
    for chunk in rough:
        # lalu split lagi tiap potongan di kata sambung kontras
        sub_parts = re.split(CONTRAST_WORDS, chunk, flags=re.IGNORECASE)
        parts.extend(sub_parts)

    # buang potongan kosong atau cuma sisa kata sambung/kepanjangan-nya
    return [p.strip() for p in parts if p.strip() and len(p.strip().split()) >= 3]
