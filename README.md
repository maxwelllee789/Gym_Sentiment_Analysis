# Reputation Detection & Business Insight — Final Project

Analisa & bandingkan reputational risk 2 tempat kebugaran (gym/fitness center) berdasarkan
review Google Maps, menggunakan model Deep Learning (LSTM) untuk klasifikasi sentimen 3
kelas (Positif / Negatif / Netral), lalu ditampilkan sebagai perbandingan side-by-side di
Streamlit.

## Struktur Folder 

```
final_project/
├── requirements.txt
├── README.md
├── eda.py # STANDALONE, bukan bagian alur running app — lihat bagian EDA di bawah
├── data/
│ ├── raw/ # taruh gym_reviews_NY_clean.csv & gym_reviews_VT_clean.csv di sini
│ └── prepare_dataset.py # load CSV -> cleaning -> label -> split -> CSV bersih
├── preprocessing.py # cleaning teks + split_sentences (dipakai training & app)
├── train_model.py # training LSTM (oversampling) + tokenizer + evaluasi
├── scrape_reviews.py # scraping toko demo pakai Apify -> CSV cache
├── model/ # output: model .keras, tokenizer .pkl, label encoder
├── data/cache/ # hasil scraping toko-toko (link -> data)
└── app/
├── app.py # Streamlit utama
├── inference.py # load model + predict per kalimat + agregasi
└── recommendation.py # logic rule-based rekomendasi & kesimpulan
```

## Sumber Dataset

`gym_reviews_NY_clean.csv` + `gym_reviews_VT_clean.csv`: subset UCSD Google Local Reviews
(New York + Vermont) yang sudah difilter manual khusus kategori kebugaran (gym, fitness,
yoga, pilates, crossfit, martial arts, dll — termasuk konsep seperti Hyrox). Total ~26.179
review dari ~3.826 tempat kebugaran, bahasa Inggris. Taruh kedua file ini di `data/raw/`
sebelum menjalankan `prepare_dataset.py`.

## Alur Kerja (urutan menjalankan)

1. Taruh `gym_reviews_NY_clean.csv` dan `gym_reviews_VT_clean.csv` di `data/raw/`
2. `python data/prepare_dataset.py` — gabungkan kedua file, cleaning teks, generate label
   dari rating, split train/val/test
3. `python train_model.py` — training LSTM (dengan oversampling kelas minoritas di data
   training untuk mengatasi imbalance), simpan model (`model/sentiment_lstm_model.keras`) + tokenizer
4. Isi `.env` dengan `APIFY_API_TOKEN` (lihat `.env.example`)
5. `streamlit run app/app.py` — jalankan aplikasi. User paste link Google Maps toko A &
   B langsung → kalau link itu belum pernah discrape, app otomatis scrape live saat itu
   juga (butuh beberapa detik) dan menyimpannya ke `data/cache/`; kalau link sudah pernah
   dipakai sebelumnya, hasilnya langsung tampil dari cache (instan, tidak scrape ulang).
   Toko yang pernah dicari juga otomatis muncul di dropdown untuk dipilih ulang.

   (Opsional) Untuk menyiapkan beberapa toko ke cache sekaligus di awal — misalnya
   sebelum presentasi, biar tidak perlu menunggu scraping saat demo — bisa jalankan
   `python scrape_reviews.py --links-file links.txt` (1 link per baris) terlebih dahulu.

## EDA (Exploratory Data Analysis) — opsional, untuk laporan saja

`eda.py` **bukan bagian dari alur running aplikasi** — tidak dipanggil oleh `train_model.py`
maupun `app.py`. Ini script terpisah yang dijalankan manual, sekali, setelah `prepare_dataset.py`
selesai, untuk menghasilkan grafik yang dipakai di laporan/slide presentasi final project: