import pytest
import sys
import os

# Memastikan folder utama bisa di-import oleh pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dari file app/inference.py
from app.inference import predict_sentiment

def test_predict_sentiment_positive():
    # Arrange
    teks_positif = "Tempatnya sangat bersih, AC dingin, dan alat gym sangat lengkap!"
    
    # Act
    hasil = predict_sentiment(teks_positif)
    
    # Assert
    # Jika output lu berupa teks:
    assert str(hasil).lower() in ["positif", "positive"]
    
    # ⚠️ JIKA OUTPUT LU BERUPA ANGKA (misal 1 untuk positif, 0 untuk negatif):
    # Hapus baris assert di atas, ganti jadi:
    # assert hasil == 1

def test_predict_sentiment_negative():
    teks_negatif = "Lantainya kotor, banyak alat yang rusak dan pelayanannya lambat."
    hasil = predict_sentiment(teks_negatif)
    
    # Jika output lu berupa teks:
    assert str(hasil).lower() in ["negatif", "negative"]
    
    # ⚠️ JIKA OUTPUT LU BERUPA ANGKA (misal 0 untuk negatif):
    # assert hasil == 0
