import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Config & Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MAX_WORDS = 10000 
MAX_LEN = 120 
EMBED_DIM = 128 
BATCH_SIZE = 64
EPOCHS = 15  

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Data Processed
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Penanganan teks NaN / kosong
    train_df['clean_text'] = train_df['clean_text'].fillna('')
    val_df['clean_text'] = val_df['clean_text'].fillna('')
    test_df['clean_text'] = test_df['clean_text'].fillna('')

    # 2. Encode Label (String -> Integer -> One-Hot)
    le = LabelEncoder()
    y_train_num = le.fit_transform(train_df['label'])
    y_val_num = le.transform(val_df['label'])
    y_test_num = le.transform(test_df['label'])

    num_classes = len(le.classes_)
    y_train = to_categorical(y_train_num, num_classes=num_classes)
    y_val = to_categorical(y_val_num, num_classes=num_classes)
    y_test = to_categorical(y_test_num, num_classes=num_classes)

    # 3. Hitung Class Weights (Pengganti Oversampling Duplikasi)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_num),
        y=y_train_num
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n--- Class Weights Disetting ---")
    for cls_idx, weight in class_weight_dict.items():
        print(f"Kelas '{le.classes_[cls_idx]}': {weight:.4f}")

    # 4. Tokenization & Sequence Padding (padding='post', truncating='post')
    print("\nTokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['clean_text'])

    X_train_seq = tokenizer.texts_to_sequences(train_df['clean_text'])
    X_val_seq = tokenizer.texts_to_sequences(val_df['clean_text'])
    X_test_seq = tokenizer.texts_to_sequences(test_df['clean_text'])

    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_val = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # 5. Build Arsitektur Bidirectional LSTM
    print("\nBuilding Bidirectional LSTM model...")
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, dropout=0.2, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    # 6. Training Model dengan EarlyStopping + Class Weights
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,  # Penyeimbang kelas tanpa menggandakan data
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Evaluasi Akhir pada Test Set
    print("\n================ EVALUASI TEST SET ================")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test_num, y_pred, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_num, y_pred))

    # 8. Simpan Model & Artifacts
    model_path = os.path.join(MODEL_DIR, "sentiment_lstm_model.keras")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    print(f"\nModel & artifacts berhasil disimpan di folder: {MODEL_DIR}")

if __name__ == "__main__":
    main()