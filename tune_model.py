"""
Hyperparameter tuning for LSTM sentiment model.
Tests multiple combinations of embedding dim, LSTM units, dropout, and learning rate.
Logs results to CSV for reporting.
"""

import os
import pickle
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULT_DIR = os.path.join(BASE_DIR, "tuning_results")
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_WORDS = 10000
MAX_LEN = 120
BATCH_SIZE = 64
EPOCHS = 10


# ============================================================
# HYPERPARAMETER GRID
# ============================================================
PARAM_GRID = {
    "embed_dim":     [64, 128],
    "lstm_units":    [32, 64, 128],
    "dropout":       [0.2, 0.4],
    "learning_rate": [0.001, 0.0005],
}


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    for df in (train_df, val_df):
        df["clean_text"] = df["clean_text"].fillna("")
    return train_df, val_df


def build_model(embed_dim, lstm_units, dropout, num_classes):
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=embed_dim, input_length=MAX_LEN),
        SpatialDropout1D(dropout),
        Bidirectional(LSTM(lstm_units, dropout=dropout)),
        Dropout(dropout + 0.1),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def main():
    print("=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    train_df, val_df = load_data()

    # Encode
    le = LabelEncoder()
    y_train_num = le.fit_transform(train_df["label"])
    y_val_num = le.transform(val_df["label"])
    num_classes = len(le.classes_)

    y_train = to_categorical(y_train_num, num_classes=num_classes)
    y_val = to_categorical(y_val_num, num_classes=num_classes)

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_num),
        y=y_train_num,
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Tokenize once (biar cepet)
    print("\nTokenizing...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_df["clean_text"]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(val_df["clean_text"]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )

    # Generate semua kombinasi
    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*PARAM_GRID.values()))

    print(f"\nTotal kombinasi: {len(combinations)}")
    print("=" * 70)

    results = []

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        print(f"\n[{i}/{len(combinations)}] {params}")
        print("-" * 70)

        # Build model
        model = build_model(
            embed_dim=params["embed_dim"],
            lstm_units=params["lstm_units"],
            dropout=params["dropout"],
            num_classes=num_classes,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            metrics=["accuracy"],
        )

        # Train
        early_stop = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=0,
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=[early_stop],
            verbose=0,
        )

        # Evaluate
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc = accuracy_score(y_val_num, y_pred)
        f1_macro = f1_score(y_val_num, y_pred, average="macro")
        best_epoch = len(history.history["loss"])
        best_val_loss = min(history.history["val_loss"])

        print(f"  val_accuracy  : {acc:.4f}")
        print(f"  val_f1_macro  : {f1_macro:.4f}")
        print(f"  best_val_loss : {best_val_loss:.4f}")
        print(f"  epochs_ran    : {best_epoch}")

        results.append({
            **params,
            "val_accuracy": round(acc, 4),
            "val_f1_macro": round(f1_macro, 4),
            "best_val_loss": round(best_val_loss, 4),
            "epochs_ran": best_epoch,
        })

    # Save results
    results_df = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False)
    csv_path = os.path.join(RESULT_DIR, "tuning_results.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("HASIL TUNING (diurutkan by F1-macro)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print()
    print(f"Hasil lengkap disimpan di: {csv_path}")

    # Best params
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    print(f"  embed_dim     : {int(best['embed_dim'])}")
    print(f"  lstm_units    : {int(best['lstm_units'])}")
    print(f"  dropout       : {best['dropout']}")
    print(f"  learning_rate : {best['learning_rate']}")
    print(f"  val_f1_macro  : {best['val_f1_macro']}")


if __name__ == "__main__":
    main()