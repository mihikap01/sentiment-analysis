"""Train a Bidirectional LSTM model for sentiment analysis on IMDB reviews."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def main():
    # Load the dataset
    df = pd.read_csv(DATA_DIR / "IMDB_Dataset.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # Tokenization and Padding
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df["review"])
    sequences = tokenizer.texts_to_sequences(df["review"])
    data = pad_sequences(sequences, maxlen=200)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(
        data, df["sentiment"], test_size=0.2, random_state=42
    )

    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

    # Save the model
    model.save(str(MODELS_DIR / "bi_lstm_model.keras"))


if __name__ == "__main__":
    main()
