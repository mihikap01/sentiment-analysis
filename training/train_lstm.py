"""Train an LSTM model for sentiment analysis on IMDB reviews."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle


def main():
    # Load the dataset
    df = pd.read_csv(DATA_DIR / "IMDB_Dataset.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # Tokenize text
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df["review"])
    sequences = tokenizer.texts_to_sequences(df["review"])
    data = pad_sequences(sequences, maxlen=200)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data, df["sentiment"].values, test_size=0.2, random_state=42
    )

    # Build the model
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=200))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))

    # Save the model and tokenizer
    model.save(str(MODELS_DIR / "reviews_model.keras"))
    with open(MODELS_DIR / "tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Positive reviews: {len(df[df['sentiment'] == 1])}")
    print(f"Negative reviews: {len(df[df['sentiment'] == 0])}")


if __name__ == "__main__":
    main()
