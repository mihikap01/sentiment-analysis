"""Train a Bi-LSTM with Attention model (with negation handling) for sentiment analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR

import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Bidirectional, Flatten, Dropout
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def custom_tokenize(texts):
    """Handle negations like 'not bad' by joining them with underscore."""
    updated_texts = []
    for text in texts:
        text = re.sub(r"\bnot\s+", "not_", text)
        updated_texts.append(text)
    return updated_texts


def attention_layer(inputs, units):
    """Attention mechanism layer."""
    attention_score = Dense(units, activation="tanh")(inputs)
    attention_weights = Dense(1, activation="softmax")(attention_score)
    context_vector = tf.matmul(attention_weights, inputs, transpose_a=True)
    context_vector = tf.squeeze(context_vector, -2)
    return context_vector


def main():
    # Load dataset
    df = pd.read_csv(DATA_DIR / "IMDB_Dataset.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # Custom tokenizer function to handle negations
    df["review"] = custom_tokenize(df["review"])

    # Data Preprocessing
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df["review"])
    sequences = tokenizer.texts_to_sequences(df["review"])
    data = pad_sequences(sequences, maxlen=200)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, df["sentiment"], test_size=0.2, random_state=42
    )

    # Model building
    input_layer = Input(shape=(200,))
    embedding_layer = Embedding(10000, 128, input_length=200)(input_layer)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    attention = attention_layer(lstm_layer, 128)
    dropout = Dropout(0.5)(attention)
    output_layer = Dense(1, activation="sigmoid")(Flatten()(dropout))
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Model summary
    model.summary()

    # Train the model
    history = model.fit(
        X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test)
    )

    # Save the model
    model.save(str(MODELS_DIR / "bi_lstm_attention_model_with_negation_handling.keras"))

    # Plotting training and validation error
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
