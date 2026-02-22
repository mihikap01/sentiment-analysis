"""Evaluate all trained sentiment analysis models on the IMDB test dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def evaluate_model(model, X_test, y_test, is_neural_network=True):
    """Evaluate a model and return accuracy, precision, recall, and F1 score."""
    if is_neural_network:
        predictions = model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


def main():
    # Load pre-trained models
    lstm_model = load_model(str(MODELS_DIR / "reviews_model.keras"))
    bi_lstm_model = load_model(str(MODELS_DIR / "bi_lstm_model.keras"))
    ann_model = load_model(str(MODELS_DIR / "bi_lstm_attention_model.keras"))
    with open(MODELS_DIR / "log_reg_model.pkl", "rb") as f:
        log_reg_model = pickle.load(f)
    with open(MODELS_DIR / "naive_bayes_model.pkl", "rb") as f:
        naive_bayes_model = pickle.load(f)

    # Load tokenizer and vectorizer
    with open(MODELS_DIR / "tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Load test data
    test_df = pd.read_csv(DATA_DIR / "IMDB_Dataset.csv")
    X_test = test_df["review"]
    y_test = test_df["sentiment"].map({"positive": 1, "negative": 0}).values

    # Preprocess test data for LSTM models
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_sequences, maxlen=200)

    # Preprocess test data for Logistic Regression and Naive Bayes
    X_test_vectorized = vectorizer.transform(X_test)

    # Evaluate neural network models (LSTM, Bi-LSTM, ANN)
    for model, name in zip(
        [lstm_model, bi_lstm_model, ann_model], ["LSTM", "Bi-LSTM", "ANN"]
    ):
        accuracy, precision, recall, f1 = evaluate_model(model, X_test_padded, y_test)
        print(
            f"{name} - Accuracy: {accuracy}, Precision: {precision}, "
            f"Recall: {recall}, F1 Score: {f1}"
        )

    # Evaluate sklearn models (Logistic Regression, Naive Bayes)
    for model, name in zip(
        [log_reg_model, naive_bayes_model], ["Logistic Regression", "Naive Bayes"]
    ):
        accuracy, precision, recall, f1 = evaluate_model(
            model, X_test_vectorized, y_test, is_neural_network=False
        )
        print(
            f"{name} - Accuracy: {accuracy}, Precision: {precision}, "
            f"Recall: {recall}, F1 Score: {f1}"
        )


if __name__ == "__main__":
    main()
