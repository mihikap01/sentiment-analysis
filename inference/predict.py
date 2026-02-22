"""Run sentiment prediction on a single text input using the trained attention model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_resources():
    """Load the trained model, tokenizer, and vectorizer."""
    ann_model = load_model(str(MODELS_DIR / "bi_lstm_attention_model_with_negation_handling.keras"))

    with open(MODELS_DIR / "tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return ann_model, tokenizer, vectorizer


def predict_sentiment(model, tokenizer, user_input):
    """Predict sentiment for a given text input."""
    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=200)
    prediction = model.predict(padded_seq)
    if prediction[0] > 0.5:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    ann_model, tokenizer, vectorizer = load_resources()

    user_input = "The prequel was good last time but this time it was a snooze fest"
    result = predict_sentiment(ann_model, tokenizer, user_input)
    print(result)
