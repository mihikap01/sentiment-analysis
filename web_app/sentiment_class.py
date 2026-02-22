"""Sentiment analysis class for the Flask web application."""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Sentiment:
    def __init__(self):
        self.ann_model = load_model(
            str(MODELS_DIR / "bi_lstm_attention_model_with_negation_handling.keras")
        )
        with open(MODELS_DIR / "tokenizer.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict_sentiment(self, user_input):
        """Predict sentiment for a given review text.

        Args:
            user_input: A string containing the review text.

        Returns:
            True if sentiment is positive, False if negative.
        """
        seq = self.tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=200)
        prediction = self.ann_model.predict(padded_seq)
        if prediction[0] > 0.5:
            return True
        else:
            return False
