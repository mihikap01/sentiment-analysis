"""Streamlit web app for sentiment analysis with multiple model selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR

import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models
lstm_model = load_model(str(MODELS_DIR / "reviews_model.keras"))
bi_lstm_model = load_model(str(MODELS_DIR / "bi_lstm_model.keras"))
ann_model = load_model(str(MODELS_DIR / "bi_lstm_attention_model_with_negation_handling.keras"))
with open(MODELS_DIR / "log_reg_model.pkl", "rb") as f:
    log_reg_model = pickle.load(f)
with open(MODELS_DIR / "naive_bayes_model.pkl", "rb") as f:
    naive_bayes_model = pickle.load(f)

# Load tokenizer and vectorizer
with open(MODELS_DIR / "tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)
with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Model dictionary
models = {
    "LSTM": lstm_model,
    "Bi-LSTM": bi_lstm_model,
    "ANN": ann_model,
    "Logistic Regression": log_reg_model,
    "Naive Bayes": naive_bayes_model,
}

# Streamlit app
st.title("Sentiment Analysis Web App")
st.write("Enter a review and select a model to predict the sentiment")

# User input
user_input = st.text_area("Enter the review:")

# Model selection
model_option = st.selectbox("Choose the model for prediction:", models.keys())


# Predict function
def predict_sentiment(model, user_input, model_type):
    """Predict sentiment using the selected model."""
    if model_type in ["LSTM", "Bi-LSTM", "ANN"]:
        # Preprocess for LSTM-based models
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=200)
        prediction = model.predict(padded_seq)
        return "Positive" if prediction[0] > 0.5 else "Negative"
    elif model_type in ["Logistic Regression", "Naive Bayes"]:
        # Preprocess for Logistic Regression and Naive Bayes models
        processed_input = vectorizer.transform([user_input])
        prediction = model.predict(processed_input)
        return "Positive" if prediction[0] == 1 else "Negative"


# On button click
if st.button("Predict Sentiment"):
    prediction = predict_sentiment(models[model_option], user_input, model_option)
    st.write(f"Predicted Sentiment: {prediction}")
