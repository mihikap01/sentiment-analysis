"""Train a Logistic Regression model for sentiment analysis on IMDB reviews."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR

import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_and_save_model():
    """Train a logistic regression model and save it along with the vectorizer."""
    df = pd.read_csv(DATA_DIR / "IMDB_Dataset.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=200)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_tfidf, y_train)

    # Evaluate
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = log_reg.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    with open(MODELS_DIR / "log_reg_model.pkl", "wb") as f:
        pickle.dump(log_reg, f)
    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


if __name__ == "__main__":
    train_and_save_model()
