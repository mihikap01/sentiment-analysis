"""Train a Naive Bayes model for sentiment analysis on IMDB reviews."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def main():
    # Load the dataset
    df = pd.read_csv(DATA_DIR / "IMDB_Dataset.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    # Vectorize the text
    vectorizer = CountVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = nb_model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Save the model and vectorizer
    with open(MODELS_DIR / "naive_bayes_model.pkl", "wb") as f:
        pickle.dump(nb_model, f)
    with open(MODELS_DIR / "vectorizer_nb.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


if __name__ == "__main__":
    main()
