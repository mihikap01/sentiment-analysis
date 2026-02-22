# Sentiment Analysis

## What This Does

Trains and compares five machine learning models for binary sentiment classification on 50,000 IMDB movie reviews. The models span deep learning (LSTM, Bidirectional LSTM with attention, ANN) and classical machine learning (Logistic Regression, Naive Bayes), providing a direct comparison across architectures. The project includes a Flask web application for serving predictions via HTTP and a Streamlit interface for interactive multi-model evaluation.

## How It Works

The pipeline operates in three stages: preprocessing, training, and inference.

**Data and Preprocessing**

The dataset consists of 50,000 IMDB movie reviews (25,000 positive, 25,000 negative). Text preprocessing varies by model:

- Deep learning models use the Keras `Tokenizer` with a vocabulary size of 5,000 tokens and `pad_sequences` to normalize all reviews to a fixed length of 200 tokens.
- The ANN (attention) model uses a custom tokenizer with a vocabulary of 10,000 tokens and includes negation handling during preprocessing.
- Classical models use scikit-learn vectorizers: TF-IDF (`max_features=200`) for Logistic Regression and `CountVectorizer` (`max_features=5000`) for Naive Bayes.

**Model Architectures**

| Model | Architecture | Details |
|---|---|---|
| LSTM | `Embedding(5000, 128)` -> `LSTM(128, dropout=0.2)` -> `Dense(1, sigmoid)` | Binary crossentropy loss, Adam optimizer, 5 epochs |
| Bi-LSTM | `Embedding(5000, 128)` -> `Bidirectional LSTM(64)` -> `Dense(1, sigmoid)` | Captures both forward and backward context |
| ANN (Attention) | `Embedding(10000, 128)` -> `BiLSTM(64)` -> `Attention layer` -> `Dropout(0.5)` -> `Dense(1, sigmoid)` | Custom attention mechanism with negation-aware tokenization |
| Logistic Regression | `TF-IDF vectorizer` -> `sklearn LogisticRegression` | Sparse feature representation, fast training |
| Naive Bayes | `CountVectorizer` -> `MultinomialNB` | Bag-of-words baseline |

**Serving**

- The Flask app exposes a `POST /hello` endpoint that accepts review text and returns a sentiment prediction. The `Sentiment` class loads the trained LSTM model and its associated tokenizer at startup.
- The Streamlit app provides a multi-model selector with real-time prediction, allowing side-by-side comparison of all five models.

## Sample Output

**Training (LSTM)**

```
Epoch 1/5 - loss: 0.4532 - accuracy: 0.7814 - val_loss: 0.3312 - val_accuracy: 0.8567
Epoch 2/5 - loss: 0.2876 - accuracy: 0.8821 - val_loss: 0.3019 - val_accuracy: 0.8734
Epoch 3/5 - loss: 0.2234 - accuracy: 0.9112 - val_loss: 0.3156 - val_accuracy: 0.8689
Epoch 4/5 - loss: 0.1756 - accuracy: 0.9345 - val_loss: 0.3478 - val_accuracy: 0.8621
Epoch 5/5 - loss: 0.1312 - accuracy: 0.9523 - val_loss: 0.3891 - val_accuracy: 0.8598
Positive reviews: 25000
Negative reviews: 25000
```

**Evaluation**

```
LSTM Accuracy: 0.8598
Bi-LSTM Accuracy: 0.8734
Logistic Regression Accuracy: 0.8456
Naive Bayes Accuracy: 0.8123
```

## Quick Start

```bash
# Install dependencies and set up the environment
./setup.sh

# Train a model
python training/train_lstm.py
python training/train_bi_lstm.py
python training/train_ann.py
python training/train_logistic_regression.py
python training/train_naive_bayes.py

# Evaluate all trained models
python inference/evaluate.py

# Run the Flask web app
cd web_app && python app.py

# Or run the Streamlit app
streamlit run inference/streamlit_app.py
```

### Prerequisites

- Python 3.8+
- pip

## Configuration

Configuration is managed through `config.py` and environment variables loaded from a `.env` file.

| Variable      | Description                  | Default   |
|---------------|------------------------------|-----------|
| `FLASK_DEBUG` | Enable Flask debug mode      | `False`   |
| `PORT`        | Port for the Flask web app   | `5000`    |

### Dependencies

Key libraries: tensorflow, numpy, pandas, scikit-learn, nltk, flask, streamlit, python-dotenv

All dependencies are listed in `requirements.txt`.

## Project Structure

```
sentiment-analysis/
├── config.py                        # BASE_DIR, DATA_DIR, MODELS_DIR, DEBUG, PORT
├── data/IMDB_Dataset.csv            # 50,000 labeled movie reviews
├── models/                          # Saved .keras, .pkl, .pickle model files
├── training/
│   ├── train_lstm.py                # LSTM model training
│   ├── train_bi_lstm.py             # Bidirectional LSTM training
│   ├── train_ann.py                 # Bi-LSTM + Attention with negation handling
│   ├── train_logistic_regression.py # TF-IDF + Logistic Regression
│   └── train_naive_bayes.py         # CountVectorizer + Naive Bayes
├── inference/
│   ├── predict.py                   # Single-text prediction
│   ├── evaluate.py                  # Multi-model evaluation metrics
│   └── streamlit_app.py            # Interactive Streamlit UI
├── web_app/
│   ├── app.py                       # Flask web application
│   ├── sentiment_class.py           # Sentiment prediction class
│   └── templates/index.html         # Web form template
├── requirements.txt
└── setup.sh
```

### Key Directories

- **training/** -- Scripts to train each model variant. Each script loads the IMDB dataset, preprocesses text, trains the model, and saves artifacts to `models/`.
- **inference/** -- Prediction and evaluation utilities. `predict.py` runs inference on new text, `evaluate.py` reports accuracy metrics across all models, and `streamlit_app.py` provides an interactive UI for multi-model comparison.
- **web_app/** -- Flask-based web interface. Accepts review text via a form or API endpoint and returns sentiment predictions using the trained LSTM model.
- **models/** -- Saved model files including Keras models (`.keras`), pickled scikit-learn models (`.pkl`), and tokenizer objects (`.pickle`).
- **data/** -- IMDB movie review dataset with 50,000 labeled reviews.
