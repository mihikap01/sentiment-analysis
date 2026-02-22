"""Flask web application for sentiment analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEBUG, PORT

from flask import Flask, render_template, request, jsonify
from sentiment_class import Sentiment

app = Flask(__name__)

s = Sentiment()


@app.route("/")
def main_page():
    return render_template("index.html")


@app.route("/hello", methods=["GET", "POST"])
def get_sentiment():
    if request.method == "POST":
        review = request.form["sentiment"]
        output = str(s.predict_sentiment(review))
        return jsonify({"output": f"(The review is {output}"})


if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT)
