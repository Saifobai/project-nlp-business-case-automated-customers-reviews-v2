import os
import json
import re
import threading
from collections import Counter

import torch
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob

# -------------------------
# Config / paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_DIR = os.path.join(BASE_DIR, "sentiment_model")
CLUSTERING_PKL = os.path.join(BASE_DIR, "clustering.pkl")
REVIEWS_FILE = os.path.join(BASE_DIR, "reviews.json")
SUMMARIZER_MODEL = "google/flan-t5-small"

app = Flask(__name__)
CORS(app)

write_lock = threading.Lock()


# -------------------------
# Helpers
# -------------------------
def cpu_joblib_load(path):
    """Load joblib objects saved with GPU tensors on CPU."""
    orig = torch.load

    def _torch_load_cpu(*args, **kwargs):
        kwargs["map_location"] = torch.device("cpu")
        return orig(*args, **kwargs)

    torch.load = _torch_load_cpu
    try:
        obj = joblib.load(path)
    finally:
        torch.load = orig
    return obj


def save_reviews_file(reviews):
    with write_lock:
        with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)


def load_reviews_file():
    if not os.path.exists(REVIEWS_FILE):
        # Seed with 5 sample reviews including genres and ratings
        initial = [
            {
                "id": 1,
                "movie": "Inception",
                "text": "Mind-bending and amazing!",
                "s": "positive",
                "genre": "Sci-Fi",
                "rating": 9,
            },
            {
                "id": 2,
                "movie": "The Room",
                "text": "So bad it's funny",
                "s": "neutral",
                "genre": "Drama",
                "rating": 3,
            },
            {
                "id": 3,
                "movie": "Cats",
                "text": "Awful movie, waste of time",
                "s": "negative",
                "genre": "Musical",
                "rating": 2,
            },
            {
                "id": 4,
                "movie": "The Dark Knight",
                "text": "Heath Ledger was phenomenal as the Joker.",
                "s": "positive",
                "genre": "Action",
                "rating": 10,
            },
            {
                "id": 5,
                "movie": "Titanic",
                "text": "An emotional rollercoaster, but a bit too long.",
                "s": "neutral",
                "genre": "Romance",
                "rating": 7,
            },
        ]
        save_reviews_file(initial)
        return initial
    with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_wordcloud_from_reviews(reviews, top_k=12):
    text = " ".join(r["text"] for r in reviews)
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stopwords = {
        "the",
        "and",
        "this",
        "that",
        "with",
        "have",
        "has",
        "was",
        "for",
        "but",
        "not",
        "it's",
        "its",
        "movie",
    }
    tokens = [t for t in tokens if t not in stopwords]
    counts = Counter(tokens).most_common(top_k)
    if not counts:
        return []
    max_count = counts[0][1]
    min_count = counts[-1][1]

    def size(c):
        if max_count == min_count:
            return 20
        return int(14 + (c - min_count) / (max_count - min_count) * 26)

    return [{"word": w, "size": size(c)} for w, c in counts]


# -------------------------
# Load models
# -------------------------
sent_tokenizer, sent_model = None, None

try:
    if os.path.isdir(SENTIMENT_DIR):
        sent_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
        sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_DIR)
    else:
        print("No sentiment_model directory found; using TextBlob fallback.")
except Exception as e:
    print("Failed to load sentiment_model:", e)
    sent_model, sent_tokenizer = None, None


def predict_sentiment_model(text):
    if sent_model and sent_tokenizer:
        inputs = sent_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = sent_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        id2 = (
            sent_model.config.id2label if hasattr(sent_model.config, "id2label") else {}
        )
        labels = []
        n = probs.shape[0]
        for i in range(n):
            lbl = id2.get(i) if i in id2 else id2.get(str(i))
            labels.append(lbl.lower() if lbl else f"label_{i}")

        scores_list = [
            {"label": labels[i], "score": float(probs[i])} for i in range(len(labels))
        ]
        predicted_label = labels[int(probs.argmax())]
        return scores_list, predicted_label

    # Fallback: TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        predicted_label = "positive"
    elif polarity < -0.2:
        predicted_label = "negative"
    else:
        predicted_label = "neutral"

    pos = max(0.0, polarity)
    neg = max(0.0, -polarity)
    neu = max(0.0, 1.0 - (pos + neg))
    scores_list = [
        {"label": "negative", "score": float(neg)},
        {"label": "neutral", "score": float(neu)},
        {"label": "positive", "score": float(pos)},
    ]
    return scores_list, predicted_label


embedder, kmeans = None, None
try:
    if os.path.exists(CLUSTERING_PKL):
        embedder, kmeans = cpu_joblib_load(CLUSTERING_PKL)
    else:
        print("clustering.pkl not found; cluster endpoints disabled.")
except Exception as e:
    print("Failed to load clustering.pkl:", e)


summarizer = None
try:
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=device)
except Exception as e:
    print("Failed to initialize summarizer:", e)


# -------------------------
# Dashboard computation
# -------------------------
def compute_dashboard():
    reviews = load_reviews_file()
    total = len(reviews)

    ratings = [r.get("rating") for r in reviews if r.get("rating") is not None]
    avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else 0

    sentiment_counts = Counter([r.get("s", "neutral").lower() for r in reviews])
    sentiment_data = [
        {"name": "positive", "value": sentiment_counts.get("positive", 0)},
        {"name": "neutral", "value": sentiment_counts.get("neutral", 0)},
        {"name": "negative", "value": sentiment_counts.get("negative", 0)},
    ]

    positive_share = (
        round(sentiment_counts.get("positive", 0) / total * 100, 1) if total else 0
    )

    genres = Counter([r.get("genre", "Unknown") for r in reviews])
    genre_list = [{"genre": g, "count": c} for g, c in genres.items()]
    top_genre = (
        max(genre_list, key=lambda x: x["count"])["genre"] if genre_list else None
    )

    wordcloud = simple_wordcloud_from_reviews(reviews)

    stats = {
        "totalReviews": total,
        "avgRating": avg_rating,
        "topGenre": top_genre,
        "positiveShare": positive_share,
    }

    return {
        "stats": stats,
        "genres": genre_list,
        "wordcloud": wordcloud,
        "reviews": reviews,
        "sentiment": sentiment_data,
    }


# -------------------------
# Routes
# -------------------------
@app.route("/dashboard", methods=["GET"])
def route_dashboard():
    return jsonify(compute_dashboard())


@app.route("/reviews", methods=["GET"])
def route_get_reviews():
    return jsonify(load_reviews_file())


@app.route("/reviews", methods=["POST"])
def route_post_review():
    data = request.get_json(force=True, silent=True) or {}
    movie = (data.get("movie") or "Unknown Movie").strip()
    text = (data.get("text") or "").strip()
    rating = data.get("rating")
    genre = data.get("genre", "Unknown")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment_scores, predicted_label = predict_sentiment_model(text)

    reviews = load_reviews_file()
    new_id = max([r["id"] for r in reviews]) + 1 if reviews else 1
    review_obj = {
        "id": new_id,
        "movie": movie,
        "text": text,
        "s": predicted_label.lower(),
        "genre": genre,
    }
    if rating is not None:
        review_obj["rating"] = rating

    reviews.append(review_obj)
    save_reviews_file(reviews)

    summary = None
    try:
        if summarizer:
            summary = summarizer(text, max_length=60, min_length=15, do_sample=False)[
                0
            ]["summary_text"]
    except Exception:
        summary = None

    return jsonify(
        {
            "review": review_obj,
            "summary": summary,
            "sentiment": sentiment_scores,
            "dashboard": compute_dashboard(),
        }
    )


@app.route("/sentiment", methods=["POST"])
def route_sentiment():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    scores, label = predict_sentiment_model(text)
    return jsonify(scores)


@app.route("/summarize", methods=["POST"])
def route_summarize():
    data = request.get_json(force=True, silent=True) or {}
    reviews = data.get("reviews") or []
    if not isinstance(reviews, list) or len(reviews) == 0:
        return jsonify({"summary": ""})
    text = " ".join(reviews)
    try:
        if summarizer:
            out = summarizer(text, max_length=150, min_length=30, do_sample=False)
            return jsonify({"summary": out[0]["summary_text"]})
    except Exception:
        pass
    words = text.split()
    summary = " ".join(words[:50]) + ("..." if len(words) > 50 else "")
    return jsonify({"summary": summary})


@app.route("/cluster", methods=["POST"])
def route_cluster():
    if (not embedder) or (not kmeans):
        return jsonify({"error": "Cluster model not available"}), 500
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    emb = embedder.encode([text])
    cluster_id = int(kmeans.predict(emb)[0])
    return jsonify({"cluster": int(cluster_id)})


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print(
        "Server starting. Models loaded?:",
        {
            "sentiment": bool(sent_model),
            "cluster": bool(embedder),
            "summarizer": bool(summarizer),
        },
    )
    app.run(host="0.0.0.0", port=5000, debug=True)
