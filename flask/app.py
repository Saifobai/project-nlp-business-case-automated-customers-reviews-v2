import os
import json
import re
import threading
from collections import Counter

from dotenv import load_dotenv

load_dotenv()  # ✅ Load environment variables from .env file

import torch
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from datasets import load_dataset
from openai import OpenAI


# -------------------------
# Paths and constants
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_DIR = os.path.join(BASE_DIR, "sentiment_model")
CLUSTERING_PKL = os.path.join(BASE_DIR, "clustering.pkl")
REVIEWS_FILE = os.path.join(BASE_DIR, "reviews.json")
IMDB_DIR = os.path.join(BASE_DIR, "IMDb_Movies", "IMDB_DATA")
IMDB_FILE = os.path.join(IMDB_DIR, "IMDB_Dataset.csv")

# -------------------------
# OpenAI client
# -------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY is not set. Please set it in a .env file.")

client = OpenAI(api_key=api_key)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)
CORS(app)

write_lock = threading.Lock()
imdb_df = None


# -------------------------
# Helpers
# -------------------------
def cpu_joblib_load(path):
    """Ensure clustering.pkl loads safely on CPU."""
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
    with write_lock:
        if not os.path.exists(REVIEWS_FILE):
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
            ]
            save_reviews_file(initial)
            return initial
        with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def load_imdb_data():
    """Ensure IMDb dataset is present or fetch from Hugging Face."""
    global imdb_df
    try:
        if not os.path.exists(IMDB_DIR):
            os.makedirs(IMDB_DIR, exist_ok=True)

        if not os.path.exists(IMDB_FILE):
            print("⚡ IMDb CSV not found, downloading from Hugging Face...")
            ds = load_dataset("Q-b1t/IMDB-Dataset-of-50K-Movie-Reviews-Backup")
            df = ds["train"].to_pandas()
            df.to_csv(IMDB_FILE, index=False)
            print(f"✅ IMDb dataset saved at {IMDB_FILE}")
        else:
            print(f"✅ IMDb CSV found at {IMDB_FILE}")

        imdb_df = pd.read_csv(IMDB_FILE)
        imdb_df.columns = [c.strip().lower() for c in imdb_df.columns]
        print("IMDb dataset loaded:", imdb_df.shape)
    except Exception as e:
        print("❌ IMDb load failed:", e)
        imdb_df = None


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
    max_count, min_count = counts[0][1], counts[-1][1]

    def size(c):
        return (
            20
            if max_count == min_count
            else int(14 + (c - min_count) / (max_count - min_count) * 26)
        )

    return [{"word": w, "size": size(c)} for w, c in counts]


# -------------------------
# Sentiment
# -------------------------
sent_tokenizer, sent_model = None, None
try:
    if os.path.isdir(SENTIMENT_DIR):
        sent_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
        sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_DIR)
    else:
        print("⚠ sentiment_model not found; using TextBlob fallback")
except Exception as e:
    print("❌ Sentiment model load failed:", e)
    sent_model, sent_tokenizer = None, None


def predict_sentiment_model(text):
    if sent_model and sent_tokenizer:
        inputs = sent_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = sent_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        id2 = getattr(sent_model.config, "id2label", {})
        labels = [
            id2.get(i) or id2.get(str(i), f"label_{i}") for i in range(len(probs))
        ]
        scores_list = [
            {"label": labels[i].lower(), "score": float(probs[i])}
            for i in range(len(labels))
        ]
        predicted_label = labels[int(probs.argmax())].lower()
        return scores_list, predicted_label

    # fallback with TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        predicted_label = "positive"
    elif polarity < -0.2:
        predicted_label = "negative"
    else:
        predicted_label = "neutral"
    pos, neg, neu = (
        max(0.0, polarity),
        max(0.0, -polarity),
        1.0 - (max(0.0, polarity) + max(0.0, -polarity)),
    )
    scores_list = [
        {"label": "negative", "score": float(neg)},
        {"label": "neutral", "score": float(neu)},
        {"label": "positive", "score": float(pos)},
    ]
    return scores_list, predicted_label


# -------------------------
# GPT Summarizer
# -------------------------
def gpt_summarize_reviews(reviews):
    """Summarize a list of reviews using GPT, fallback to TextBlob if GPT fails."""
    text = "\n".join(reviews)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # you can switch to gpt-4o if you like
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes movie reviews.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following movie review in 3 to 5 sentences:\n\n{text}",
                },
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT summarization failed, falling back:", e)
        # Simple fallback: use TextBlob sentence summary
        blob = TextBlob(text)
        sentences = blob.sentences
        if len(sentences) > 3:
            return " ".join(str(s) for s in sentences[:3])
        return text or "No summary available."


# -------------------------
# Dashboard
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
    new_id = max((r.get("id", 0) for r in reviews), default=0) + 1
    review_obj = {
        "id": new_id,
        "movie": movie,
        "text": text,
        "s": predicted_label,
        "genre": genre,
    }
    if rating is not None:
        try:
            review_obj["rating"] = max(1, min(10, int(rating)))
        except ValueError:
            review_obj["rating"] = None
    reviews.append(review_obj)
    save_reviews_file(reviews)

    # ✅ Summarize only the new review text
    summary_text = gpt_summarize_reviews([text])

    return jsonify(
        {
            "review": review_obj,
            "sentiment": sentiment_scores,
            "summary": summary_text,
            "dashboard": compute_dashboard(),
        }
    )


@app.route("/search", methods=["GET"])
def route_search():
    q = (request.args.get("q") or "").strip().lower()
    if not q:
        return jsonify([])
    local_reviews = load_reviews_file()
    filtered_local = [
        r
        for r in local_reviews
        if q in (r.get("movie") or "").lower() or q in (r.get("text") or "").lower()
    ]
    filtered_local.sort(key=lambda r: r.get("id", 0), reverse=True)
    local_result = filtered_local[:1]
    imdb_matches = []
    if imdb_df is not None and "review" in imdb_df.columns:
        df_results = imdb_df[
            imdb_df["review"].astype(str).str.lower().str.contains(q, na=False)
        ]
        for idx, row in df_results.head(3).iterrows():
            imdb_matches.append(
                {
                    "id": f"imdb_{idx}",
                    "movie": f"IMDb Review {idx}",
                    "text": row.get("review"),
                    "s": row.get("sentiment"),
                    "genre": "N/A",
                    "rating": "N/A",
                }
            )
    return jsonify(local_result + imdb_matches)


# -------------------------
# Init
# -------------------------
if __name__ == "__main__":
    load_imdb_data()
    print("🚀 Server starting. Models loaded?:", {"sentiment": bool(sent_model)})
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
