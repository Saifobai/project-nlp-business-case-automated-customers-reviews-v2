# 🎬 AutoReview Project

An AI-powered pipeline for analyzing movie reviews — combining sentiment classification, clustering, and summarization.
Built with Transformers, MLflow, Flask/React, and MLOps best practices.

### download these files and put them in their folders and run the app

#### cluster https://drive.google.com/file/d/16DGximUPdSkjbhhY5DVIwPKHG-Ouxl6f/view?usp=drive_link

-

### sentiment

- download the folder and replace it with the same folder name

#### Sentiment https://drive.google.com/drive/folders/1jZh0xKM8mTqYR30JmhjW1BJWImiRTlrS?usp=drive_link

### IMDB_Data 50k

#### https://drive.google.com/drive/folders/1XF9oDOU-lBAJgBBUixKT1gb0OcEW3ULr?usp=drive_link

## 🚀 Features

- Sentiment Analysis
- Fine-tuned DistilBERT (with LoRA adapters) → outputs NEGATIVE / POSITIVE / NEUTRAL.

- Movie Clustering
- SentenceTransformers embeddings + KMeans for grouping by genre, director, actor.

- Review Summarization
- Concise human-readable summaries powered by Flan-T5.

- Visualization Tools

- Loss curves (training vs. evaluation)

- Accuracy & F1 score trends

- Confusion matrices (raw + normalized)

- Cluster distribution histograms

- MLOps with MLflow

- Automatic experiment tracking

- Metrics, parameters, and artifact logging

- Model registry

🛠️ Installation

# Clone repository

git clone https://github.com/your-username/AutoReview.git
cd AutoReview

# Create virtual environment

python -m venv venv

# Linux / Mac

source venv/bin/activate

# Windows

venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

## 📂 Project Structure

```
AutoReview/
│── notebooks/
│   └── AutoReviewModel.ipynb   # Training & experimentation
│── backend/
│   └── app.py                  # Flask/FastAPI backend
│── frontend/
│   └── ...                     # React / Streamlit frontend
│── sentiment_model/            # Saved sentiment model
│── clustering.pkl               # Saved clustering artifacts
│── imdb_data.csv                # IMDb dataset
│── README.md
│── requirements.txt

```

## 🧠 Model Training

- Sentiment Model

- Dataset: IMDb (50k reviews) + synthetic neutral class

- Base model: distilbert-base-uncased (LoRA fine-tuned)

- Optimizer: AdamW

- Epochs: 5

- Clustering

- Embeddings: all-MiniLM-L6-v2 (SentenceTransformers)

- Algorithm: KMeans (n=5)

- Summarization

- Model: google/flan-t5-small

- Produces 3–5 sentence summaries of review sets

- 📊 Visualizations

- 📉 Loss Curves – Training vs. Evaluation

- ✅ Accuracy & F1 Trends

- 🔲 Confusion Matrices – Counts & Percentages

- 🎭 Cluster Distribution Histograms

- 🏗️ System Architecture
  graph TD
  A[User] -->|Submits review / query| B[React Frontend]
  B --> C[Flask/FastAPI Backend]
  C --> D[Sentiment Model (DistilBERT+LoRA)]
  C --> E[Clustering (SentenceTransformers + KMeans)]
  C --> F[Summarization (Flan-T5 / GPT)]
  C --> G[MLflow Tracking]
  D --> H[Results: Sentiment]
  E --> H
  F --> H
  H --> B

🔄 Workflow Pipeline
flowchart LR
A[Raw Movie Reviews] --> B[Preprocessing]
B --> C[Sentiment Analysis]
B --> D[Clustering Embeddings]
B --> E[Summarization]
C --> F[Visualizations & Metrics]
D --> F
E --> F
F --> G[Dashboard (React + Flask API)]

📦 Saving & Reloading

# Save trained models

trainer.model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
joblib.dump((embedder, kmeans), "clustering.pkl")

# Reload for inference

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
"./sentiment_model", num_labels=3
)
embedder, kmeans = joblib.load("clustering.pkl")

⚡ MLOps with MLflow

Track experiments, parameters, and metrics:

mlflow ui

➡️ Open in browser: http://127.0.0.1:5000

🔍 Example Usage

# Sentiment

sentiment_pipe("Loved the movie, amazing acting!")

# → POSITIVE

# Clustering

"Dune Part Two (2025)" → Cluster 2

# Summarization

["Great visuals!", "Story was slow.", "Loved the music."]

# → "Amazing visuals and soundtrack, but pacing was slow. Overall great experience."

🔮 Future Improvements

🎯 Deploy with FastAPI / Streamlit

🔍 Add movie search & recent reviews feature

📡 Connect to live review feeds (IMDb, Rotten Tomatoes, Twitter)

⚙️ Full CI/CD pipeline with automated retraining & deployment

✨ AutoReview combines ML, LLMs, and modern web tools into a powerful end-to-end review analysis system.
