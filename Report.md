# Movie Review Analysis and Summarization

## Project Overview

This project is a Movie Review Analysis Dashboard built with Flask as the backend and a React-based dashboard as the frontend.
The application allows users to:

Submit new movie reviews

Automatically analyze sentiment (Positive, Neutral, Negative)

Generate clean summaries of multiple reviews

View statistics, charts, and word clouds in a dashboard

The goal of this project was to combine Natural Language Processing (NLP) techniques with data visualization to create a simple but effective review analysis system.

Objectives

Build a system to collect and store reviews persistently.

Apply sentiment analysis using machine learning models.

Summarize reviews into concise highlights.

Provide a user-friendly dashboard that includes:

Average rating

Sentiment distribution

Most frequent words

Genre distribution

Integrate an external IMDb dataset to allow richer search results.

Technology Stack

Backend: Python (Flask), Torch, Hugging Face Transformers

Frontend: React with Tailwind CSS for styling and visualization

NLP Models:

DistilBERT for sentiment classification (with fallback to TextBlob)

PEGASUS-XSum for lightweight abstractive summarization

Data Sources:

Local JSON storage (reviews.json) for submitted reviews

IMDb 50k Movie Reviews dataset as an additional reference

Project Structure

app.py: Flask backend containing API routes for sentiment analysis, summarization, and search.

reviews.json: Local file for persistent review storage.

sentiment_model/: Folder for pretrained or fine-tuned sentiment models.

clustering.pkl: Pre-saved clustering model (optional).

IMDb_Movies/IMDB_DATA/: Dataset containing IMDb reviews.

frontend/: React-based dashboard interface.

Methodology

Data Ingestion
Users submit reviews through the frontend. Each review contains the movie name, text review, rating (1–10), and optionally a genre.

Sentiment Analysis
The system primarily uses a DistilBERT model to classify reviews as positive, neutral, or negative.
If the model is unavailable, a fallback rule-based method using TextBlob polarity is applied.

Summarization
Summarization is handled by the PEGASUS-XSum model.

If there are three or fewer reviews, they are combined and summarized directly.

If there are more than three reviews, each review is summarized individually, then a second summarization pass is applied on the collection of summaries.
This two-pass method avoids repetition and ensures concise results.

Dashboard Insights
The frontend provides interactive visualizations, including:

Total reviews, average rating, positive review share

Sentiment distribution pie chart

Genre distribution bar chart

Word cloud of most frequent words

List of recent reviews

Search functionality across local reviews and IMDb dataset

Example Results

Input Review:
"A divisive but grand finale to Christopher Nolan's Batman trilogy, praised for its epic scale, thrilling action, and strong performances from Christian Bale."

Sentiment Result:
Positive (confidence: 82%)

Summarized Insights from multiple reviews:
"A thrilling and visually stunning film with strong performances, though opinions remain divided."

How to Run the Project

Backend (Flask API)

Install dependencies:
pip install flask flask-cors torch transformers textblob datasets joblib

Run the Flask app:
python app.py

Frontend (React Dashboard)

Navigate to frontend folder: cd frontend

Install dependencies: npm install

Start the frontend: npm run dev

The frontend will connect to the Flask backend running at http://localhost:5000.

Testing and Results

The system was tested with:

Positive reviews

Negative reviews

Mixed-opinion reviews

Long descriptive reviews

Results showed that:

Sentiment detection aligned well with human interpretation.

Summaries produced by PEGASUS were short, clear, and non-repetitive.

Dashboard statistics and visualizations updated dynamically after new reviews.

Conclusion

This project demonstrates how Flask, React, and NLP models can be integrated to create a movie review analysis system.
The pipeline successfully handles:

Real-time review submission

Automatic sentiment analysis

Summarization of multiple opinions

Visualization of trends and insights in an interactive dashboard

It is a lightweight but effective application suitable for a student-level project.

Future Work

Fine-tune PEGASUS for movie reviews to improve summarization quality.

Add user authentication and personalized dashboards.

Deploy the system on a cloud platform (Heroku, Render, or Hugging Face Spaces).

Incorporate real-time data from social media platforms like Twitter or Reddit for trending movie reviews.
