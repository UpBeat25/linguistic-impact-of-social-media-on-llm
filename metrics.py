import pandas as pd
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import evaluate

# === Setup ===
vader_analyzer = SentimentIntensityAnalyzer()
sentiment_model = SentenceTransformer("all-MiniLM-L6-v2")
perplexity_metric = evaluate.load("perplexity", module_type="metric")

PERSPECTIVE_API_KEY = "API KEY HERE"
PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"


# === Sentiment Analysis ===
def analyze_sentiment(text):
    blob_polarity = TextBlob(text).sentiment.polarity
    vader_score = vader_analyzer.polarity_scores(text)["compound"]
    return {"textblob": blob_polarity, "vader": vader_score}


# === Toxicity Detection ===
def check_toxicity(text):
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}}
        }
        response = requests.post(f"{PERSPECTIVE_URL}?key={PERSPECTIVE_API_KEY}", json=data, headers=headers)
        return response.json()["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
    except Exception:
        return None


# === Coherence (semantic similarity) ===
def measure_coherence(user_input, bot_response):
    embeddings = sentiment_model.encode([user_input, bot_response])
    similarity_score = util.cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()


# === Repetition ===
def repetition_score(text):
    tokens = text.lower().split()
    unique = len(set(tokens))
    return unique / len(tokens) if tokens else 0


# === Length (verbosity) ===
def response_length(text):
    return len(text.split())


# === Combined Logger ===
def log_interaction(user_input, bot_response):
    sentiment = analyze_sentiment(bot_response)
    user_toxicity = check_toxicity(user_input)
    bot_toxicity = check_toxicity(bot_response)
    coherence = measure_coherence(user_input, bot_response)
    repetition = repetition_score(bot_response)
    length = response_length(bot_response)
    
    interaction = {
        "sentiment_textblob": sentiment["textblob"],
        "sentiment_vader": sentiment["vader"],
        "user_toxicity": user_toxicity,
        "bot_toxicity": bot_toxicity,
        "coherence": coherence,
        "repetition": repetition,
        "length": length,
    }
    
    return interaction
