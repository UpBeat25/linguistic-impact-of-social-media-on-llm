# Linguistic Impact of Social Media on Large Language Models

⚠️ **WARNING:** This repository contains examples of harmful language, and reader discretion is recommended.

This repository contains the code and experimental setup used to investigate how exposure to social-media-style language affects the behavior of large language models (LLMs).

The study analyzes how incremental exposure to conversational data influences:
- Toxicity
- Sentiment
- Semantic similarity

---

## 📌 Overview

Large language models are highly sensitive to the data they are trained on. This project explores how even small amounts of social-media-style data can alter model behavior.

The experiments are conducted using a controlled, single-model setup with staged fine-tuning, allowing precise observation of behavioral changes across exposure levels.

---

## ⚙️ Methodology

- Base Model: Qwen 2.5 0.5B Instruct  
- Framework: Hugging Face Transformers  
- Training: Staged supervised fine-tuning  
- Evaluation Metrics:
  - Toxicity → Perspective API
  - Sentiment → VADER
  - Semantic Similarity → MiniLM (Sentence-BERT embeddings)

---

## 📊 Evaluation Pipeline

For each stage:
1. Fixed input prompts are used
2. Model generates responses
3. Metrics are computed:
   - Toxicity (Perspective API)
   - Sentiment (VADER)
   - Semantic Similarity (cosine similarity of embeddings)

---

## Question❓ Why VADER was used instead of TextBlob

Although both VADER and TextBlob were initially considered for sentiment analysis, only VADER was used in the final evaluation.

This is because:
- VADER is specifically designed for **social-media-style text**
- It handles **informal language, slang, and punctuation** more effectively
- It provides **more consistent results** for short conversational inputs

TextBlob, while useful for general-purpose sentiment analysis, is less suitable for noisy and informal data typically found in social media. Therefore, it was not included in the final evaluation pipeline to maintain consistency and reliability of results.

---
