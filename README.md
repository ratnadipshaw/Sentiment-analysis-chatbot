# Sentiment Analysis Chatbot

A web-based chatbot that analyzes user sentiment using NLP techniques.

## Features
- Text preprocessing (punctuation, repetition handling)
- Typo correction with user validation
- Gibberish detection
- Hybrid sentiment analysis (VADER + TextBlob)
- Dynamic chatbot responses
- Chat-style UI using Streamlit

## Model Approach
- VADER compound score
- TextBlob polarity
- Final score = average of both

## Evaluation
- IMDb Dataset → ~60% accuracy
- Google GoEmotions Dataset → ~81% accuracy

### Insight:
Model performs better on short, emotion-rich text (GoEmotions) compared to long complex reviews (IMDb).

## Contributors
- Ratnadip Shaw
- Sabeeka Firdous

## Run Locally

```bash
pip install -r requirements.txt
streamlit run chatbot_ui_2.py
