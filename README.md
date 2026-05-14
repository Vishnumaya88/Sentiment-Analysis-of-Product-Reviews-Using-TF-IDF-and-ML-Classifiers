# Sentiment-Analysis-of-Product-Reviews-Using-TF-IDF-and-ML-Classifiers

| Name       | Roll No | Course |
|------------|----------|--------|
| Adithya    | 253314   | GIS    |
| Gayathri   | 253019    | CSDA   |
| Vishnumaya | 253013   | CSDA   |
## 1. Problem Statement and Motivation
The goal of this project is to classify customer reviews into Positive, Neutral, and Negative sentiments using traditional Machine Learning. Unlike deep learning "black-box" models, this approach focuses on high interpretability and efficiency by combining TF-IDF vectorization with hand-crafted linguistic features. This allows businesses to understand not just what the sentiment is, but which specific linguistic markers (like negation or punctuation) drive customer feedback.
## 2. Dataset Description
- **Source:** `Amazon_Reviews.csv` (Scraped Amazon Customer Reviews).
- **Size:** 21,214 records.
- **Features:**
    - `Review Text`: Raw textual feedback.
    - `Rating`: Star ratings (1-5), mapped to sentiment classes.
    - `Linguistic Features`: Sentence length, punctuation density, average word length, and negation presence.
- **Class Distribution:**
    - **Negative (Ratings 1-2):** 14,350
    - **Positive (Ratings 4-5):** 5,820
    - **Neutral (Rating 3):** 885
