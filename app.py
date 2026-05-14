import streamlit as st
import joblib
import re
import string

# =====================================================
# Load Saved Model Files
# =====================================================

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")
selector = joblib.load("chi_selector.pkl")

# =====================================================
# Text Cleaning Function
# =====================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = text.strip()

    return text

# =====================================================
# Streamlit Page Settings
# =====================================================

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# =====================================================
# App Title
# =====================================================

st.title("😊 Sentiment Analysis Application")

st.write(
    "Enter a sentence and predict its sentiment"
)

# =====================================================
# User Input
# =====================================================

user_input = st.text_area(
    "Enter Text Here"
)

# =====================================================
# Prediction Button
# =====================================================

if st.button("Predict Sentiment"):

    if user_input.strip() == "":

        st.warning("Please enter some text")

    else:

        # Clean text
        cleaned_text = clean_text(user_input)

        # Convert text to TF-IDF
        text_vector = tfidf.transform([cleaned_text])

        # Apply Chi-Square Selection
        selected_features = selector.transform(text_vector)

        # Predict sentiment
        prediction = model.predict(selected_features)

        # Decode prediction
        sentiment = encoder.inverse_transform(prediction)

        # Display result
        st.success(f"Predicted Sentiment: {sentiment[0]}")