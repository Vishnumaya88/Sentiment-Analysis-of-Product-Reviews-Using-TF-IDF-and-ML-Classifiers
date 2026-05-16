import streamlit as st
import joblib
import numpy as np
import re
import string

# =====================================================
# LOAD SAVED FILES
# =====================================================

model = joblib.load("sentiment_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")

encoder = joblib.load("label_encoder.pkl")

selector = joblib.load("chi_selector.pkl")

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 42px;
        color: #1f77b4;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TEXT CLEANING FUNCTION
# =====================================================

def clean_text(text):

    text = str(text).lower()

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
# APP TITLE
# =====================================================

st.markdown(
    "<div class='title'>😊 Sentiment Analysis App</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict customer review sentiment using Machine Learning</div>",
    unsafe_allow_html=True
)

st.write("")

# =====================================================
# USER INPUT
# =====================================================

user_input = st.text_area(
    "Enter Customer Review",
    height=180,
    placeholder="Type your review here..."
)

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("Predict Sentiment"):

    if user_input.strip() == "":

        st.warning(
            "⚠ Please enter some text"
        )

    else:

        # =====================================================
        # CLEAN TEXT
        # =====================================================

        cleaned_text = clean_text(
            user_input
        )

        # =====================================================
        # SIMPLE NEUTRAL CHECK
        # =====================================================

        neutral_keywords = [
            "arrived",
            "ordered",
            "purchased",
            "delivered",
            "received",
            "bought",
            "package",
            "shipment",
            "item"
        ]

        if any(
            word in cleaned_text
            for word in neutral_keywords
        ):

            st.info(
                "😐 Predicted Sentiment: Neutral"
            )

        else:

            # =====================================================
            # TF-IDF FEATURES
            # =====================================================

            text_vector = tfidf.transform(
                [cleaned_text]
            ).toarray()

            # =====================================================
            # LINGUISTIC FEATURES
            # =====================================================

            sentence_length = len(cleaned_text)

            avg_word_length = (
                np.mean(
                    [
                        len(word)
                        for word in cleaned_text.split()
                    ]
                )
                if len(cleaned_text.split()) > 0
                else 0
            )

            punct_density = (
                sum(
                    1 for char in cleaned_text
                    if char in string.punctuation
                ) / len(cleaned_text)
                if len(cleaned_text) > 0
                else 0
            )

            neg_words = [
                "not",
                "no",
                "never",
                "none",
                "nothing",
                "n't"
            ]

            negation = int(
                any(
                    word in cleaned_text.lower()
                    for word in neg_words
                )
            )

            # =====================================================
            # COMBINE FEATURES
            # =====================================================

            extra_features = np.array(
                [[
                    sentence_length,
                    avg_word_length,
                    punct_density,
                    negation
                ]]
            )

            combined_features = np.hstack(
                [text_vector, extra_features]
            )

            # =====================================================
            # FEATURE SELECTION
            # =====================================================

            selected_features = (
                selector.transform(
                    combined_features
                )
            )

            # =====================================================
            # PREDICTION
            # =====================================================

            prediction = model.predict(
                selected_features
            )

            sentiment = (
                encoder.inverse_transform(
                    prediction
                )
            )[0]

            # =====================================================
            # DISPLAY RESULT
            # =====================================================

            if sentiment == "Positive":

                st.success(
                    f"😊 Predicted Sentiment: {sentiment}"
                )

            elif sentiment == "Negative":

                st.error(
                    f"😠 Predicted Sentiment: {sentiment}"
                )

            elif sentiment == "Neutral":

                st.info(
                    f"😐 Predicted Sentiment: {sentiment}"
                )

            else:

                st.warning(
                    f"Prediction: {sentiment}"
                )

# =====================================================
# FOOTER
# =====================================================

st.write("")

st.caption(
    "Built using TF-IDF, Chi-Square Feature Selection and Machine Learning"
)