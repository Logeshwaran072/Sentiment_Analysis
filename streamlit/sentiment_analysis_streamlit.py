import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

# Define constants
MODEL_PATH = (
    MODEL_PATH = os.path.join(os.getcwd(), "RNN_Models", "lstm_sentiment_model.h5")
)
TOKENIZER_PATH = "D:/Project/Sentiment_Analysis/Model/tokenizer.pkl"

VOCAB_SIZE = 20000
MAX_LENGTH = 50

# Load the trained LSTM model
best_model = tf.keras.models.load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)


# Function to preprocess input text
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")
    return padded_sequence


# Function to predict sentiment
def predict_sentiment(text):
    processed_input = preprocess_text(text)
    prediction = best_model.predict(processed_input)

    if prediction.shape[1] == 1:  # Binary classification
        sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😡"
        confidence = prediction[0][0]
    else:  # Multi-class classification (Softmax output)
        predicted_class = np.argmax(prediction)
        sentiment_labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
        sentiment = sentiment_labels[predicted_class]
        confidence = prediction[0][predicted_class]

    return sentiment, confidence


# Function to clean and split text into sentences
def split_text(text):
    sentences = re.split(r"[.!?]", text)
    return [s.strip() for s in sentences if s.strip()]


# --- Streamlit UI ---
st.set_page_config(page_title="Sentiment Analysis", page_icon="🔍", layout="centered")

st.title("🔍 Sentiment Analysis with LSTM")
st.write("### Analyze text sentiment & track trends! 📊")

# Store predictions for visualization
if "sentiment_data" not in st.session_state:
    st.session_state.sentiment_data = []

user_input = st.text_area(
    "✍️ Enter your text (single or multiple sentences):", height=150
)

if st.button("Predict Sentiment 🧠"):
    if user_input.strip():
        sentences = split_text(user_input)
        sentiment_results = []

        for sentence in sentences:
            sentiment, confidence = predict_sentiment(sentence)
            sentiment_results.append(
                {"Text": sentence, "Sentiment": sentiment, "Confidence": confidence}
            )
            st.session_state.sentiment_data.append(
                {"Text": sentence, "Sentiment": sentiment, "Confidence": confidence}
            )

        # Display results in a table
        df = pd.DataFrame(sentiment_results)
        st.write("### Sentiment Results 📝")
        st.dataframe(df)
    else:
        st.warning("⚠️ Please enter some text for analysis.")

# Show Sentiment Distribution Chart if data exists
if len(st.session_state.sentiment_data) > 0:
    df = pd.DataFrame(st.session_state.sentiment_data)

    # 📊 Pie Chart for sentiment distribution
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    st.write("### Sentiment Distribution 📊")
    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map={
            "Negative 😡": "#FF4B4B",
            "Neutral 😐": "#FFC107",
            "Positive 😊": "#4CAF50",
        },
        hole=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📈 Confidence Score Over Time
    st.write("### Confidence Score Trend 📈")
    fig2 = px.line(
        df, y="Confidence", x=df.index, markers=True, title="Confidence Score Over Time"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 📊 Sentiment Over Time (Bar Chart) without Names
    st.write("### Sentiment Over Time 📊")
    fig3 = px.bar(
        df,
        x=df.index,
        y="Confidence",
        color="Sentiment",
        title="Sentiment Confidence Over Time",
    )
    fig3.update_layout(xaxis_tickangle=-45)  # Rotate text labels for readability
    st.plotly_chart(fig3, use_container_width=True)

    # 🔍 Most & Least Confident Predictions
    st.write("### Confidence Analysis 🔎")
    most_confident = df.loc[df["Confidence"].idxmax()]
    least_confident = df.loc[df["Confidence"].idxmin()]
    st.write(
        f"✅ **Most Confident Prediction:** {most_confident['Text']} ({most_confident['Sentiment']}, {most_confident['Confidence']:.2f})"
    )
    st.write(
        f"⚠️ **Least Confident Prediction:** {least_confident['Text']} ({least_confident['Sentiment']}, {least_confident['Confidence']:.2f})"
    )

    # 📋 Word Frequency Table
    st.write("### Word Frequency Table 📋")
    words = re.findall(r"\w+", " ".join(df["Text"]))
    word_freq = Counter(words).most_common(10)
    word_freq_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
    st.dataframe(word_freq_df)

# Reset the session state
if st.button("Reset"):
    st.session_state.sentiment_data = []
    st.success("Session has been reset! 🔄")
