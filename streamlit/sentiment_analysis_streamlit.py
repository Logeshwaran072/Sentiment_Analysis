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

# --- Constants ---
MODEL_PATH = "D:/Project/Sentiment_Analysis/Model/RNN_Models/lstm_sentiment_model.h5"
TOKENIZER_PATH = "D:/Project/Sentiment_Analysis/Model/tokenizer.pkl"
VOCAB_SIZE = 20000
MAX_LENGTH = 50

# --- Load Model & Tokenizer ---
best_model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)


# --- Preprocessing Function ---
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")
    return padded_sequence


# --- Predict Sentiment ---
def predict_sentiment(text):
    processed_input = preprocess_text(text)
    prediction = best_model.predict(processed_input)
    if prediction.shape[1] == 1:
        sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😡"
        confidence = prediction[0][0]
    else:
        predicted_class = np.argmax(prediction)
        sentiment_labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
        sentiment = sentiment_labels[predicted_class]
        confidence = prediction[0][predicted_class]
    return sentiment, confidence


# --- Text Splitting ---
def split_text(text):
    sentences = re.split(r"[.!?]", text)
    return [s.strip() for s in sentences if s.strip()]


# --- Session Reset ---
def reset_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("🔄 Session has been reset!")
    st.experimental_rerun()


# --- Streamlit UI ---
st.set_page_config(page_title="Sentiment Analysis", page_icon="🔍", layout="centered")

st.title("🔍 Sentiment Analysis with LSTM")
st.write("### Analyze text sentiment & track trends! 📊")

# Button to reset session
if st.button("🔄 Reset All"):
    reset_all()

# Store data in session
if "sentiment_data" not in st.session_state:
    st.session_state.sentiment_data = []

# --- User Input Text ---
st.markdown("### ✍️ Enter your text (single or multiple sentences):")
user_input = st.text_area("Input Text:", height=150, key="user_input")

# --- Prediction for Text Area ---
if st.button("Predict Sentiment 🧠"):
    if user_input.strip():
        sentences = split_text(user_input)
        sentiment_results = []
        for sentence in sentences:
            sentiment, confidence = predict_sentiment(sentence)
            result = {
                "Text": sentence,
                "Sentiment": sentiment,
                "Confidence": float(confidence),
            }
            sentiment_results.append(result)
            st.session_state.sentiment_data.append(result)

        df = pd.DataFrame(sentiment_results)
        st.write("### Sentiment Results 📝")
        st.dataframe(df)
    else:
        st.warning("⚠️ Please enter some text for analysis.")

# --- File Upload ---
st.markdown("### 📁 Or Upload a CSV/TXT File for Bulk Sentiment Analysis")
uploaded_file = st.file_uploader("Upload a .csv or .txt file", type=["csv", "txt"])
bulk_df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            bulk_df = pd.read_csv(uploaded_file)
            if "Text" not in bulk_df.columns:
                st.warning("⚠️ CSV must have a column named 'Text'")
                bulk_df = None
        else:
            lines = uploaded_file.read().decode("utf-8").splitlines()
            bulk_df = pd.DataFrame(lines, columns=["Text"])
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")


# --- Prediction for Uploaded File ---
if bulk_df is not None and st.button("📊 Predict Uploaded Data"):
    st.write("### Analyzing Uploaded Text...")
    bulk_results = []
    for sentence in bulk_df["Text"]:
        sentiment, confidence = predict_sentiment(sentence)
        result = {
            "Text": sentence,
            "Sentiment": sentiment,
            "Confidence": float(confidence),
        }
        bulk_results.append(result)
        st.session_state.sentiment_data.append(result)

    result_df = pd.DataFrame(bulk_results)
    st.write("### 📋 Bulk Sentiment Results")
    st.dataframe(result_df)

    # # Download button
    # csv = result_df.to_csv(index=False).encode("utf-8")
    # st.download_button(
    #     label="⬇️ Download CSV",
    #     data=csv,
    #     file_name="sentiment_results.csv",
    #     mime="text/csv",
    # )

# --- Visualizations ---
if len(st.session_state.get("sentiment_data", [])) > 0:
    df = pd.DataFrame(st.session_state["sentiment_data"])
    # ✅ Summary Section
    st.write("### 📊 Summary Insights")
    total = len(df)
    avg_conf = df["Confidence"].mean()
    dominant = df["Sentiment"].value_counts().idxmax()
    st.info(f"**Total Sentences Analyzed:** {total}")
    st.success(f"**Most Common Sentiment:** {dominant}")
    st.warning(f"**Average Confidence Score:** {avg_conf:.2f}")

    # pie chart
    st.write("### Sentiment Distribution 📊")
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
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
    # line chart
    st.write("### Confidence Score Trend 📈")
    df["Index"] = df.index.astype(str)
    fig2 = px.line(
        df, y="Confidence", x="Index", markers=True, title="Confidence Score Over Time"
    )
    st.plotly_chart(fig2, use_container_width=True)
    # bar chart
    st.write("### Sentiment Over Time 📊")
    fig3 = px.bar(
        df,
        x="Index",
        y="Confidence",
        color="Sentiment",
        color_discrete_map={
            "Negative 😡": "#FF4B4B",
            "Neutral 😐": "#FFC107",
            "Positive 😊": "#4CAF50",
        },
        title="Sentiment Confidence Over Time",
    )
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

    # most & least confident
    st.write("### Confidence Analysis 🔎")
    most_confident = df.loc[df["Confidence"].idxmax()]
    least_confident = df.loc[df["Confidence"].idxmin()]
    st.write(
        f"✅ **Most Confident Prediction:** {most_confident['Text']} ({most_confident['Sentiment']}, {most_confident['Confidence']:.2f})"
    )
    st.write(
        f"⚠️ **Least Confident Prediction:** {least_confident['Text']} ({least_confident['Sentiment']}, {least_confident['Confidence']:.2f})"
    )

    # Word Frequency
    st.write("### Word Frequency Table 📋")
    words = re.findall(r"\w+", " ".join(df["Text"]))
    word_freq = Counter(words).most_common(10)
    word_freq_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
    st.dataframe(word_freq_df)

    # 📤 Download
    st.download_button(
        "📥 Download Results as CSV",
        data=df.to_csv(index=False),
        file_name="sentiment_results.csv",
        mime="text/csv",
    )

    # 📨 Feedback Form (optional user feedback)
    st.write("### 📝 Feedback (Optional)")
    feedback_text = st.text_area(
        "Was any sentiment prediction incorrect? Let us know here:"
    )
    if st.button("Submit Feedback"):
        if feedback_text.strip():
            with open("feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(feedback_text + "\n")
            st.success("✅ Feedback submitted successfully!")
        else:
            st.warning("⚠️ Feedback is empty!")
