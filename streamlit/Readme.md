# Streamlit Sentiment Analysis App

## Overview
This Streamlit application performs **Sentiment Analysis** using a trained LSTM model. It allows users to input text and receive sentiment predictions, including **Positive 😊, Neutral 😐, and Negative 😡** sentiments. Additionally, the app provides various visualizations such as sentiment distribution, confidence trends, and word frequency analysis.

## Features
- **Text Sentiment Analysis**: Predicts sentiment of user-inputted text.
- **Multiple Sentence Processing**: Splits and analyzes text sentence by sentence.
- **Confidence Score Visualization**: Displays confidence levels for predictions.
- **Interactive Charts**:
  - **Pie Chart**: Sentiment distribution.
  - **Line Chart**: Confidence score trend over time.
  - **Bar Chart**: Sentiment confidence levels over time.
- **Word Frequency Analysis**: Highlights the most common words used in input texts.
- **Session Reset**: Option to clear session data and start fresh.

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** and the required libraries installed:

```sh
pip install streamlit tensorflow numpy pandas plotly matplotlib
```

### Clone the Repository
```sh
git clone <your-repository-link>
cd <repository-folder>
```

### Run the Streamlit App
```sh
streamlit run app.py
```

## Folder Structure
```
.
├── app.py                  # Main Streamlit application
├── model/                  # Contains trained LSTM model & tokenizer
│   ├── lstm_sentiment_model.h5
│   ├── tokenizer.pkl
├── data/                   # (Optional) Folder for sample input text
├── README.md               # Project documentation (this file)
```

## How to Use
1. Open the app in your browser after running `streamlit run app.py`.
2. Enter your text in the provided text box.
3. Click **Predict Sentiment 🧠**.
4. View sentiment results in the table and charts.
5. Reset session if needed.

## Model Details
- Uses a **pretrained LSTM model** for sentiment analysis.
- Processes text using **Tokenization & Padding**.
- Predicts **Positive, Neutral, or Negative** sentiment.

## Future Enhancements
- Add **GRU model support** for comparison.
- Implement **real-time sentiment tracking**.
- Deploy the app on **Streamlit Cloud or Hugging Face Spaces**.

## License
This project is licensed under the **MIT License**.

---
_Developed with ❤️ using TensorFlow & Streamlit_ 🚀

