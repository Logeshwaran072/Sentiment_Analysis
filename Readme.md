# Sentiment Analysis Using LSTM

## 📌 Project Overview
This project focuses on sentiment analysis using deep learning and traditional machine learning models. The models trained include:
- **LSTM (Long Short-Term Memory) [Final Model]**
- **GRU (Gated Recurrent Unit)**
- **Logistic Regression**
- **Naive Bayes**

The best-performing model, **LSTM**, was selected for deployment using **Streamlit**.

---

## 🚀 Features
- Predicts sentiment using LSTM
- Displays sentiment distribution with interactive charts
- Provides confidence scores for predictions
- Preprocesses textual data
- Trained ML models (Logistic Regression, Naive Bayes) and RNN models (LSTM, GRU)
- Uses LSTM for final sentiment classification
- Provides an interactive UI using Streamlit
- Logs experiment details with MLflow
- Generates classification reports and confusion matrices

---

## 🐂 Dataset
- **Dataset Source:** [Kaggle](https://www.kaggle.com/)(Sentiment140 Dataset) 
- **Dataset Files:** `sentiment140.csv
- **Preprocessing:** Tokenization, Stop-word Removal, Lemmatization

---

## 📂 Folder Structure
```
/sentiment_analysis_project
│── models/                # Saved trained models
│   ├── ml_models/         # Machine Learning models (Logistic Regression, Naive Bayes)
│   ├── rnn_models/        # RNN models (LSTM, GRU)
│   ├── lstm_model.h5
│── data/                  # Dataset folder
│   ├── Sentiment140
│── notebooks/             # Jupyter notebooks for EDA & model training
│── app.py                 # Streamlit app for deployment
│── requirements.txt       # Dependencies
│── README.md              # Project Documentation
```

---

## 🛠️ Installation & Setup
1. **Clone the repository**
```bash
git clone https://github.com/Logeshwaran072/Sentiment_Analysis.git
cd sentiment-analysis-lstm
```

2. **Create a virtual environment and install dependencies**
```bash
pip install -r requirements.txt
```


---

## 🏋️ Model Training
Run the training script to preprocess data and train the models:
```bash
python run the notebook folder file (Jupyter file)
```
This will:
- Load and preprocess text data
- Train ML models (Logistic Regression, Naive Bayes)
- Train RNN models (LSTM, GRU)
- Save the trained models in the `models/` folder

---

## 🏆 Model Evaluation
The evaluation script computes metrics and logs them with **MLflow**:
```
specified in the trained file itself.
```
It generates:
- Accuracy, Precision, Recall, F1-score for all models
- Confusion Matrix
- Classification Report
- Model comparison for LSTM, GRU, Logistic Regression, and Naive Bayes

---

## 🌐 Deploy with Streamlit
To launch the interactive app for sentiment prediction:
```bash
streamlit sentiment_analysis_streamlit.py
```
This will start a **web-based UI** where users can enter text and receive sentiment predictions using the **LSTM model**.

---

## 📊 MLflow Tracking
To track model performance, start an MLflow server:
```bash
mlflow ui
```
Then open **http://localhost:5000** to view logged experiments.

---

## 🚀 Deployment (Optional)
### **Deploy to Streamlit Cloud**
1. Push your project to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click **Deploy an app**, select your repo, and deploy!

### **Deploy on Heroku**
1. Install Heroku CLI
2. Create a `Procfile`:
```bash
web: streamlit run app.py --server.port=$PORT
```
3. Push to Heroku:
```bash
git init
git add .
git commit -m "Deploy Streamlit app"
heroku create my-sentiment-app
git push heroku main
```

---

## 🐝 Screenshots (Optional)
Include screenshots or GIFs of the **Streamlit app UI**.

---

## 📝 License
This project is licensed under the MIT License.

---

## 🤝 Contact
For any queries, reach out via:
- **Email**: your-email@example.com
- **GitHub**: [your-username](https://github.com/your-username)

