# Sentiment Analysis Model

This project builds a sentiment analysis model that classifies text into **Positive, Negative, or Neutral** categories.

## 📌 Project Structure
```
📦 sentiment-analysis
 ┣ 📂 data
 ┃ ┣ 📜 sentiment_data.csv  # Dataset
 ┣ 📂 models
 ┃ ┣ 📜 sentiment_model.pkl  # Saved Model
 ┃ ┣ 📜 vectorizer.pkl  # TF-IDF Vectorizer
 ┣ 📂 scripts
 ┃ ┣ 📜 train_model.py  # Python script for training
 ┃ ┣ 📜 predict.py  # Script for testing the model
 ┣ 📜 README.md  # Project Documentation
 ┣ 📜 requirements.txt  # Required Python libraries
```

## 📌 Setup Instructions

### Install dependencies:
```
pip install -r requirements.txt
```

### Train the Model:
```
python scripts/train_model.py
```

### Predict Sentiment:
Run:
```
python scripts/predict.py
```
Then enter a sentence for analysis.

---
🚀 **Built with Python, Scikit-Learn, and Joblib**
