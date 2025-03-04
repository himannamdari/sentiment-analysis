import joblib

# Load saved model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_sentiment(text):
    """Predict the sentiment of a given text."""
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return prediction

if __name__ == "__main__":
    text = input("Enter a sentence for sentiment analysis: ")
    sentiment = predict_sentiment(text)
    print(f"Predicted Sentiment: {sentiment}")
