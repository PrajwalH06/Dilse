import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import random

def load_data():
    with open('static/entries.json', 'r') as f:
        data = json.load(f)
    return data

def categorize_sentiment(score):
    if score <= -0.3:
        return 'sadness'
    elif score >= 0.3:
        return 'happiness'
    else:
        return 'neutral'

def generate_sentiment_sentence(score):
    if score <= -0.8:
        sentences = [
            "A profound sense of melancholy fills the atmosphere.",
            "The emotional weight feels particularly heavy.",
            "A deep sense of sadness permeates the mood."
        ]
    elif score <= -0.5:
        sentences = [
            "The atmosphere carries a noticeable weight of sadness.",
            "A gentle cloud of melancholy lingers in the air.",
            "The mood reflects a quiet sense of sorrow."
        ]
    elif score <= -0.3:
        sentences = [
            "A slightly downcast mood prevails.",
            "The atmosphere has a subtle hint of sadness.",
            "The emotional tone feels a bit heavy."
        ]
    elif score <= -0.1:
        sentences = [
            "The mood is slightly subdued.",
            "A gentle sense of melancholy lingers.",
            "The atmosphere is quiet and reflective."
        ]
    elif score < 0.1:
        sentences = [
            "The emotional state remains balanced and centered.",
            "A calm and composed atmosphere prevails.",
            "The mood is stable and peaceful."
        ]
    elif score < 0.3:
        sentences = [
            "There's a subtle sense of positivity in the air.",
            "A gentle warmth of contentment fills the atmosphere.",
            "A light sense of joy begins to emerge."
        ]
    elif score < 0.5:
        sentences = [
            "The atmosphere is noticeably positive and uplifting.",
            "A warm sense of happiness fills the air.",
            "The mood is bright and cheerful."
        ]
    elif score < 0.8:
        sentences = [
            "The atmosphere radiates with genuine happiness.",
            "A strong sense of joy and positivity prevails.",
            "The mood is vibrant and uplifting."
        ]
    else:
        sentences = [
            "An overwhelming sense of joy and positivity fills the air!",
            "The atmosphere is absolutely radiant with happiness!",
            "A profound sense of fulfillment and joy prevails!"
        ]
    
    return random.choice(sentences)

def prepare_data(data):
    # Extract sentiment scores
    X = np.array([entry['sentiment'] for entry in data]).reshape(-1, 1)
    
    # Create labels based on sentiment scores
    y = np.array([categorize_sentiment(entry['sentiment']) for entry in data])
    
    return X, y

def train_model():
    # Load and prepare data
    data = load_data()
    X, y = prepare_data(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    return model

def predict_sentiment(model, score):
    prediction = model.predict([[score]])[0]
    probability = model.predict_proba([[score]])[0]
    sentence = generate_sentiment_sentence(score)
    return prediction, probability, sentence

if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    # Test with some example scores
    test_scores = [-0.8, -0.2, 0.0, 0.2, 0.8]
    print("\nPredictions for test scores:")
    for score in test_scores:
        prediction, probability, sentence = predict_sentiment(model, score)
        print(f"\nScore: {score:.2f}")
        print(f"Predicted category: {prediction}")
        print(f"Description: {sentence}")
        print("Probabilities:")
        for category, prob in zip(model.classes_, probability):
            print(f"{category}: {prob:.2%}") 