import spacy
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

# Load tools
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
analyzer = SentimentIntensityAnalyzer()
translator = Translator()

def preprocess_text(text):
    """Lemmatize and clean using spaCy."""
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

def analyze_sentiment(text):
    """Get sentiment score and label using VADER."""
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.1:
        sentiment = "Positive"
    elif score <= -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, score

def remove_repetitive_phrases(text):
    """Remove repetitive or filler phrases from the text."""
    repetitive_phrases = [
        "I don't know what to say", 
        "anything that comes to my mind", 
        "please give me the sentiment analysis", 
        "okay thank you", 
        "I'm completely bored"
    ]
    for phrase in repetitive_phrases:
        text = text.replace(phrase, "")
    return text.strip()

def process_text(text, language):
    """Summarize, translate if needed, and analyze sentiment."""

    # Step 1: Remove filler phrases (optional)
    cleaned_text = remove_repetitive_phrases(text)

    # Step 2: Set dynamic summarization limits based on word count
    word_count = len(cleaned_text.split())

    if word_count < 40:
        summary = "Transcript too short to generate a meaningful summary."
    else:
        # Dynamically adjust summary length based on input size
        max_len = 45  # cap max length to 100
        min_len = 35       # minimum threshold
        
        # Step 3: Summarize
        try:
            summary_result = summarizer(cleaned_text, max_length=max_len, min_length=min_len, do_sample=False)
            summary = summary_result[0]["summary_text"]
        except Exception as e:
            summary = "Summarization failed. Error: " + str(e)

    # Step 4: Sentiment on raw text
    sentiment, score = analyze_sentiment(text)

    # Step 5: Translate if required
    if language != "en":
        try:
            summary = translator.translate(summary, dest=language).text
        except Exception as e:
            summary = f"Translation failed: {e}"

    return summary, sentiment, score
