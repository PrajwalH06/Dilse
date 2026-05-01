# Dilse

Dilse is an intelligent audio journaling and sentiment analysis web application. It allows users to record or upload voice entries, which are then transcribed, summarized, and analyzed for sentiment to help track emotional well-being over time.

## Features

- **Speech-to-Text Conversion**: Uses Google Cloud Speech-to-Text API to accurately transcribe voice entries into text.
- **Automated Summarization**: Employs the Hugging Face `facebook/bart-large-cnn` model via Transformers to generate concise summaries of longer audio journals.
- **Sentiment Analysis**: Evaluates the emotional tone of entries using VADER Sentiment Analysis and categorizes them using a custom Logistic Regression model.
- **Multi-language Support**: Automatically translates summaries to the desired language using Google Translate.
- **Journal History & Visualization**: Saves entries locally and visualizes your emotional journey over time, calculating an average sentiment score to describe your overall mood.

## Technologies Used

- **Backend**: Flask (Python)
- **AI/ML**: 
  - `transformers` (Summarization)
  - `vaderSentiment` (Sentiment Scoring)
  - `spacy` (Text Preprocessing)
  - `scikit-learn` (Sentiment Categorization Model)
- **Cloud Services**: Google Cloud Speech-to-Text API

## Setup & Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. Setup Google Cloud Credentials:
   - Obtain a JSON service account key with Speech-to-Text API access.
   - Place it in `credentials/dilse_apikey.json`.

4. Run the application:
   ```bash
   python app.py
   ```
   The app will be accessible at `http://127.0.0.1:5000/`.

## Architecture

- `app.py`: Main Flask application, handles routing and API endpoints.
- `speech_to_text.py`: Integrates with Google Cloud to transcribe audio files.
- `summarizer.py`: Orchestrates text preprocessing, summarization, and VADER sentiment analysis.
- `sentiment_model.py`: Trains and applies a logistic regression classifier on sentiment scores to generate descriptive mood categorizations.

## License

This project is licensed under the MIT License.
