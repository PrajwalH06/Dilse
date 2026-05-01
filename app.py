from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np

from datetime import datetime
from werkzeug.utils import secure_filename
from speech_to_text import transcribe_audio
from summarizer import process_text
from sentiment_model import generate_sentiment_sentence

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# JSON file path
ENTRY_FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'entries.json')

# Ensure entries.json exists
if not os.path.exists(ENTRY_FILE_PATH):
    with open(ENTRY_FILE_PATH, 'w') as f:
        json.dump([], f)

# Google Cloud credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials", "dilse_apikey.json")


# ---------- Routes ----------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route("/analyze", methods=["POST"])
def analyze():
    language = request.form.get("language")
    audio = request.files.get("audio")
    date = request.form.get("date")  # Get date from form

    if not audio:
        return "No audio file uploaded.", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio.filename))
    audio.save(filepath)

    transcript = transcribe_audio(filepath)

    if len(transcript.split()) < 5:
        summary = "Text too short for meaningful sentiment analysis."
        sentiment = "😐 Neutral"
        score = 0.0
    else:
        summary, sentiment, score = process_text(transcript, language)

    return render_template("result.html", summary=summary, sentiment=sentiment,
                           transcript=transcript, score=score, date=date)

@app.route("/save", methods=["POST"])
def save_entry():
    data = request.get_json()

    date = data.get("date")
    summary = data.get("summary")
    transcript = data.get("transcript")
    sentiment = data.get("sentiment")
    score = data.get("score")

    if not date or not summary:
        return jsonify({"status": "error", "message": "Date and summary are required."})

    try:
        with open(ENTRY_FILE_PATH, 'r') as f:
            entries = json.load(f)
    except (IOError, json.JSONDecodeError):
        entries = []

    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "date": date,
        "summary": summary,
        "transcript": transcript,
        "sentiment": sentiment,
        "score": score
    }
    entries.append(entry)

    try:
        with open(ENTRY_FILE_PATH, 'w') as f:
            json.dump(entries, f, indent=4)
        return jsonify({"status": "success", "message": "Entry saved!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error saving entry: {str(e)}"})

@app.route("/get_entries", methods=["GET"])
def get_entries():
    try:
        with open(ENTRY_FILE_PATH, 'r') as f:
            entries = json.load(f)
            
        # Sort entries by date in ascending order
        entries_sorted = sorted(entries, key=lambda x: x['date'])
        
        # Ensure all scores are floats and within valid range
        for entry in entries_sorted:
            try:
                score = float(entry['score'])
                # Ensure score is between -1 and 1
                entry['score'] = max(min(score, 1.0), -1.0)
            except (ValueError, TypeError):
                entry['score'] = 0.0
                
        return jsonify(entries_sorted)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching entries: {str(e)}"})

@app.route("/delete_entry/<entry_id>", methods=["DELETE"])
def delete_entry(entry_id):
    try:
        with open(ENTRY_FILE_PATH, 'r') as f:
            entries = json.load(f)

        # Find the entry to delete
        entry_to_delete = None
        for entry in entries:
            if str(entry.get('id')) == str(entry_id):
                entry_to_delete = entry
                break

        if not entry_to_delete:
            return jsonify({"status": "error", "message": "Entry not found"}), 404

        # Remove the entry
        entries = [entry for entry in entries if str(entry.get('id')) != str(entry_id)]

        with open(ENTRY_FILE_PATH, 'w') as f:
            json.dump(entries, f, indent=4)

        return jsonify({"status": "success", "message": "Entry deleted!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error deleting entry: {str(e)}"}), 500


@app.route("/edit_entry/<entry_id>", methods=["POST"])
def edit_entry(entry_id):
    new_data = request.get_json()
    try:
        with open(ENTRY_FILE_PATH, 'r') as f:
            entries = json.load(f)

        for entry in entries:
            if entry['id'] == entry_id:
                entry['transcript'] = new_data.get("transcript", entry['transcript'])
                entry['summary'] = new_data.get("summary", entry['summary'])
                entry['sentiment'] = new_data.get("sentiment", entry['sentiment'])
                entry['score'] = new_data.get("score", entry['score'])
                break

        with open(ENTRY_FILE_PATH, 'w') as f:
            json.dump(entries, f, indent=4)

        return jsonify({"status": "success", "message": "Entry updated!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error editing entry: {str(e)}"})

@app.route("/entries", methods=["GET"])
def entries():
    try:
        with open(ENTRY_FILE_PATH, 'r') as f:
            entries = json.load(f)

        entries_sorted = sorted(entries, key=lambda x: x['date'], reverse=True)
        return render_template("entries.html", entries=entries_sorted)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error loading entries: {str(e)}"})

def calculate_average_sentiment():
    try:
        with open(ENTRY_FILE_PATH, 'r') as f:
            data = json.load(f)
        scores = [entry['score'] for entry in data]  # Changed from 'sentiment' to 'score' to match the data structure
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"Error calculating average sentiment: {str(e)}")
        return 0.0

@app.route('/chart')
def chart():
    avg_sentiment = calculate_average_sentiment()
    sentiment_sentence = generate_sentiment_sentence(avg_sentiment)
    return render_template('chart.html', 
                         average_sentiment=avg_sentiment,
                         sentiment_sentence=sentiment_sentence)

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
