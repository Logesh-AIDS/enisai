import os
import librosa
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to extract features from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    
    # Extracting MFCCs, Chroma, Spectral Contrast, and Zero Crossing Rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    
    features = {
        "mfcc_mean": mfccs.mean(axis=1),  # Mean MFCC values
        "chroma_mean": chroma.mean(axis=1),  # Mean Chroma values
        "contrast_mean": spectral_contrast.mean(axis=1),  # Mean Spectral Contrast values
        "zcr_mean": zcr.mean()  # Mean Zero Crossing Rate
    }
    
    return features

# Function to recommend songs based on features
def recommend_songs(features):
    # Expanded song database with more categories and Tamil songs
    songs_db = {
        "Unplugged Acoustic": [
            "Nenjukkul Peidhidum - Vaaranam Aayiram",
            "Anbil Avan - Pariyerum Perumal",
            "Kannukkul Kannai - Dheena",
            "Vennilave - Minsara Kanavu"
        ],
        "High Pitch Classical": [
            "Tham Tham - Dasavatharam",
            "Kaatru Veliyidai - Kaatru Veliyidai",
            "Vennilave - Minsara Kanavu",
            "Suttrum Vizhi - Kandukondain Kandukondain"
        ],
        "Soft Lyrical": [
            "Vaanam Kottattum - Vaanam Kottattum",
            "Kaathalae Kaathalae - 96",
            "Suttrum Vizhi - Kandukondain Kandukondain",
            "Pudhu Vellai Mazhai - Roja"
        ],
        "Romantic Ballads": [
            "Enna Solla Pogirai - Kandukondain Kandukondain",
            "Kaadhal Rojave - Roja",
            "Chandralekha - Thiruda Thiruda",
            "Thendral Vandhu - Baasha"
        ],
        "Energetic Beats": [
            "Aathichudi - TN 07 AL 4777",
            "Zinda - Bhaag Milkha Bhaag",
            "Surviva - Vivegam",
            "Vivegam Theme - Vivegam"
        ],
        "Classical": [
            "Madhura - Ilaiyaraaja",
            "Oru Vannarapettai - M.S. Subbulakshmi",
            "Vandhanam - K.V. Mahadevan",
            "Manathil Urudhi Vendum - Rajinikanth"
        ],
        # Add more categories as required
    }

    # Actual recommendation logic based on MFCC values (just an example)
    if features["mfcc_mean"][0] > 0.5:
        return "Unplugged Acoustic", songs_db["Unplugged Acoustic"]
    elif features["mfcc_mean"][1] > 0.3:
        return "High Pitch Classical", songs_db["High Pitch Classical"]
    elif features["mfcc_mean"][2] > 0.4:
        return "Soft Lyrical", songs_db["Soft Lyrical"]
    elif features["mfcc_mean"][3] > 0.2:
        return "Romantic Ballads", songs_db["Romantic Ballads"]
    elif features["mfcc_mean"][4] > 0.5:
        return "Energetic Beats", songs_db["Energetic Beats"]
    else:
        return "Classical", songs_db["Classical"]

# Home route (Render HTML with upload form)
@app.route('/')
def home():
    return render_template('index.html')

# Upload and process audio file
@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No file part"})
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save the uploaded file
    upload_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(upload_path)
    
    # Extract features from the uploaded audio file
    features = extract_features(upload_path)
    
    # Recommend songs based on the extracted features
    song_type, recommended_songs = recommend_songs(features)
    
    # Return the recommended song type and songs
    return jsonify({
        "song_type": song_type,
        "recommended_songs": recommended_songs
    })

if __name__ == "__main__":
    app.run(debug=True)
