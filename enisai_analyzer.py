from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to extract audio features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = {
        "mfcc_mean": np.mean(mfcc, axis=1),
        "chroma_mean": np.mean(chroma, axis=1),
        "zcr_mean": np.mean(zcr),
        "contrast_mean": np.mean(contrast, axis=1),
    }
    return features

# Songs database (expanded with 13 categories)
def recommend_songs(features):
    songs_db = {
        "Pop Smooth Vocals": [
            "Blinding Lights - The Weeknd", "Shape of You - Ed Sheeran", "Vaseegara - Minnale",
            "As It Was - Harry Styles", "Someone You Loved - Lewis Capaldi", "Love Story - Taylor Swift",
            "Perfect - Ed Sheeran", "Thinking Out Loud - Ed Sheeran", "Unakkenna Venum Sollu - Yennai Arindhaal",
            "Ennavale Adi Ennavale - Kadhalan"
        ],
        "Rock Raspy Vocals": [
            "Smells Like Teen Spirit - Nirvana", "Highway to Hell - AC/DC", "In the End - Linkin Park",
            "Iris - Goo Goo Dolls", "Numb - Linkin Park", "Oru Maalai - Ghajini",
            "Back in Black - AC/DC", "It's My Life - Bon Jovi", "Zombie - The Cranberries",
            "Sweet Child O' Mine - Guns N' Roses"
        ],
        "High Pitch Classical": [
            "Nessun Dorma - Pavarotti", "Kurai Ondrum Illai - M.S. Subbulakshmi", "O Paalanhaare - Lagaan",
            "Vennilave Vennilave - Minsara Kanavu", "Ave Maria - Schubert", "Ava Enna - Vaaranam Aayiram",
            "Queen of the Night Aria - Mozart", "Paadariyen Padippariyen - Sindhu Bhairavi",
            "Vidai Kodu Engal Naadae - Kannathil Muthamittal", "Kanmani Anbodu - Guna"
        ],
        "Soulful Ballads": [
            "Someone Like You - Adele", "Photograph - Ed Sheeran", "Hello - Lionel Richie",
            "Munbe Vaa - Sillunu Oru Kaadhal", "Kaathalae Kaathalae - 96", "Tujh Mein Rab Dikhta Hai - RNBDJ",
            "Jeene Laga Hoon - Ramaiya Vastavaiya", "Say You Won't Let Go - James Arthur",
            "All of Me - John Legend", "Engeyum Kadhal - Engeyum Kadhal"
        ],
        "Energetic Hip-Hop/Rap": [
            "Lose Yourself - Eminem", "Bad Guy - Billie Eilish", "Sicko Mode - Travis Scott",
            "Surviva - Vivegam", "Venom - Eminem", "Vaathi Coming - Master",
            "Without Me - Eminem", "Rap God - Eminem", "DNA - Kendrick Lamar",
            "Aaluma Doluma - Vedalam"
        ],
        "Smooth Jazz Vocals": [
            "Fly Me to the Moon - Frank Sinatra", "Come Away With Me - Norah Jones",
            "Sway - Michael Bublé", "L-O-V-E - Nat King Cole", "Neela Kuyil - Ilaiyaraaja",
            "Feeling Good - Nina Simone", "Cheek to Cheek - Ella Fitzgerald",
            "Save the Last Dance for Me - Michael Bublé", "Kadhal Sadugudu - Alaipayuthey",
            "Summertime - Louis Armstrong"
        ],
        "Heavy Metal Screaming": [
            "Enter Sandman - Metallica", "Painkiller - Judas Priest", "The Trooper - Iron Maiden",
            "Duality - Slipknot", "Killing in the Name - Rage Against the Machine",
            "Psychosocial - Slipknot", "Master of Puppets - Metallica",
            "Hallowed Be Thy Name - Iron Maiden", "Valhalla Calling Me - Miracle of Sound",
            "Before I Forget - Slipknot"
        ],
        "Low Pitch Bass Vocals": [
            "The Sound of Silence - Disturbed", "Way Down We Go - Kaleo", "Believer - Imagine Dragons",
            "Fix You - Coldplay", "Can't Help Falling In Love - Elvis Presley",
            "Unakkul Naane - Pachaikili Muthucharam", "Let Her Go - Passenger",
            "Tennessee Whiskey - Chris Stapleton", "I See Fire - Ed Sheeran",
            "Kannodu Kanbathellam - Jeans"
        ],
        "Acoustic Unplugged": [
            "Tears in Heaven - Eric Clapton", "Wish You Were Here - Pink Floyd",
            "Hey There Delilah - Plain White T’s", "Thinking Out Loud - Ed Sheeran",
            "Idhu Varai - Goa", "Blackbird - The Beatles",
            "Hotel California (Acoustic) - Eagles", "Layla (Unplugged) - Eric Clapton",
            "Nenjukkul Peidhidum - Vaaranam Aayiram", "Yellow - Coldplay"
        ],
        "Folk Traditional": [
            "Jimmiki Kammal - Velipadinte Pusthakam", "Cotton Fields - Lead Belly",
            "Oyatha Yathraiye - LKG", "The Gambler - Kenny Rogers", "Mayakkama Kalakkama - M.S. Viswanathan",
            "Ring of Fire - Johnny Cash", "Country Roads - John Denver",
            "Achy Breaky Heart - Billy Ray Cyrus", "Raasathi Raasathi - Thiruda Thiruda",
            "Take Me Home, Country Roads - John Denver"
        ],
        "Dance EDM Vocals": [
            "Titanium - David Guetta ft. Sia", "Wake Me Up - Avicii", "Levitating - Dua Lipa",
            "Onnume Puriyala - Jigarthanda", "Animals - Martin Garrix",
            "Don't You Worry Child - Swedish House Mafia", "Clarity - Zedd ft. Foxes",
            "Stay - Zedd & Alessia Cara", "Faded - Alan Walker", "Call on Me - Eric Prydz"
        ],
        "Sad/Melancholic Vocals": [
            "Yesterday - The Beatles", "Fix You - Coldplay", "Boulevard of Broken Dreams - Green Day",
            "Konjam Nilavu - Thiruda Thiruda", "Unakkenna Venum Sollu - Yennai Arindhaal",
            "Let Her Go - Passenger", "Someone Like You - Adele", "Tujhe Bhula Diya - Anjaana Anjaani",
            "My Immortal - Evanescence", "Tears Dry on Their Own - Amy Winehouse"
        ],
        "Opera Powerful Vocals": [
            "O Sole Mio - Luciano Pavarotti", "Time to Say Goodbye - Andrea Bocelli & Sarah Brightman",
            "Amigos Para Siempre - Jose Carreras", "Va Pensiero - Giuseppe Verdi",
            "Nella Fantasia - Sarah Brightman", "Con te partirò - Andrea Bocelli",
            "Di Capua - O Sole Mio", "The Prayer - Andrea Bocelli & Celine Dion",
            "La Donna è Mobile - Verdi", "Casta Diva - Bellini"
        ]# Add all the song categories as in your original code
    }

    # Basic matching logic (later replaced with ML)
    mfcc_energy = np.mean(features["mfcc_mean"])
    zcr_value = features["zcr_mean"]

    if mfcc_energy > 100 and zcr_value > 0.05:
        return "Energetic Hip-Hop/Rap", songs_db["Energetic Hip-Hop/Rap"]
    elif mfcc_energy > 50:
        return "Pop Smooth Vocals", songs_db["Pop Smooth Vocals"]
    elif mfcc_energy < -50:
        return "Sad/Melancholic Vocals", songs_db["Sad/Melancholic Vocals"]
    elif features["contrast_mean"].mean() > 30:
        return "Rock Raspy Vocals", songs_db["Rock Raspy Vocals"]
    elif features["chroma_mean"].mean() > 0.5:
        return "Dance EDM Vocals", songs_db["Dance EDM Vocals"]
    else:
        return "Soulful Ballads", songs_db["Soulful Ballads"]

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    features = extract_features(file_path)
    category, songs = recommend_songs(features)

    return render_template('index.html', song_type=category, recommended_songs=songs)

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
