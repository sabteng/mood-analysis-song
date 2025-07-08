import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
import lyricsgenius
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Genius API setup
genius = lyricsgenius.Genius(
    os.environ["GENIUS_API_TOKEN"],
    skip_non_songs=True,
    remove_section_headers=True
)

# VADER setup
sia = SentimentIntensityAnalyzer()


def analyze_song(song_path):
    y, sr = librosa.load(song_path)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if tempo < 100:
        tempo *= 2

    y_harmonic, _ = librosa.effects.hpss(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    key = np.mean(tonnetz, axis=1)

    energy = np.mean(librosa.feature.rms(y=y))

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast)
    spectral_contrast_std = np.std(spectral_contrast)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return tempo, key, energy, spectral_contrast_mean, spectral_contrast_std, mfcc_mean, mfcc_std


def classify_mood(
    tempo, key, energy,
    spectral_contrast_mean, spectral_contrast_std,
    mfcc_mean, mfcc_std,
    compound_sentiment
):
    key_centrality = np.mean(key)
    mfcc_brightness = np.mean(mfcc_mean)
    mfcc_variability = np.mean(mfcc_std)

    candidates = []

    if compound_sentiment < -0.2:
        print(energy, tempo)
        if energy < 0.3 and tempo < 130:
            candidates.append("Sad")
        if energy >= 0.1 and mfcc_brightness < 2:
            candidates.append("Dark")
        if energy > 0.25 and tempo > 110 and mfcc_variability > 4:
            candidates.append("Energetic")
        if tempo < 140 and energy < 0.4:
            candidates.append("Melancholic")

    if compound_sentiment > 0.2:
        if energy > 0.15 and tempo >= 90:
            candidates.append("Happy")
        if energy > 0.2 and tempo >= 110:
            candidates.append("Uplifting")
        if energy > 0.25 and tempo > 110 and mfcc_variability > 4:
            candidates.append("Energetic")
        if tempo < 140 and energy < 0.4:
            candidates.append("Melancholic")

    if energy < 0.15 and tempo < 110 and compound_sentiment > -0.3:
        candidates.append("Calm")

    priority = ["Calm", "Energetic", "Uplifting", "Happy", "Melancholic", "Sad", "Dark"]

    for mood in priority:
        if mood in candidates:
            return mood

    return "Neutral"


def process_song(song_path):
    # Analyse
    tempo, key, energy, spectral_contrast_mean, spectral_contrast_std, mfcc_mean, mfcc_std = analyze_song(song_path)

    # Infer artist and title
    base = os.path.basename(song_path)
    parts = os.path.splitext(base)[0].split("-")
    if len(parts) >= 2:
        artist = parts[0].strip()
        title = parts[1].strip()
    else:
        artist = ""
        title = parts[0].strip()

    # Lyrics sentiment
    compound_sentiment = 0
    if artist and title:
        try:
            song = genius.search_song(title, artist)
            if song and song.lyrics:
                sentiment = sia.polarity_scores(song.lyrics)
                compound_sentiment = sentiment["compound"]
        except Exception as e:
            print(f"Lyrics fetch failed for {title}: {e}")
            compound_sentiment = 0

    # Classify
    mood = classify_mood(
        tempo, key, energy,
        spectral_contrast_mean, spectral_contrast_std,
        mfcc_mean, mfcc_std,
        compound_sentiment
    )

    return mood


def create_gui():
    window = tk.Tk()
    window.title("Music Mood Classifier")

    label = tk.Label(window, text="Upload a song to predict its mood!", font=("Arial", 16))
    label.pack()

    def handle_upload():
        song_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        if song_path:
            mood = process_song(song_path)
            label.config(text=f"Mood: {mood}")

    button = tk.Button(window, text="Upload Song", command=handle_upload)
    button.pack()
    window.mainloop()


def batch_process(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".mp3") or file.endswith(".wav"):
            song_path = os.path.join(folder_path, file)
            mood = process_song(song_path)
            print(f"{file} => Mood: {mood}")


if __name__ == "__main__":
    # Choose one or the other:
    # create_gui()
    # or
    batch_process("test_songs")

