import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
import lyricsgenius
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

genius = lyricsgenius.Genius(
    "Wj7TpmXRRUzEo5buJAmUrFLKUm1zr00itM56iQyFgjczN2y6eq9cvErPiKztjxKT",
    skip_non_songs=True,
    remove_section_headers=True
)

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
    tempo, key, energy, spectral_contrast_mean, spectral_contrast_std, mfcc_mean, mfcc_std = analyze_song(song_path)

    base = os.path.basename(song_path)
    parts = os.path.splitext(base)[0].split("-")
    if len(parts) >= 2:
        artist = parts[0].strip()
        title = parts[1].strip()
    else:
        artist = ""
        title = parts[0].strip()

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

    mood = classify_mood(
        tempo, key, energy,
        spectral_contrast_mean, spectral_contrast_std,
        mfcc_mean, mfcc_std,
        compound_sentiment
    )

    return mood


import tkinter as tk
from tkinter import filedialog
from tkinter import ttk  # <-- for the progress bar
import os
import threading

def create_gui():
    window = tk.Tk()
    window.title("Music Mood Classifier")

    label = tk.Label(window, text="Upload song(s) to predict mood!\n File name: 'artist - song_name'", font=("Arial", 16))
    label.pack(pady=10)

    text_output = tk.Text(window, height=20, width=60, font=("Arial", 12), state="disabled")

    text_output.pack(pady=10)

    progress = ttk.Progressbar(window, orient="horizontal", mode="determinate", length=400)
    progress.pack(pady=10)

    def log_to_output(text):
        text_output.tk.call(text_output._w, 'configure', '-state', 'normal')
        text_output.insert(tk.END, text + '\n')
        text_output.see(tk.END)
        text_output.tk.call(text_output._w, 'configure', '-state', 'disabled')
        window.update_idletasks()

    def disable_buttons():
        upload_song_button.config(state="disabled")
        upload_folder_button.config(state="disabled")

    def enable_buttons():
        upload_song_button.config(state="normal")
        upload_folder_button.config(state="normal")

    def handle_song_upload():
        song_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        progress["maximum"] = 1
        progress["value"] = 0
        disable_buttons()
        if song_path:
            log_to_output(f"Processing {os.path.basename(song_path)}...")
            mood = process_song(song_path)
            log_to_output(f"→ Mood: {mood}\n")
            progress["value"] = 1
        enable_buttons()

    def handle_folder_upload():
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        def process_folder():
            disable_buttons()
            log_to_output(f"Processing folder: {folder_path}\n")

            files = [f for f in os.listdir(folder_path) if f.endswith(".mp3") or f.endswith(".wav")]
            total_files = len(files)

            if total_files == 0:
                log_to_output("No audio files found in folder.\n")
                return

            progress["maximum"] = total_files
            progress["value"] = 0

            for idx, file in enumerate(files, start=1):
                song_path = os.path.join(folder_path, file)
                log_to_output(f"Processing {file}...")
                window.update_idletasks()
                try:
                    mood = process_song(song_path)
                    log_to_output(f"→ Mood: {mood}\n")
                except Exception as e:
                    log_to_output(f"→ Error: {e}\n")

                progress["value"] = idx
                window.update_idletasks()

            log_to_output("✅ Folder processing complete.\n")
            enable_buttons()

        threading.Thread(target=process_folder, daemon=True).start()

    upload_song_button = tk.Button(window, text="Upload Song", command=handle_song_upload, width=30)
    upload_song_button.pack(pady=5)

    upload_folder_button = tk.Button(window, text="Upload Folder", command=handle_folder_upload, width=30)
    upload_folder_button.pack(pady=5)

    window.mainloop()


if __name__ == "__main__":
    create_gui()

