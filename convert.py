import os
from pydub import AudioSegment

# Root data folder
DATASET_PATH = "data"

for accent in os.listdir(DATASET_PATH):
    accent_path = os.path.join(DATASET_PATH, accent)

    if os.path.isdir(accent_path):
        for file in os.listdir(accent_path):
            if file.endswith(".mp3"):
                mp3_path = os.path.join(accent_path, file)
                wav_path = os.path.join(accent_path, file.replace(".mp3", ".wav"))

                print(f"Converting {file}...")

                audio = AudioSegment.from_mp3(mp3_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(wav_path, format="wav")

                print(f"Saved: {wav_path}")

print("All conversions completed!")