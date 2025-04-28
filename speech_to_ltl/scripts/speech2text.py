import openai
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import yaml


output_file = "instruction.mp3"
# Function to record and save audio
def record_and_save():
    # Set the sampling frequency and duration
    fs = 44100  # 44.1 kHz
    duration = int(input("Enter seconds required to record the command: \n"))  # seconds

    print("Recording will stop in ",str(duration), " seconds")
    audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    
    # Convert to AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=fs,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )

    # Save as MP3
    audio_segment.export(output_file, format="mp3")
    print(f"Recording finished. Audio saved as {output_file}")

# Function to convert audio to text
def speech2instruction(openai_key):
    record_and_save()
    audio_file_path = output_file
    audio_file = open(audio_file_path, "rb")

    transcription = openai.Audio.translate(file=audio_file,api_key=openai_key,model="whisper-1")
    audio_file.close()
    return transcription["text"]

# Record and save audio

if __name__ == "__main__":

    with open("config/config_openai.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    openai_key = config['openai_api_key']

    instruc = speech2instruction(openai_key)
    print(instruc)


