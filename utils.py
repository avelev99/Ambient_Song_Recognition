# utils.py
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sounddevice as sd
import soundfile as sf

class AudioDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.audio_files = ["1.mp3", "2.mp3", "3.mp3"]
        self.labels = list(range(len(self.audio_files)))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_folder, self.audio_files[idx])
        waveform, _ = torchaudio.load(filepath)
        return waveform, self.labels[idx]

def recognize_song(model, device):
    def record_audio():
        duration = 5  # seconds
        fs = 44100
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        return audio

    model.eval()
    audio = record_audio()
    sf.write("temp.wav", audio, 44100)

    waveform, _ = torchaudio.load("temp.wav")
    waveform = waveform.to(device)
    with torch.no_grad():
        outputs = model(waveform.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        print("Predicted song:", predicted.item())