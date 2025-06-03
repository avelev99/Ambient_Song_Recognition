from __future__ import annotations

import sounddevice as sd
import soundfile as sf
import torch
import torchaudio


def record_audio(duration: int = 5, sample_rate: int = 44100, filename: str = "temp.wav") -> str:
    """Record audio from the microphone and save it to a file."""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    sf.write(filename, audio, sample_rate)
    return filename


def predict(model: torch.nn.Module, device: torch.device, filepath: str) -> int:
    """Predict which song is present in the given audio file."""
    model.eval()
    waveform, _ = torchaudio.load(filepath)
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    transform = torchaudio.transforms.MelSpectrogram()
    mel = transform(waveform).to(device)
    with torch.no_grad():
        output = model(mel.unsqueeze(0))
        _, pred = torch.max(output, 1)
    return int(pred.item())
