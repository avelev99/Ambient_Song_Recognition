from __future__ import annotations

import os
from typing import List

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Dataset reading audio files from a directory and converting them to mel spectrograms."""

    def __init__(self, data_dir: str, extensions: tuple[str, ...] = (".wav", ".mp3")):
        self.data_dir = data_dir
        self.extensions = extensions
        self.filepaths: List[str] = [
            os.path.join(data_dir, f)
            for f in sorted(os.listdir(data_dir))
            if f.lower().endswith(extensions)
        ]
        if not self.filepaths:
            raise RuntimeError(f"No audio files with extensions {extensions} found in {data_dir}")
        self.transform = torchaudio.transforms.MelSpectrogram()

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        waveform, _ = torchaudio.load(path)
        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        mel = self.transform(waveform)
        # Resize to a fixed size for batching
        mel = torch.nn.functional.interpolate(
            mel.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
        ).squeeze(0)
        label = idx  # Each file is its own label
        return mel, label
