from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .data import AudioDataset
from .model import SongRecognizer


def train(data_dir: str, epochs: int = 10, batch_size: int = 4, lr: float = 1e-3, device: str | torch.device = None) -> SongRecognizer:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = AudioDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SongRecognizer(num_classes=len(dataset)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
