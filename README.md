# Song Recognizer

This project provides a minimal example of training a neural network to recognise short audio clips. It is built on top of **PyTorch** and **torchaudio** and exposes a small command line interface for training and prediction.

## Features

- Convert audio files to Mel-spectrograms on the fly
- Simple convolutional neural network architecture
- CLI commands for training and live microphone prediction

## Installation

1. Create a virtual environment (optional but recommended)
2. Install the dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training

Place your audio files (e.g. WAV or MP3) in a directory and run:

```bash
python main.py train /path/to/audio
```

The trained model will be saved to `song_recognizer.pth`.

### Prediction

To make a prediction using the microphone run:

```bash
python main.py predict
```

or provide a prerecorded file:

```bash
python main.py predict --input_file sample.wav
```

## Project Structure

```
.
├── song_recognizer
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── recognition.py
│   └── train.py
├── main.py
├── requirements.txt
└── README.md
```

## License

MIT
