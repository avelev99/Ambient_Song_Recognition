# 🎵 Song Recognizer 🎵

A Python program that uses PyTorch to train a neural network to recognize a song from microphone input. The program trains on 3 mp3 files from the local directory and utilizes pandas and numpy. After training, it attempts to recognize the song from the microphone.

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or later
- pip

### Installation

1. Clone the repository:
git clone https://github.com/your_username/song-recognizer.git

2. Change to the project directory:
cd song-recognizer

3. Install the dependencies:
pip install -r requirements.txt

## 🎤 Usage

1. Add your training songs (1.mp3, 2.mp3, and 3.mp3) to the `data` folder.

2. Run the program:
python main.py

3. After the model has been trained, it will prompt you to record a 5-second audio sample from your microphone.

4. The program will predict the song from the recorded sample and display the result.

## 📁 Project Structure

📁 project
├── 📁 data
│ ├── 🎵 1.mp3
│ ├── 🎵 2.mp3
│ ├── 🎵 3.mp3
├── 📄 main.py
├── 📄 model.py
├── 📄 utils.py
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md

## 📚 Libraries and Frameworks

- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/stable/index.html)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- [soundfile](https://pysoundfile.readthedocs.io/)

## 📝 License

Unlicensed, do what your heart desires with this :D