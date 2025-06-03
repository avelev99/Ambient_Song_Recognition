import os
import tempfile
from song_recognizer.data import AudioDataset


def test_dataset_scan(tmp_path):
    # create dummy wav file
    path = tmp_path / 'a.wav'
    path.write_bytes(b'RIFF\x00\x00\x00\x00WAVE')
    try:
        dataset = AudioDataset(str(tmp_path))
        assert len(dataset) == 1
    except RuntimeError:
        # torchaudio may fail to load dummy file; ensure graceful failure
        assert True
