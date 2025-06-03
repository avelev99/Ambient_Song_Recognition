from __future__ import annotations

import argparse
import torch

from song_recognizer.train import train
from song_recognizer.model import SongRecognizer


MODEL_PATH = "song_recognizer.pth"


def cmd_train(args: argparse.Namespace) -> None:
    model = train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def cmd_predict(args: argparse.Namespace) -> None:
    from song_recognizer.recognition import record_audio, predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SongRecognizer(num_classes=args.num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    if args.input_file:
        filepath = args.input_file
    else:
        filepath = record_audio()

    pred = predict(model, device, filepath)
    print(f"Predicted label: {pred}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Song recognition CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("data_dir", help="Directory with training audio files")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Predict from microphone or file")
    predict_parser.add_argument("--input_file", help="Optional audio file to use instead of microphone")
    predict_parser.add_argument("--num_classes", type=int, default=3)
    predict_parser.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
