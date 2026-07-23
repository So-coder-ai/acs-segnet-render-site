"""Create a smaller FP16 inference checkpoint during the Render build."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def convert_to_fp16(value):
    if isinstance(value, torch.Tensor):
        return value.half() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: convert_to_fp16(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert_to_fp16(item) for item in value]
    if isinstance(value, tuple):
        return tuple(convert_to_fp16(item) for item in value)
    return value


def main(source: Path, destination: Path) -> None:
    if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
        print(f"Using existing FP16 checkpoint: {destination}")
        return

    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch releases do not support weights_only.
        checkpoint = torch.load(source, map_location="cpu")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(convert_to_fp16(checkpoint), temporary)
    temporary.replace(destination)
    print(f"Created FP16 checkpoint: {destination}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python convert_checkpoint.py SOURCE.pth DESTINATION.pth")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
