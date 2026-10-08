#!/usr/bin/env bash
set -e

CHECKPOINT_PATH="${1:-checkpoints/classifier/best.pt}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  echo "Tip: pass a path explicitly, for example:"
  echo "  bash scripts/show_best_checkpoint.sh checkpoints/classifier/best.pt"
  exit 1
fi

python - "$CHECKPOINT_PATH" <<'PYTHON'
import sys
from src.utils.checkpoint import load_checkpoint

checkpoint_path = sys.argv[1]
ckpt = load_checkpoint(checkpoint_path, map_location="cpu")

print("Best checkpoint summary")
print("-----------------------")
print("path:", checkpoint_path)
print("epoch:", ckpt["epoch"])
print("train_loss:", ckpt["train_loss"])
print("val_loss:", ckpt["val_loss"])
print("val_accuracy:", ckpt["val_accuracy"])
print("val_precision:", ckpt["val_precision"])
print("val_recall:", ckpt["val_recall"])
print("val_f1:", ckpt["val_f1"])
PYTHON
