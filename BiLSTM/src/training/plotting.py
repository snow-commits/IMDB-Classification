from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_DIR / "results"

BASELINE_LOSS_PLOT_PATH = DEFAULT_FIGURES_DIR / "baseline_train_val_loss.png"
BASELINE_VAL_ACCURACY_PLOT_PATH = DEFAULT_FIGURES_DIR / "baseline_val_accuracy.png"
BASELINE_VAL_F1_PLOT_PATH = DEFAULT_FIGURES_DIR / "baseline_val_f1.png"
BASELINE_LABEL_DISTRIBUTION_PLOT_PATH = DEFAULT_FIGURES_DIR / "baseline_label_distribution.png"
BASELINE_TOKEN_LENGTH_DISTRIBUTION_PLOT_PATH = DEFAULT_FIGURES_DIR / "baseline_token_length_distribution.png"
BASELINE_CONFUSION_MATRIX_PLOT_PATH = DEFAULT_FIGURES_DIR / "baseline_confusion_matrix.png"


def _ensure_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_train_val_loss_plot(history: Sequence[dict[str, float | int]], output_path: str | Path) -> Path:
    """Save a train-versus-validation loss plot for the provided epoch history."""
    history_rows = list(history)
    if not history_rows:
        raise ValueError("history must contain at least one epoch result.")

    path = _ensure_output_path(output_path)
    epochs = [int(epoch_result["epoch"]) for epoch_result in history_rows]
    train_losses = [float(epoch_result["train_loss"]) for epoch_result in history_rows]
    val_losses = [float(epoch_result["val_loss"]) for epoch_result in history_rows]

    figure, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", color="#1f77b4", marker="o", linewidth=2.0)
    ax.plot(epochs, val_losses, label="Validation loss", color="#ff7f0e", marker="s", linestyle="--", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def save_validation_metric_plot(
    history: Sequence[dict[str, float | int]],
    metric_key: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> Path:
    """Save a validation metric curve across epochs."""
    history_rows = list(history)
    if not history_rows:
        raise ValueError("history must contain at least one epoch result.")

    path = _ensure_output_path(output_path)
    epochs = [int(epoch_result["epoch"]) for epoch_result in history_rows]
    metric_values = [float(epoch_result[metric_key]) for epoch_result in history_rows]

    figure, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, metric_values, color="#2ca02c", marker="o", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def save_label_distribution_plot(train_labels: Sequence[int], test_labels: Sequence[int], output_path: str | Path) -> Path:
    """Save a train/test label distribution bar chart."""
    path = _ensure_output_path(output_path)
    train_negative = sum(label == 0 for label in train_labels)
    train_positive = sum(label == 1 for label in train_labels)
    test_negative = sum(label == 0 for label in test_labels)
    test_positive = sum(label == 1 for label in test_labels)

    figure, ax = plt.subplots(figsize=(8, 5))
    positions = [0, 1]
    width = 0.35
    ax.bar([position - width / 2 for position in positions], [train_negative, train_positive], width=width, label="Train")
    ax.bar([position + width / 2 for position in positions], [test_negative, test_positive], width=width, label="Test")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution")
    ax.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def save_token_length_distribution_plot(
    train_lengths: Sequence[int],
    test_lengths: Sequence[int],
    output_path: str | Path,
) -> Path:
    """Save a token-length distribution histogram for train and test splits."""
    path = _ensure_output_path(output_path)
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_lengths, bins=50, alpha=0.6, label="Train")
    ax.hist(test_lengths, bins=50, alpha=0.6, label="Test")
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Length Distribution")
    ax.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def save_confusion_matrix_plot(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: str | Path,
) -> Path:
    """Save a confusion matrix heatmap for binary classification."""
    path = _ensure_output_path(output_path)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Negative", "Positive"])
    display.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
