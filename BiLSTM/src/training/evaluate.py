from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_labeled_batch
from src.data.dataset import LabeledExample, LabeledTextDataset
from src.data.processed_artifacts import load_processed_artifacts_from_config
from src.data.vocab import PAD_TOKEN
from src.models.bilstm_classifier import BiLSTMClassifier
from src.training.losses import compute_supervised_loss
from src.training.plotting import DEFAULT_FIGURES_DIR, DEFAULT_RESULTS_DIR, save_confusion_matrix_plot
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_and_validate_config
from src.utils.device import amp_enabled_for_device, select_device

DEFAULT_BASELINE_CONFIG_PATH = "configs/baseline.yaml"
DEFAULT_BEST_CHECKPOINT_PATH = "checkpoints/classifier/best.pt"


class EvaluationOutputs(TypedDict):
    metrics: dict[str, float]
    labels: list[int]
    predictions: list[int]


def _run_evaluation_pass(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    amp_enabled: bool,
) -> EvaluationOutputs:
    """Run evaluation once and return metrics plus raw predictions."""
    from src.utils.metrics import compute_classification_metrics

    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with autocast_context:
                logits = model(input_ids, mask)
                loss = compute_supervised_loss(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item()
            num_batches += 1
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    if num_batches == 0:
        raise ValueError("Evaluation dataloader is empty.")

    metrics = compute_classification_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / num_batches
    return {
        "metrics": metrics,
        "labels": all_labels,
        "predictions": all_predictions,
    }


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    amp_enabled: bool = False,
) -> dict[str, float]:
    """Evaluate the classifier and return loss plus binary classification metrics."""
    outputs = _run_evaluation_pass(model=model, dataloader=dataloader, device=device, amp_enabled=amp_enabled)
    return outputs["metrics"]


def collect_misclassified_examples(
    examples: list[LabeledExample],
    predictions: list[int],
) -> list[dict[str, object]]:
    """Collect misclassified examples with raw text for qualitative analysis."""
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length.")

    misclassified_examples: list[dict[str, object]] = []
    for index, (example, prediction) in enumerate(zip(examples, predictions, strict=True)):
        if example["label"] == prediction:
            continue
        misclassified_examples.append(
            {
                "index": index,
                "text": example["text"],
                "true_label": example["label"],
                "predicted_label": prediction,
                "token_count": len(example["input_ids"]),
            }
        )
    return misclassified_examples


def save_evaluation_outputs(
    metrics: dict[str, float],
    labels: list[int],
    predictions: list[int],
    test_examples: list[LabeledExample],
    run_name: str,
) -> None:
    """Persist confusion matrix, metrics, and misclassified examples for report writing."""
    metrics_path = DEFAULT_RESULTS_DIR / f"{run_name}_metrics.json"
    misclassified_path = DEFAULT_RESULTS_DIR / f"{run_name}_misclassified_examples.json"
    confusion_matrix_path = DEFAULT_FIGURES_DIR / f"{run_name}_confusion_matrix.png"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, ensure_ascii=False, indent=2)

    save_confusion_matrix_plot(labels, predictions, confusion_matrix_path)

    misclassified_examples = collect_misclassified_examples(test_examples, predictions)
    with misclassified_path.open("w", encoding="utf-8") as file_handle:
        json.dump(misclassified_examples, file_handle, ensure_ascii=False, indent=2)


def run_baseline_evaluation(
    config_path: str = DEFAULT_BASELINE_CONFIG_PATH,
    checkpoint_path: str | Path = DEFAULT_BEST_CHECKPOINT_PATH,
) -> dict[str, float]:
    """Load the best baseline checkpoint and evaluate it on the held-out test split."""
    config = load_and_validate_config(config_path, "baseline")
    artifacts = load_processed_artifacts_from_config(config)

    vocab = artifacts["vocab"]
    test_examples = artifacts["test_examples"]
    pad_value = vocab.token_to_id[PAD_TOKEN]

    model_config = config["model"]
    training_config = config["training"]
    if not isinstance(model_config, dict):
        raise TypeError("config['model'] must be a dictionary.")
    if not isinstance(training_config, dict):
        raise TypeError("config['training'] must be a dictionary.")

    device = select_device()
    amp_enabled = amp_enabled_for_device(bool(training_config.get("amp", False)), device)
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        bidirectional=bool(model_config["bidirectional"]),
        classifier_hidden_dim=int(model_config["classifier_hidden_dim"]),
        embedding_dropout=float(model_config["embedding_dropout"]),
        classifier_dropout=float(model_config["classifier_dropout"]),
    ).to(device)

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = LabeledTextDataset(test_examples)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        collate_fn=lambda batch: collate_labeled_batch(batch, pad_value=pad_value),
    )
    outputs = _run_evaluation_pass(model=model, dataloader=test_dataloader, device=device, amp_enabled=amp_enabled)
    run_name = Path(config_path).stem
    save_evaluation_outputs(outputs["metrics"], outputs["labels"], outputs["predictions"], test_examples, run_name)
    return outputs["metrics"]


def main(
    config_path: str = DEFAULT_BASELINE_CONFIG_PATH,
    checkpoint_path: str | Path = DEFAULT_BEST_CHECKPOINT_PATH,
) -> None:
    """Run held-out baseline evaluation from config and checkpoint paths."""
    metrics = run_baseline_evaluation(config_path=config_path, checkpoint_path=checkpoint_path)
    print("Evaluation complete.")
    print(
        f"loss={metrics['loss']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"precision={metrics['precision']:.4f} | "
        f"recall={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    import sys

    selected_config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE_CONFIG_PATH
    selected_checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BEST_CHECKPOINT_PATH
    main(selected_config_path, selected_checkpoint_path)
