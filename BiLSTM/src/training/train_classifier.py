from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TypedDict

import torch
# Unused duplicate baseline entrypoint support; canonical entrypoint is baseline_runner.
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader

# from src.data.collate import collate_labeled_batch
# from src.data.dataset import LabeledExample, LabeledTextDataset
# from src.data.preprocess import run_preprocessing_from_config
# from src.data.vocab import PAD_TOKEN
# from src.models.bilstm_classifier import BiLSTMClassifier
from src.training.evaluate import evaluate_classifier
from src.training.losses import compute_supervised_loss
from src.utils.checkpoint import save_checkpoint
# from src.utils.config import load_and_validate_config
# from src.utils.device import amp_enabled_for_device, select_device
# from src.utils.seed import set_seed

# DEFAULT_BASELINE_CONFIG_PATH = "configs/baseline.yaml"
# DEFAULT_LATEST_CHECKPOINT_PATH = "checkpoints/classifier/latest.pt"
# DEFAULT_BEST_CHECKPOINT_PATH = "checkpoints/classifier/best.pt"
# DEFAULT_VAL_SPLIT_RATIO = 0.1


class EpochResult(TypedDict):
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float


# Unused duplicate of baseline_runner.split_labeled_examples; kept commented for traceability.
# def split_labeled_examples(
#     examples: list[LabeledExample],
#     val_split_ratio: float,
#     split_seed: int,
# ) -> tuple[list[LabeledExample], list[LabeledExample]]:
#     """Create a reproducible stratified train/validation split from labeled examples."""
#     if not 0.0 < val_split_ratio < 1.0:
#         raise ValueError("val_split_ratio must be between 0 and 1.")
#
#     labels = [example["label"] for example in examples]
#     train_examples, val_examples = train_test_split(
#         examples,
#         test_size=val_split_ratio,
#         random_state=split_seed,
#         shuffle=True,
#         stratify=labels,
#     )
#     return list(train_examples), list(val_examples)
#

def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    grad_accum_steps: int,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
) -> float:
    """Run one supervised training epoch and return the average loss."""
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1.")

    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    total_steps = len(dataloader)

    for step_index, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
        with autocast_context:
            logits = model(input_ids, mask)
            loss = compute_supervised_loss(logits, labels)
            scaled_loss = loss / grad_accum_steps

        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()

        should_step = step_index % grad_accum_steps == 0 or step_index == total_steps
        if should_step:
            if scaler is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Dataloader is empty.")

    return total_loss / num_batches


def train_model(
    model: torch.nn.Module,
    train_dataloader,
    val_dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    grad_clip_norm: float,
    grad_accum_steps: int,
    amp_enabled: bool,
    checkpoint_path: str | Path,
    best_checkpoint_path: str | Path,
    early_stopping_patience: int,
) -> list[EpochResult]:
    """Train the classifier across epochs with validation, checkpointing, and early stopping."""
    history: list[EpochResult] = []
    best_val_f1 = float("-inf")
    epochs_without_improvement = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            grad_accum_steps=grad_accum_steps,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        val_metrics = evaluate_classifier(
            model=model,
            dataloader=val_dataloader,
            device=device,
            amp_enabled=amp_enabled,
        )

        epoch_result: EpochResult = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_result)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "amp_enabled": amp_enabled,
            "grad_accum_steps": grad_accum_steps,
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        save_checkpoint(checkpoint, checkpoint_path)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            save_checkpoint(checkpoint, best_checkpoint_path)
            print(f"New best model saved with val_f1={best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs. "
                f"No improvement in val_f1 for {early_stopping_patience} consecutive epochs."
            )
            break

    return history


# Unused duplicate CLI entrypoint; scripts/train_baseline.sh uses src.training.baseline_runner.
#
# def run_baseline_training(config_path: str = DEFAULT_BASELINE_CONFIG_PATH) -> list[EpochResult]:
#     """Run the baseline classifier training pipeline from the baseline config."""
#     config = load_and_validate_config(config_path, "baseline")
#     artifacts = run_preprocessing_from_config(config)
#
#     vocab = artifacts["vocab"]
#     pad_value = vocab.token_to_id[PAD_TOKEN]
#     all_train_examples = artifacts["train_examples"]
#
#     training_config = config["training"]
#     model_config = config["model"]
#     data_config = config["data"]
#
#     if not isinstance(training_config, dict):
#         raise TypeError("config['training'] must be a dictionary.")
#     if not isinstance(model_config, dict):
#         raise TypeError("config['model'] must be a dictionary.")
#     if not isinstance(data_config, dict):
#         raise TypeError("config['data'] must be a dictionary.")
#
#     seed_value = int(config.get("seed", 42))
#     set_seed(seed_value)
#
#     val_split_ratio = float(data_config.get("val_split_ratio", DEFAULT_VAL_SPLIT_RATIO))
#     train_examples, val_examples = split_labeled_examples(
#         all_train_examples,
#         val_split_ratio=val_split_ratio,
#         split_seed=seed_value,
#     )
#
#     train_dataset = LabeledTextDataset(train_examples)
#     val_dataset = LabeledTextDataset(val_examples)
#
#     batch_size = int(training_config["batch_size"])
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=lambda batch: collate_labeled_batch(batch, pad_value=pad_value),
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=lambda batch: collate_labeled_batch(batch, pad_value=pad_value),
#     )
#
#     device = select_device()
#     amp_enabled = amp_enabled_for_device(bool(training_config.get("amp", False)), device)
#     model = BiLSTMClassifier(
#         vocab_size=len(vocab),
#         embedding_dim=int(model_config["embedding_dim"]),
#         hidden_dim=int(model_config["hidden_dim"]),
#         num_layers=int(model_config["num_layers"]),
#         bidirectional=bool(model_config["bidirectional"]),
#         classifier_hidden_dim=int(model_config["classifier_hidden_dim"]),
#         embedding_dropout=float(model_config["embedding_dropout"]),
#         classifier_dropout=float(model_config["classifier_dropout"]),
#     ).to(device)
#
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=float(training_config["lr"]),
#         weight_decay=float(training_config["weight_decay"]),
#     )
#
#     return train_model(
#         model=model,
#         train_dataloader=train_dataloader,
#         val_dataloader=val_dataloader,
#         optimizer=optimizer,
#         device=device,
#         num_epochs=int(training_config["num_epochs"]),
#         grad_clip_norm=float(training_config["grad_clip_norm"]),
#         grad_accum_steps=int(training_config["grad_accum_steps"]),
#         amp_enabled=amp_enabled,
#         checkpoint_path=DEFAULT_LATEST_CHECKPOINT_PATH,
#         best_checkpoint_path=DEFAULT_BEST_CHECKPOINT_PATH,
#         early_stopping_patience=int(training_config["early_stopping_patience"]),
#     )
#
#
# def main(config_path: str = DEFAULT_BASELINE_CONFIG_PATH) -> None:
#     """Run baseline training from the configured baseline YAML file."""
#     history = run_baseline_training(config_path)
#     if not history:
#         raise ValueError("Training produced no history.")
#
#     final_epoch = history[-1]
#     print("Training complete.")
#     print(
#         f"Final epoch {final_epoch['epoch']} | "
#         f"train_loss={final_epoch['train_loss']:.4f} | "
#         f"val_loss={final_epoch['val_loss']:.4f} | "
#         f"val_acc={final_epoch['val_accuracy']:.4f} | "
#         f"val_f1={final_epoch['val_f1']:.4f}"
#     )
#
#
# if __name__ == "__main__":
#     main()
