from __future__ import annotations

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_labeled_batch
from src.data.dataset import LabeledExample, LabeledTextDataset
from src.data.processed_artifacts import load_processed_artifacts_from_config
from src.data.vocab import PAD_TOKEN
from src.models.bilstm_classifier import BiLSTMClassifier
from src.training.plotting import (
    BASELINE_LABEL_DISTRIBUTION_PLOT_PATH,
    BASELINE_LOSS_PLOT_PATH,
    BASELINE_TOKEN_LENGTH_DISTRIBUTION_PLOT_PATH,
    BASELINE_VAL_ACCURACY_PLOT_PATH,
    BASELINE_VAL_F1_PLOT_PATH,
    save_label_distribution_plot,
    save_token_length_distribution_plot,
    save_train_val_loss_plot,
    save_validation_metric_plot,
)
from src.training.train_classifier import EpochResult, train_model
from src.utils.config import load_and_validate_config
from src.utils.device import amp_enabled_for_device, select_device
from src.utils.seed import set_seed

DEFAULT_BASELINE_CONFIG_PATH = "configs/baseline.yaml"
DEFAULT_LATEST_CHECKPOINT_PATH = "checkpoints/classifier/latest.pt"
DEFAULT_BEST_CHECKPOINT_PATH = "checkpoints/classifier/best.pt"
DEFAULT_VAL_SPLIT_RATIO = 0.1


def split_labeled_examples(
    examples: list[LabeledExample],
    val_split_ratio: float,
    split_seed: int,
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    """Create a reproducible stratified train/validation split from labeled examples."""
    if not 0.0 < val_split_ratio < 1.0:
        raise ValueError("val_split_ratio must be between 0 and 1.")

    labels = [example["label"] for example in examples]
    train_examples, val_examples = train_test_split(
        examples,
        test_size=val_split_ratio,
        random_state=split_seed,
        shuffle=True,
        stratify=labels,
    )
    return list(train_examples), list(val_examples)


def run_baseline_training(config_path: str = DEFAULT_BASELINE_CONFIG_PATH) -> list[EpochResult]:
    """Run the baseline classifier training pipeline from the baseline config."""
    config = load_and_validate_config(config_path, "baseline")
    artifacts = load_processed_artifacts_from_config(config)

    vocab = artifacts["vocab"]
    pad_value = vocab.token_to_id[PAD_TOKEN]
    all_train_examples = artifacts["train_examples"]
    test_examples = artifacts["test_examples"]
    training_config = config["training"]
    model_config = config["model"]
    data_config = config["data"]

    if not isinstance(training_config, dict):
        raise TypeError("config['training'] must be a dictionary.")
    if not isinstance(model_config, dict):
        raise TypeError("config['model'] must be a dictionary.")
    if not isinstance(data_config, dict):
        raise TypeError("config['data'] must be a dictionary.")

    seed_value = int(config.get("seed", 42))
    set_seed(seed_value)

    val_split_ratio = float(data_config.get("val_split_ratio", DEFAULT_VAL_SPLIT_RATIO))
    train_examples, val_examples = split_labeled_examples(
        all_train_examples,
        val_split_ratio=val_split_ratio,
        split_seed=seed_value,
    )

    train_dataset = LabeledTextDataset(train_examples)
    val_dataset = LabeledTextDataset(val_examples)

    batch_size = int(training_config["batch_size"])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_labeled_batch(batch, pad_value=pad_value),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_labeled_batch(batch, pad_value=pad_value),
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["lr"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=int(training_config["num_epochs"]),
        grad_clip_norm=float(training_config["grad_clip_norm"]),
        grad_accum_steps=int(training_config["grad_accum_steps"]),
        amp_enabled=amp_enabled,
        checkpoint_path=DEFAULT_LATEST_CHECKPOINT_PATH,
        best_checkpoint_path=DEFAULT_BEST_CHECKPOINT_PATH,
        early_stopping_patience=int(training_config["early_stopping_patience"]),
    )

    save_train_val_loss_plot(history, BASELINE_LOSS_PLOT_PATH)
    save_validation_metric_plot(
        history,
        metric_key="val_accuracy",
        output_path=BASELINE_VAL_ACCURACY_PLOT_PATH,
        title="Validation Accuracy",
        ylabel="Accuracy",
    )
    save_validation_metric_plot(
        history,
        metric_key="val_f1",
        output_path=BASELINE_VAL_F1_PLOT_PATH,
        title="Validation F1",
        ylabel="F1",
    )
    save_label_distribution_plot(
        train_labels=[example["label"] for example in all_train_examples],
        test_labels=[example["label"] for example in test_examples],
        output_path=BASELINE_LABEL_DISTRIBUTION_PLOT_PATH,
    )
    save_token_length_distribution_plot(
        train_lengths=[len(example["input_ids"]) for example in all_train_examples],
        test_lengths=[len(example["input_ids"]) for example in test_examples],
        output_path=BASELINE_TOKEN_LENGTH_DISTRIBUTION_PLOT_PATH,
    )
    return history


def main(config_path: str = DEFAULT_BASELINE_CONFIG_PATH) -> None:
    """Run baseline training from the configured baseline YAML file."""
    history = run_baseline_training(config_path)
    if not history:
        raise ValueError("Training produced no history.")

    final_epoch = history[-1]
    print("Training complete.")
    print(
        f"Final epoch {final_epoch['epoch']} | "
        f"train_loss={final_epoch['train_loss']:.4f} | "
        f"val_loss={final_epoch['val_loss']:.4f} | "
        f"val_acc={final_epoch['val_accuracy']:.4f} | "
        f"val_f1={final_epoch['val_f1']:.4f}"
    )


if __name__ == "__main__":
    import sys

    selected_config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE_CONFIG_PATH
    main(selected_config_path)
