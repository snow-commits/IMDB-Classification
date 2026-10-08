# BiLSTM IMDb

## How to run the BiLSTM pipeline
The BiLSTM pipeline can be reproduced from the project root using the small bash entrypoints in `scripts/`. These scripts wrap the underlying Python modules and keep the command sequence consistent with the saved configuration and artifacts used in this notebook.

A typical run is:

```bash
bash scripts/download_data.sh
bash scripts/run_preprocess.sh
bash scripts/train_baseline.sh
bash scripts/show_best_checkpoint.sh checkpoints/classifier/best.pt
bash scripts/evaluate_baseline.sh
```

The commands follow the same order as the pipeline described above. `download_data.sh` retrieves the IMDb data, `run_preprocess.sh` creates the vocabulary and processed JSON artifacts, `train_baseline.sh` trains the recurrent baseline, `show_best_checkpoint.sh` prints the validation metrics stored in the selected checkpoint, and `evaluate_baseline.sh` produces the held-out test metrics and figures. The main hyperparameters are read from `configs/baseline.yaml`, so changing that file changes the subsequent preprocessing, training, or evaluation behavior without rewriting the notebook text.

