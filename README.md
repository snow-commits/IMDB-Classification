# IMDb Sentiment Classification

This project compares five approaches to binary sentiment classification on IMDb movie reviews. The methods range from sparse lexical baselines to neural sequence models and a parameter-efficient large language model adaptation.

## Methods

- **Logistic Regression** — TF-IDF review features with validation-based hyperparameter selection.
- **Multinomial Naive Bayes** — bag-of-words features, comparing unigram and unigram-plus-bigram settings.
- **Embedding + TextCNN** — a controlled comparison of random, semantic, sentiment-aware, and combined pretrained embeddings.
- **BiLSTM** — an ordered-token sequence classifier with masked pooling.
- **Qwen3-0.6B** — zero-shot and few-shot prompting, plus LoRA fine-tuning.

## Dataset and evaluation

The project uses the [IMDb Sentiment dataset](https://huggingface.co/datasets/Kwaai/IMDB_Sentiment), which contains 50,000 labelled reviews balanced between positive and negative sentiment. The official 25,000-review training split was divided into 20,000 training reviews and 5,000 validation reviews. The official 25,000-review test split was held out for final evaluation. The additional 50,000 unlabelled reviews were used only in the relevant embedding-pretraining experiment.

All models are compared using accuracy, precision, recall, and F1-score on the test set. The table reports the project’s final results; precision, recall, and F1 are for the positive class.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.89% | 81.72% | 84.74% | 83.20% |
| Multinomial Naive Bayes | 84.92% | 86.20% | 83.14% | 84.64% |
| Embedding + TextCNN | 88.63% | 87.22% | 90.53% | 88.84% |
| BiLSTM | 89.68% | 88.17% | 91.65% | 89.88% |
| Qwen3-0.6B with LoRA | **95.14%** | **95.75%** | 94.48% | **95.11%** |

Qwen3-0.6B with LoRA achieved the highest test F1-score. BiLSTM was the strongest non-LLM model. The TextCNN experiments show that embedding pretraining improved on random initialization, with the semantic-plus-sentiment-plus-unlabelled condition producing the best TextCNN test F1-score.

## Repository layout

- `COMP9444_Notebook_SR.ipynb` — consolidated project code for data exploration, preprocessing, experiments, and result analysis. Its code cells are retained without saved outputs; run the cells to regenerate results.
- `LR/` — Logistic Regression notebook and plot.
- `Naive_bayes/` — Naive Bayes notebook and confusion matrix.
- `Embedding_CNN/` — TextCNN notebooks, saved embedding results, and plots.
- `BiLSTM/` — BiLSTM source code, configuration, scripts, and project-specific instructions in [`BiLSTM/README.md`](BiLSTM/README.md).
- `Qwen3_0.6B_IMDB/` — Qwen3 prompting and LoRA notebooks with saved result summaries and figures in `final_result/`.

## Running the experiments

The consolidated notebook can be opened in Jupyter. Its cells load the dataset from Hugging Face and use the files in the corresponding model folders. Install the dependencies required by the notebook cells and model before running them.

The BiLSTM pipeline has a `requirements.txt` and shell scripts. From a Bash-compatible terminal, run:

```bash
cd BiLSTM
pip install -r requirements.txt
bash scripts/download_data.sh
bash scripts/run_preprocess.sh
bash scripts/train_baseline.sh
bash scripts/evaluate_baseline.sh
```

The Qwen3 experiments require the Transformers and PEFT stack, access to the base model, and a suitable GPU for fine-tuning. See the notebooks in `Qwen3_0.6B_IMDB/` for their specific setup and execution steps.

## Notes

- The raw dataset and generated intermediate files are not tracked in this repository; use the download/preprocessing steps to recreate them.
- Notebook outputs are cleared in the consolidated notebook. Running cells will generate new outputs and figures.
- The reported scores are the results recorded in the project notebook; reproducing them may require the same data split, package versions, model checkpoint, and hardware setup.
