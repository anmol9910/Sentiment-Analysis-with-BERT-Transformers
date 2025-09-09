# Sentiment Analysis with BERT

## Project Overview

A production-ready PyTorch + Hugging Face implementation for multi-class sentiment classification on app reviews using **BERT (bert-base-cased)**. This repository covers data preprocessing, dataset & dataloader creation, model definition, training loop with AdamW + linear scheduler, evaluation (classification report & confusion matrix), and inference on raw text.

---

## Features

* Converts 1–5 star ratings into 3 sentiment classes: **negative (0)**, **neutral (1)**, **positive (2)**
* Uses `bert-base-cased` tokenizer + model from Hugging Face
* Custom `Dataset` and `DataLoader` for efficient batching
* Training with gradient clipping and learning-rate scheduling
* Model checkpointing (best validation accuracy)
* Evaluation with precision/recall/F1 and confusion matrix
* Simple inference API for single sentences

---

## Repository Structure

```
├── data/
│   └── reviews.csv            # raw dataset
├── src/
│   ├── dataset.py             # GPReviewDataset class
│   ├── model.py               # SentimentClassifier class
│   ├── train.py               # training & validation loops
│   ├── eval.py                # evaluation & get_predictions
│   └── inference.py           # script to run inference on raw text
├── best_model_state.bin       # saved best model (generated after training)
├── requirements.txt
└── README.md                  # this file
```

---

## Requirements

* Python 3.8+
* PyTorch
* transformers (Hugging Face)
* scikit-learn
* pandas, numpy
* matplotlib, seaborn

Install with pip:

```bash
pip install -r requirements.txt
```

Example `requirements.txt` (minimal):

```
torch
transformers
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## Quick Start

1. Place `reviews.csv` in the `data/` folder. The expected columns are: `userName, userImage, content, score, thumbsUpCount, reviewCreatedVersion, at, replyContent, repliedAt, sortOrder, appId`.
2. Adjust hyperparameters in `train.py` (e.g. `MAX_LEN`, `BATCH_SIZE`, `EPOCHS`, `PRE_TRAINED_MODEL_NAME`).
3. Run training:

```bash
python src/train.py --data_path data/reviews.csv --output best_model_state.bin
```

4. Evaluate on the test set:

```bash
python src/eval.py --model_path best_model_state.bin --data_path data/reviews.csv
```

5. Run inference on a raw sentence:

```bash
python src/inference.py --model_path best_model_state.bin --text "I love completing my todos!"
```

---

## Data Preprocessing

* Convert star ratings (`score`) to sentiment labels using: `<=2 -> negative (0)`, `3 -> neutral (1)`, `>=4 -> positive (2)`.
* Tokenize with `BertTokenizer.from_pretrained('bert-base-cased')` using `encode_plus` with `add_special_tokens=True`, `max_length=MAX_LEN`, `pad_to_max_length=True`, `return_attention_mask=True` and `return_tensors='pt'`.
* Use `attention_mask` to ignore `[PAD]` tokens.

---

## Model

`SentimentClassifier` wraps `BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)`, followed by a `Dropout` and a linear layer mapping `hidden_size` to `n_classes`.

Forward signature:

```py
def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.drop(pooled_output)
    return self.out(output)
```

Loss: `nn.CrossEntropyLoss()`

---

## Training Details

* Optimizer: `AdamW(model.parameters(), lr=2e-5, correct_bias=False)`
* Scheduler: `get_linear_schedule_with_warmup` with `num_warmup_steps=0` and `num_training_steps = len(train_loader) * EPOCHS`
* Gradient clipping: `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
* Save best model by validation accuracy: `torch.save(model.state_dict(), 'best_model_state.bin')`

Recommended defaults used in experiments:

```py
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 10
RANDOM_SEED = 42
```

---

## Evaluation & Results

* Use `classification_report` from `sklearn.metrics` to print precision, recall, and f1-score per class.
* Visualize confusion matrix with `seaborn.heatmap`.

**Example results from the experiment**:

* Test accuracy: `~0.883` (on a held-out test set)
* Per-class performance (example):

  * negative: precision 0.89 | recall 0.87
  * neutral:  precision 0.83 | recall 0.85
  * positive: precision 0.92 | recall 0.93

Note: neutral class is typically harder to classify — consider class re-balancing or focal loss if needed.

---

## Inference

A simple inference workflow:

1. Tokenize input using the same tokenizer and `MAX_LEN`.
2. Move `input_ids` and `attention_mask` to the same device as the model.
3. Run `model(input_ids, attention_mask)` and apply `softmax` to get probabilities.
4. Map `argmax` to `['negative', 'neutral', 'positive']`.

---

## Tips to Improve Performance

* Increase dataset size or perform data augmentation (back-translation, paraphrasing).
* Try larger models (e.g., `bert-large-cased`) or distilled versions for faster inference.
* Use class weighting or focal loss when classes are imbalanced.
* Experiment with learning rates, schedulers, and warmup steps.
* Fine-tune tokenizer `MAX_LEN` and batch size based on GPU memory.

---

## Reproducibility

* Set `RANDOM_SEED` for `numpy`, `torch`, and other libs.
* Save model weights and training hyperparameters alongside logs.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with a clear description of the change.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Contact

For questions or collaboration, contact: **Anmol**

*Prepared using PyTorch and Hugging Face transformers.*
