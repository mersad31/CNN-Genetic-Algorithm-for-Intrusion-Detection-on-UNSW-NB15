# 🛡️ CNN + Genetic Algorithm for Network Intrusion Detection (UNSW-NB15)

A 1D CNN that classifies network traffic as *normal* or *attack*, with its hyperparameters tuned by a Genetic Algorithm instead of by hand.

---

## 🎯 The Problem

Intrusion-detection models are sensitive to hyperparameters (learning rate, filter size, kernel size, dropout, optimizer...). Tuning them manually is slow and rarely finds the best combination. This project automates the search with an evolutionary algorithm.

## 💡 The Approach

1. **Preprocess** the UNSW-NB15 CSV files: drop irrelevant columns, one-hot encode categorical features, scale numeric features and reshape for a 1D CNN.
2. **Search** with a Genetic Algorithm: each individual is a set of CNN hyperparameters, and its fitness is the validation accuracy of the trained model.
3. **Train** the final CNN with the best hyperparameters and evaluate it.

| GA element | Detail |
|---|---|
| Genes | learning rate, filter size, kernel size, activation, dropout, optimizer |
| Selection | the best individuals are kept as parents |
| Variation | crossover and mutation |
| Generations | 10 (`N_GENERATIONS` in `genetic_algorithm.py`) |
| Fitness | validation accuracy |

## 🏗️ Model

`Conv1D → Activation → BatchNorm` (×2) → `MaxPooling` → `Dense` → `Dropout` → `Sigmoid` output, trained with binary cross-entropy.

## 📊 Results

`train.py` reports Accuracy, Precision, Recall, F1-score, Specificity and False Positive Rate, and plots a confusion matrix plus training curves.

<!-- TODO: add the final metrics table (GA-tuned vs manually tuned baseline) -->
<!-- TODO: add the confusion-matrix image -->

## 📂 Repository Layout

| File | Purpose |
|---|---|
| `preprocess.py` | Loads and prepares UNSW-NB15 |
| `cnn_model.py` | Configurable 1D CNN |
| `genetic_algorithm.py` | Hyperparameter search |
| `train.py` | Final training, evaluation and plots |
| `requirements.txt` | Python dependencies |

## 🚀 Getting Started

```bash
git clone https://github.com/mersad31/CNN-Genetic-Algorithm-for-Intrusion-Detection-on-UNSW-NB15.git
cd CNN-Genetic-Algorithm-for-Intrusion-Detection-on-UNSW-NB15
pip install -r requirements.txt
```

1. Download the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and set the file paths in `preprocess.py`.
2. *(Optional)* run the hyperparameter search:

```bash
python genetic_algorithm.py
```

3. Train and evaluate the final model:

```bash
python train.py
```

## 🔭 Future Work

- Multi-class classification with `attack_cat`
- Evaluation on other IDS datasets (CICIDS2017, NSL-KDD)
- Other optimizers (PSO, Bayesian optimization)
- Real-time detection

## 📝 License

MIT License.
