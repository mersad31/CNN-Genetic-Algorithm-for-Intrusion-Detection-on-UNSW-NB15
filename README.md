# CNN-Genetic-Algorithm-for-Intrusion-Detection-on-UNSW-NB15
This project presents a hybrid deep learning framework for Network Intrusion Detection Systems (NIDS) using the UNSW-NB15 dataset. A 1D Convolutional Neural Network (CNN) is employed to classify traffic as normal or attack, while a Genetic Algorithm (GA) is used to optimize hyperparameters such as learning rate, filter size, kernel size, activation functions, dropout, and optimizer type. By integrating evolutionary optimization with deep learning, the system achieves higher detection accuracy, robustness against diverse attack types, and improved generalization compared to manually tuned models.

This project implements a **1D Convolutional Neural Network (CNN)** optimized with a **Genetic Algorithm (GA)** for detecting network intrusions using the **UNSW-NB15 dataset**.

**Dataset:**  
- UNSW-NB15 contains modern synthetic network traffic.  
- Binary classification: Normal (`label=0`) vs Attack (`label=1`).  
- Optional multi-class classification via `attack_cat`.

**Modules:**  
1. **preprocess.py** – Loads CSVs, drops irrelevant columns, applies one-hot encoding, scales numerical features, reshapes data for CNN.  
2. **cnn_model.py** – Defines configurable 1D CNN (Conv1D → Activation → BatchNorm ×2 → MaxPooling → Dense → Dropout → Sigmoid output) with binary crossentropy loss.  
3. **genetic_algorithm.py** – Optimizes CNN hyperparameters using a Genetic Algorithm (population, generations, mutation, crossover) with fitness based on validation accuracy.  
4. **train.py** – Trains CNN with GA-selected hyperparameters, evaluates with Accuracy, Precision, Recall, F1, Specificity, FPR, and visualizes confusion matrix and training curves.

**Usage:**  
1. Clone the repo:  
```bash
git clone https://github.com/mersad31/CNN-Genetic-Algorithm-for-Intrusion-Detection-on-UNSW-NB15.git
```
2. Install dependencies:
 ```bash
pip install -r requirements.txt
```
3. Download UNSW-NB15 dataset and update file paths in preprocess.py.

4. (Optional) Run GA to find best hyperparameters:
```bash
python genetic_algorithm.py
```
5. Train the final model and evaluate:
 ```bash
python train.py
```
Results:
- Metrics include Accuracy, Precision, Recall, F1-Score, Specificity, FPR.
- Visualization includes confusion matrix heatmap and training accuracy/loss curves.

Future Work:
- Multi-class classification using attack_cat.
- Test on other IDS datasets (CICIDS2017, NSL-KDD).
- Integration with real-time intrusion detection.
- Advanced optimization techniques (PSO, Bayesian).

License: MIT License

Contributing: Pull requests and issues are welcome.


