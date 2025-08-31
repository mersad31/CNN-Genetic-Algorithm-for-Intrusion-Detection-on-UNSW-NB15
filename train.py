# FILE: train.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names, cm):
    """Plots a confusion matrix using seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == '__main__':
    # 1. DEFINE HYPERPARAMETERS
    # Use the best set found by your genetic_algorithm.py script
    hyperparameters = {
        'learning_rate': 0.001,
        'filters1': 32,
        'filters2': 128,
        'kernel_size': 3,
        'activation_str': 'leaky-relu',
        'dropout_rate': 0.3,
        'optimizer_str': 'adam'
    }
    print("Using hyperparameters:")
    print(hyperparameters)

    # 2. LOAD AND PREPROCESS THE DATA
    X_train, y_train, X_test, y_test = preprocess_unsw_nb15()
    if X_train is None:
        exit()
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # 3. CREATE AND TRAIN THE 1D CNN MODEL
    input_shape = X_train.shape[1:]
    model = create_cnn_model(input_shape=input_shape, **hyperparameters)
    print("\n1D CNN Model Architecture:")
    model.summary()

    print("\nTraining the final model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=256,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 4. GET PREDICTIONS
    print("\nGenerating predictions on the test set...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype("int32")

    # =========================================================================
    # == STEP 5: DETAILED EVALUATION AND COMPARISON METRICS ==
    # =========================================================================
    print("\n--- Detailed Performance Metrics for Comparison ---")

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}\n")

    # --- Calculate Standard Metrics ---
    # Add a small epsilon to denominators to avoid division by zero
    epsilon = 1e-9

    # Accuracy: (TP + TN) / Total
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    print(f"✅ Accuracy: {accuracy:.4f}")

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP + epsilon)
    print(f"🎯 Precision: {precision:.4f}")

    # Recall (Sensitivity): TP / (TP + FN)
    recall = TP / (TP + FN + epsilon)
    print(f"📈 Recall (Sensitivity): {recall:.4f}")

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    print(f"📊 F1-Score: {f1_score:.4f}")

    # Specificity: TN / (TN + FP)
    specificity = TN / (TN + FP + epsilon)
    print(f"🛡️ Specificity: {specificity:.4f}")

    # False Positive Rate (FPR): FP / (TN + FP)
    fpr = FP / (TN + FP + epsilon)
    print(f"⚠️ False Positive Rate (FPR): {fpr:.4f}")
    print("-------------------------------------------------")

    # --- Print scikit-learn's summary report ---
    print("\nClassification Report (from scikit-learn):")
    class_names = ['Normal (0)', 'Attack (1)']
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Visualize the results ---
    plot_confusion_matrix(y_test, y_pred, class_names, cm)
    print("\n--- Training History Plot ---")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()