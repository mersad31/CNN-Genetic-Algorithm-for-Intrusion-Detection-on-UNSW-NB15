# FILE: preprocess.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_unsw_nb15():
    """
    Loads and correctly preprocesses the UNSW_NB15 dataset for a 1D CNN.
    This version streamlines the one-hot encoding process.
    """
    # Step 1: Load the official train and test sets
    try:
        train_df = pd.read_csv('write your train file path')
        test_df = pd.read_csv('write your test file path')
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        return None, None, None, None

    # Keep track of original lengths
    num_train_samples = len(train_df)

    # Step 2: Combine, clean, and identify feature types
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df.drop(columns=['id', 'attack_cat'], inplace=True)

    categorical_features = combined_df.select_dtypes(include=['object']).columns
    numerical_features = combined_df.select_dtypes(include=np.number).drop(columns=['label']).columns

    print(f"\nCategorical features: {list(categorical_features)}")
    print(f"Numerical features: {list(numerical_features)}\n")

    # Step 3: One-hot encode categorical features on the combined data
    # This ensures all categories from both train and test sets are handled correctly.
    combined_df = pd.get_dummies(combined_df, columns=categorical_features, dtype=int)

    # Step 4: Separate back into training and testing sets BEFORE scaling
    train_processed = combined_df.iloc[:num_train_samples]
    test_processed = combined_df.iloc[num_train_samples:]

    X_train = train_processed.drop(columns=['label'])
    y_train = train_processed['label']
    X_test = test_processed.drop(columns=['label'])
    y_test = test_processed['label']

    # Step 5: Normalize numerical features (NO DATA LEAKAGE)
    scaler = MinMaxScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Step 6: Reshape data for 1D CNN input
    X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    print("Data preprocessing complete.")
    print(f"Final X_train shape: {X_train_reshaped.shape}")
    print(f"Final X_test shape: {X_test_reshaped.shape}")

    return X_train_reshaped, y_train.values, X_test_reshaped, y_test.values
