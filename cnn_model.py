# FILE: cnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, Sequential

def create_cnn_model(input_shape, learning_rate, filters1, filters2, kernel_size, activation_str, dropout_rate, optimizer_str):
    """
    Dynamically creates and compiles a 1D CNN model for BINARY classification.
    """
    model = Sequential(name="Binary_CNN_for_NIDS")

    # --- Feature Extraction Block ---
    model.add(layers.Conv1D(filters=filters1, kernel_size=kernel_size, padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.1) if activation_str == 'leaky_relu' else layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(filters=filters2, kernel_size=kernel_size, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1) if activation_str == 'leaky_relu' else layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling1D(pool_size=2))

    # --- Classification Block ---
    model.add(layers.Flatten())

    model.add(layers.Dense(units=500))
    model.add(layers.LeakyReLU(alpha=0.1) if activation_str == 'leaky_relu' else layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(units=100))
    model.add(layers.LeakyReLU(alpha=0.1) if activation_str == 'leaky_relu' else layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))

    # --- KEY CHANGE: Binary Output Layer ---
    # A single neuron with a sigmoid activation is standard for binary classification.
    model.add(layers.Dense(units=1, activation='sigmoid'))

    # --- Optimizer and Compilation ---
    if optimizer_str.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_str.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_str}")

    # --- KEY CHANGE: Binary Loss Function ---
    # Compile with binary_crossentropy for a binary (0/1) problem.
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model