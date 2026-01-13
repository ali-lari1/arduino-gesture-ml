#!/usr/bin/env python3
"""
Gesture Recognition Model Training Script
Trains a CNN model on collected gesture data
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
SAMPLE_RATE = 200  # Hz
SAMPLE_DURATION = 2.0  # seconds
EXPECTED_SAMPLES = int(SAMPLE_DURATION * SAMPLE_RATE)  # 400 samples per gesture

# Training parameters
TEST_SIZE = 0.2  # 80/20 split
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 16
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42


def detect_motion_window(features, threshold=0.15, window_size=100):
    """
    Detect the window where significant motion occurs.

    Args:
        features: Array of shape (n_samples, n_features)
        threshold: Motion threshold for detection (lowered to 0.15 to catch swipes)
        window_size: Desired window size to extract

    Returns:
        Extracted motion window of shape (window_size, n_features)
    """
    # Calculate total acceleration magnitude for motion detection
    accel = features[:, :3]  # aX, aY, aZ
    gyro = features[:, 3:]   # gX, gY, gZ

    # Compute motion intensity - use acceleration variation for better swipe detection
    accel_mag = np.sqrt(np.sum(accel**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))

    # Also consider acceleration changes (derivative) for quick motions like swipes
    accel_diff = np.concatenate([[0], np.sqrt(np.sum(np.diff(accel, axis=0)**2, axis=1))])

    # Combined motion signal
    motion_intensity = accel_mag + gyro_mag + accel_diff

    # Find regions above threshold
    motion_mask = motion_intensity > threshold

    if not np.any(motion_mask):
        # No significant motion detected, use last portion
        start_idx = max(0, len(features) - window_size)
        return features[start_idx:start_idx + window_size]

    # Find first and last motion points
    motion_indices = np.where(motion_mask)[0]
    motion_start = motion_indices[0]
    motion_end = motion_indices[-1]

    # Extract window centered around motion with some context
    motion_center = (motion_start + motion_end) // 2
    start_idx = max(0, motion_center - window_size // 2)
    end_idx = start_idx + window_size

    # Adjust if we're at the end
    if end_idx > len(features):
        end_idx = len(features)
        start_idx = max(0, end_idx - window_size)

    motion_window = features[start_idx:end_idx]

    # Pad if necessary
    if len(motion_window) < window_size:
        padding = np.zeros((window_size - len(motion_window), features.shape[1]))
        motion_window = np.vstack([motion_window, padding])

    return motion_window


def load_and_preprocess_data(data_files, use_motion_detection=True, window_size=100):
    """
    Load gesture data from CSV files and preprocess it.

    Args:
        data_files: List of CSV file paths
        use_motion_detection: Whether to extract motion windows (True) or use full sequence (False)
        window_size: Size of motion window to extract (samples)

    Returns:
        X: Array of shape (n_gestures, n_samples, n_features)
        y: Array of gesture labels
    """
    print("Loading data files...")

    all_data = []
    for file_idx, file in enumerate(data_files):
        df = pd.read_csv(file)
        # Add file identifier to make gesture instances unique across files
        df['file_id'] = file_idx
        all_data.append(df)

    # Combine all data
    data = pd.concat(all_data, ignore_index=True)

    print(f"Total rows loaded: {len(data)}")
    print(f"Gesture distribution:\n{data['gesture'].value_counts()}\n")

    # Extract features (aX, aY, aZ, gX, gY, gZ)
    feature_columns = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

    # Group by gesture instance (gesture + gesture_num + file_id)
    gesture_instances = []
    labels = []

    for (gesture, gesture_num, file_id), group in data.groupby(['gesture', 'gesture_num', 'file_id']):
        # Extract feature values
        features = group[feature_columns].values

        if use_motion_detection:
            # Detect and extract motion window
            features = detect_motion_window(features, window_size=window_size)
        else:
            # Use last portion of recording (where gesture actually happens)
            # This works better for quick gestures like swipes
            if len(features) > window_size:
                # Take last window_size samples
                features = features[-window_size:]
            elif len(features) < window_size:
                # Pad at beginning with zeros
                padding = np.zeros((window_size - len(features), len(feature_columns)))
                features = np.vstack([padding, features])

        gesture_instances.append(features)
        labels.append(gesture)

    X = np.array(gesture_instances)
    y = np.array(labels)

    print(f"Motion detection: {'Enabled' if use_motion_detection else 'Disabled'}")
    if use_motion_detection:
        print(f"Window size: {window_size} samples (~{window_size/SAMPLE_RATE:.2f} seconds)")
    print(f"Processed {len(X)} gesture instances")
    print(f"Data shape: {X.shape}")
    print(f"Features per sample: {X.shape[2]} (aX, aY, aZ, gX, gY, gZ)")

    # Show per-class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nGesture instance counts:")
    for gesture, count in zip(unique, counts):
        print(f"  {gesture}: {count} instances")

    return X, y


def augment_data(X, y, augment_factor=2):
    """
    Apply data augmentation by adding noise and time shifts.

    Args:
        X: Input data of shape (n_samples, n_timesteps, n_features)
        y: Labels
        augment_factor: How many augmented versions to create per sample

    Returns:
        Augmented X and y
    """
    X_augmented = [X]
    y_augmented = [y]

    for _ in range(augment_factor - 1):
        # Add small random noise
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X + noise

        # Small time shifts (roll by random amount)
        X_shifted = np.zeros_like(X)
        for i in range(len(X)):
            shift = np.random.randint(-5, 6)
            X_shifted[i] = np.roll(X[i], shift, axis=0)

        X_augmented.append(X_noisy)
        X_augmented.append(X_shifted)
        y_augmented.extend([y, y])

    X_final = np.vstack(X_augmented)
    y_final = np.hstack(y_augmented)

    return X_final, y_final


def normalize_data(X_train, X_test):
    """
    Normalize features using training set statistics.

    Args:
        X_train: Training data
        X_test: Test data

    Returns:
        Normalized X_train and X_test
    """
    # Calculate mean and std from training data
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))

    # Avoid division by zero
    std[std == 0] = 1.0

    # Normalize both sets using training statistics
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    print(f"Data normalized - Mean: {mean}")
    print(f"Data normalized - Std: {std}\n")

    return X_train_norm, X_test_norm


def build_model(input_shape, num_classes):
    """
    Build the CNN model architecture.

    Args:
        input_shape: Shape of input data (n_samples, n_features)
        num_classes: Number of gesture classes

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv1D(filters=16, kernel_size=5, padding='same',
                     input_shape=input_shape, name='conv1d_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Activation('relu', name='relu_1'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_1'),
        layers.Dropout(0.2, name='dropout_1'),

        # Second convolutional block
        layers.Conv1D(filters=32, kernel_size=5, padding='same',
                     name='conv1d_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Activation('relu', name='relu_2'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_2'),
        layers.Dropout(0.2, name='dropout_2'),

        # Third convolutional block
        layers.Conv1D(filters=64, kernel_size=3, padding='same',
                     name='conv1d_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.Activation('relu', name='relu_3'),
        layers.GlobalAveragePooling1D(name='global_avg_pool'),

        # Dense layers
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.4, name='dropout_3'),
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.Dropout(0.3, name='dropout_4'),

        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])

    # Compile model with learning rate schedule
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history, save_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")


def main():
    print("=" * 60)
    print("Gesture Recognition Model Training")
    print("=" * 60)
    print()

    # Create models directory
    MODEL_DIR.mkdir(exist_ok=True)

    # Find all CSV data files
    data_files = sorted(DATA_DIR.glob("gesture_data_*.csv"))

    if not data_files:
        print(f"ERROR: No data files found in {DATA_DIR}")
        print("Please run collect_data.py first to collect gesture data.")
        return

    print(f"Found {len(data_files)} data files:")
    for f in data_files:
        print(f"  - {f.name}")
    print()

    # Load and preprocess data WITHOUT motion detection
    # Motion detection was missing swipes - just use the last portion where gestures happen
    X, y = load_and_preprocess_data(data_files, use_motion_detection=False, window_size=120)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Gesture classes: {list(label_encoder.classes_)}")
    print(f"Number of classes: {len(label_encoder.classes_)}\n")

    # Split data FIRST (before augmentation) to prevent data leakage
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE
    )

    print(f"Original training samples: {len(X_train)}")
    print(f"Original test samples: {len(X_test)}")

    # Apply data augmentation ONLY to training data
    print("\nApplying data augmentation to training set only...")
    X_train_augmented, y_train_augmented = augment_data(X_train, y_train_encoded, augment_factor=4)
    y_train = to_categorical(y_train_augmented)
    y_test = to_categorical(y_test_encoded)

    print(f"Augmented training samples: {len(X_train_augmented)} (from {len(X_train)})")
    print(f"Test samples (unchanged): {len(y_test)}\n")

    # Update X_train to augmented version
    X_train = X_train_augmented

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")

    # Normalize data
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (samples, features)
    num_classes = len(label_encoder.classes_)

    print("Building model...")
    model = build_model(input_shape, num_classes)
    model.summary()
    print()

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Train model
    print("=" * 60)
    print("Training model...")
    print("=" * 60)

    history = model.fit(
        X_train_norm, y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("\n" + "=" * 60)
    print("Evaluating model on test set...")
    print("=" * 60)

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test_norm, y_test, verbose=0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Get predictions for confusion matrix
    y_pred = model.predict(X_test_norm, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Print classification report
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(
        y_test_classes,
        y_pred_classes,
        target_names=label_encoder.classes_,
        digits=3
    ))

    # Create confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    confusion_matrix_path = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    print(f"\nConfusion matrix saved to {confusion_matrix_path}")

    if test_accuracy >= 0.85:
        print("\nTarget accuracy of 85% achieved!")
    else:
        print("\nWarning: Accuracy below 85% target.")
        print("Check confusion matrix to see which gestures are being confused.")
        print("Consider collecting more data for confused gesture pairs.")

    # Save model
    model_path = MODEL_DIR / "gesture_model.h5"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save label encoder
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {encoder_path}")

    # Save training history plot
    plot_path = MODEL_DIR / "training_history.png"
    plot_training_history(history, plot_path)

    # Save model summary
    summary_path = MODEL_DIR / "model_summary.txt"
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
