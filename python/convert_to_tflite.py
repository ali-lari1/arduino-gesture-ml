#!/usr/bin/env python3
"""
Convert trained Keras model to TensorFlow Lite format with INT8 quantization.
This script prepares the model for deployment on Arduino Nano 33 BLE.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle

# Configuration
MODEL_PATH = Path("../models/gesture_model.h5")
TFLITE_MODEL_PATH = Path("../models/gesture_model.tflite")
LABEL_ENCODER_PATH = Path("../models/label_encoder.pkl")
DATA_DIR = Path("../data")
HEADER_FILE_PATH = Path("../models/gesture_model.h")

# Model parameters (should match training script)
WINDOW_SIZE = 120
N_FEATURES = 6  # aX, aY, aZ, gX, gY, gZ

def load_representative_dataset(num_samples=100):
    """
    Load representative dataset for quantization calibration.
    Uses real training data to determine the range of values for INT8 quantization.

    Args:
        num_samples: Number of samples to use for calibration

    Yields:
        Input arrays for the model
    """
    print(f"Loading representative dataset ({num_samples} samples)...")

    import pandas as pd

    # Load all data files
    data_files = sorted(DATA_DIR.glob("gesture_data_*.csv"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")

    all_data = []
    for file_idx, file in enumerate(data_files):
        df = pd.read_csv(file)
        df['file_id'] = file_idx
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)

    # Extract features
    feature_columns = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

    # Group by gesture instance
    samples = []
    for (gesture, gesture_num, file_id), group in data.groupby(['gesture', 'gesture_num', 'file_id']):
        features = group[feature_columns].values

        # Use last portion of recording (same as training)
        if len(features) > WINDOW_SIZE:
            features = features[-WINDOW_SIZE:]
        elif len(features) < WINDOW_SIZE:
            padding = np.zeros((WINDOW_SIZE - len(features), len(feature_columns)))
            features = np.vstack([padding, features])

        samples.append(features)

        if len(samples) >= num_samples:
            break

    # Normalize the data (approximate normalization for quantization)
    X = np.array(samples, dtype=np.float32)

    # Apply normalization per feature (approximate values)
    mean = np.array([0.0747, 0.0261, 0.9021, 19.0034, 4.8718, 0.9769], dtype=np.float32)
    std = np.array([0.2232, 0.1471, 0.1513, 26.7914, 30.4082, 81.6065], dtype=np.float32)

    X = (X - mean) / std

    print(f"Loaded {len(X)} representative samples")
    print(f"Sample shape: {X.shape}")

    return X

def representative_dataset_gen(num_samples=100):
    """
    Generator function for representative dataset.
    Required by TFLite converter for quantization.
    """
    dataset = load_representative_dataset(num_samples)

    for i in range(len(dataset)):
        # Yield single sample as float32
        yield [dataset[i:i+1].astype(np.float32)]

def convert_to_tflite():
    """
    Convert Keras model to TensorFlow Lite with INT8 quantization.
    """
    print(f"\nLoading model from {MODEL_PATH}...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

    # Load the trained Keras model
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nModel loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Create TFLite converter
    print("\nConfiguring TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set optimization to DEFAULT (enables weight quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide representative dataset for full integer quantization
    converter.representative_dataset = representative_dataset_gen

    # Set target ops to TFLITE_BUILTINS_INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set input and output types to INT8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converter configuration:")
    print(f"  - Optimization: DEFAULT")
    print(f"  - Target ops: TFLITE_BUILTINS_INT8")
    print(f"  - Input type: INT8")
    print(f"  - Output type: INT8")

    # Convert the model
    print("\nConverting model to TFLite format...")
    try:
        tflite_model = converter.convert()
        print("Conversion successful!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise

    # Save the TFLite model
    print(f"\nSaving TFLite model to {TFLITE_MODEL_PATH}...")
    TFLITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_MODEL_PATH.write_bytes(tflite_model)

    # Check file size
    file_size = len(tflite_model)
    file_size_kb = file_size / 1024

    print(f"\nModel saved successfully!")
    print(f"File size: {file_size} bytes ({file_size_kb:.2f} KB)")

    if file_size_kb > 15:
        print(f"⚠️  WARNING: Model size ({file_size_kb:.2f} KB) exceeds 15KB target!")
    else:
        print(f"✓ Model size is within 15KB target")

    return tflite_model

def convert_to_header_file(tflite_model):
    """
    Convert TFLite model to C header file for Arduino.
    """
    print(f"\nConverting to C header file...")

    # Generate C header file content
    header_content = """// Gesture recognition model for Arduino Nano 33 BLE
// Auto-generated from TensorFlow Lite model
// Do not edit manually

#ifndef GESTURE_MODEL_H
#define GESTURE_MODEL_H

// Model data
const unsigned int gesture_model_len = {model_size};
alignas(8) const unsigned char gesture_model[] = {{
{model_data}
}};

// Gesture labels (in the order the model was trained)
const char* GESTURES[] = {{
    "idle",
    "punch",
    "rotate_ccw",
    "rotate_cw",
    "shake",
    "swipe_down",
    "swipe_left",
    "swipe_right",
    "swipe_up"
}};

const int NUM_GESTURES = 9;

#endif  // GESTURE_MODEL_H
"""

    # Convert model bytes to C array format
    hex_array = []
    for i, byte in enumerate(tflite_model):
        if i % 12 == 0:
            hex_array.append("\n  ")
        hex_array.append(f"0x{byte:02x}, ")

    # Remove trailing comma and space
    model_data_str = ''.join(hex_array).rstrip(', ')

    # Fill in the template
    header_content = header_content.format(
        model_size=len(tflite_model),
        model_data=model_data_str
    )

    # Save header file
    print(f"Saving C header file to {HEADER_FILE_PATH}...")
    HEADER_FILE_PATH.write_text(header_content)

    print(f"✓ Header file saved successfully!")
    print(f"  Include in Arduino sketch: #include \"gesture_model.h\"")

def verify_tflite_model(tflite_model):
    """
    Verify the quantized TFLite model by running test inference.
    """
    print("\nVerifying quantized model...")

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nModel details:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input type: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output type: {output_details[0]['dtype']}")

    # Get quantization parameters
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]

    print(f"\nQuantization parameters:")
    print(f"  Input scale: {input_scale:.6f}, zero point: {input_zero_point}")
    print(f"  Output scale: {output_scale:.6f}, zero point: {output_zero_point}")

    # Load a test sample
    dataset = load_representative_dataset(num_samples=5)

    print("\nRunning test inference on 5 samples...")

    for i in range(min(5, len(dataset))):
        # Quantize input
        input_data = dataset[i:i+1].astype(np.float32)
        input_quantized = (input_data / input_scale + input_zero_point).astype(np.int8)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_quantized)
        interpreter.invoke()

        # Get output
        output_quantized = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize output
        output_data = (output_quantized.astype(np.float32) - output_zero_point) * output_scale

        # Get predicted class
        predicted_class = np.argmax(output_data[0])
        confidence = output_data[0][predicted_class]

        print(f"  Sample {i+1}: Predicted class {predicted_class}, confidence: {confidence:.4f}")

    print("\n✓ Model verification complete!")

def main():
    """
    Main conversion pipeline.
    """
    print("=" * 60)
    print("TensorFlow Lite Model Conversion for Arduino")
    print("=" * 60)

    # Step 1: Convert to TFLite
    tflite_model = convert_to_tflite()

    # Step 2: Convert to C header file
    convert_to_header_file(tflite_model)

    # Step 3: Verify the model
    verify_tflite_model(tflite_model)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {TFLITE_MODEL_PATH}")
    print(f"  - {HEADER_FILE_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Copy {HEADER_FILE_PATH.name} to your Arduino src/ directory")
    print(f"  2. Include it in your sketch: #include \"gesture_model.h\"")
    print(f"  3. Use the model array: gesture_model, gesture_model_len")

if __name__ == "__main__":
    main()
