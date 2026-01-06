#!/usr/bin/env python3
"""
Gesture Data Collection Script
Connects to Arduino MPU-6050 and records labeled gesture samples
"""

import serial
import csv
import time
from datetime import datetime
import os
import sys

# Configuration
SERIAL_PORT = '/dev/cu.usbmodem1201'  # Update this to match your Arduino port
BAUD_RATE = 115200
SAMPLE_DURATION = 2.0  # seconds
SAMPLE_RATE = 100  # Hz (matches Arduino)
EXPECTED_SAMPLES = int(SAMPLE_DURATION * SAMPLE_RATE)

# Gesture labels with descriptions
GESTURES = [
    ("idle", "Keep sensor still on table"),
    ("swipe_left", "Quick horizontal swipe to the left"),
    ("swipe_right", "Quick horizontal swipe to the right"),
    ("swipe_up", "Quick upward motion"),
    ("swipe_down", "Quick downward motion"),
    ("punch", "Forward punching motion"),
    ("shake", "Rapid side-to-side shaking"),
    ("rotate_cw", "Rotate sensor clockwise in a circle"),
    ("rotate_ccw", "Rotate sensor counter-clockwise"),
    ("tap", "Single quick tap/flick motion")
]


def list_serial_ports():
    """List available serial ports"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def connect_arduino(port, baud_rate, timeout=2):
    """Connect to Arduino and wait for it to be ready"""
    try:
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        time.sleep(2)  # Wait for Arduino to reset

        # Read and discard any initial data
        ser.reset_input_buffer()

        # Read header line
        header = ser.readline().decode('utf-8').strip()
        print(f"Connected! Header: {header}")

        return ser
    except serial.SerialException as e:
        print(f"Error connecting to {port}: {e}")
        print("\nAvailable ports:")
        for p in list_serial_ports():
            print(f"  {p}")
        sys.exit(1)


def record_gesture(ser, duration):
    """Record sensor data for specified duration"""
    readings = []
    start_time = time.time()

    print(f"Recording for {duration} seconds...")

    while time.time() - start_time < duration:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line and not line.startswith('ERROR'):
                # Parse CSV line: timestamp,aX,aY,aZ,gX,gY,gZ,temp
                values = line.split(',')
                if len(values) == 8:
                    readings.append(values)
        except UnicodeDecodeError:
            continue

    print(f"Captured {len(readings)} readings (expected ~{EXPECTED_SAMPLES})")
    return readings


def save_gesture(readings, gesture_label, output_file, gesture_number):
    """Save gesture readings to CSV file with gesture label"""
    file_exists = os.path.isfile(output_file)

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow(['gesture', 'gesture_num', 'timestamp', 'aX', 'aY', 'aZ', 'gX', 'gY', 'gZ', 'temp'])

        # Write all readings with gesture label and gesture number
        for reading in readings:
            writer.writerow([gesture_label, gesture_number] + reading)

    print(f"Saved gesture #{gesture_number} with {len(readings)} readings to {output_file}")


def main():
    print("=" * 60)
    print("Gesture Data Collection Tool")
    print("=" * 60)

    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"gesture_data_{timestamp}.csv")

    print(f"\nOutput file: {output_file}")

    # Connect to Arduino
    print(f"\nConnecting to Arduino on {SERIAL_PORT}...")
    ser = connect_arduino(SERIAL_PORT, BAUD_RATE)

    print("\n" + "=" * 60)
    print("Available Gestures:")
    for i, (gesture_name, description) in enumerate(GESTURES, 1):
        print(f"  {i}. {gesture_name:15} - {description}")
    print("  q. Quit")
    print("=" * 60)

    gesture_count = 0

    try:
        while True:
            print(f"\n[Gestures recorded: {gesture_count}]")
            choice = input("\nSelect gesture number (or 'q' to quit): ").strip()

            if choice.lower() == 'q':
                break

            try:
                gesture_idx = int(choice) - 1
                if 0 <= gesture_idx < len(GESTURES):
                    gesture_name, description = GESTURES[gesture_idx]

                    print(f"\nPrepare to perform: {gesture_name.upper()}")
                    print(f"How: {description}")
                    input("Press ENTER when ready...")

                    # Countdown
                    for i in range(3, 0, -1):
                        print(f"{i}...")
                        time.sleep(1)
                    print("GO!")

                    # Record gesture
                    readings = record_gesture(ser, SAMPLE_DURATION)

                    if len(readings) > 0:
                        gesture_count += 1
                        save_gesture(readings, gesture_name, output_file, gesture_count)
                        print(f"✓ Gesture recorded successfully!")
                    else:
                        print("✗ No data captured. Try again.")
                else:
                    print("Invalid gesture number!")
            except ValueError:
                print("Invalid input! Enter a number or 'q'.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        ser.close()
        print(f"\n{'=' * 60}")
        print(f"Session complete! Recorded {gesture_count} gestures.")
        print(f"Data saved to: {output_file}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
