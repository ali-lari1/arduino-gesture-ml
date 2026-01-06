# Gesture Data Collection

Python script for collecting labeled gesture data from the Arduino MPU-6050 sensor.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Find your Arduino's serial port:
```bash
# macOS/Linux
ls /dev/cu.*

# Windows
# Check Device Manager or Arduino IDE
```

3. Update `SERIAL_PORT` in `collect_data.py` to match your Arduino port.

## Usage

1. Upload the Arduino sketch to your board
2. Run the data collection script:
```bash
python3 collect_data.py
```

3. Follow the prompts:
   - Select a gesture number
   - Press ENTER when ready
   - Perform the gesture after the countdown
   - Repeat for multiple samples

## Gestures

The script includes these pre-defined gestures:
- idle (no movement)
- swipe_left
- swipe_right
- swipe_up
- swipe_down
- shake
- rotate_cw (clockwise)
- rotate_ccw (counter-clockwise)
- tap

## Output

Data is saved to `data/gesture_data_YYYYMMDD_HHMMSS.csv` with columns:
- gesture: Label for the gesture
- sample_id: Unique ID for each recording
- timestamp: Milliseconds since Arduino boot
- aX, aY, aZ: Accelerometer data (g)
- gX, gY, gZ: Gyroscope data (°/s)
- temp: Temperature (°C)

## Tips

- Collect 10-20 samples per gesture for good training data
- Perform gestures consistently
- Include variations (different speeds, angles)
- Keep the Arduino powered via USB during collection
