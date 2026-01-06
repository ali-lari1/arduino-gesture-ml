// Human-readable version for Serial Monitor testing
// (c) Michael Schoeffler 2017, http://www.mschoeffler.de

#include <Arduino.h>
#include <Wire.h> // This library allows you to communicate with I2C devices.

const int MPU_ADDR = 0x68; // I2C address of the MPU-6050. If AD0 pin is set to HIGH, the I2C address will be 0x69.

// Conversion constants
const float ACCEL_SCALE = 16384.0; // LSB/g for ±2g range
const float GYRO_SCALE = 131.0;    // LSB/(°/s) for ±250°/s range

int16_t accelerometer_x, accelerometer_y, accelerometer_z; // variables for accelerometer raw data
int16_t gyro_x, gyro_y, gyro_z; // variables for gyro raw data
int16_t temperature; // variables for temperature data

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Check if MPU-6050 is connected
  Wire.beginTransmission(MPU_ADDR);
  byte error = Wire.endTransmission();

  if (error != 0) {
    Serial.println("ERROR: MPU-6050 not found - Check wiring!");
    while(1); // Halt if sensor not found
  }

  Wire.beginTransmission(MPU_ADDR); // Begins a transmission to the I2C slave (GY-521 board)
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0); // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);

  delay(100);

  // Print header
  Serial.println("========================================");
  Serial.println("MPU-6050 Gesture Sensor - Readable Mode");
  Serial.println("========================================");
  Serial.println();
}

void loop() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // starting with register 0x3B (ACCEL_XOUT_H) [MPU-6000 and MPU-6050 Register Map and Descriptions Revision 4.2, p.40]
  Wire.endTransmission(false); // the parameter indicates that the Arduino will send a restart. As a result, the connection is kept active.
  Wire.requestFrom(MPU_ADDR, 7*2, true); // request a total of 7*2=14 registers

  // "Wire.read()<<8 | Wire.read();" means two registers are read and stored in the same variable
  accelerometer_x = Wire.read()<<8 | Wire.read(); // reading registers: 0x3B (ACCEL_XOUT_H) and 0x3C (ACCEL_XOUT_L)
  accelerometer_y = Wire.read()<<8 | Wire.read(); // reading registers: 0x3D (ACCEL_YOUT_H) and 0x3E (ACCEL_YOUT_L)
  accelerometer_z = Wire.read()<<8 | Wire.read(); // reading registers: 0x3F (ACCEL_ZOUT_H) and 0x40 (ACCEL_ZOUT_L)
  temperature = Wire.read()<<8 | Wire.read(); // reading registers: 0x41 (TEMP_OUT_H) and 0x42 (TEMP_OUT_L)
  gyro_x = Wire.read()<<8 | Wire.read(); // reading registers: 0x43 (GYRO_XOUT_H) and 0x44 (GYRO_XOUT_L)
  gyro_y = Wire.read()<<8 | Wire.read(); // reading registers: 0x45 (GYRO_YOUT_H) and 0x46 (GYRO_YOUT_L)
  gyro_z = Wire.read()<<8 | Wire.read(); // reading registers: 0x47 (GYRO_ZOUT_H) and 0x48 (GYRO_ZOUT_L)

  // Convert to real-world units
  float accel_x_g = accelerometer_x / ACCEL_SCALE;
  float accel_y_g = accelerometer_y / ACCEL_SCALE;
  float accel_z_g = accelerometer_z / ACCEL_SCALE;

  float gyro_x_dps = gyro_x / GYRO_SCALE;
  float gyro_y_dps = gyro_y / GYRO_SCALE;
  float gyro_z_dps = gyro_z / GYRO_SCALE;

  float temp_c = temperature / 340.00 + 36.53;

  // Human-readable output - compact format
  Serial.println("================================================");

  Serial.print("Time: ");
  Serial.print(millis() / 1000.0, 2);
  Serial.println(" s");

  Serial.println();
  Serial.println("Accel (g)   |  Gyro (deg/s)  | Temp");
  Serial.println("------------|----------------|--------");

  // X values
  Serial.print("X: ");
  if (accel_x_g >= 0) Serial.print(" ");
  Serial.print(accel_x_g, 2);
  Serial.print("  | ");
  if (gyro_x_dps >= 0) Serial.print(" ");
  Serial.print(gyro_x_dps, 1);
  Serial.print("    | ");
  Serial.print(temp_c, 1);
  Serial.println(" C");

  // Y values
  Serial.print("Y: ");
  if (accel_y_g >= 0) Serial.print(" ");
  Serial.print(accel_y_g, 2);
  Serial.print("  | ");
  if (gyro_y_dps >= 0) Serial.print(" ");
  Serial.println(gyro_y_dps, 1);

  // Z values
  Serial.print("Z: ");
  if (accel_z_g >= 0) Serial.print(" ");
  Serial.print(accel_z_g, 2);
  Serial.print("  | ");
  if (gyro_z_dps >= 0) Serial.print(" ");
  Serial.println(gyro_z_dps, 1);

  // Slower refresh for readability (1000ms = 1Hz)
  delay(1000);
}
