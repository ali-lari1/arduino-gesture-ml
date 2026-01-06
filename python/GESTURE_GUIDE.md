# Gesture Performance Guide

This guide explains how to perform each gesture for data collection. Hold the Arduino/MPU-6050 in your hand comfortably, with the sensor secure but not too tight.

## General Tips

- **Consistency is key**: Perform each gesture the same way every time
- **Natural movements**: Don't be too robotic, but stay consistent
- **Full motion**: Complete the entire gesture within the 2-second window
- **Return to neutral**: End in a resting position
- **Multiple samples**: Collect 10-20 samples per gesture for best results

---

## 1. Idle
**Description**: Keep sensor still on table

**How to perform**:
- Place the Arduino flat on a stable table
- Don't touch it during recording
- This captures the "no gesture" baseline
- Sensor should be completely stationary

**What the sensor sees**:
- aZ ≈ 1g (gravity pointing up)
- All gyro values near zero (no rotation)

---

## 2. Swipe Left
**Description**: Quick horizontal swipe to the left

**How to perform**:
1. Hold sensor in your hand, arm relaxed
2. When countdown says "GO!", quickly swipe your hand to the left
3. Move about 30-40cm (1-1.5 feet) horizontally
4. Keep the motion sharp and decisive
5. Return to resting position

**What the sensor sees**:
- **aX** spikes negative during the swipe
- Brief gyroscope activity

**Tips**: Like swiping a phone screen, but bigger and in the air

---

## 3. Swipe Right
**Description**: Quick horizontal swipe to the right

**How to perform**:
1. Hold sensor in your hand, arm relaxed
2. When countdown says "GO!", quickly swipe your hand to the right
3. Move about 30-40cm horizontally
4. Keep the motion sharp and decisive
5. Return to resting position

**What the sensor sees**:
- **aX** spikes positive during the swipe
- Brief gyroscope activity

**Tips**: Mirror image of swipe left

---

## 4. Swipe Up
**Description**: Quick upward motion

**How to perform**:
1. Start with hand at waist level
2. When countdown says "GO!", quickly move hand upward
3. Move about 30-40cm vertically
4. Like raising your hand to answer a question
5. Return to resting position

**What the sensor sees**:
- **aZ** increases significantly during upward acceleration
- Then decreases as you decelerate

**Tips**: Sharp upward motion, not slow lifting

---

## 5. Swipe Down
**Description**: Quick downward motion

**How to perform**:
1. Start with hand at chest/shoulder level
2. When countdown says "GO!", quickly move hand downward
3. Move about 30-40cm vertically
4. Like chopping motion or swatting downward
5. Return to resting position

**What the sensor sees**:
- **aZ** decreases (can briefly go negative)
- Opposite pattern of swipe up

**Tips**: Controlled but quick downward motion

---

## 6. Punch
**Description**: Forward punching motion

**How to perform**:
1. Hold sensor in your fist
2. Start with arm bent, hand near your shoulder
3. When countdown says "GO!", quickly extend arm forward in a punching motion
4. Extend about 30-40cm forward
5. Pull back to starting position

**What the sensor sees**:
- **aY** spikes positive (forward acceleration)
- Brief spike in opposite direction when stopping
- Possible rotation from wrist twist

**Tips**:
- Like a boxing jab - quick extension and retraction
- Keep wrist relatively stable
- Don't hyperextend your elbow

---

## 7. Shake
**Description**: Rapid side-to-side shaking

**How to perform**:
1. Hold sensor in your hand
2. When countdown says "GO!", rapidly shake hand left-right
3. Small quick movements (about 10-15cm side to side)
4. Do 4-6 shakes within the 2 seconds
5. Fast oscillating motion

**What the sensor sees**:
- **aX** oscillates rapidly positive and negative
- High frequency changes in acceleration
- **gY** or **gZ** shows rotation if you rotate while shaking

**Tips**: Like shaking a spray paint can or dice

---

## 8. Rotate Clockwise (CW)
**Description**: Rotate sensor clockwise in a circle

**How to perform**:
1. Hold sensor flat (like holding a phone screen-up)
2. When countdown says "GO!", rotate your hand clockwise
3. Make 1-2 complete circles within 2 seconds
4. Keep the rotation smooth and continuous
5. Imagine stirring a pot

**What the sensor sees**:
- **gZ** shows positive rotation (clockwise)
- Accelerometer shows circular pattern as orientation changes
- Gravity vector rotates through different axes

**Tips**:
- Keep sensor relatively flat
- Smooth circular motion
- Looking down at the sensor, rotate it like a clock hand

---

## 9. Rotate Counter-Clockwise (CCW)
**Description**: Rotate sensor counter-clockwise

**How to perform**:
1. Hold sensor flat (like holding a phone screen-up)
2. When countdown says "GO!", rotate your hand counter-clockwise
3. Make 1-2 complete circles within 2 seconds
4. Keep the rotation smooth and continuous
5. Opposite direction from clockwise

**What the sensor sees**:
- **gZ** shows negative rotation (counter-clockwise)
- Accelerometer shows circular pattern (opposite direction from CW)

**Tips**: Mirror image of clockwise rotation

---

## 10. Tap
**Description**: Single quick tap/flick motion

**How to perform**:
1. Hold sensor in your hand
2. When countdown says "GO!", make a quick sharp flick/tap motion
3. Like flicking water off your fingers
4. Very brief, sudden motion
5. Can be in any direction - just make it sharp and quick

**What the sensor sees**:
- Very brief spike in one or more acceleration axes
- Short duration (under 0.5 seconds)
- Distinct from longer gestures

**Tips**:
- Think "snap" motion
- Quick wrist flick
- The sharper the better for distinguishing from other gestures

---

## Data Collection Strategy

For best ML model performance:

1. **Start with idle**: Collect several idle samples first
2. **Vary your samples**:
   - Different speeds (fast, medium)
   - Different amplitudes (big swipe, small swipe)
   - Different orientations (if relevant)
3. **Collect in batches**: Do 5-10 samples of one gesture, then move to next
4. **Take breaks**: Rest between gestures to stay consistent
5. **Minimum samples**: At least 10 per gesture, ideally 15-20

## Troubleshooting

**Not capturing data?**
- Check Serial Monitor shows data streaming
- Verify baud rate is 115200
- Ensure Arduino is connected to correct port

**Gestures look too similar?**
- Make motions more exaggerated
- Increase speed differences
- Add more distinctive characteristics

**Low accuracy later?**
- Collect more samples (20+ per gesture)
- Perform gestures more consistently
- Add variation within each gesture type
