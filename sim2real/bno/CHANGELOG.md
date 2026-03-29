# IMU Viewer Changelog

## Recent Updates

### Fix: Quaternion Unpacking Order (Critical Fix)
- ✅ **Fixed quaternion reading from BNO08x library**
- BNO08x returns `(i, j, k, real)` = `(x, y, z, w)` format
- bare.py was unpacking as `(w, x, y, z)` - WRONG!
- Changed to correct order: `qx, qy, qz, qw = self.bno.quaternion`
- This affects all orientation calculations

### Removed Unnecessary Axis Transformations
- ✅ **Removed all axis transformation code from viewer.py**
- Previous transformations were debugging incorrect quaternion data
- Sensor orientation now matches visualization directly
- Cleaner, simpler codebase

## Features

### 3D Visualization
- ✅ Real-time IMU orientation (quaternion-based)
- ✅ Black cuboid representing sensor (3cm × 2.5cm × 2mm)
- ✅ Dark red LED indicator on top surface
- ✅ World frame with Z-up convention
- ✅ Ground plane for reference (30cm × 30cm)
- ✅ Updates at 40Hz

### Time Series Plots
- ✅ Gyroscope (rad/s) - X, Y, Z
- ✅ Accelerometer (m/s²) - X, Y, Z
- ✅ Magnetometer (µT) - X, Y, Z
- ✅ Linear Acceleration (m/s²) - X, Y, Z
- ✅ Euler Angles (degrees) - Roll, Pitch, Yaw
- ✅ Rolling window display (configurable buffer)
- ✅ Updates at 10Hz (downsampled for performance)

## Coordinate System

### World Frame (East-North-Up)
- X-axis: East (tangent to ground)
- Y-axis: North (magnetic, tangent to ground)
- Z-axis: Up (perpendicular to ground)

### Quaternion
Rotation from World Frame → Sensor Frame

### Euler Angles (ZYX Convention)
- Roll: Rotation around X-axis
- Pitch: Rotation around Y-axis
- Yaw: Rotation around Z-axis
