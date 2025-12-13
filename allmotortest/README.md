# Multi-Motor Dynamixel Controller

<!--
rumi-custom-quadruped — Reinforcement-learning-based quadruped-robot control framework for custom quadruped
Copyright (C) 2025  Vishwanath R

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

High-frequency controller for simultaneous control of multiple Dynamixel motors using GroupSyncRead/Write with automatic motor discovery and power measurement support.

## Features

- ✅ **Automatic Motor Discovery**: Broadcast ping to detect all connected motors
- ✅ **High-Frequency Control**: 80-120Hz with GroupSyncRead/Write
- ✅ **Power Measurement**: Real-time current, voltage, power, and energy tracking
- ✅ **Incremental Control**: Smooth oscillation with direction reversal at limits
- ✅ **Multi-Motor Support**: Easily scale from 3 to 12+ motors
- ✅ **Robust Error Handling**: Automatic retries and graceful degradation

## Quick Start

```bash
# First, discover connected motors
python3 ping_motors.py

# Scan with custom settings
python3 ping_motors.py --device /dev/ttyUSB1 --protocol 2.0 --baudrate 1000000

# Test 3 motors at 100Hz (default step size: 40 units)
python3 multi_motor_controller.py

# Test 12 motors at 50Hz with larger steps (faster cycle)
python3 multi_motor_controller.py -n 12 -r 50 -s 100

# Test with fine control (small steps)
python3 multi_motor_controller.py -s 20 -r 50

# View all options
python3 multi_motor_controller.py -h
```

## What It Does

1. **Auto-discovers motors** via broadcast ping (shows model numbers and firmware versions)
2. **Selects first N motors** by ID and enables torque
3. **Reads current motor positions** from all motors simultaneously
4. **Calculates next position**: `current_pos + step_size` (oscillates between 0-4000)
5. **Writes positions** to ALL motors in ONE packet (GroupSyncWrite)
6. **Reads back positions** from ALL motors in ONE packet (GroupSyncRead)
7. **Measures power** (optional, for supported models): current, voltage, energy consumption
8. **Repeats incrementally** - no absolute sequence, just relative steps

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n N` | Number of motors | 3 |
| `-r HZ` | Control frequency (Hz) | 100 |
| `-p PORT` | Serial port | /dev/ttyUSB0 |
| `-b BAUD` | Baudrate (bps) | 2000000 |
| `-d SEC` | Write-read delay (sec) | 0.0005 |
| `-s STEP` | Position step size (units) | 40 |

## Test Pattern (Incremental Positioning)

**Approach**: Read current position → Add step → Write new position → Repeat

- **Step size**: Configurable (default: 40 units = ~3.5°)
- **Range**: 0 to 4000 units
- **Pattern**: Starts from current position, increments by step_size, reverses at limits

**Benefits**:
- No sudden jumps at sequence boundaries
- Handles intermittent read failures gracefully (continues with last known position)
- Smoother motion
- More robust for multiple motors

**Step Size Examples:**

| Step Size | Degrees/Step | Safe at 20Hz? | Safe at 100Hz? |
|-----------|--------------|---------------|----------------|
| 20 units | 1.8° | ✓ Yes | ✓ Yes |
| 40 units | 3.5° | ✓ Yes | ✓ Yes |
| 100 units | 8.8° | ✓ Yes | ✗ Marginal |
| 200 units | 17.6° | ✓ Yes | ✗ No |

**Safe step sizes** (motors can reach target in 1 cycle):
  - 20Hz: ≤215 units
  - 100Hz: ≤43 units

## Usage in Code

```python
from multi_motor_controller import MultiMotorController

# Initialize (motors will be auto-discovered)
controller = MultiMotorController(num_motors=3)
controller.connect()  # Discovers motors and selects first 3 by ID

# Write positions (Dynamixel units)
controller.write_positions({1: 2048, 2: 3000, 3: 1500})

# Read positions
positions = controller.read_positions()  # Returns {1: 2048, 2: 3000, 3: 1500}

# Or use radians
controller.write_positions_radians({1: 0.0, 2: 0.5, 3: -0.3})
positions_rad = controller.read_positions_radians()

# Read power measurements (if supported by motors)
power_data = controller.read_power_measurements()
if power_data:
    print(f"Total current: {power_data['total_current_A']:.3f} A")
    print(f"Average voltage: {power_data['avg_voltage_V']:.2f} V")
    print(f"Total power: {power_data['total_power_W']:.2f} W")
    # Per-motor data available in power_data['per_motor']

controller.disconnect()
```

## Power Measurement

Power measurement is automatically enabled for motors that support it (Protocol 2.0 motors with model number ≥ 300).

**Features:**
- Real-time current, voltage, and power readings
- Per-motor and total system measurements
- Automatic energy consumption tracking (Wh)
- Statistics: min/max/average values during session

**Supported Motors:**
- ✅ X-Series (XM, XH, XW)
- ✅ P-Series
- ❌ MX-Series with Protocol 1.0 (models < 300)

**Output Example:**
```
[POWER STATISTICS] Electrical Measurements:
  Samples collected: 150
  Total Current:
    Average: 1.234 A
    Min:     0.456 A
    Max:     2.567 A
  Average Voltage:
    Average: 11.8 V
    Min:     11.5 V
    Max:     12.1 V
  Total Power:
    Average: 14.56 W
    Min:     5.23 W
    Max:     31.06 W
  Total Energy:
    Estimated: 0.243 Wh (243.0 mWh)
```

## Troubleshooting

**No motors discovered**:
- Check port is correct (`/dev/ttyUSB0`, `/dev/ttyUSB1`, etc.)
- Verify motors are powered on
- Ensure baudrate matches motor configuration
- Check cable connections

**Insufficient motors found**:
- Program will prompt to continue with fewer motors
- Check all motors are powered and connected
- Verify motor IDs are sequential (1, 2, 3...)

**Read errors**:
- Increase delay `-d 0.001` or reduce frequency `-r 20`
- Check for loose connections or EMI

**Low frequency**:
- Use higher baudrate `-b 3000000` or `-b 4000000`
- Reduce number of motors
- Minimize write-read delay `-d 0`

## Files

- `ping_motors.py` - Motor discovery utility to detect all connected Dynamixel motors
- `multi_motor_controller.py` - Main controller class and test script with auto-discovery and power measurement
- `mx_64.md` - Motor specifications and control table
- `README.md` - This documentation file
