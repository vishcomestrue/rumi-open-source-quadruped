#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bare import IMUReader
from viewer import IMUViewer
import time


def main():
    print("="*80)
    print("BNO080 IMU with Viser Visualization")
    print("="*80)

    try:
        imu = IMUReader()
        viz = IMUViewer(port=8080, buffer_size=250)   # 5s at 50Hz

        print(f"\n{'='*80}")
        print("Visualization running at: http://localhost:8080")
        print(f"{'='*80}")
        print("\nStarting data acquisition at 50Hz...")
        print("Press Ctrl+C to stop\n")

        last_print = 0.0
        while True:
            data = imu.read()
            if data is not None:
                viz.update(data)
                # if data['timestamp'] - last_print >= 1.0:
                if data['timestamp'] != last_print:
                    qx, qy, qz, qw = data['quaternion']
                    gx, gy, gz = data['gyro']
                    print(f"[{data['timestamp']:.1f}s] "
                          f"quat=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})  "
                          f"gyro=({gx:.3f},{gy:.3f},{gz:.3f})")
                    last_print = data['timestamp']
            time.sleep(1 / 50)

    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
        print("Closing visualization server...")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
