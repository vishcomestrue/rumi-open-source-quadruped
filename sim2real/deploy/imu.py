"""BNO080 IMU interface for deploy — background thread, non-blocking reads."""

import time
import threading
import numpy as np
from bno080 import IMUReader


class IMU:
    """Runs IMUReader in a background thread so read_quaternion() never blocks.

    I2C latency and sensor reconnects are absorbed in the thread — the
    control loop always gets the latest valid quaternion instantly.
    Also exposes read_accel() and read_gyro() for tasks that need them.
    """

    def __init__(self, i2c_address: int = 0x4b):
        print(f"[IMU] Initializing BNO080 at I2C 0x{i2c_address:02x} ...")
        self._reader = IMUReader(address=i2c_address)
        self._quat  = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._accel = np.zeros(3, dtype=np.float32)
        self._gyro  = np.zeros(3, dtype=np.float32)
        self._seq   = 0        # incremented every time fresh sensor data arrives
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[IMU] Ready.")

    def _run(self):
        consec_none = 0
        while True:
            try:
                data = self._reader.read()
            except Exception:
                data = None
            if data is not None:
                consec_none = 0
                qx, qy, qz, qw = data['quaternion']   # bno080.py returns (x, y, z, w)
                ax, ay, az = data['accel']
                gx, gy, gz = data['gyro']
                with self._lock:
                    self._quat  = np.array([qw, qx, qy, qz], dtype=np.float32)
                    self._accel = np.array([ax, ay, az],      dtype=np.float32)
                    self._gyro  = np.array([gx, gy, gz],      dtype=np.float32)
                    self._seq  += 1
            else:
                consec_none += 1
                # After 500 ms of silence, the BNO080 has likely auto-reset.
                # Force a reconnect so the freeze doesn't last 2+ seconds.
                if consec_none >= 100:   # 100 × 5 ms = 500 ms
                    consec_none = 0
                    try:
                        self._reader._connect()
                    except Exception:
                        pass
                time.sleep(0.005)

    def read_quaternion(self) -> np.ndarray:
        """Return latest (w, x, y, z) quaternion — non-blocking."""
        with self._lock:
            return self._quat.copy()

    def read_accel(self) -> np.ndarray:
        """Return latest linear acceleration (x, y, z) in m/s² — non-blocking."""
        with self._lock:
            return self._accel.copy()

    def read_gyro(self) -> np.ndarray:
        """Return latest angular velocity (x, y, z) in rad/s — non-blocking."""
        with self._lock:
            return self._gyro.copy()

    def read_all(self) -> tuple:
        """Return (quat_wxyz, accel_xyz, gyro_xyz) atomically — non-blocking."""
        with self._lock:
            return self._quat.copy(), self._accel.copy(), self._gyro.copy()

    def read_all_with_seq(self) -> tuple:
        """Return (quat_wxyz, accel_xyz, gyro_xyz, seq) atomically.

        seq increments each time fresh data arrives from the sensor.
        If seq is unchanged since your last call, the data is stale (repeated frame).
        """
        with self._lock:
            return self._quat.copy(), self._accel.copy(), self._gyro.copy(), self._seq
