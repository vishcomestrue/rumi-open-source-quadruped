"""BNO080 IMU interface for deploy — background thread, non-blocking reads."""

import time
import threading
import numpy as np
from bno080 import IMUReader


class IMU:
    """Runs IMUReader in a background thread so read_quaternion() never blocks.

    I2C latency and sensor reconnects are absorbed in the thread — the
    control loop always gets the latest valid quaternion instantly.
    """

    def __init__(self, i2c_address: int = 0x4b):
        print(f"[IMU] Initializing BNO080 at I2C 0x{i2c_address:02x} ...")
        self._reader = IMUReader(address=i2c_address)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[IMU] Ready.")

    def _run(self):
        while True:
            data = self._reader.read()
            if data is not None:
                qx, qy, qz, qw = data['quaternion']   # bno080.py returns (x, y, z, w)
                with self._lock:
                    self._quat = np.array([qw, qx, qy, qz], dtype=np.float32)
            else:
                time.sleep(0.005)   # brief pause during errors / reconnects

    def read_quaternion(self) -> np.ndarray:
        """Return latest (w, x, y, z) quaternion — non-blocking."""
        with self._lock:
            return self._quat.copy()
