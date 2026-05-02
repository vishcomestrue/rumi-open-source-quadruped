import sys
import time
import types
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_ROTATION_VECTOR,
    _separate_batch,
)

# BNO_REPORT_ACCELEROMETER = specific force (gravity-included), matching MuJoCo's
# <accelerometer> sensor which also outputs specific force (~+9.8 in z when upright).
_FEATURES = [BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE, BNO_REPORT_ROTATION_VECTOR]


def _quiet_handle_packet(self, packet):
    """Drop-in for BNO08X._handle_packet without the unconditional print."""
    try:
        _separate_batch(packet, self._packet_slices)
        while self._packet_slices:
            self._process_report(*self._packet_slices.pop())
    except Exception as error:
        raise error


class IMUReader:
    def __init__(self, address=0x4b):
        self._address = address
        self._i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self._connect()

    def _connect(self):
        for _ in range(10):
            try:
                self._bno = BNO08X_I2C(self._i2c, address=self._address, debug=False)
                # Silence the unconditional print(packet) for unknown/unsupported reports
                self._bno._handle_packet = types.MethodType(_quiet_handle_packet, self._bno)
                for f in _FEATURES:
                    self._bno.enable_feature(f, report_interval=20000)   # 5ms = 200Hz
                    time.sleep(0.05)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("BNO080 failed to initialize")

    def read(self):
        """
        Returns {'timestamp', 'quaternion' (x,y,z,w), 'accel' (x,y,z), 'gyro' (x,y,z)}
        or None on reset or corrupt frame (caller should skip).
        """
        try:
            quat = self._bno.quaternion   # (i, j, k, real) = (x, y, z, w)
            accel = self._bno.acceleration
            gyro = self._bno.gyro
        except OSError:
            time.sleep(0.5)   # wait for sensor reset to complete
            try:
                self._connect()
            except RuntimeError:
                pass           # still recovering, will retry next read()
            return None
        except Exception:
            return None

        # Drop frames where sensor hasn't produced data yet or I2C bit errors:
        #   - quaternion components must be in [-1, 1] (unit quaternion)
        #   - gyro must be within BNO080 hardware range (±2000 dps = ±34.9 rad/s)
        if quat is None or accel is None or gyro is None:
            return None
        if any(abs(v) > 1.0 for v in quat):
            return None
        if any(abs(v) > 35.0 for v in gyro):
            return None
        if any(abs(v) > 78.5 for v in accel):   # BNO080 SH-2 configures accel for ±8g = ±78.5 m/s²
            return None

        return {'timestamp': time.time(), 'quaternion': quat, 'accel': accel, 'gyro': gyro}
