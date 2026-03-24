"""MX-64 hardware interface — position control mode.

Uses the MX-64 built-in position controller (Extended Position Control Mode,
operating mode 4, which allows multi-turn).  Goal position is written in raw
ticks; state is read back as position (rad) and velocity (rad/s).

Typical usage:
    hw = MX64Hardware()        # motor_id=None → auto-detected on connect()
    hw.connect()               # pings bus, finds motor ID automatically
    hw.enable()                # sets mode, enables torque

    hw.set_position_rad(0.5)
    state = hw.read_state()
    print(state.pos_rad, state.vel_rads)

    hw.shutdown()
"""

from __future__ import annotations

import time

import numpy as np
from dynamixel_sdk import (
    COMM_SUCCESS,
    PacketHandler,
    PortHandler,
)

# ── Default hardware config ────────────────────────────────────────────────────
DEFAULT_DEVICE   = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 4_000_000
DEFAULT_PROTOCOL = 2.0
DEFAULT_MOTOR_ID = None   # auto-detect via scan if not specified

# ── MX-64 control table addresses ─────────────────────────────────────────────
_ADDR_OPERATING_MODE   = 11
_ADDR_TORQUE_ENABLE    = 64
_ADDR_POSITION_D_GAIN  = 80    # 2 bytes  Kd_raw = Kd * 16
_ADDR_POSITION_I_GAIN  = 82    # 2 bytes  Ki_raw = Ki * 65536
_ADDR_POSITION_P_GAIN  = 84    # 2 bytes  Kp_raw = Kp * 128
_ADDR_GOAL_POSITION    = 116   # 4 bytes
_ADDR_MOVING_STATUS    = 123   # 1 byte   — bit0: in-position, bit1: instruction ongoing, bit2: profile ongoing
_ADDR_PRESENT_CURRENT  = 126   # 2 bytes  \
_ADDR_PRESENT_VELOCITY = 128   # 4 bytes   | contiguous 10-byte read
_ADDR_PRESENT_POSITION = 132   # 4 bytes  /
_READ_BASE             = 126
_READ_LEN              = 10

_MODE_EXTENDED_POSITION = 4    # multi-turn position control

# Gain conversions (from datasheet):
#   Kp = Kp_gain(84) / 128   →  raw = Kp * 128
#   Ki = Ki_gain(82) / 65536 →  raw = Ki * 65536
#   Kd = Kd_gain(80) / 16    →  raw = Kd * 16
DEFAULT_KP = 1.0    # Nm/rad  (written on every enable())
DEFAULT_KD = 0.0    # Nm·s/rad

# ── Physical constants ─────────────────────────────────────────────────────────
_TICKS_PER_REV  = 4096
_RAD_PER_TICK   = 2.0 * np.pi / _TICKS_PER_REV
_RPM_PER_LSB    = 0.229
_RADS_PER_LSB   = _RPM_PER_LSB * 2.0 * np.pi / 60.0
_KT             = 1.463           # Nm/A
_CURRENT_LSB_A  = 2.69e-3
_KT_PER_LSB     = _CURRENT_LSB_A * _KT

_MAX_POSITION_RAW = 2**31 - 1   # extended position is signed 32-bit


# ── Motor scanner ──────────────────────────────────────────────────────────────

def scan_motors(
    device:   str   = DEFAULT_DEVICE,
    baudrate: int   = DEFAULT_BAUDRATE,
    protocol: float = DEFAULT_PROTOCOL,
    id_range: range = range(1, 21),
) -> list[int]:
    """Ping all IDs in id_range and return a list of responding motor IDs."""
    port = PortHandler(device)
    ph   = PacketHandler(protocol)

    if not port.openPort():
        raise RuntimeError(f"Cannot open port {device}")
    if not port.setBaudRate(baudrate):
        port.closePort()
        raise RuntimeError(f"Cannot set baudrate {baudrate}")

    found: list[int] = []
    for mid in id_range:
        _, result, _ = ph.ping(port, mid)
        if result == COMM_SUCCESS:
            print(f"[scan] Found motor ID {mid:2d}")
            found.append(mid)

    port.closePort()
    print(f"[scan] Done. Found {len(found)} motor(s): {found}")
    return found


# ── MotorState ─────────────────────────────────────────────────────────────────

class MotorState:
    """Snapshot of one sensor read. All values in physical units."""
    __slots__ = ("pos_rad", "vel_rads", "torque_nm")

    def __init__(self, pos_raw: int, vel_raw: int, cur_raw: int) -> None:
        self.pos_rad  = _to_signed32(pos_raw) * _RAD_PER_TICK
        self.vel_rads = _to_signed32(vel_raw) * _RADS_PER_LSB
        self.torque_nm = _to_signed16(cur_raw) * _KT_PER_LSB


# ── MX64Hardware ──────────────────────────────────────────────────────────────

class MX64Hardware:
    """High-level interface to a single MX-64 in extended position control mode.

    Args:
        device:   Serial port (default '/dev/ttyUSB0').
        baudrate: Bus baud rate (default 4 000 000).
        motor_id: Dynamixel ID (default 17).
        protocol: Dynamixel protocol version (default 2.0).
    """

    def __init__(
        self,
        device:   str        = DEFAULT_DEVICE,
        baudrate: int        = DEFAULT_BAUDRATE,
        motor_id: int | None = DEFAULT_MOTOR_ID,
        protocol: float      = DEFAULT_PROTOCOL,
    ) -> None:
        self._device   = device
        self._baudrate = baudrate
        self._motor_id = motor_id
        self._protocol = protocol
        self._port: PortHandler | None   = None
        self._ph:   PacketHandler | None = None

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._port = PortHandler(self._device)
        self._ph   = PacketHandler(self._protocol)
        if not self._port.openPort():
            raise RuntimeError(f"Cannot open port {self._device}")
        if not self._port.setBaudRate(self._baudrate):
            raise RuntimeError(f"Cannot set baudrate {self._baudrate}")
        if self._motor_id is None:
            found = []
            for mid in range(1, 21):
                _, result, _ = self._ph.ping(self._port, mid)
                if result == COMM_SUCCESS:
                    found.append(mid)
            if not found:
                self._port.closePort()
                raise RuntimeError("No Dynamixel motors found on the bus.")
            if len(found) > 1:
                print(f"[MX64] Warning: multiple motors found {found}, using ID {found[0]}")
            self._motor_id = found[0]
            print(f"[MX64] Auto-detected motor ID: {self._motor_id}")
        print(f"[MX64] Connected on {self._device} @ {self._baudrate} baud  (ID {self._motor_id}).")

    def disconnect(self) -> None:
        if self._port is not None:
            self._port.closePort()
            self._port = None
            print("[MX64] Port closed.")

    # ── Enable / disable ───────────────────────────────────────────────────────

    def enable(self, kp: float = DEFAULT_KP, kd: float = DEFAULT_KD) -> None:
        """Set extended position mode, write PID gains, enable torque output.

        Gains must be written after the mode switch because the firmware resets
        them when the operating mode changes.

        Args:
            kp: Position P gain (physical units). Default 10.
            kd: Position D gain (physical units). Default 0.
        """
        self._set_operating_mode(_MODE_EXTENDED_POSITION)
        self._write_position_gains(kp=kp, kd=kd)
        self._torque_enable(True)
        print(f"[MX64] Extended position control enabled  kp={kp}  kd={kd}.")

    def disable(self) -> None:
        self._torque_enable(False)
        print("[MX64] Torque disabled.")

    # ── Write ──────────────────────────────────────────────────────────────────

    def set_position_rad(self, pos_rad: float) -> None:
        """Command a goal position in radians (multi-turn)."""
        raw = int(round(pos_rad / _RAD_PER_TICK))
        raw = int(np.clip(raw, -_MAX_POSITION_RAW, _MAX_POSITION_RAW))
        if raw < 0:
            raw += 2**32   # two's complement for unsigned SDK write
        result, error = self._ph.write4ByteTxRx(
            self._port, self._motor_id, _ADDR_GOAL_POSITION, raw
        )
        self._check(result, error, "set_position")

    # ── Read ───────────────────────────────────────────────────────────────────

    def read_state(self) -> MotorState | None:
        """Read position, velocity, and estimated torque in one round-trip."""
        data, result, error = self._ph.readTxRx(
            self._port, self._motor_id, _READ_BASE, _READ_LEN
        )
        if result != COMM_SUCCESS or error != 0 or len(data) < _READ_LEN:
            return None

        cur_raw = (data[0]) | (data[1] << 8)
        vel_raw = (data[2]) | (data[3] << 8) | (data[4] << 16) | (data[5] << 24)
        pos_raw = (data[6]) | (data[7] << 8) | (data[8] << 16) | (data[9] << 24)
        return MotorState(pos_raw, vel_raw, cur_raw)

    def read_moving_status(self) -> dict | None:
        """Read Moving Status (addr 123, 1 byte).

        Returns a dict with keys:
            in_position         (bit 0) — target position reached
            instruction_ongoing (bit 1) — processing a goal instruction
            profile_ongoing     (bit 2) — velocity profile still in progress
            raw                         — raw byte value
        Returns None on comm failure.
        """
        data, result, error = self._ph.readTxRx(
            self._port, self._motor_id, _ADDR_MOVING_STATUS, 1
        )
        if result != COMM_SUCCESS or error != 0 or len(data) < 1:
            return None
        raw = data[0]
        return {
            "in_position":         bool(raw & 0x01),
            "instruction_ongoing": bool(raw & 0x02),
            "profile_ongoing":     bool(raw & 0x04),
            "raw":                 raw,
        }

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Disable torque and close port."""
        try:
            self._torque_enable(False)
            time.sleep(0.05)
        except Exception as exc:
            print(f"[MX64] Warning during shutdown: {exc}")
        finally:
            self.disconnect()
        print("[MX64] Shutdown complete.")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check(self, result: int, error: int, label: str) -> bool:
        if result != COMM_SUCCESS:
            print(f"[MX64][COMM ERROR] {label}: {self._ph.getTxRxResult(result)}")
            return False
        if error != 0:
            print(f"[MX64][PKT ERROR]  {label}: {self._ph.getRxPacketError(error)}")
            return False
        return True

    def _write1(self, addr: int, value: int, label: str) -> None:
        result, error = self._ph.write1ByteTxRx(
            self._port, self._motor_id, addr, value
        )
        if not self._check(result, error, label):
            raise RuntimeError(f"Write failed: {label}")

    def _torque_enable(self, enable: bool) -> None:
        self._write1(_ADDR_TORQUE_ENABLE, 1 if enable else 0,
                     "torque_enable" if enable else "torque_disable")

    def _set_operating_mode(self, mode: int) -> None:
        try:
            self._torque_enable(False)
        except RuntimeError:
            pass  # transient comm error on pre-disable is non-fatal
        time.sleep(0.05)
        self._write1(_ADDR_OPERATING_MODE, mode, f"set_mode({mode})")
        time.sleep(0.05)

    def _write2(self, addr: int, value: int, label: str) -> None:
        result, error = self._ph.write2ByteTxRx(
            self._port, self._motor_id, addr, value
        )
        if not self._check(result, error, label):
            raise RuntimeError(f"Write failed: {label}")

    def _write_position_gains(self, kp: float, kd: float) -> None:
        """Write Kp and Kd to the motor registers using datasheet conversions.

        Kp_raw = Kp * 128   (addr 84)
        Kd_raw = Kd * 16    (addr 80)
        Ki is left at 0.
        """
        kp_raw = int(np.clip(round(kp * 128),  0, 16383))
        kd_raw = int(np.clip(round(kd * 16),   0, 16383))
        self._write2(_ADDR_POSITION_P_GAIN, kp_raw, "position_p_gain")
        self._write2(_ADDR_POSITION_D_GAIN, kd_raw, "position_d_gain")
        self._write2(_ADDR_POSITION_I_GAIN, 0,      "position_i_gain")
        print(f"[MX64] Gains written — Kp={kp} (raw {kp_raw})  "
              f"Kd={kd} (raw {kd_raw})  Ki=0")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_signed16(value: int) -> int:
    return value - 65536 if value > 32767 else value


def _to_signed32(value: int) -> int:
    return value - 2**32 if value > 2**31 - 1 else value
