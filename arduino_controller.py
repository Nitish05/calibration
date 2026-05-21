"""Arduino Uno + CNC Shield V3 motor controller.

Custom serial protocol for 6 actuators:
  - X stepper: X<steps>
  - Z stepper: Z<steps>
  - BYJ1 stepper: B<steps>
  - BYJ2 stepper: J<steps>
  - Servo: O<angle> (0-180)
  - DC motor: F<speed> (forward), G<speed> (reverse), S (stop)
  - P: query positions
  - R: reset all, H: reset steppers
"""

import os
import threading
import time


class ArduinoController:
    def __init__(self):
        self._serial = None
        self._lock = threading.Lock()
        # Firmware reports stepper positions via `P`, but servo angle and DC
        # state are write-only — track them host-side so the recorder can
        # snapshot a full machine state.
        self.current_servo_angle = None
        self.current_dc_state = {"action": "stop", "speed": 0}

    # ------------------------------------------------------------------
    # Auto-detect
    # ------------------------------------------------------------------
    @staticmethod
    def find_serial_port():
        """Find the Arduino serial port, avoiding the main CNC (CH340)."""
        symlink = "/dev/cnc_aux"
        if os.path.exists(symlink):
            print(f"[Arduino] Using symlink: {symlink}")
            return symlink

        try:
            import serial.tools.list_ports
        except ImportError:
            return None

        arduino_port = None
        fallback_port = None

        for port in serial.tools.list_ports.comports():
            vid = port.vid or 0
            # Exclude CH340 (main CNC)
            if vid == 0x1A86:
                continue
            # Prefer Arduino VID or description
            if vid == 0x2341 or "Arduino" in (port.description or ""):
                print(f"[Arduino] Auto-detected: {port.device} ({port.description})")
                return port.device
            # Fallback to ttyACM devices
            if "ttyACM" in port.device and fallback_port is None:
                fallback_port = port

        if fallback_port:
            print(f"[Arduino] Fallback: {fallback_port.device} ({fallback_port.description})")
            return fallback_port.device

        return None

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------
    def connect(self, port, baud=115200):
        """Open serial connection and wait for 'Control Ready' handshake."""
        import serial
        ser = serial.Serial(port, baud, timeout=5)
        # Wait for ready signal
        deadline = time.time() + 5
        ready = False
        while time.time() < deadline:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if "Control Ready" in line:
                ready = True
                break
        if not ready:
            print("[Arduino] Warning: did not receive 'Control Ready', continuing anyway")
        ser.reset_input_buffer()
        with self._lock:
            self._serial = ser
        print(f"[Arduino] Connected to {port} @ {baud}")

    def disconnect(self):
        """Close serial connection."""
        with self._lock:
            if self._serial:
                self._serial.close()
                self._serial = None
        print("[Arduino] Disconnected")

    @property
    def connected(self):
        with self._lock:
            return self._serial is not None

    # ------------------------------------------------------------------
    # Core command
    # ------------------------------------------------------------------
    def _send_inline(self, ser, cmd):
        """Send a command and read one line response. Caller must hold _lock.

        Drains the RX buffer before writing so any stale, unread response from
        a prior aborted transaction can't be misread as this command's reply.
        """
        try:
            ser.reset_input_buffer()
            ser.write((cmd + "\n").encode())
            resp = ser.readline().decode("ascii", errors="replace").strip()
            return resp if resp else None
        except (OSError, Exception) as e:
            print(f"[Arduino] Command '{cmd}' failed: {e}")
            return None

    def send_command(self, cmd):
        """Send a command string and read one line response. Returns string or None."""
        with self._lock:
            ser = self._serial
            if ser is None:
                return None
            return self._send_inline(ser, cmd)

    # ------------------------------------------------------------------
    # Position query
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_positions(resp):
        """Parse a 'X: N | Z: N | BYJ1: N | BYJ2: N' line. Returns dict or None."""
        if resp is None:
            return None
        try:
            positions = {}
            for part in resp.split("|"):
                part = part.strip()
                if ":" not in part:
                    continue
                key, val = part.split(":", 1)
                key = key.strip().lower()
                val = int(val.strip())
                if key in ("x", "z", "byj1", "byj2"):
                    positions[key] = val
            return positions if positions else None
        except (ValueError, IndexError):
            return None

    def query_positions(self):
        """Send 'P' and parse position response.

        Expected format: "X: N | Z: N | BYJ1: N | BYJ2: N"
        Returns dict {x, z, byj1, byj2} or None.
        """
        with self._lock:
            ser = self._serial
            if ser is None:
                return None
            return self._parse_positions(self._send_inline(ser, "P"))

    # ------------------------------------------------------------------
    # Stepper motors
    # ------------------------------------------------------------------
    def move_x(self, steps):
        return self.send_command(f"X{steps}")

    def move_z(self, steps):
        return self.send_command(f"Z{steps}")

    def move_byj1(self, steps):
        return self.send_command(f"B{steps}")

    def move_byj2(self, steps):
        return self.send_command(f"J{steps}")

    # ------------------------------------------------------------------
    # Servo
    # ------------------------------------------------------------------
    def set_servo(self, angle):
        angle = max(0, min(180, int(angle)))
        resp = self.send_command(f"O{angle}")
        self.current_servo_angle = angle
        return resp

    # ------------------------------------------------------------------
    # DC motor
    # ------------------------------------------------------------------
    def dc_forward(self, speed):
        speed = max(0, min(100, int(speed)))
        resp = self.send_command(f"F{speed}")
        self.current_dc_state = {"action": "forward", "speed": speed}
        return resp

    def dc_reverse(self, speed):
        speed = max(0, min(100, int(speed)))
        resp = self.send_command(f"G{speed}")
        self.current_dc_state = {"action": "reverse", "speed": speed}
        return resp

    def dc_stop(self):
        resp = self.send_command("S")
        self.current_dc_state = {"action": "stop", "speed": 0}
        return resp

    # ------------------------------------------------------------------
    # Absolute stepper targeting (for sequence playback)
    # ------------------------------------------------------------------
    _MOTOR_CMD = {"x": "X", "z": "Z", "byj1": "B", "byj2": "J"}

    def move_to_abs(self, motor, target_steps):
        """Move a stepper to an absolute step count. Atomic under _lock.

        Firmware accepts only relative moves, so we query the current position
        and send the difference — but the whole P-then-move sequence runs
        under one lock acquisition so another thread can't change the
        position between read and write. Returns firmware response, "no-op"
        if already at target, or None on error.
        """
        motor = motor.lower()
        if motor not in self._MOTOR_CMD:
            return None
        with self._lock:
            ser = self._serial
            if ser is None:
                return None
            positions = self._parse_positions(self._send_inline(ser, "P"))
            if positions is None or motor not in positions:
                return None
            delta = int(target_steps) - int(positions[motor])
            if delta == 0:
                return "no-op"
            return self._send_inline(ser, f"{self._MOTOR_CMD[motor]}{delta}")

    def move_all_to_abs(self, targets):
        """Atomic multi-motor absolute move.

        `targets` is a dict with any subset of {x, z, byj1, byj2} → int steps.
        One P query, then one move per non-zero delta, all under one lock.
        Returns {motor: response_or_"no-op"} or None if the P query failed.
        """
        with self._lock:
            ser = self._serial
            if ser is None:
                return None
            positions = self._parse_positions(self._send_inline(ser, "P"))
            if positions is None:
                return None
            results = {}
            for motor, cmd_char in self._MOTOR_CMD.items():
                if motor not in targets:
                    continue
                try:
                    target = int(targets[motor])
                except (TypeError, ValueError):
                    results[motor] = None
                    continue
                delta = target - int(positions.get(motor, 0))
                if delta == 0:
                    results[motor] = "no-op"
                else:
                    results[motor] = self._send_inline(ser, f"{cmd_char}{delta}")
            return results

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        return self.send_command("R")

    def reset_steppers(self):
        return self.send_command("H")
