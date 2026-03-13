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
    def send_command(self, cmd):
        """Send a command string and read one line response. Returns string or None."""
        with self._lock:
            ser = self._serial
            if ser is None:
                return None
            try:
                ser.write((cmd + "\n").encode())
                resp = ser.readline().decode("ascii", errors="replace").strip()
                return resp if resp else None
            except (OSError, Exception) as e:
                print(f"[Arduino] Command '{cmd}' failed: {e}")
                return None

    # ------------------------------------------------------------------
    # Position query
    # ------------------------------------------------------------------
    def query_positions(self):
        """Send 'P' and parse position response.

        Expected format: "X: N | Z: N | BYJ1: N | BYJ2: N"
        Returns dict {x, z, byj1, byj2} or None.
        """
        resp = self.send_command("P")
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
                if key == "x":
                    positions["x"] = val
                elif key == "z":
                    positions["z"] = val
                elif key == "byj1":
                    positions["byj1"] = val
                elif key == "byj2":
                    positions["byj2"] = val
            return positions if positions else None
        except (ValueError, IndexError):
            return None

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
        return self.send_command(f"O{angle}")

    # ------------------------------------------------------------------
    # DC motor
    # ------------------------------------------------------------------
    def dc_forward(self, speed):
        speed = max(0, min(100, int(speed)))
        return self.send_command(f"F{speed}")

    def dc_reverse(self, speed):
        speed = max(0, min(100, int(speed)))
        return self.send_command(f"G{speed}")

    def dc_stop(self):
        return self.send_command("S")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        return self.send_command("R")

    def reset_steppers(self):
        return self.send_command("H")
