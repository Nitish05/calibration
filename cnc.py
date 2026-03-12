"""Reusable CNC serial controller class for Grbl-based machines."""

import threading
import time


class CNCController:
    def __init__(self):
        self._serial = None
        self._lock = threading.Lock()
        self._wco = {"x": 0.0, "y": 0.0, "z": 0.0}

    # ------------------------------------------------------------------
    # Auto-detect
    # ------------------------------------------------------------------
    @staticmethod
    def find_serial_port():
        """Find the CNC serial port by scanning available USB serial devices."""
        try:
            import serial.tools.list_ports
        except ImportError:
            return None
        for port in serial.tools.list_ports.comports():
            if "CH340" in (port.description or "") or "CH340" in (port.manufacturer or ""):
                print(f"[CNC] Auto-detected serial port: {port.device} ({port.description})")
                return port.device
        for port in serial.tools.list_ports.comports():
            if "ttyUSB" in port.device or "ttyACM" in port.device:
                print(f"[CNC] Auto-detected serial port: {port.device} ({port.description})")
                return port.device
        return None

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------
    def connect(self, port, baud=115200):
        """Open serial connection to Grbl CNC controller."""
        import serial
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(1)
        ser.reset_input_buffer()
        with self._lock:
            self._serial = ser
        self._fetch_initial_wco()
        print(f"[CNC] Connected to {port} @ {baud}  WCO: {self._wco}")

    def disconnect(self):
        """Close serial connection."""
        with self._lock:
            if self._serial:
                self._serial.close()
                self._serial = None
        print("[CNC] Disconnected")

    @property
    def connected(self):
        with self._lock:
            return self._serial is not None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def query_status(self):
        """Send ? and parse Grbl status. Returns dict or None."""
        with self._lock:
            ser = self._serial
        if ser is None:
            return None
        try:
            ser.write(b"?")
        except (OSError, Exception):
            return None
        deadline = time.time() + 1.0
        while time.time() < deadline:
            try:
                resp = ser.readline().decode("ascii", errors="replace").strip()
            except (OSError, Exception):
                return None
            if resp.startswith("<") and resp.endswith(">"):
                inner = resp[1:-1]
                parts = inner.split("|")
                state = parts[0]
                wpos = None
                for p in parts[1:]:
                    if p.startswith("MPos:"):
                        c = p[5:].split(",")
                        if len(c) >= 3:
                            wpos = {
                                "x": float(c[0]) - self._wco["x"],
                                "y": float(c[1]) - self._wco["y"],
                                "z": float(c[2]) - self._wco["z"],
                            }
                    elif p.startswith("WPos:"):
                        c = p[5:].split(",")
                        if len(c) >= 3:
                            wpos = {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
                    elif p.startswith("WCO:"):
                        c = p[4:].split(",")
                        if len(c) >= 3:
                            self._wco = {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
                if wpos is None:
                    wpos = {"x": 0.0, "y": 0.0, "z": 0.0}
                return {"state": state, "wpos": wpos}
        return None

    # ------------------------------------------------------------------
    # Jog cancel
    # ------------------------------------------------------------------
    def jog_cancel(self):
        """Send realtime jog cancel (0x85) to immediately stop a jog."""
        with self._lock:
            ser = self._serial
        if ser:
            ser.write(b"\x85")

    # ------------------------------------------------------------------
    # Send G-code
    # ------------------------------------------------------------------
    def send_line(self, line):
        """Send one G-code line and wait for ok/error response."""
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("("):
            return True
        with self._lock:
            ser = self._serial
        if ser is None:
            return "error: not connected"
        ser.write((line + "\n").encode())
        while True:
            resp = ser.readline().decode("ascii", errors="replace").strip()
            if not resp:
                continue
            if resp.startswith("<"):
                continue
            if resp == "ok":
                return True
            if resp.startswith("error"):
                return resp
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _fetch_initial_wco(self):
        """Query status repeatedly until we get a WCO value to cache."""
        with self._lock:
            ser = self._serial
        if ser is None:
            return
        for _ in range(50):
            try:
                ser.write(b"?")
                deadline = time.time() + 0.2
                while time.time() < deadline:
                    resp = ser.readline().decode("ascii", errors="replace").strip()
                    if resp.startswith("<") and resp.endswith(">"):
                        for p in resp[1:-1].split("|"):
                            if p.startswith("WCO:"):
                                c = p[4:].split(",")
                                if len(c) >= 3:
                                    self._wco = {
                                        "x": float(c[0]),
                                        "y": float(c[1]),
                                        "z": float(c[2]),
                                    }
                                    return
            except Exception:
                return
            time.sleep(0.05)
        print("[CNC] Warning: could not fetch initial WCO, using 0,0,0")
