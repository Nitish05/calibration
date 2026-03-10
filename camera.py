"""
Camera abstraction that works on both desktop (OpenCV) and Raspberry Pi (picamera2).
Set use_picamera=True to use picamera2 (for CSI cameras), otherwise uses OpenCV (for USB cameras).
"""

import cv2


class Camera:
    def __init__(self, width=1920, height=1080, device=0, use_picamera=False):
        self.width = width
        self.height = height
        self._picam2 = None
        self._cap = None

        if use_picamera:
            try:
                from picamera2 import Picamera2
                self._picam2 = Picamera2()
                config = self._picam2.create_video_configuration(
                    main={"size": (width, height), "format": "BGR888"},
                )
                self._picam2.configure(config)
                self._picam2.start()
                print(f"Camera: picamera2 ({width}x{height})")
                return
            except (ImportError, RuntimeError) as e:
                print(f"picamera2 failed ({e}), falling back to OpenCV")

        self._cap = cv2.VideoCapture(device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock_controls()
        print(f"Camera: OpenCV v4l2 device {device} ({width}x{height})")

    def _lock_controls(self):
        """Lock C920 controls for consistent imaging."""
        if self._cap is None:
            return
        import subprocess, time
        dev = f"/dev/video{int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) or 0}"
        # Find the actual device path
        try:
            backend = self._cap.getBackendName()
        except Exception:
            pass
        # Read a few frames to let the camera initialize
        for _ in range(5):
            self._cap.read()
        time.sleep(0.5)
        controls = [
            ("focus_automatic_continuous", "0"),
            ("focus_absolute", "35"),
            ("white_balance_automatic", "0"),
            ("white_balance_temperature", "4195"),
            ("auto_exposure", "1"),
            ("exposure_time_absolute", "329"),
            ("exposure_dynamic_framerate", "0"),
            ("gain", "66"),
            ("sharpness", "128"),
        ]
        for ctrl, val in controls:
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", "/dev/video0", f"--set-ctrl={ctrl}={val}"],
                    capture_output=True, timeout=2,
                )
                if ctrl in ("focus_automatic_continuous", "white_balance_automatic", "auto_exposure"):
                    time.sleep(0.3)
            except Exception:
                pass

    def read(self):
        if self._picam2 is not None:
            frame = self._picam2.capture_array()
            return True, frame
        else:
            return self._cap.read()

    def release(self):
        if self._picam2 is not None:
            self._picam2.stop()
        elif self._cap is not None:
            self._cap.release()
