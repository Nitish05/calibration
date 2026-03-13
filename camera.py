"""
Camera abstraction that works on both desktop (OpenCV) and Raspberry Pi (picamera2).
Set use_picamera=True to use picamera2 (for CSI cameras), otherwise uses OpenCV (for USB cameras).
"""

import re

import cv2


class Camera:
    @staticmethod
    def find_usb_camera():
        """Find the first USB camera (e.g. C920) by scanning v4l2 devices."""
        import os
        import subprocess
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True, timeout=5,
            )
            lines = result.stdout.splitlines()
            for i, line in enumerate(lines):
                if "usb" in line.lower():
                    # Next indented lines are device paths
                    for j in range(i + 1, len(lines)):
                        dev = lines[j].strip()
                        if not dev or not dev.startswith("/dev/video"):
                            break
                        # Return the first video device for this USB camera
                        return dev
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def __init__(self, width=1920, height=1080, device=None, use_picamera=False):
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

        if device is None:
            device = self.find_usb_camera() or 0
        # OpenCV needs an integer index (not path string) for FOURCC/resolution
        # changes to work via the V4L2 backend
        if isinstance(device, str):
            m = re.search(r"video(\d+)", device)
            device = int(m.group(1)) if m else device
        self._cap = cv2.VideoCapture(device)
        # Set MJPEG format first — C920 only does 1080p via MJPEG, not YUYV
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._lock_controls()
        print(f"Camera: OpenCV v4l2 device {device} ({actual_w}x{actual_h})")

    def _lock_controls(self):
        """Lock C920 controls for consistent imaging via OpenCV API."""
        if self._cap is None:
            return
        # Read a few frames to let the camera initialize
        for _ in range(5):
            self._cap.read()
        # Disable auto modes first
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=manual, 3=auto
        # Then set manual values
        self._cap.set(cv2.CAP_PROP_FOCUS, 35)
        self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4195)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, 312)
        self._cap.set(cv2.CAP_PROP_GAIN, 66)
        self._cap.set(cv2.CAP_PROP_SHARPNESS, 128)
        self._cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        self._cap.set(cv2.CAP_PROP_CONTRAST, 128)
        self._cap.set(cv2.CAP_PROP_SATURATION, 128)

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
