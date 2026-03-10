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
