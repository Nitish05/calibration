"""
Camera abstraction that works on both desktop (OpenCV) and Raspberry Pi (picamera2).
"""

import cv2
import numpy as np


class Camera:
    def __init__(self, width=640, height=480, device=1):
        self.width = width
        self.height = height
        self._picam2 = None
        self._cap = None

        try:
            from picamera2 import Picamera2
            self._picam2 = Picamera2()
            config = self._picam2.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"},
            )
            self._picam2.configure(config)
            self._picam2.start()
            print(f"Camera: picamera2 ({width}x{height})")
            self._is_picam = True
        except (ImportError, RuntimeError):
            self._cap = cv2.VideoCapture(device)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Camera: OpenCV v4l2 device {device} ({width}x{height})")

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
