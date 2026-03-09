"""
Shared MJPEG streaming server.
Usage:
    from stream import StreamServer
    server = StreamServer(port=8080)
    server.start()
    # In your loop:
    server.update_frame(frame)
    # When done:
    server.stop()
"""

import threading
import time
import cv2
from flask import Flask, Response, render_template_string

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { margin: 0; background: #111; display: flex; flex-direction: column;
               align-items: center; justify-content: center; height: 100vh; font-family: monospace; }
        h1 { color: #0f0; margin-bottom: 10px; }
        img { max-width: 100%; max-height: 90vh; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <img src="/feed">
</body>
</html>
"""


class StreamServer:
    def __init__(self, port=8080, title="Camera Stream", stream_width=640):
        self.port = port
        self.title = title
        self.frame = None
        self.stream_width = stream_width
        self.lock = threading.Lock()
        self.app = Flask(__name__)
        self.thread = None

        @self.app.route("/")
        def index():
            return render_template_string(HTML_PAGE, title=self.title)

        @self.app.route("/feed")
        def feed():
            return Response(self._generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _generate(self):
        while True:
            with self.lock:
                if self.frame is None:
                    time.sleep(0.03)
                    continue
                _, jpeg = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = jpeg.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            time.sleep(0.033)  # ~30 fps cap

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def start(self):
        self.thread = threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=self.port, threaded=True),
            daemon=True,
        )
        self.thread.start()
        print(f"Stream running at http://0.0.0.0:{self.port}")

    def stop(self):
        pass  # daemon thread exits with main process
