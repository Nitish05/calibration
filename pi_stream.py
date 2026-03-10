"""Stream the Pi camera as MJPEG over HTTP using picamera2 + flask."""

from picamera2 import Picamera2
from flask import Flask, Response
import io
import time

app = Flask(__name__)

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1640, 1232), "format": "RGB888"},
    controls={"FrameRate": 84.0},
)
picam2.configure(config)
picam2.start()
time.sleep(1)


def generate_frames():
    while True:
        buf = io.BytesIO()
        picam2.capture_file(buf, format="jpeg")
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n")


@app.route("/")
def index():
    return """<html><body style="margin:0;background:#000">
    <img src="/stream" style="width:100vw;height:100vh;object-fit:contain">
    </body></html>"""


@app.route("/stream")
def stream():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture")
def capture():
    buf = io.BytesIO()
    picam2.capture_file(buf, format="jpeg")
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
