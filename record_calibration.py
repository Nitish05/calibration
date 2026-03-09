"""
Record calibration video with live stream preview.
Usage:
    python record_calibration.py --headless --duration 60
    Then open http://<rpi-ip>:8080 in your browser.
    Press Ctrl+C to stop early.
"""

import cv2
import argparse
import time
from stream import StreamServer

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds")
parser.add_argument("--output", type=str, default="intrinsics/cam_00/intrinsics.mp4")
args = parser.parse_args()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.output, fourcc, 30.0, (1280, 720))

stream = None
if args.headless:
    stream = StreamServer(port=args.port, title="Calibration Recording")
    stream.start()

print(f"Recording to {args.output} for {args.duration}s")
if not args.headless:
    print("Press 'q' to stop early")
else:
    print("Press Ctrl+C to stop early")
print("-" * 60)

start = time.time()
frames = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start
        remaining = args.duration - elapsed
        if remaining <= 0:
            break

        out.write(frame)
        frames += 1

        # Overlay recording info
        display = frame.copy()
        cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)  # red dot
        cv2.putText(display, f"REC {elapsed:.1f}s / {args.duration}s",
                    (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, "Move checkerboard slowly to all corners & angles",
                    (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if args.headless:
            stream.update_frame(display)
        else:
            cv2.imshow("Calibration Recording", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopped early.")

out.release()
cap.release()
if not args.headless:
    cv2.destroyAllWindows()

print(f"Saved {frames} frames to {args.output}")
