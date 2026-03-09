"""
Tennis racket contour segmentation + world coordinate extraction.

Segments the racket's outer frame contour from the camera feed,
converts pixel coordinates to real-world mm (relative to AprilTag),
and outputs them as CSV or JSON for an XYZ gantry to follow.
"""

import argparse
import csv
import json
import sys

import cv2
import numpy as np
from pupil_apriltags import Detector

from camera import Camera
from stream import StreamServer
from transforms import FX, FY, CX, CY, TAG_SIZE, CAMERA_MATRIX, pixels_to_world

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Segment tennis racket contour")
parser.add_argument("--headless", action="store_true", help="Stream via HTTP instead of cv2.imshow")
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--output", type=str, default="racket_contour", help="Output filename (without extension)")
parser.add_argument("--format", choices=["csv", "json"], default="csv")
parser.add_argument("--continuous", action="store_true", help="Live preview; press 's' to save or wait for Ctrl+C")
parser.add_argument("--epsilon", type=float, default=2.0, help="approxPolyDP epsilon (px)")
parser.add_argument("--min-area", type=int, default=50000, help="Minimum contour area (px^2) to consider")
parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    default=[130, 100, 870, 620],
                    help="Region of interest (x y w h) in pixels — only segment inside this box")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# AprilTag detector (same config as detect_apriltag.py)
# ---------------------------------------------------------------------------
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=2.0,
    quad_sigma=0.0,
    decode_sharpening=0.25,
    refine_edges=True,
)

# ---------------------------------------------------------------------------
# Camera + stream
# ---------------------------------------------------------------------------
cap = Camera()
stream = None
if args.headless:
    stream = StreamServer(port=args.port, title="Racket Segmentation")
    stream.start()


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------
def segment_racket(frame: np.ndarray, min_area: int, epsilon: float, roi=None):
    """Return the simplified outer contour of the racket, or None.

    roi: (x, y, w, h) — if provided, only search inside this region.
         Returned contour coordinates are in full-frame space.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Crop to ROI for segmentation
    ox, oy = 0, 0
    if roi is not None:
        rx, ry, rw, rh = roi
        gray = gray[ry:ry + rh, rx:rx + rw]
        ox, oy = rx, ry

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold — dark racket on lighter board becomes white foreground
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=10,
    )

    # Morphological close (fill gaps) then open (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area, pick largest
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not big:
        return None

    largest = max(big, key=cv2.contourArea)

    # Simplify for smooth gantry-friendly output
    simplified = cv2.approxPolyDP(largest, epsilon, closed=True)

    # Offset contour back to full-frame coordinates
    if roi is not None:
        simplified[:, :, 0] += ox
        simplified[:, :, 1] += oy

    return simplified


def detect_tag(gray: np.ndarray):
    """Detect first AprilTag and return (R, t) or (None, None)."""
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[FX, FY, CX, CY],
        tag_size=TAG_SIZE,
    )
    if not detections:
        return None, None
    det = detections[0]
    return det.pose_R, det.pose_t


def draw_overlay(frame, contour, R, t, world_pts, roi=None):
    """Draw contour, tag axes, and status text on the frame."""
    # ROI box in yellow dashed (draw as thin rectangle)
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 200, 200), 1)

    # Contour in cyan
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)

    # Tag axes
    if R is not None and t is not None:
        axis_len = TAG_SIZE * 0.5
        axis_pts = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]])
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, CAMERA_MATRIX, None)
        img_pts = img_pts.astype(int).reshape(-1, 2)
        origin = tuple(img_pts[0])
        cv2.line(frame, origin, tuple(img_pts[1]), (0, 0, 255), 2)   # X red
        cv2.line(frame, origin, tuple(img_pts[2]), (0, 255, 0), 2)   # Y green
        cv2.line(frame, origin, tuple(img_pts[3]), (255, 0, 0), 2)   # Z blue

    # Status text
    tag_status = "TAG OK" if R is not None else "NO TAG"
    contour_status = f"{len(contour)} pts" if contour is not None else "no contour"
    cv2.putText(frame, f"{tag_status} | contour: {contour_status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if world_pts is not None:
        x_span = world_pts[:, 0].max() - world_pts[:, 0].min()
        y_span = world_pts[:, 1].max() - world_pts[:, 1].min()
        cv2.putText(frame, f"span: {x_span:.0f} x {y_span:.0f} mm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_output(world_pts: np.ndarray, fmt: str, filename: str):
    """Save world points to CSV or JSON."""
    ext = f".{fmt}"
    path = filename if filename.endswith(ext) else filename + ext

    if fmt == "csv":
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_mm", "y_mm"])
            for x, y in world_pts:
                writer.writerow([f"{x:.1f}", f"{y:.1f}"])
    else:
        data = [{"x_mm": round(float(x), 1), "y_mm": round(float(y), 1)} for x, y in world_pts]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Saved {len(world_pts)} points to {path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
roi = tuple(args.roi)
print(f"Racket segmentation | epsilon={args.epsilon} | min_area={args.min_area} | roi={roi}")
print(f"Output: {args.output}.{args.format}")
if args.continuous:
    save_key = "'s' to save" if not args.headless else "Ctrl+C to save last"
    print(f"Continuous mode — {save_key}, 'q' to quit")
else:
    print("One-shot mode — capturing first valid frame")
print("-" * 60)

last_world_pts = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contour = segment_racket(frame, args.min_area, args.epsilon, roi=roi)
        R, t = detect_tag(gray)

        world_pts = None
        if contour is not None and R is not None:
            # Extract (u, v) from OpenCV contour shape (N, 1, 2)
            pixels = contour.reshape(-1, 2).astype(np.float64)
            world_pts = pixels_to_world(pixels, R, t)
            last_world_pts = world_pts

        draw_overlay(frame, contour, R, t, world_pts, roi=roi)

        if args.headless:
            stream.update_frame(frame)
        else:
            cv2.imshow("Racket Segmentation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and last_world_pts is not None:
                save_output(last_world_pts, args.format, args.output)

        # One-shot mode: save and exit as soon as we get a valid result
        if not args.continuous and world_pts is not None:
            save_output(world_pts, args.format, args.output)
            break

except KeyboardInterrupt:
    print("\nStopping...")
    if last_world_pts is not None:
        save_output(last_world_pts, args.format, args.output)

cap.release()
if not args.headless:
    cv2.destroyAllWindows()
