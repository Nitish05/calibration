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
import threading

import cv2
import numpy as np
from flask import request, jsonify
from pupil_apriltags import Detector

from camera import Camera
from stream import StreamServer
from transforms import FX, FY, CX, CY, TAG_SIZE, CAMERA_MATRIX, pixels_to_world

# ---------------------------------------------------------------------------
# Interactive HTML with draggable ROI overlay
# ---------------------------------------------------------------------------
ROI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #111; display: flex; flex-direction: column;
               align-items: center; height: 100vh; font-family: monospace; overflow: hidden; }
        h1 { color: #0f0; margin: 8px 0; font-size: 1.2em; }
        .wrap { position: relative; display: inline-block; line-height: 0; }
        .wrap img { display: block; max-width: 100vw; max-height: 88vh; }
        .wrap canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
        #info { color: #0ff; font-size: 13px; margin-top: 6px; }
        #coord-tooltip {
            position: fixed; pointer-events: none; z-index: 9999;
            font-family: monospace; font-size: 13px; padding: 4px 8px;
            background: rgba(0,0,0,0.8); border-radius: 4px;
            white-space: nowrap; display: none;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="wrap">
        <img id="stream" src="/feed">
        <canvas id="overlay"></canvas>
    </div>
    <div id="info">Loading...</div>
    <div id="coord-tooltip"></div>
<script>
const img = document.getElementById('stream');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const tooltip = document.getElementById('coord-tooltip');

const CAM_W = 1920, CAM_H = 1080;
const HANDLE = 10;

let roi = {x:0, y:0, w:100, h:100};
let drag = null;       // null | 'move' | 'nw'|'ne'|'sw'|'se'|'n'|'s'|'e'|'w'
let dragStart = {};
let roiStart = {};

// --- coordinate helpers ---
function s() { return img.clientWidth / CAM_W; }
function toCanvas(px, py) { const sc=s(); return [px*sc, py*sc]; }
function toCam(cx, cy) { const sc=s(); return [cx/sc, cy/sc]; }

// --- fetch initial ROI ---
fetch('/roi').then(r=>r.json()).then(d=>{ roi=d; draw(); });

// --- drawing ---
function draw() {
    const sc = s();
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const rx=roi.x*sc, ry=roi.y*sc, rw=roi.w*sc, rh=roi.h*sc;

    // dim outside ROI
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.clearRect(rx, ry, rw, rh);

    // ROI border
    ctx.strokeStyle = '#0ff';
    ctx.lineWidth = 2;
    ctx.setLineDash([6,4]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);

    // handles
    ctx.fillStyle = '#0ff';
    const hs = HANDLE;
    const corners = [
        [rx-hs/2, ry-hs/2],
        [rx+rw-hs/2, ry-hs/2],
        [rx-hs/2, ry+rh-hs/2],
        [rx+rw-hs/2, ry+rh-hs/2],
    ];
    const mids = [
        [rx+rw/2-hs/2, ry-hs/2],
        [rx+rw/2-hs/2, ry+rh-hs/2],
        [rx-hs/2, ry+rh/2-hs/2],
        [rx+rw-hs/2, ry+rh/2-hs/2],
    ];
    [...corners, ...mids].forEach(([hx,hy]) => ctx.fillRect(hx, hy, hs, hs));

    info.textContent = `ROI: x=${Math.round(roi.x)} y=${Math.round(roi.y)} w=${Math.round(roi.w)} h=${Math.round(roi.h)}  —  drag box to move, drag handles to resize`;
}

// --- hit test ---
function hitTest(mx, my) {
    const sc=s(), hs=HANDLE;
    const rx=roi.x*sc, ry=roi.y*sc, rw=roi.w*sc, rh=roi.h*sc;
    const near = (a,b) => Math.abs(a-b) < hs;

    // corners
    if (near(mx,rx) && near(my,ry)) return 'nw';
    if (near(mx,rx+rw) && near(my,ry)) return 'ne';
    if (near(mx,rx) && near(my,ry+rh)) return 'sw';
    if (near(mx,rx+rw) && near(my,ry+rh)) return 'se';
    // edges
    if (near(my,ry) && mx>rx && mx<rx+rw) return 'n';
    if (near(my,ry+rh) && mx>rx && mx<rx+rw) return 's';
    if (near(mx,rx) && my>ry && my<ry+rh) return 'w';
    if (near(mx,rx+rw) && my>ry && my<ry+rh) return 'e';
    // inside
    if (mx>rx && mx<rx+rw && my>ry && my<ry+rh) return 'move';
    return null;
}

const cursors = {nw:'nw-resize',ne:'ne-resize',sw:'sw-resize',se:'se-resize',
                 n:'n-resize',s:'s-resize',w:'w-resize',e:'e-resize',move:'grab'};

// --- world coord tooltip ---
let wcAborter = null;
let wcLastTime = 0;
const WC_INTERVAL = 100; // ms → max 10 req/sec

function fetchWorldCoord(camU, camV, clientX, clientY) {
    const now = Date.now();
    if (now - wcLastTime < WC_INTERVAL) return;
    wcLastTime = now;
    if (wcAborter) wcAborter.abort();
    wcAborter = new AbortController();
    fetch('/world_coord', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({u: Math.round(camU), v: Math.round(camV)}),
        signal: wcAborter.signal
    })
    .then(r => r.json())
    .then(d => {
        tooltip.style.display = 'block';
        tooltip.style.left = (clientX + 14) + 'px';
        tooltip.style.top  = (clientY + 14) + 'px';
        if (d.error) {
            tooltip.style.color = '#f44';
            tooltip.textContent = d.error;
        } else {
            tooltip.style.color = '#0f0';
            tooltip.textContent = 'X: ' + d.x_mm.toFixed(1) + ' mm  Y: ' + d.y_mm.toFixed(1) + ' mm';
        }
    })
    .catch(() => {});
}

// --- mouse events ---
canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (!drag) {
        const h = hitTest(mx, my);
        canvas.style.cursor = h ? (cursors[h]||'default') : 'default';
        const [camU, camV] = toCam(mx, my);
        fetchWorldCoord(camU, camV, e.clientX, e.clientY);
        return;
    }
    const dx = mx - dragStart.x, dy = my - dragStart.y;
    const sc = s();
    const dxC = dx/sc, dyC = dy/sc;

    let {x,y,w,h} = roiStart;
    if (drag==='move') { x+=dxC; y+=dyC; }
    else {
        if (drag.includes('w')) { x+=dxC; w-=dxC; }
        if (drag.includes('e')) { w+=dxC; }
        if (drag.includes('n')) { y+=dyC; h-=dyC; }
        if (drag.includes('s')) { h+=dyC; }
    }
    // clamp
    w = Math.max(50, w); h = Math.max(50, h);
    x = Math.max(0, Math.min(x, CAM_W-w));
    y = Math.max(0, Math.min(y, CAM_H-h));
    roi = {x,y,w,h};
    draw();
});

canvas.addEventListener('mousedown', e => {
    tooltip.style.display = 'none';
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const h = hitTest(mx, my);
    if (!h) return;
    drag = h;
    dragStart = {x: mx, y: my};
    roiStart = {...roi};
    if (h==='move') canvas.style.cursor = 'grabbing';
});

canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

window.addEventListener('mouseup', () => {
    if (drag) {
        drag = null;
        canvas.style.cursor = 'default';
        // send to server
        fetch('/roi', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({x:Math.round(roi.x), y:Math.round(roi.y),
                                  w:Math.round(roi.w), h:Math.round(roi.h)})
        });
    }
});

// redraw on image load / resize
img.addEventListener('load', draw);
window.addEventListener('resize', draw);
</script>
</body>
</html>
"""

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
parser.add_argument("--hsv-low", type=int, nargs=3, metavar=("H", "S", "V"),
                    default=[10, 30, 50], help="Lower HSV bound for board color")
parser.add_argument("--hsv-high", type=int, nargs=3, metavar=("H", "S", "V"),
                    default=[25, 200, 220], help="Upper HSV bound for board color")
parser.add_argument("--dark-thresh", type=int, default=40,
                    help="Exclude pixels with V < this value (hole shadows)")
parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    default=[130, 100, 870, 620],
                    help="Region of interest (x y w h) in pixels — only segment inside this box")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Shared ROI state (thread-safe for Flask <-> main loop)
# ---------------------------------------------------------------------------
roi_lock = threading.Lock()
shared_roi = list(args.roi)  # [x, y, w, h]

pose_lock = threading.Lock()
shared_pose = (None, None)  # (R, t)

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
    stream = StreamServer(port=args.port, title="Racket Segmentation", html=ROI_HTML)

    @stream.app.route("/roi", methods=["GET", "POST"])
    def roi_endpoint():
        global shared_roi
        if request.method == "POST":
            data = request.get_json()
            with roi_lock:
                shared_roi = [int(data["x"]), int(data["y"]),
                              int(data["w"]), int(data["h"])]
            print(f"ROI updated: {shared_roi}")
            return jsonify(ok=True)
        with roi_lock:
            r = list(shared_roi)
        return jsonify(x=r[0], y=r[1], w=r[2], h=r[3])

    @stream.app.route("/world_coord", methods=["POST"])
    def world_coord_endpoint():
        data = request.get_json()
        u, v = float(data["u"]), float(data["v"])
        with pose_lock:
            R, t = shared_pose
        if R is None:
            return jsonify(error="NO TAG")
        pt = pixels_to_world(np.array([[u, v]]), R, t)
        return jsonify(x_mm=round(float(pt[0, 0]), 1),
                       y_mm=round(float(pt[0, 1]), 1),
                       z_mm=0.0)

    stream.start()


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------
def segment_racket(frame: np.ndarray, min_area: int, epsilon: float,
                   roi=None, hsv_low=(10, 30, 50), hsv_high=(25, 200, 220),
                   dark_thresh=40):
    """Return the simplified outer contour of the racket, or None.

    roi: (x, y, w, h) — if provided, only search inside this region.
         Returned contour coordinates are in full-frame space.
    hsv_low / hsv_high: HSV bounds for the board color (excluded).
    dark_thresh: pixels with V < this are excluded (hole shadows).
    """
    # Crop to ROI first (operate on color frame)
    ox, oy = 0, 0
    crop = frame
    if roi is not None:
        rx, ry, rw, rh = roi
        crop = frame[ry:ry + rh, rx:rx + rw]
        ox, oy = rx, ry

    # HSV masking — identify board color and dark shadows
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    board_mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    dark_mask = cv2.inRange(hsv[:, :, 2], 0, dark_thresh)
    background = board_mask | dark_mask
    foreground = cv2.bitwise_not(background)

    # Grayscale adaptive threshold
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=10,
    )

    # Combine: only keep threshold pixels that are NOT board/shadow
    combined = thresh & foreground

    # Morphological close (fill gaps) then open (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
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
print(f"Racket segmentation | epsilon={args.epsilon} | min_area={args.min_area} | roi={list(shared_roi)}")
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

        # Read current ROI (may be updated from web UI)
        with roi_lock:
            roi = tuple(shared_roi)

        contour = segment_racket(frame, args.min_area, args.epsilon, roi=roi,
                                 hsv_low=tuple(args.hsv_low),
                                 hsv_high=tuple(args.hsv_high),
                                 dark_thresh=args.dark_thresh)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        R, t = detect_tag(gray)

        with pose_lock:
            shared_pose = (R, t)

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
