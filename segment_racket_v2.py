"""
Tennis racket contour segmentation v2 — ellipse-fit approach.

Uses LAB color segmentation + morphological filtering + parametric ellipse
fitting to extract the racket frame centerline. Produces a smooth, evenly
spaced contour suitable for CNC/XYZ gantry path following.

Key differences from v1 (segment_racket.py):
  - LAB L-channel threshold instead of adaptive threshold (ignores strings)
  - Morphological open/close to remove thin strings and fill frame gaps
  - cv2.fitEllipse on the head region → smooth parametric curve
  - Handle contour stitched via smoothing spline
  - Fallback to periodic spline if ellipse fit is poor
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
from scipy.spatial.distance import cdist

from camera import Camera
from stream import StreamServer
from transforms import FX, FY, CX, CY, TAG_SIZE, CAMERA_MATRIX, pixels_to_world, world_to_pixels

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
               align-items: center; height: 100vh; font-family: monospace; }
        h1 { color: #0f0; margin: 4px 0; font-size: 1.1em; }
        .wrap { position: relative; display: inline-block; line-height: 0; flex: 1; min-height: 0; }
        .wrap img { display: block; max-width: 100vw; max-height: calc(100vh - 100px); }
        .wrap canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
        #info { color: #0ff; font-size: 13px; margin-top: 6px; }
        #coord-tooltip {
            position: fixed; pointer-events: none; z-index: 9999;
            font-family: monospace; font-size: 13px; padding: 4px 8px;
            background: rgba(0,0,0,0.8); border-radius: 4px;
            white-space: nowrap; display: none;
        }
        #controls { color: #ddd; font-size: 13px; margin-top: 6px;
                     display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
        #controls label { display: flex; align-items: center; gap: 4px; }
        #controls input[type=range] { width: 150px; accent-color: #f0f; vertical-align: middle; }
        #controls .val { color: #f0f; min-width: 36px; }
        #controls button { background: #a0a; color: #fff; border: none; padding: 5px 14px;
                           border-radius: 3px; cursor: pointer; font-family: monospace; font-size: 13px; }
        #controls button:hover { background: #c0c; }
        #gcode-status { font-size: 12px; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="wrap">
        <img id="stream" src="/feed">
        <canvas id="overlay"></canvas>
    </div>
    <div id="info">Loading...</div>
    <div id="controls">
        <label>Offset (mm):
            <input type="range" id="offset" min="-100" max="100" value="5" step="0.5">
            <span class="val" id="offset_val">5.0</span>
        </label>
        <button id="gen-gcode">Generate G-code</button>
        <span id="gcode-status"></span>
    </div>
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

// --- offset slider + gcode button ---
const offsetSlider = document.getElementById('offset');
const offsetVal = document.getElementById('offset_val');
const gcodeBtn = document.getElementById('gen-gcode');
const gcodeStatus = document.getElementById('gcode-status');

fetch('/offset').then(r=>r.json()).then(d=>{
    offsetSlider.value = d.offset_mm;
    offsetVal.textContent = d.offset_mm.toFixed(1);
});

offsetSlider.addEventListener('input', () => {
    offsetVal.textContent = parseFloat(offsetSlider.value).toFixed(1);
    fetch('/offset', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({offset_mm: parseFloat(offsetSlider.value)})
    });
});

gcodeBtn.addEventListener('click', () => {
    gcodeStatus.textContent = 'Generating...';
    gcodeStatus.style.color = '#ff0';
    fetch('/gcode')
    .then(r => {
        if (!r.ok) return r.json().then(d => { throw new Error(d.error); });
        return r.blob();
    })
    .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'racket_path.gcode'; a.click();
        URL.revokeObjectURL(url);
        gcodeStatus.textContent = 'Downloaded!';
        gcodeStatus.style.color = '#0f0';
        setTimeout(() => gcodeStatus.textContent = '', 3000);
    })
    .catch(e => {
        gcodeStatus.textContent = e.message;
        gcodeStatus.style.color = '#f44';
        setTimeout(() => gcodeStatus.textContent = '', 4000);
    });
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Segment tennis racket contour (v2 — ellipse fit)")
parser.add_argument("--headless", action="store_true", help="Stream via HTTP instead of cv2.imshow")
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--output", type=str, default="racket_contour", help="Output filename (without extension)")
parser.add_argument("--format", choices=["csv", "json"], default="csv")
parser.add_argument("--continuous", action="store_true", help="Live preview; press 's' to save or wait for Ctrl+C")
parser.add_argument("--smooth-points", type=int, default=200,
                    help="Number of evenly-spaced points on the final contour (0 = raw stitched points)")
parser.add_argument("--min-area", type=int, default=15000, help="Minimum contour area (px^2) to consider")
parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    default=[130, 100, 870, 620],
                    help="Region of interest (x y w h) in pixels")
parser.add_argument("--epsilon", type=float, default=2.0, help="(compat only, unused in v2)")

# v2-specific arguments
parser.add_argument("--l-thresh", type=int, default=72,
                    help="LAB L-channel threshold (pixels with L < this are 'dark'/frame)")
parser.add_argument("--h-low", type=int, default=0, help="HSV H lower bound")
parser.add_argument("--h-high", type=int, default=20, help="HSV H upper bound")
parser.add_argument("--s-low", type=int, default=0, help="HSV S lower bound")
parser.add_argument("--s-high", type=int, default=85, help="HSV S upper bound")
parser.add_argument("--v-low", type=int, default=50, help="HSV V lower bound")
parser.add_argument("--v-high", type=int, default=220, help="HSV V upper bound")
parser.add_argument("--invert-hsv", action="store_true", default=True,
                    help="Invert HSV mask (HSV range defines background to exclude)")
parser.add_argument("--no-invert-hsv", dest="invert_hsv", action="store_false",
                    help="Don't invert HSV mask")
parser.add_argument("--no-lab", action="store_true",
                    help="Skip LAB L-threshold, use only HSV mask (for non-black rackets)")
parser.add_argument("--open-kernel", type=int, default=9,
                    help="Morph opening kernel diameter (> string width, < frame width)")
parser.add_argument("--close-kernel", type=int, default=25,
                    help="Morph closing kernel diameter (fills frame gaps)")
parser.add_argument("--head-dist-thresh", type=float, default=20.0,
                    help="Max px distance from ellipse to classify as 'head'")
parser.add_argument("--min-head-ratio", type=float, default=0.4,
                    help="Min fraction of head points for valid ellipse fit")
parser.add_argument("--debug-vis", action="store_true",
                    help="Show intermediate masks and ellipse overlay")

# Offset / G-code arguments
parser.add_argument("--offset", type=float, default=5.0,
                    help="Ellipse offset in mm (positive = outward)")
parser.add_argument("--feed-rate", type=float, default=1000.0,
                    help="G-code feed rate in mm/min")
parser.add_argument("--gcode-output", type=str, default="racket_path.gcode",
                    help="G-code output filename")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Shared ROI state (thread-safe for Flask <-> main loop)
# ---------------------------------------------------------------------------
roi_lock = threading.Lock()
shared_roi = list(args.roi)  # [x, y, w, h]

pose_lock = threading.Lock()
shared_pose = (None, None)  # (R, t)

# Debug visualization state shared with draw_overlay
debug_info_lock = threading.Lock()
shared_debug_info = {}

# Offset ellipse state
offset_lock = threading.Lock()
shared_offset_mm = args.offset
shared_offset_world = None  # latest (N, 2) offset world points for G-code
shared_offset_heading = None  # latest (N,) heading angles in degrees for G-code

# ---------------------------------------------------------------------------
# AprilTag detector
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
    stream = StreamServer(port=args.port, title="Racket Segmentation v2", html=ROI_HTML)

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

    @stream.app.route("/offset", methods=["GET", "POST"])
    def offset_endpoint():
        global shared_offset_mm
        if request.method == "POST":
            data = request.get_json()
            with offset_lock:
                shared_offset_mm = float(data["offset_mm"])
            return jsonify(ok=True)
        with offset_lock:
            return jsonify(offset_mm=shared_offset_mm)

    @stream.app.route("/gcode", methods=["GET"])
    def gcode_endpoint():
        from flask import send_file
        import io
        with offset_lock:
            pts = shared_offset_world
            hdg = shared_offset_heading
        if pts is None:
            return jsonify(error="No offset path yet (need tag + contour)"), 400
        gcode = generate_gcode(pts, feed_rate=args.feed_rate, heading_deg=hdg)
        # Also save to disk
        with open(args.gcode_output, "w") as f:
            f.write(gcode)
        print(f"G-code saved to {args.gcode_output} ({len(pts)} points)")
        buf = io.BytesIO(gcode.encode())
        buf.seek(0)
        return send_file(buf, mimetype="text/plain",
                         as_attachment=True, download_name=args.gcode_output)

    stream.start()


# ---------------------------------------------------------------------------
# Ellipse helper functions
# ---------------------------------------------------------------------------
def sample_ellipse(cx, cy, w, h, angle_deg, n):
    """Sample n points from a parametric ellipse.

    Parameters match cv2.fitEllipse output: center (cx, cy), axes (w, h), angle.
    Returns (n, 2) float64 array.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return ellipse_point(t, cx, cy, w, h, angle_deg)


def ellipse_point(t, cx, cy, w, h, angle_deg):
    """Evaluate ellipse at parametric angle(s) t. Vectorized.

    Returns (len(t), 2) float64 if t is array, or (2,) if scalar.
    """
    t = np.atleast_1d(t)
    a, b = w / 2.0, h / 2.0
    cos_a = np.cos(np.radians(angle_deg))
    sin_a = np.sin(np.radians(angle_deg))
    x = a * np.cos(t)
    y = b * np.sin(t)
    pts = np.stack([
        cx + x * cos_a - y * sin_a,
        cy + x * sin_a + y * cos_a,
    ], axis=-1)
    return pts



# ---------------------------------------------------------------------------
# Offset path + G-code
# ---------------------------------------------------------------------------
def compute_offset_path(world_pts, offset_mm):
    """Offset a closed 2D path outward by offset_mm using point normals.

    Positive offset = outward (away from centroid), negative = inward.
    """
    center = world_pts.mean(axis=0)

    # Central-difference tangents (wrapping)
    tangents = np.roll(world_pts, -1, axis=0) - np.roll(world_pts, 1, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-6)
    tangents /= lengths

    # Normals: rotate tangent +90°
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # Flip normals that point toward center (we want outward)
    to_center = center - world_pts
    dots = np.sum(normals * to_center, axis=1)
    normals[dots > 0] *= -1

    # Camera faces outward for positive offset, inward for negative
    if offset_mm >= 0:
        cam_dir = normals   # away from center
    else:
        cam_dir = -normals  # toward center

    heading_rad = np.arctan2(cam_dir[:, 1], cam_dir[:, 0])
    heading_rad = np.unwrap(heading_rad)  # smooth out 360° jumps
    heading_deg = np.degrees(heading_rad)

    return world_pts + offset_mm * normals, heading_deg


def generate_gcode(offset_pts, feed_rate=1000.0, heading_deg=None):
    """Generate G-code for a closed XY path with optional A-axis heading."""
    def _rot(i):
        if heading_deg is None:
            return ""
        return f" Z{heading_deg[i]:.2f}"

    lines = [
        "; Racket offset path — generated by segment_racket_v2.py",
        f"; {len(offset_pts)} points",
        "G21 ; mm",
        "G90 ; absolute",
        f"G0 X{offset_pts[0, 0]:.2f} Y{offset_pts[0, 1]:.2f}{_rot(0)} ; rapid to start",
    ]
    for i, pt in enumerate(offset_pts[1:], start=1):
        lines.append(f"G1 X{pt[0]:.2f} Y{pt[1]:.2f}{_rot(i)} F{feed_rate:.0f}")
    lines.append(f"G1 X{offset_pts[0, 0]:.2f} Y{offset_pts[0, 1]:.2f}{_rot(0)} ; close loop")
    lines.append("M2")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Segmentation (v2)
# ---------------------------------------------------------------------------
def segment_racket(frame, min_area, roi=None, l_thresh=90,
                   open_kernel=9, close_kernel=15,
                   head_dist_thresh=20.0, min_head_ratio=0.4,
                   smooth_points=200, debug_vis=False,
                   hsv_range=None, invert_hsv=False, no_lab=False):
    """Segment racket head using LAB thresholding + ellipse fit.

    Returns the head ellipse as a sampled contour (N, 1, 2) int32, or None.
    Coordinates are in full-frame space.
    """
    # Crop to ROI
    ox, oy = 0, 0
    crop = frame
    if roi is not None:
        rx, ry, rw, rh = roi
        crop = frame[ry:ry + rh, rx:rx + rw]
        ox, oy = rx, ry

    # --- Step 1: Build mask ---
    if no_lab:
        # Pure HSV mask (no LAB L-threshold) — works for any color racket
        mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255
    else:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        mask = (L < l_thresh).astype(np.uint8) * 255

    # --- Step 1b: Optional HSV masking ---
    if hsv_range is not None:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
        if invert_hsv:
            hsv_mask = cv2.bitwise_not(hsv_mask)
        mask = mask & hsv_mask
    dark_mask = mask

    # --- Step 2: Morphological opening — remove strings ---
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    opened = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # --- Step 3: Morphological closing — fill frame gaps ---
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=1)

    # Store debug masks
    if debug_vis:
        with debug_info_lock:
            shared_debug_info["dark_mask"] = dark_mask
            shared_debug_info["opened"] = opened
            shared_debug_info["closed"] = closed

    # --- Step 4: Find contours, take largest ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"[v2] no contours found (dark_px={np.count_nonzero(dark_mask)}, "
              f"after_open={np.count_nonzero(opened)}, after_close={np.count_nonzero(closed)})")
        return None

    areas_sorted = sorted([(cv2.contourArea(c), c) for c in contours],
                          key=lambda x: x[0], reverse=True)
    top_areas = [int(a) for a, _ in areas_sorted[:5]]
    big = [(a, c) for a, c in areas_sorted if a >= min_area]
    if not big:
        print(f"[v2] no contour >= min_area={min_area}  top areas: {top_areas}")
        return None

    # Merge the largest contours that are likely two halves of the racket ring.
    # If the 2nd-largest is at least 40% the size of the largest, merge them.
    merged = big[0][1]
    if len(big) >= 2 and big[1][0] >= big[0][0] * 0.4:
        merged = np.vstack([big[0][1], big[1][1]])
        print(f"[v2] merged top 2 contours: {int(big[0][0])} + {int(big[1][0])}")

    # Need at least 5 points for fitEllipse
    if len(merged) < 5:
        return None

    # --- Step 5–7: Iterative ellipse fit ---
    # Initial fit on all contour points, then iteratively refit on head-only
    # points. Each pass the ellipse converges toward the true head shape.
    contour_pts = merged.reshape(-1, 2).astype(np.float64)
    ellipse = cv2.fitEllipse(merged)
    MAX_ITERS = 5

    for iteration in range(MAX_ITERS):
        (ecx, ecy), (ew, eh), eangle = ellipse
        ellipse_pts = sample_ellipse(ecx, ecy, ew, eh, eangle, 720)
        dists = cdist(contour_pts, ellipse_pts).min(axis=1)

        head_mask = dists < head_dist_thresh
        head_points = contour_pts[head_mask]
        head_ratio = head_mask.sum() / len(head_mask)

        if len(head_points) < 5:
            break

        new_ellipse = cv2.fitEllipse(head_points.reshape(-1, 1, 2).astype(np.int32))

        # Check convergence — if center moved less than 1px, stop
        (ncx, ncy), _, _ = new_ellipse
        if abs(ncx - ecx) < 1.0 and abs(ncy - ecy) < 1.0:
            ellipse = new_ellipse
            break

        ellipse = new_ellipse

    (hcx, hcy), (hw, hh), hangle = ellipse

    if debug_vis:
        with debug_info_lock:
            shared_debug_info["ellipse"] = ellipse
            shared_debug_info["head_ratio"] = head_ratio
            shared_debug_info["fallback"] = head_ratio < min_head_ratio
            shared_debug_info["iterations"] = iteration + 1

    # --- Step 8: Sample the full head ellipse as output contour ---
    n_pts = smooth_points if smooth_points > 0 else 360
    sampled = sample_ellipse(hcx, hcy, hw, hh, hangle, n_pts)

    # Offset to full-frame coordinates
    sampled[:, 0] += ox
    sampled[:, 1] += oy

    result = sampled.round().astype(np.int32).reshape(-1, 1, 2)
    return result


# ---------------------------------------------------------------------------
# Tag detection
# ---------------------------------------------------------------------------
def detect_tag(gray):
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


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def draw_overlay(frame, contour, R, t, world_pts, roi=None, debug_vis=False,
                 offset_pixels=None, offset_mm=0.0, heading_deg=None,
                 offset_world=None):
    """Draw contour, tag axes, offset path, heading arrows, and status text."""
    # ROI box in yellow
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 200, 200), 1)

    # Contour in cyan
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)

    # Offset contour in magenta
    if offset_pixels is not None:
        pts = offset_pixels.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(frame, [pts], -1, (255, 0, 255), 2)
        cv2.putText(frame, f"offset: {offset_mm:+.1f} mm",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Heading arrows on offset path
    if (heading_deg is not None and offset_world is not None
            and R is not None and t is not None):
        arrow_len_mm = 8.0
        step = max(1, len(heading_deg) // 20)  # ~20 arrows around the loop
        for i in range(0, len(heading_deg), step):
            rad = np.radians(heading_deg[i])
            tip_world = offset_world[i] + arrow_len_mm * np.array([np.cos(rad), np.sin(rad)])
            base_px = world_to_pixels(offset_world[i:i+1], R, t)[0].astype(int)
            tip_px = world_to_pixels(tip_world.reshape(1, 2), R, t)[0].astype(int)
            cv2.arrowedLine(frame, tuple(base_px), tuple(tip_px),
                            (0, 255, 255), 2, tipLength=0.3)

    # Debug: ellipse overlay + mask thumbnail
    if debug_vis:
        with debug_info_lock:
            di = dict(shared_debug_info)

        if di.get("ellipse") is not None:
            (ecx, ecy), (ew, eh), eangle = di["ellipse"]
            dox, doy = 0, 0
            if roi is not None:
                dox, doy = roi[0], roi[1]
            cv2.ellipse(frame,
                        (int(ecx + dox), int(ecy + doy)),
                        (int(ew / 2), int(eh / 2)),
                        eangle, 0, 360, (0, 255, 0), 1)

        if di.get("head_ratio") is not None:
            iters = di.get("iterations", "?")
            cv2.putText(frame, f"head: {di['head_ratio']:.0%}  iters: {iters}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Inset mask thumbnail in bottom-left corner
        mask = di.get("closed", di.get("dark_mask"))
        if mask is not None:
            th = 200
            tw = int(mask.shape[1] * th / mask.shape[0])
            thumb = cv2.resize(mask, (tw, th))
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            fh, fw = frame.shape[:2]
            frame[fh - th:fh, 0:tw] = thumb_bgr

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
def save_output(world_pts, fmt, filename):
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
print(f"Racket segmentation v2 | l_thresh={args.l_thresh} | open_kernel={args.open_kernel} | "
      f"close_kernel={args.close_kernel} | smooth_points={args.smooth_points} | min_area={args.min_area}")
print(f"Head params: dist_thresh={args.head_dist_thresh} | min_ratio={args.min_head_ratio}")
print(f"Output: {args.output}.{args.format}")
if args.continuous:
    save_key = "'s' to save" if not args.headless else "Ctrl+C to save last"
    print(f"Continuous mode — {save_key}, 'q' to quit")
else:
    print("One-shot mode — capturing first valid frame")
print("-" * 60)

last_world_pts = None

# Local mode: offset trackbar (0–40 maps to -20..+20 mm)
if not args.headless:
    cv2.namedWindow("Racket Segmentation v2")
    _offset_center = 100

    def _offset_cb(val):
        global shared_offset_mm
        with offset_lock:
            shared_offset_mm = float(val - _offset_center)

    cv2.createTrackbar("Offset mm", "Racket Segmentation v2",
                       int(args.offset) + _offset_center, 200, _offset_cb)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # Read current ROI (may be updated from web UI)
        with roi_lock:
            roi = tuple(shared_roi)

        hsv_range = (
            np.array([args.h_low, args.s_low, args.v_low]),
            np.array([args.h_high, args.s_high, args.v_high]),
        )

        contour = segment_racket(
            frame, args.min_area, roi=roi,
            l_thresh=args.l_thresh,
            open_kernel=args.open_kernel,
            close_kernel=args.close_kernel,
            head_dist_thresh=args.head_dist_thresh,
            min_head_ratio=args.min_head_ratio,
            smooth_points=args.smooth_points,
            debug_vis=args.debug_vis,
            hsv_range=hsv_range,
            invert_hsv=args.invert_hsv,
            no_lab=args.no_lab,
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        R, t = detect_tag(gray)

        with pose_lock:
            shared_pose = (R, t)

        world_pts = None
        offset_pixel_pts = None
        cur_offset = 0.0
        heading_deg = None
        off_world = None
        if contour is not None and R is not None:
            pixels = contour.reshape(-1, 2).astype(np.float64)
            world_pts = pixels_to_world(pixels, R, t)
            last_world_pts = world_pts

            # Compute offset path
            with offset_lock:
                cur_offset = shared_offset_mm
            heading_deg = None
            off_world = None
            if abs(cur_offset) > 0.01:
                off_world, heading_deg = compute_offset_path(world_pts, cur_offset)
                with offset_lock:
                    shared_offset_world = off_world
                    shared_offset_heading = heading_deg
                offset_pixel_pts = world_to_pixels(off_world, R, t)
            else:
                with offset_lock:
                    shared_offset_world = world_pts
                    shared_offset_heading = None

        draw_overlay(frame, contour, R, t, world_pts, roi=roi,
                     debug_vis=args.debug_vis, offset_pixels=offset_pixel_pts,
                     offset_mm=cur_offset, heading_deg=heading_deg,
                     offset_world=off_world)

        # Debug: show intermediate masks in separate windows
        if args.debug_vis and not args.headless:
            with debug_info_lock:
                di = dict(shared_debug_info)
            if "dark_mask" in di:
                cv2.imshow("LAB Dark Mask", di["dark_mask"])
            if "opened" in di:
                cv2.imshow("After Opening", di["opened"])
            if "closed" in di:
                cv2.imshow("After Closing", di["closed"])

        if args.headless:
            stream.update_frame(frame)
        else:
            cv2.imshow("Racket Segmentation v2", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and last_world_pts is not None:
                save_output(last_world_pts, args.format, args.output)
            if key == ord('g'):
                with offset_lock:
                    gpts = shared_offset_world
                    ghdg = shared_offset_heading
                if gpts is not None:
                    gcode = generate_gcode(gpts, feed_rate=args.feed_rate, heading_deg=ghdg)
                    with open(args.gcode_output, "w") as f:
                        f.write(gcode)
                    print(f"G-code saved to {args.gcode_output} ({len(gpts)} points)")
                else:
                    print("No offset path yet (need tag + contour)")

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
