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
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist

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
parser = argparse.ArgumentParser(description="Segment tennis racket contour (v2 — ellipse fit)")
parser.add_argument("--headless", action="store_true", help="Stream via HTTP instead of cv2.imshow")
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--output", type=str, default="racket_contour", help="Output filename (without extension)")
parser.add_argument("--format", choices=["csv", "json"], default="csv")
parser.add_argument("--continuous", action="store_true", help="Live preview; press 's' to save or wait for Ctrl+C")
parser.add_argument("--smooth-points", type=int, default=200,
                    help="Number of evenly-spaced points on the final contour (0 = raw stitched points)")
parser.add_argument("--min-area", type=int, default=50000, help="Minimum contour area (px^2) to consider")
parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    default=[130, 100, 870, 620],
                    help="Region of interest (x y w h) in pixels")
parser.add_argument("--epsilon", type=float, default=2.0, help="(compat only, unused in v2)")

# v2-specific arguments
parser.add_argument("--l-thresh", type=int, default=90,
                    help="LAB L-channel threshold (pixels with L < this are 'dark'/frame)")
parser.add_argument("--open-kernel", type=int, default=9,
                    help="Morph opening kernel diameter (> string width, < frame width)")
parser.add_argument("--close-kernel", type=int, default=15,
                    help="Morph closing kernel diameter (fills frame gaps)")
parser.add_argument("--head-dist-thresh", type=float, default=20.0,
                    help="Max px distance from ellipse to classify as 'head'")
parser.add_argument("--min-head-ratio", type=float, default=0.4,
                    help="Min fraction of head points for valid ellipse fit")
parser.add_argument("--debug-vis", action="store_true",
                    help="Show intermediate masks and ellipse overlay")
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


def point_to_ellipse_angle(px, py, cx, cy, w, h, angle_deg):
    """Map a point to its nearest parametric angle on the ellipse.

    Transforms the point into the ellipse's local frame, then uses atan2.
    Returns angle in [0, 2*pi).
    """
    cos_a = np.cos(np.radians(angle_deg))
    sin_a = np.sin(np.radians(angle_deg))
    # Rotate into ellipse-local frame
    dx = px - cx
    dy = py - cy
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    # Normalize by semi-axes to get angle
    a, b = w / 2.0, h / 2.0
    angle = np.arctan2(local_y / b, local_x / a)
    if angle < 0:
        angle += 2 * np.pi
    return angle


def smooth_contour_fallback(contour, num_points):
    """Fit a periodic cubic spline through contour points and resample evenly.

    Same as v1 smooth_contour — used as fallback when ellipse fit fails.
    """
    x = contour[:, 0, 0].astype(np.float64)
    y = contour[:, 0, 1].astype(np.float64)
    tck, _ = splprep([x, y], s=0, per=True, k=3)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    result = np.stack([x_new, y_new], axis=-1).round().astype(np.int32)
    return result.reshape(-1, 1, 2)


# ---------------------------------------------------------------------------
# Segmentation (v2)
# ---------------------------------------------------------------------------
def segment_racket(frame, min_area, roi=None, l_thresh=90,
                   open_kernel=9, close_kernel=15,
                   head_dist_thresh=20.0, min_head_ratio=0.4,
                   smooth_points=200, debug_vis=False):
    """Segment racket using LAB thresholding + ellipse fit.

    Returns the contour as (N, 1, 2) int32, or None.
    Coordinates are in full-frame space.
    """
    # Crop to ROI
    ox, oy = 0, 0
    crop = frame
    if roi is not None:
        rx, ry, rw, rh = roi
        crop = frame[ry:ry + rh, rx:rx + rw]
        ox, oy = rx, ry

    # --- Step 1: LAB color segmentation ---
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    dark_mask = (L < l_thresh).astype(np.uint8) * 255

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
        return None

    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not big:
        return None

    largest = max(big, key=cv2.contourArea)

    # Need at least 5 points for fitEllipse
    if len(largest) < 5:
        return None

    # --- Step 5: Fit ellipse to all contour points ---
    ellipse_params = cv2.fitEllipse(largest)
    (ecx, ecy), (ew, eh), eangle = ellipse_params

    # --- Step 6: Classify head vs handle points ---
    # Sample the fitted ellipse densely
    ellipse_pts = sample_ellipse(ecx, ecy, ew, eh, eangle, 720)  # (720, 2)

    # Contour points as (M, 2)
    contour_pts = largest.reshape(-1, 2).astype(np.float64)

    # Min distance from each contour point to the ellipse
    dists = cdist(contour_pts, ellipse_pts).min(axis=1)  # (M,)

    head_mask = dists < head_dist_thresh
    head_ratio = head_mask.sum() / len(head_mask)

    # --- Fallback: if ellipse fit is poor, use spline smoothing ---
    if head_ratio < min_head_ratio:
        if debug_vis:
            with debug_info_lock:
                shared_debug_info["ellipse"] = None
                shared_debug_info["junctions"] = None
                shared_debug_info["fallback"] = True
        # Offset to full-frame
        largest[:, :, 0] += ox
        largest[:, :, 1] += oy
        if smooth_points > 0 and len(largest) >= 4:
            return smooth_contour_fallback(largest, smooth_points)
        return largest

    # --- Step 7: Refit ellipse on head points only ---
    head_points = contour_pts[head_mask]
    if len(head_points) < 5:
        # Not enough head points, use initial ellipse
        head_ellipse = ellipse_params
    else:
        head_contour = head_points.reshape(-1, 1, 2).astype(np.int32)
        head_ellipse = cv2.fitEllipse(head_contour)

    (hcx, hcy), (hw, hh), hangle = head_ellipse

    # --- Step 8: Find junction points (head↔handle transitions) ---
    # Smooth the boolean head_mask to remove noise
    head_smooth = uniform_filter1d(head_mask.astype(np.float64), size=11, mode='wrap')
    head_bool = head_smooth > 0.5

    # Find transitions (head→handle and handle→head)
    transitions = np.where(np.diff(head_bool.astype(int)) != 0)[0]

    junction_a_idx = None
    junction_b_idx = None

    if len(transitions) >= 2:
        # Pick the pair enclosing the longest non-head (handle) run
        # Treat the array as circular
        n_pts = len(head_bool)
        best_gap = 0
        best_pair = (transitions[0], transitions[1])

        for i in range(len(transitions)):
            t_start = transitions[i]
            t_end = transitions[(i + 1) % len(transitions)]

            # Calculate gap length (circular)
            if t_end > t_start:
                gap = t_end - t_start
            else:
                gap = (n_pts - t_start) + t_end

            # Check if this gap is in the non-head region
            mid_idx = (t_start + gap // 2) % n_pts
            if not head_bool[mid_idx] and gap > best_gap:
                best_gap = gap
                best_pair = (t_start, t_end)

        junction_a_idx = best_pair[0]
        junction_b_idx = best_pair[1]
    else:
        # No clear transitions — entire contour is head (or handle)
        # Fall back to spline
        largest[:, :, 0] += ox
        largest[:, :, 1] += oy
        if smooth_points > 0 and len(largest) >= 4:
            return smooth_contour_fallback(largest, smooth_points)
        return largest

    # --- Step 9: Stitch head ellipse arc + handle spline ---
    # Junction points in contour coordinates
    jA = contour_pts[junction_a_idx]
    jB = contour_pts[junction_b_idx]

    # Map junction points to ellipse parametric angles
    angle_a = point_to_ellipse_angle(jA[0], jA[1], hcx, hcy, hw, hh, hangle)
    angle_b = point_to_ellipse_angle(jB[0], jB[1], hcx, hcy, hw, hh, hangle)

    # Head arc: go from angle_a to angle_b the "long way" (away from handle)
    # The handle is between junction_a and junction_b in contour order.
    # The head arc should go the other way around the ellipse.
    # Determine which direction around the ellipse is "away from handle"
    # by checking which arc contains the centroid of head points.
    head_centroid = head_points.mean(axis=0)
    head_centroid_angle = point_to_ellipse_angle(
        head_centroid[0], head_centroid[1], hcx, hcy, hw, hh, hangle
    )

    # Arc from a→b going counterclockwise
    if angle_b > angle_a:
        arc_ccw = angle_b - angle_a
    else:
        arc_ccw = (2 * np.pi - angle_a) + angle_b

    # Check if head centroid is in the ccw arc
    def angle_in_arc(angle, start, arc_len):
        """Check if angle is within arc starting at start going ccw for arc_len."""
        diff = (angle - start) % (2 * np.pi)
        return diff < arc_len

    if angle_in_arc(head_centroid_angle, angle_a, arc_ccw):
        # Head is in the ccw arc (a→b counterclockwise)
        arc_start = angle_a
        arc_length = arc_ccw
    else:
        # Head is in the cw arc (b→a counterclockwise, i.e., a→b clockwise)
        arc_start = angle_b
        arc_length = 2 * np.pi - arc_ccw

    # Sample head arc — number of points proportional to arc length
    a_semi, b_semi = hw / 2.0, hh / 2.0
    # Approximate ellipse perimeter (Ramanujan)
    ellipse_perim = np.pi * (3 * (a_semi + b_semi) - np.sqrt((3 * a_semi + b_semi) * (a_semi + 3 * b_semi)))
    head_arc_frac = arc_length / (2 * np.pi)
    head_arc_len = head_arc_frac * ellipse_perim

    # Handle: extract contour points from junction_a to junction_b (the non-head run)
    n_contour = len(contour_pts)
    if junction_b_idx > junction_a_idx:
        handle_indices = list(range(junction_a_idx, junction_b_idx + 1))
    else:
        handle_indices = list(range(junction_a_idx, n_contour)) + list(range(0, junction_b_idx + 1))
    handle_pts = contour_pts[handle_indices]

    # Compute handle arc length
    handle_diffs = np.diff(handle_pts, axis=0)
    handle_arc_len = np.sqrt((handle_diffs ** 2).sum(axis=1)).sum()

    total_arc_len = head_arc_len + handle_arc_len

    if smooth_points > 0:
        n_head_pts = max(4, int(smooth_points * head_arc_len / total_arc_len))
        n_handle_pts = max(4, smooth_points - n_head_pts)
    else:
        # No resampling requested — use proportional point counts
        n_head_pts = max(4, int(len(contour_pts) * head_arc_frac))
        n_handle_pts = max(4, len(handle_pts))

    # Sample head arc from the refined ellipse
    head_t = np.linspace(arc_start, arc_start + arc_length, n_head_pts, endpoint=False)
    head_sampled = ellipse_point(head_t, hcx, hcy, hw, hh, hangle)

    # Fit smoothing spline through handle points
    if len(handle_pts) >= 4:
        hx = handle_pts[:, 0]
        hy = handle_pts[:, 1]
        # s > 0 for smoothing (not interpolation)
        s_val = len(handle_pts) * 2.0  # smoothing factor
        try:
            tck, _ = splprep([hx, hy], s=s_val, per=False, k=3)
            u_new = np.linspace(0, 1, n_handle_pts)
            hx_new, hy_new = splev(u_new, tck)
            handle_sampled = np.stack([hx_new, hy_new], axis=-1)
        except (ValueError, TypeError):
            # Spline fit failed, use raw points evenly subsampled
            idx = np.linspace(0, len(handle_pts) - 1, n_handle_pts).astype(int)
            handle_sampled = handle_pts[idx]
    else:
        handle_sampled = handle_pts

    # Stitch head + handle
    stitched = np.vstack([head_sampled, handle_sampled])

    # Store debug info
    if debug_vis:
        with debug_info_lock:
            shared_debug_info["ellipse"] = head_ellipse
            shared_debug_info["initial_ellipse"] = ellipse_params
            shared_debug_info["junctions"] = (jA + np.array([ox, oy]),
                                              jB + np.array([ox, oy]))
            shared_debug_info["head_ratio"] = head_ratio
            shared_debug_info["fallback"] = False

    # Offset to full-frame coordinates
    stitched[:, 0] += ox
    stitched[:, 1] += oy

    # Convert to OpenCV contour format (N, 1, 2) int32
    result = stitched.round().astype(np.int32).reshape(-1, 1, 2)
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
def draw_overlay(frame, contour, R, t, world_pts, roi=None, debug_vis=False):
    """Draw contour, tag axes, and status text on the frame."""
    # ROI box in yellow
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 200, 200), 1)

    # Contour in cyan
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)

    # Debug: ellipse overlay + junction points
    if debug_vis:
        with debug_info_lock:
            di = dict(shared_debug_info)

        if di.get("ellipse") is not None:
            (ecx, ecy), (ew, eh), eangle = di["ellipse"]
            ox, oy = 0, 0
            if roi is not None:
                ox, oy = roi[0], roi[1]
            cv2.ellipse(frame,
                        (int(ecx + ox), int(ecy + oy)),
                        (int(ew / 2), int(eh / 2)),
                        eangle, 0, 360, (0, 255, 0), 1)

        if di.get("junctions") is not None:
            jA, jB = di["junctions"]
            cv2.circle(frame, (int(jA[0]), int(jA[1])), 6, (0, 0, 255), -1)
            cv2.circle(frame, (int(jB[0]), int(jB[1])), 6, (0, 0, 255), -1)

        if di.get("head_ratio") is not None:
            cv2.putText(frame, f"head: {di['head_ratio']:.0%}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if di.get("fallback"):
            cv2.putText(frame, "FALLBACK (spline)",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # Read current ROI (may be updated from web UI)
        with roi_lock:
            roi = tuple(shared_roi)

        contour = segment_racket(
            frame, args.min_area, roi=roi,
            l_thresh=args.l_thresh,
            open_kernel=args.open_kernel,
            close_kernel=args.close_kernel,
            head_dist_thresh=args.head_dist_thresh,
            min_head_ratio=args.min_head_ratio,
            smooth_points=args.smooth_points,
            debug_vis=args.debug_vis,
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        R, t = detect_tag(gray)

        with pose_lock:
            shared_pose = (R, t)

        world_pts = None
        if contour is not None and R is not None:
            pixels = contour.reshape(-1, 2).astype(np.float64)
            world_pts = pixels_to_world(pixels, R, t)
            last_world_pts = world_pts

        draw_overlay(frame, contour, R, t, world_pts, roi=roi, debug_vis=args.debug_vis)

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
