"""Hole vector measurement tool.

Captures images of string holes from both sides via a Pi camera,
detects the bright circle, records CNC XY positions, and computes
3D vectors through each hole for CNC path planning.
"""

import argparse
import atexit
import base64
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, request

from cnc import CNCController

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Hole vector measurement tool")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--pi-host", default="192.168.0.123")
parser.add_argument("--pi-port", type=int, default=8080)
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument("--serial-port", default="auto")
parser.add_argument("--threshold", type=int, default=128)
parser.add_argument("--min-area", type=int, default=100)
parser.add_argument("--z-working", type=float, default=50.0)
parser.add_argument("--fx", type=float, default=None)
parser.add_argument("--fy", type=float, default=None)
parser.add_argument("--cx-intrinsic", type=float, default=None)
parser.add_argument("--cy-intrinsic", type=float, default=None)
args = parser.parse_args()

PI_URL = f"http://{args.pi_host}:{args.pi_port}"

# ---------------------------------------------------------------------------
# Launch pi_stream.py as a subprocess
# ---------------------------------------------------------------------------
_pi_stream_proc = None
_pi_stream_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_stream.py")

if os.path.exists(_pi_stream_script):
    print(f"Launching pi_stream.py on port {args.pi_port}...")
    _pi_stream_proc = subprocess.Popen(
        [sys.executable, _pi_stream_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    def _kill_pi_stream():
        if _pi_stream_proc and _pi_stream_proc.poll() is None:
            _pi_stream_proc.terminate()
            try:
                _pi_stream_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _pi_stream_proc.kill()
            print("pi_stream.py stopped.")

    atexit.register(_kill_pi_stream)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Wait briefly for the stream server to be ready
    time.sleep(2)
else:
    print(f"Warning: {_pi_stream_script} not found, skipping auto-launch.")

# ---------------------------------------------------------------------------
# CNC setup
# ---------------------------------------------------------------------------
cnc = CNCController()
_port = args.serial_port
if _port == "auto":
    _port = CNCController.find_serial_port()
if _port:
    try:
        cnc.connect(_port, args.baud)
    except Exception as e:
        print(f"[CNC] Connection failed: {e}")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
measurements = []  # list of {side, cx, cy, radius, circularity, cnc_x, cnc_y, timestamp}
vectors = []       # list of computed vector results

# Auto-capture state
auto_mode = False
auto_last_pos = None      # last auto-capture CNC position (x, y, z)
auto_last_time = 0.0      # last auto-capture timestamp
AUTO_XY_MIN = 5.0         # mm — min XY move to trigger new capture
AUTO_Z_MIN = 10.0         # deg — min Z change to trigger at same XY
AUTO_COOLDOWN = 1.0       # seconds between auto-captures

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def capture_frame():
    """Fetch a single JPEG frame from the Pi camera and decode it."""
    resp = requests.get(f"{PI_URL}/capture", timeout=5)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def detect_circle(frame, threshold=128, min_area=100):
    """Detect the largest bright circle in the frame.

    Returns (cx, cy, radius, circularity) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Filter by min area and pick largest
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None
    best = max(valid, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(best)
    area = cv2.contourArea(best)
    perimeter = cv2.arcLength(best, True)
    circularity = (4 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
    return (float(cx), float(cy), float(radius), float(circularity))


def capture_to_world_point(cap, z_working, fx, fy, cx_intr, cy_intr):
    """Project a capture into a 3D world point on the near frame surface.

    The camera looks horizontally.  CNC X/Y position the gantry, CNC Z
    rotates the camera heading (Z=0 deg -> facing +X).

    Coordinate system:
      X, Y  – CNC horizontal plane
      H     – height (vertical, perpendicular to CNC XY plane)

    At Z=0 (facing +X):
      image right  (px +) -> world +Y   (perpendicular to viewing dir)
      image down   (py +) -> world -H   (downward)

    The point on the near surface is:
      camera_pos + z_working * view_dir + pixel lateral/vertical offsets
    """
    theta = math.radians(cap.get("cnc_z", 0.0))

    # Camera basis vectors
    view_x, view_y = math.cos(theta), math.sin(theta)   # viewing direction
    right_x, right_y = -math.sin(theta), math.cos(theta) # image-right in XY

    # Pixel offsets projected to mm at the frame surface
    px_mm = (cap["cx"] - cx_intr) / fx * z_working   # horizontal (image right)
    py_mm = (cap["cy"] - cy_intr) / fy * z_working   # vertical (image down)

    # World point on near surface of frame
    world_x = cap["cnc_x"] + z_working * view_x + px_mm * right_x
    world_y = cap["cnc_y"] + z_working * view_y + px_mm * right_y
    world_h = -py_mm  # image down = height decreases

    return (world_x, world_y, world_h)


def compute_3d_vector(cap_a, cap_b, z_working, fx, fy, cx_intr, cy_intr):
    """Compute 3D vector through a hole from two captures.

    Each capture (from opposite sides of the frame) gives a point where the
    camera ray hits the near surface.  The vector from point_a to point_b is
    the direction the hole runs through the frame.
    """
    pt_a = capture_to_world_point(cap_a, z_working, fx, fy, cx_intr, cy_intr)
    pt_b = capture_to_world_point(cap_b, z_working, fx, fy, cx_intr, cy_intr)

    vx = pt_b[0] - pt_a[0]
    vy = pt_b[1] - pt_a[1]
    vh = pt_b[2] - pt_a[2]
    mag = math.sqrt(vx * vx + vy * vy + vh * vh)
    if mag < 1e-9:
        return None
    norm = (vx / mag, vy / mag, vh / mag)
    # Angle from the XY plane normal (pure horizontal = 0 deg)
    lateral = math.sqrt(norm[0] ** 2 + norm[1] ** 2)
    angle_deg = math.degrees(math.atan2(abs(norm[2]), lateral))

    return {
        "point_a": list(pt_a),
        "point_b": list(pt_b),
        "vector": list(norm),
        "angle_deg": round(angle_deg, 2),
        "magnitude_mm": round(mag, 3),
    }


def generate_plotly_html(vecs):
    """Generate standalone HTML with Plotly 3D visualization."""
    traces = []
    for i, v in enumerate(vecs):
        a = v["point_a"]
        b = v["point_b"]
        angle = v["angle_deg"]
        # Color: green (small angle) -> red (large angle)
        t = min(angle / 30.0, 1.0)
        r = int(255 * t)
        g = int(255 * (1 - t))
        color = f"rgb({r},{g},0)"
        traces.append({
            "type": "scatter3d",
            "mode": "lines+markers",
            "x": [a[0], b[0]],
            "y": [a[1], b[1]],
            "z": [a[2], b[2]],
            "line": {"color": color, "width": 6},
            "marker": {"size": 4, "color": color},
            "name": f"Hole {i+1} ({angle:.1f} deg)",
        })

    layout = {
        "title": "Hole Vectors (3D)",
        "scene": {
            "xaxis": {"title": "X (mm)"},
            "yaxis": {"title": "Y (mm)"},
            "zaxis": {"title": "Height (mm)"},
            "aspectmode": "data",
        },
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
    }

    return f"""<!DOCTYPE html>
<html><head>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head><body style="margin:0">
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
Plotly.newPlot("plot", {json.dumps(traces)}, {json.dumps(layout)});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return INDEX_HTML



def _do_capture(side, threshold, min_area):
    """Capture a frame, detect circle, return (measurement, preview_b64).

    Raises on camera failure.
    """
    frame = capture_frame()
    result = detect_circle(frame, threshold=threshold, min_area=min_area)

    # Get CNC position
    cnc_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    if cnc.connected:
        status = cnc.query_status()
        if status:
            cnc_pos = status["wpos"]

    # Encode frame as JPEG for preview (with circle overlay)
    if result:
        cx, cy, radius, circularity = result
        cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 255, 0), -1)

    _, jpeg = cv2.imencode(".jpg", frame)
    preview_b64 = base64.b64encode(jpeg.tobytes()).decode()

    measurement = {
        "side": side,
        "cx": result[0] if result else None,
        "cy": result[1] if result else None,
        "radius": result[2] if result else None,
        "circularity": round(result[3], 3) if result else None,
        "cnc_x": cnc_pos["x"],
        "cnc_y": cnc_pos["y"],
        "cnc_z": cnc_pos["z"],
        "timestamp": datetime.now().isoformat(),
        "detected": result is not None,
    }
    return measurement, preview_b64


@app.route("/capture", methods=["POST"])
def do_capture():
    body = request.get_json(force=True)
    side = body.get("side", "A").upper()
    threshold = body.get("threshold", args.threshold)
    min_area = body.get("min_area", args.min_area)

    try:
        measurement, preview_b64 = _do_capture(side, threshold, min_area)
    except Exception as e:
        return jsonify({"error": f"Camera capture failed: {e}"}), 500

    measurements.append(measurement)

    return jsonify({
        "measurement": measurement,
        "preview": preview_b64,
        "index": len(measurements) - 1,
    })


@app.route("/measurements", methods=["GET"])
def get_measurements():
    return jsonify(measurements)


@app.route("/measurements", methods=["DELETE"])
def clear_measurements():
    measurements.clear()
    return jsonify({"status": "cleared"})


@app.route("/compute", methods=["POST"])
def compute():
    # Find last A and last B
    last_a = None
    last_b = None
    for m in reversed(measurements):
        if m["side"] == "A" and m["detected"] and last_a is None:
            last_a = m
        if m["side"] == "B" and m["detected"] and last_b is None:
            last_b = m
        if last_a and last_b:
            break

    if not last_a or not last_b:
        return jsonify({"error": "Need both a Side A and Side B capture with detected circles"}), 400

    # Pi Camera v2.1 (IMX219) at 1640x1232, close-focus estimate:
    # Nominal f=3.04mm, pixel pitch=2.24um -> 1357px. ~5% increase for close focus -> 1425px.
    fx = args.fx if args.fx else 1425.0
    fy = args.fy if args.fy else 1425.0
    cx_intr = args.cx_intrinsic if args.cx_intrinsic else 820.0  # 1640/2
    cy_intr = args.cy_intrinsic if args.cy_intrinsic else 616.0  # 1232/2

    result = compute_3d_vector(
        last_a, last_b,
        args.z_working,
        fx, fy, cx_intr, cy_intr,
    )
    if result is None:
        return jsonify({"error": "Degenerate vector (zero length)"}), 400

    result["capture_a"] = last_a
    result["capture_b"] = last_b
    vectors.append(result)
    return jsonify({"vector": result, "index": len(vectors) - 1})


@app.route("/auto/toggle", methods=["POST"])
def auto_toggle():
    global auto_mode, auto_last_pos, auto_last_time
    auto_mode = not auto_mode
    if auto_mode:
        auto_last_pos = None
        auto_last_time = 0.0
    return jsonify({"auto_mode": auto_mode})


@app.route("/auto/poll", methods=["POST"])
def auto_poll():
    global auto_last_pos, auto_last_time
    if not auto_mode:
        return jsonify({"captured": False})

    body = request.get_json(force=True) if request.data else {}
    threshold = body.get("threshold", args.threshold)
    min_area = body.get("min_area", args.min_area)

    # Get a reliable CNC position first — skip if unavailable
    if not cnc.connected:
        return jsonify({"captured": False})
    cnc_status_resp = cnc.query_status()
    if cnc_status_resp is None:
        return jsonify({"captured": False})
    cnc_pos = cnc_status_resp["wpos"]

    try:
        measurement, preview_b64 = _do_capture("auto", threshold, min_area)
    except Exception:
        return jsonify({"captured": False})

    if not measurement["detected"]:
        return jsonify({"captured": False})

    # Override with the pre-fetched position to avoid serial contention
    measurement["cnc_x"] = cnc_pos["x"]
    measurement["cnc_y"] = cnc_pos["y"]
    measurement["cnc_z"] = cnc_pos["z"]

    # Debounce: must have moved enough from last capture position
    now = time.time()
    if now - auto_last_time < AUTO_COOLDOWN:
        return jsonify({"captured": False})

    cur_pos = (measurement["cnc_x"], measurement["cnc_y"], measurement["cnc_z"])
    if auto_last_pos is not None:
        dx = cur_pos[0] - auto_last_pos[0]
        dy = cur_pos[1] - auto_last_pos[1]
        xy_dist = math.sqrt(dx * dx + dy * dy)
        dz = abs(cur_pos[2] - auto_last_pos[2])
        dz = min(dz, 360.0 - dz)  # wrapped angular difference
        if xy_dist < AUTO_XY_MIN and dz < AUTO_Z_MIN:
            return jsonify({"captured": False})

    # Passed debounce — record this capture
    auto_last_pos = cur_pos
    auto_last_time = now
    measurements.append(measurement)

    return jsonify({
        "captured": True,
        "measurement": measurement,
        "preview": preview_b64,
        "index": len(measurements) - 1,
    })


@app.route("/auto_compute", methods=["POST"])
def auto_compute():
    fx = args.fx if args.fx else 1425.0
    fy = args.fy if args.fy else 1425.0
    cx_intr = args.cx_intrinsic if args.cx_intrinsic else 820.0
    cy_intr = args.cy_intrinsic if args.cy_intrinsic else 616.0

    # Collect auto captures with detected circles
    auto_caps = [(i, m) for i, m in enumerate(measurements)
                 if m["side"] == "auto" and m["detected"]]

    if len(auto_caps) < 2:
        return jsonify({"error": "Need at least 2 auto captures with detected circles"}), 400

    # Project each capture to world XY
    world_pts = []
    for idx, cap in auto_caps:
        wp = capture_to_world_point(cap, args.z_working, fx, fy, cx_intr, cy_intr)
        world_pts.append((idx, cap, wp[0], wp[1]))  # (meas_idx, cap, world_x, world_y)

    # Build candidate pairs, filter by world XY distance and Z angular diff
    candidates = []
    for i in range(len(world_pts)):
        for j in range(i + 1, len(world_pts)):
            _, cap_i, wx_i, wy_i = world_pts[i]
            _, cap_j, wx_j, wy_j = world_pts[j]
            wdist = math.sqrt((wx_i - wx_j) ** 2 + (wy_i - wy_j) ** 2)
            dz = abs(cap_i["cnc_z"] - cap_j["cnc_z"])
            dz = min(dz, 360.0 - dz)
            if wdist < 10.0 and dz > 90.0:
                candidates.append((wdist, i, j))

    # Sort by world XY distance ascending, greedy match
    candidates.sort()
    matched = set()
    new_vectors = []
    warnings = []

    for wdist, i, j in candidates:
        if i in matched or j in matched:
            continue
        matched.add(i)
        matched.add(j)

        _, cap_i, _, _ = world_pts[i]
        _, cap_j, _, _ = world_pts[j]

        # Assign sides: smaller cnc_z -> A, larger -> B
        if cap_i["cnc_z"] <= cap_j["cnc_z"]:
            cap_a, cap_b = cap_i, cap_j
        else:
            cap_a, cap_b = cap_j, cap_i

        result = compute_3d_vector(cap_a, cap_b, args.z_working, fx, fy, cx_intr, cy_intr)
        if result is None:
            warnings.append(f"Degenerate vector for pair (world dist={wdist:.1f}mm)")
            continue

        result["capture_a"] = cap_a
        result["capture_b"] = cap_b
        vectors.append(result)
        new_vectors.append(result)

    unmatched_count = len(auto_caps) - len(matched)
    if unmatched_count > 0:
        warnings.append(f"{unmatched_count} capture(s) could not be paired")

    return jsonify({
        "vectors": new_vectors,
        "count": len(new_vectors),
        "unmatched_count": unmatched_count,
        "warnings": warnings,
    })


@app.route("/vectors", methods=["GET"])
def get_vectors():
    return jsonify(vectors)


@app.route("/plot")
def plot():
    if not vectors:
        return "<html><body><h2>No vectors computed yet.</h2></body></html>"
    return generate_plotly_html(vectors)


@app.route("/cnc/status")
def cnc_status():
    if not cnc.connected:
        return jsonify({"connected": False, "state": "Disconnected", "wpos": {"x": 0, "y": 0, "z": 0}})
    status = cnc.query_status()
    if status is None:
        return jsonify({"connected": True, "state": "No response", "wpos": {"x": 0, "y": 0, "z": 0}})
    status["connected"] = True
    return jsonify(status)


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hole Vector Measurement</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  h1 { padding: 12px 20px; background: #16213e; font-size: 1.3rem; }
  .row { display: flex; gap: 12px; padding: 12px 20px; flex-wrap: wrap; }
  .panel { background: #16213e; border-radius: 8px; padding: 12px; flex: 1; min-width: 300px; }
  .panel h3 { margin-bottom: 8px; font-size: 0.95rem; color: #8ab4f8; }
  .feeds { display: flex; gap: 12px; flex-wrap: wrap; }
  .feeds > div { flex: 1; min-width: 280px; }
  .feeds img, .feeds canvas { width: 100%; border-radius: 6px; background: #000; display: block; }
  .status-bar { padding: 10px 20px; background: #0f3460; font-family: monospace;
                 display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
  .status-bar .pos { color: #8ab4f8; }
  .status-bar .state { font-weight: bold; }
  .state-idle { color: #4caf50; }
  .state-run { color: #ff9800; }
  .state-alarm { color: #f44336; }
  .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 12px 20px; }
  label { font-size: 0.85rem; color: #aaa; }
  input[type=range] { width: 160px; }
  input[type=number] { width: 80px; background: #16213e; color: #e0e0e0; border: 1px solid #333;
                        border-radius: 4px; padding: 4px 6px; }
  button { padding: 8px 18px; border: none; border-radius: 6px; font-weight: 600;
           cursor: pointer; font-size: 0.9rem; }
  .btn-a { background: #2196f3; color: #fff; }
  .btn-b { background: #ff9800; color: #fff; }
  .btn-compute { background: #4caf50; color: #fff; }
  .btn-clear { background: #f44336; color: #fff; }
  .btn-plot { background: #9c27b0; color: #fff; }
  button:hover { opacity: 0.85; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 8px; }
  th, td { padding: 5px 8px; text-align: left; border-bottom: 1px solid #333; }
  th { color: #8ab4f8; }
  .circle-info { font-size: 0.85rem; margin-top: 6px; color: #aaa; }
  #log { max-height: 120px; overflow-y: auto; font-family: monospace; font-size: 0.8rem;
         background: #111; padding: 8px; border-radius: 4px; margin-top: 8px; }
  #log div { padding: 1px 0; }
  .log-ok { color: #4caf50; }
  .log-err { color: #f44336; }
  .btn-auto { background: #555; color: #fff; }
  .btn-auto.active { background: #00bcd4; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.7 } }
  .btn-autocompute { background: #009688; color: #fff; }
  .flash { box-shadow: 0 0 12px 4px #00bcd4; transition: box-shadow 0.2s; }
</style>
</head><body>

<h1>Hole Vector Measurement</h1>

<div class="status-bar">
  <span>CNC: <span class="pos" id="cnc-pos">X=?.?? Y=?.?? Z(angle)=?.??&deg;</span></span>
  <span class="state" id="cnc-state">--</span>
  <span style="color:#666" id="cnc-conn">disconnected</span>
</div>

<div class="row">
  <div class="panel" style="flex:2">
    <div class="feeds">
      <div>
        <h3>Pi Camera Live Feed</h3>
        <img id="live-feed" src="" alt="Live feed">
      </div>
      <div>
        <h3>Last Capture</h3>
        <canvas id="capture-canvas" width="820" height="616"></canvas>
        <div class="circle-info" id="circle-info">No capture yet</div>
      </div>
    </div>
  </div>
</div>

<div class="controls">
  <label>Threshold: <span id="thresh-val">""" + str(args.threshold) + """</span></label>
  <input type="range" id="threshold" min="0" max="255" value=\"""" + str(args.threshold) + """">
  <label>Min area:</label>
  <input type="number" id="min-area" value=\"""" + str(args.min_area) + """" min="1">

  <button class="btn-a" onclick="doCapture('A')">Capture Side A</button>
  <button class="btn-b" onclick="doCapture('B')">Capture Side B</button>
  <button class="btn-compute" onclick="doCompute()">Compute Vector</button>
  <button class="btn-auto" id="btn-auto" onclick="toggleAuto()">Auto: OFF</button>
  <button class="btn-autocompute" onclick="doAutoCompute()">Auto Compute</button>
  <button class="btn-clear" onclick="doClear()">Clear All</button>
  <button class="btn-plot" onclick="window.open('/plot','_blank')">View 3D Plot</button>
</div>

<div class="row">
  <div class="panel">
    <h3>Measurements</h3>
    <table>
      <thead><tr><th>#</th><th>Side</th><th>Circle (cx,cy,r)</th><th>Circ.</th><th>CNC X</th><th>CNC Y</th><th>Z&deg;</th><th>Time</th></tr></thead>
      <tbody id="meas-body"></tbody>
    </table>
  </div>
  <div class="panel">
    <h3>Computed Vectors</h3>
    <table>
      <thead><tr><th>#</th><th>Angle</th><th>Vector</th><th>A pos</th><th>B pos</th></tr></thead>
      <tbody id="vec-body"></tbody>
    </table>
    <div id="log"></div>
  </div>
</div>

<script>
document.getElementById('live-feed').src = '""" + PI_URL + """/stream';

const threshSlider = document.getElementById('threshold');
const threshVal = document.getElementById('thresh-val');
threshSlider.oninput = () => { threshVal.textContent = threshSlider.value; };

function log(msg, cls) {
  const d = document.getElementById('log');
  const el = document.createElement('div');
  el.className = cls || '';
  el.textContent = msg;
  d.appendChild(el);
  d.scrollTop = d.scrollHeight;
}

// CNC polling
function pollCNC() {
  fetch('/cnc/status').then(r => r.json()).then(s => {
    const p = s.wpos;
    document.getElementById('cnc-pos').textContent =
      `X=${p.x.toFixed(2)} Y=${p.y.toFixed(2)} Z(angle)=${p.z.toFixed(2)}\u00B0`;
    const st = document.getElementById('cnc-state');
    st.textContent = s.state;
    st.className = 'state state-' + s.state.toLowerCase();
    document.getElementById('cnc-conn').textContent = s.connected ? 'connected' : 'disconnected';
  }).catch(() => {});
}
setInterval(pollCNC, 500);
pollCNC();

// Capture
function doCapture(side) {
  const thresh = parseInt(threshSlider.value);
  const minArea = parseInt(document.getElementById('min-area').value);
  log(`Capturing side ${side}...`);
  fetch('/capture', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({side, threshold: thresh, min_area: minArea}),
  }).then(r => r.json()).then(data => {
    if (data.error) { log(data.error, 'log-err'); return; }
    const m = data.measurement;
    // Draw preview on canvas
    const canvas = document.getElementById('capture-canvas');
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
    };
    img.src = 'data:image/jpeg;base64,' + data.preview;

    const info = document.getElementById('circle-info');
    if (m.detected) {
      info.textContent = `Side ${m.side}: circle at (${m.cx.toFixed(1)}, ${m.cy.toFixed(1)}) r=${m.radius.toFixed(1)} circ=${m.circularity}`;
      log(`Side ${side}: detected circle r=${m.radius.toFixed(1)} circ=${m.circularity}`, 'log-ok');
    } else {
      info.textContent = `Side ${m.side}: no circle detected`;
      log(`Side ${side}: no circle detected`, 'log-err');
    }
    refreshMeasurements();
  }).catch(e => log('Capture failed: ' + e, 'log-err'));
}

function doCompute() {
  log('Computing vector...');
  fetch('/compute', {method: 'POST'}).then(r => r.json()).then(data => {
    if (data.error) { log(data.error, 'log-err'); return; }
    const v = data.vector;
    log(`Vector #${data.index + 1}: angle=${v.angle_deg} deg`, 'log-ok');
    refreshVectors();
  }).catch(e => log('Compute failed: ' + e, 'log-err'));
}

function doClear() {
  fetch('/measurements', {method: 'DELETE'}).then(() => {
    measurements = [];
    document.getElementById('meas-body').innerHTML = '';
    document.getElementById('vec-body').innerHTML = '';
    log('Cleared all measurements');
  });
}

function refreshMeasurements() {
  fetch('/measurements').then(r => r.json()).then(data => {
    const tb = document.getElementById('meas-body');
    tb.innerHTML = '';
    data.forEach((m, i) => {
      const tr = document.createElement('tr');
      const circle = m.detected ? `(${m.cx.toFixed(1)}, ${m.cy.toFixed(1)}, ${m.radius.toFixed(1)})` : '--';
      tr.innerHTML = `<td>${i+1}</td><td>${m.side}</td><td>${circle}</td>` +
        `<td>${m.circularity ?? '--'}</td><td>${m.cnc_x.toFixed(2)}</td><td>${m.cnc_y.toFixed(2)}</td>` +
        `<td>${m.cnc_z.toFixed(1)}</td><td>${m.timestamp.slice(11, 19)}</td>`;
      tb.appendChild(tr);
    });
  });
}

function refreshVectors() {
  fetch('/vectors').then(r => r.json()).then(data => {
    const tb = document.getElementById('vec-body');
    tb.innerHTML = '';
    data.forEach((v, i) => {
      const tr = document.createElement('tr');
      const vec = v.vector.map(n => n.toFixed(3)).join(', ');
      const ap = v.point_a.map(n => n.toFixed(1)).join(', ');
      const bp = v.point_b.map(n => n.toFixed(1)).join(', ');
      tr.innerHTML = `<td>${i+1}</td><td>${v.angle_deg}&deg;</td><td>(${vec})</td><td>(${ap})</td><td>(${bp})</td>`;
      tb.appendChild(tr);
    });
  });
}

// --- Auto-capture ---
let autoInterval = null;

function toggleAuto() {
  fetch('/auto/toggle', {method: 'POST'}).then(r => r.json()).then(data => {
    const btn = document.getElementById('btn-auto');
    if (data.auto_mode) {
      btn.textContent = 'Auto: ON';
      btn.classList.add('active');
      autoInterval = setInterval(autoPoll, 500);
      log('Auto-capture enabled', 'log-ok');
    } else {
      btn.textContent = 'Auto: OFF';
      btn.classList.remove('active');
      if (autoInterval) { clearInterval(autoInterval); autoInterval = null; }
      log('Auto-capture disabled');
    }
  });
}

function autoPoll() {
  const thresh = parseInt(threshSlider.value);
  const minArea = parseInt(document.getElementById('min-area').value);
  fetch('/auto/poll', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({threshold: thresh, min_area: minArea}),
  }).then(r => r.json()).then(data => {
    if (!data.captured) return;
    const m = data.measurement;
    // Flash canvas border
    const canvas = document.getElementById('capture-canvas');
    canvas.classList.add('flash');
    setTimeout(() => canvas.classList.remove('flash'), 600);
    // Draw preview
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      canvas.getContext('2d').drawImage(img, 0, 0);
    };
    img.src = 'data:image/jpeg;base64,' + data.preview;
    // Update circle info
    const info = document.getElementById('circle-info');
    info.textContent = `Auto: circle at (${m.cx.toFixed(1)}, ${m.cy.toFixed(1)}) r=${m.radius.toFixed(1)} circ=${m.circularity}`;
    log(`Auto #${data.index + 1}: r=${m.radius.toFixed(1)} circ=${m.circularity} Z=${m.cnc_z.toFixed(1)}`, 'log-ok');
    refreshMeasurements();
  }).catch(() => {});
}

function doAutoCompute() {
  log('Auto computing vectors...');
  fetch('/auto_compute', {method: 'POST'}).then(r => r.json()).then(data => {
    if (data.error) { log(data.error, 'log-err'); return; }
    log(`Auto compute: ${data.count} vector(s) paired, ${data.unmatched_count} unmatched`, 'log-ok');
    if (data.warnings) data.warnings.forEach(w => log('Warning: ' + w, 'log-err'));
    refreshVectors();
  }).catch(e => log('Auto compute failed: ' + e, 'log-err'));
}
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"Starting hole vector tool on http://localhost:{args.port}")
    print(f"Pi camera at {PI_URL}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
