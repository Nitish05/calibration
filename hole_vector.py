"""Hole vector measurement tool.

Captures images of string holes from both sides via a Pi camera,
detects the bright circle, records CNC XY positions, and computes
3D vectors through each hole for CNC path planning.
"""

import argparse
import io
import json
import math
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
parser.add_argument("--frame-thickness", type=float, default=12.0)
parser.add_argument("--z-working", type=float, default=50.0)
parser.add_argument("--fx", type=float, default=None)
parser.add_argument("--fy", type=float, default=None)
parser.add_argument("--cx-intrinsic", type=float, default=None)
parser.add_argument("--cy-intrinsic", type=float, default=None)
args = parser.parse_args()

PI_URL = f"http://{args.pi_host}:{args.pi_port}"

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


def compute_3d_vector(cap_a, cap_b, frame_thickness, z_working, fx, fy, cx_intr, cy_intr):
    """Compute 3D vector through a hole from two captures.

    Each capture (from opposite sides of the frame) gives a point where the
    camera ray hits the near surface.  The vector from point_a to point_b is
    the direction the hole runs through the frame.

    frame_thickness is recorded for reference but not used in the projection
    (the two camera positions + pixel offsets determine both endpoints).
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
        "frame_thickness_ref": frame_thickness,
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


@app.route("/stream")
def stream_proxy():
    """Proxy the Pi MJPEG stream to avoid cross-origin issues."""
    try:
        resp = requests.get(f"{PI_URL}/stream", stream=True, timeout=10)
    except requests.ConnectionError:
        return Response("Pi camera not reachable", status=502)
    def generate():
        for chunk in resp.iter_content(chunk_size=4096):
            yield chunk
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture", methods=["POST"])
def do_capture():
    body = request.get_json(force=True)
    side = body.get("side", "A").upper()
    threshold = body.get("threshold", args.threshold)
    min_area = body.get("min_area", args.min_area)

    try:
        frame = capture_frame()
    except Exception as e:
        return jsonify({"error": f"Camera capture failed: {e}"}), 500

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
    import base64
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

    # Camera intrinsics (defaults: image center, reasonable focal length)
    # Pi camera v1 at 1640x1232: ~1500px focal length is a rough estimate
    fx = args.fx if args.fx else 1500.0
    fy = args.fy if args.fy else 1500.0
    cx_intr = args.cx_intrinsic if args.cx_intrinsic else 820.0  # 1640/2
    cy_intr = args.cy_intrinsic if args.cy_intrinsic else 616.0  # 1232/2

    result = compute_3d_vector(
        last_a, last_b,
        args.frame_thickness, args.z_working,
        fx, fy, cx_intr, cy_intr,
    )
    if result is None:
        return jsonify({"error": "Degenerate vector (zero length)"}), 400

    result["capture_a"] = last_a
    result["capture_b"] = last_b
    vectors.append(result)
    return jsonify({"vector": result, "index": len(vectors) - 1})


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
  <label>Frame thickness (mm):</label>
  <input type="number" id="frame-thickness" value=\"""" + str(args.frame_thickness) + """" step="0.1">

  <button class="btn-a" onclick="doCapture('A')">Capture Side A</button>
  <button class="btn-b" onclick="doCapture('B')">Capture Side B</button>
  <button class="btn-compute" onclick="doCompute()">Compute Vector</button>
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
document.getElementById('live-feed').src = '/stream';

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
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"Starting hole vector tool on http://localhost:{args.port}")
    print(f"Pi camera at {PI_URL}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
