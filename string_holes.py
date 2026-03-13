"""String hole position capture tool.

Manually jog the CNC to string holes on a tennis racket and record
positions.  The main CNC controls X, Y, A (rotational axis in degrees).
A second CNC controls linear Z (up/down in mm).  Points are tagged
as 'inside' or 'outside' edge and visualised on an interactive 2D
Plotly scatter plot.

Optionally shows a live top-down USB camera feed with AprilTag
detection overlay and hover-to-show world XY tooltip.
"""

import argparse
import atexit
import json
import os
import threading
import time
from datetime import datetime

from flask import Flask, Response, jsonify, request

from cnc import CNCController
from arduino_controller import ArduinoController

# ---------------------------------------------------------------------------
# Optional camera / vision imports
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
    from pupil_apriltags import Detector
    from camera import Camera
    from transforms import FX, FY, CX, CY, TAG_SIZE, CAMERA_MATRIX, pixels_to_world
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="String hole position capture tool")
parser.add_argument("--port", type=int, default=5001)
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument("--serial-port", default="/dev/cnc_main")
parser.add_argument("--serial-port-z", default="/dev/cnc_aux",
                    help="Serial port for Z-axis CNC")
parser.add_argument("--no-camera", action="store_true", help="Disable camera feed")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# CNC setup — main (X, Y, A rotational)
# ---------------------------------------------------------------------------
cnc = CNCController()
_port = args.serial_port
if not os.path.exists(_port):
    print(f"[CNC] {_port} not found, auto-detecting...")
    _port = CNCController.find_serial_port()
if _port:
    try:
        cnc.connect(_port, args.baud)
    except Exception as e:
        print(f"[CNC] Connection failed: {e}")

atexit.register(cnc.disconnect)

# ---------------------------------------------------------------------------
# Arduino motor controller setup (6 actuators)
# ---------------------------------------------------------------------------
arduino = ArduinoController()
_port_z = args.serial_port_z
if not os.path.exists(_port_z):
    print(f"[Arduino] {_port_z} not found, auto-detecting...")
    _port_z = ArduinoController.find_serial_port()
if _port_z:
    try:
        arduino.connect(_port_z, args.baud)
    except Exception as e:
        print(f"[Arduino] Connection failed: {e}")
else:
    print("[Arduino] No Arduino found, motor controller not connected")

atexit.register(arduino.disconnect)

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
os.makedirs("data/sequences", exist_ok=True)
os.makedirs("data/macros", exist_ok=True)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
points = []  # [{"type": "inside"|"outside", "x": float, "y": float, "a": float, "timestamp": str}]

# Camera shared state
camera_available = False
frame_lock = threading.Lock()
shared_frame = None
shared_jpeg = None
frame_event = threading.Event()
pose_lock = threading.Lock()
shared_pose = (None, None)
tag_detected = False

# CNC / Arduino cached status (polled in background threads)
cnc_status_lock = threading.Lock()
cached_cnc_status = {"connected": False, "state": "Disconnected", "wpos": {"x": 0, "y": 0, "z": 0}}

arduino_status_lock = threading.Lock()
cached_arduino_status = {"connected": False, "positions": {"x": 0, "z": 0, "byj1": 0, "byj2": 0}}

# ---------------------------------------------------------------------------
# Camera + AprilTag background thread
# ---------------------------------------------------------------------------
if HAS_CAMERA and not args.no_camera:
    try:
        cap = Camera()
        detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            decode_sharpening=0.25,
            refine_edges=True,
        )
        camera_available = True
        print("[Camera] Initialized")
    except Exception as e:
        print(f"[Camera] Init failed: {e}")


def camera_thread():
    global shared_frame, shared_jpeg, shared_pose, tag_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[FX, FY, CX, CY],
            tag_size=TAG_SIZE,
        )

        R, t = None, None
        if detections:
            det = detections[0]
            R, t = det.pose_R, det.pose_t

            # Draw tag outline
            corners = det.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

            # Draw center
            cx_tag, cy_tag = int(det.center[0]), int(det.center[1])
            cv2.circle(frame, (cx_tag, cy_tag), 5, (0, 0, 255), -1)

            # Draw 3D axes
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

            # ID text
            cv2.putText(frame, f"ID:{det.tag_id}", (corners[0][0], corners[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO TAG", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = jpeg.tobytes()
        with frame_lock:
            shared_frame = frame
            shared_jpeg = jpeg_bytes
        with pose_lock:
            shared_pose = (R, t)
            tag_detected = R is not None
        frame_event.set()


if camera_available:
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()


# ---------------------------------------------------------------------------
# CNC / Arduino background polling threads
# ---------------------------------------------------------------------------
def cnc_poll_thread():
    global cached_cnc_status
    while True:
        if not cnc.connected:
            status = {"connected": False, "state": "Disconnected", "wpos": {"x": 0, "y": 0, "z": 0}}
        else:
            raw = cnc.query_status()
            if raw is None:
                status = {"connected": True, "state": "No response", "wpos": {"x": 0, "y": 0, "z": 0}}
            else:
                status = raw
                status["connected"] = True
        status["camera_available"] = camera_available
        status["tag_detected"] = tag_detected
        with cnc_status_lock:
            cached_cnc_status = status
        time.sleep(0.1)


def arduino_poll_thread():
    global cached_arduino_status
    while True:
        if not arduino.connected:
            status = {"connected": False, "positions": {"x": 0, "z": 0, "byj1": 0, "byj2": 0}}
        else:
            positions = arduino.query_positions()
            if positions is None:
                positions = {"x": 0, "z": 0, "byj1": 0, "byj2": 0}
            status = {"connected": True, "positions": positions}
        with arduino_status_lock:
            cached_arduino_status = status
        time.sleep(0.1)


threading.Thread(target=cnc_poll_thread, daemon=True).start()
threading.Thread(target=arduino_poll_thread, daemon=True).start()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    html = INDEX_HTML.replace("__CAMERA_AVAILABLE__", "true" if camera_available else "false")
    html = html.replace("__ARDUINO_CONNECTED__", "true" if arduino.connected else "false")
    return html


@app.route("/cnc/status")
def cnc_status():
    with cnc_status_lock:
        return jsonify(cached_cnc_status)


@app.route("/cnc/jog", methods=["POST"])
def cnc_jog():
    if not cnc.connected:
        return jsonify({"error": "CNC not connected"}), 503
    body = request.get_json(force=True)
    x = float(body.get("x", 0))
    y = float(body.get("y", 0))
    a = float(body.get("a", 0))
    feedrate = int(body.get("feedrate", 1000))
    feedrate = max(100, min(5000, feedrate))
    cmd = f"$J=G90 G21 X{x} Y{y} Z{a} F{feedrate}"
    result = cnc.send_line(cmd)
    if result is True:
        return jsonify({"ok": True})
    return jsonify({"error": str(result)}), 400


@app.route("/cnc/jog/cancel", methods=["POST"])
def cnc_jog_cancel():
    if not cnc.connected:
        return jsonify({"error": "CNC not connected"}), 503
    cnc.jog_cancel()
    return jsonify({"ok": True})


@app.route("/arduino/status")
def arduino_status():
    with arduino_status_lock:
        return jsonify(cached_arduino_status)


@app.route("/arduino/move", methods=["POST"])
def arduino_move():
    if not arduino.connected:
        return jsonify({"error": "Arduino not connected"}), 503
    body = request.get_json(force=True)
    motor = body.get("motor")
    steps = int(body.get("steps", 0))
    dispatch = {"x": arduino.move_x, "z": arduino.move_z,
                "byj1": arduino.move_byj1, "byj2": arduino.move_byj2}
    fn = dispatch.get(motor)
    if fn is None:
        return jsonify({"error": f"Unknown motor: {motor}"}), 400
    resp = fn(steps)
    return jsonify({"ok": True, "response": resp})


@app.route("/arduino/servo", methods=["POST"])
def arduino_servo():
    if not arduino.connected:
        return jsonify({"error": "Arduino not connected"}), 503
    body = request.get_json(force=True)
    angle = int(body.get("angle", 90))
    resp = arduino.set_servo(angle)
    return jsonify({"ok": True, "response": resp})


@app.route("/arduino/dc", methods=["POST"])
def arduino_dc():
    if not arduino.connected:
        return jsonify({"error": "Arduino not connected"}), 503
    body = request.get_json(force=True)
    action = body.get("action", "stop")
    speed = int(body.get("speed", 50))
    if action == "forward":
        resp = arduino.dc_forward(speed)
    elif action == "reverse":
        resp = arduino.dc_reverse(speed)
    else:
        resp = arduino.dc_stop()
    return jsonify({"ok": True, "response": resp})


@app.route("/arduino/reset", methods=["POST"])
def arduino_reset():
    if not arduino.connected:
        return jsonify({"error": "Arduino not connected"}), 503
    body = request.get_json(force=True)
    mode = body.get("mode", "all")
    if mode == "steppers":
        resp = arduino.reset_steppers()
    else:
        resp = arduino.reset()
    return jsonify({"ok": True, "response": resp})


@app.route("/feed")
def video_feed():
    if not camera_available:
        return Response("Camera not available", status=503)

    def generate():
        while True:
            frame_event.wait()
            frame_event.clear()
            with frame_lock:
                data = shared_jpeg
            if data is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/world_coord", methods=["POST"])
def world_coord():
    if not camera_available:
        return jsonify({"error": "Camera not available"}), 503
    data = request.get_json(force=True)
    u, v = float(data["u"]), float(data["v"])
    with pose_lock:
        R, t = shared_pose
    if R is None:
        return jsonify({"error": "NO TAG"})
    pt = pixels_to_world(np.array([[u, v]]), R, t)
    return jsonify(x_mm=round(float(pt[0, 0]), 1),
                   y_mm=round(float(pt[0, 1]), 1))


@app.route("/capture", methods=["POST"])
def capture():
    body = request.get_json(force=True)
    pt_type = body.get("type", "inside")
    if pt_type not in ("inside", "outside"):
        return jsonify({"error": "type must be 'inside' or 'outside'"}), 400

    cnc_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    if cnc.connected:
        status = cnc.query_status()
        if status:
            cnc_pos = status["wpos"]

    point = {
        "type": pt_type,
        "x": cnc_pos["x"],
        "y": cnc_pos["y"],
        "a": cnc_pos["z"],
        "timestamp": datetime.now().isoformat(),
    }
    points.append(point)
    return jsonify({"point": point, "index": len(points) - 1})


@app.route("/points", methods=["GET"])
def get_points():
    return jsonify(points)


@app.route("/points", methods=["DELETE"])
def clear_points():
    points.clear()
    return jsonify({"status": "cleared"})


@app.route("/points/last", methods=["DELETE"])
def undo_last():
    if not points:
        return jsonify({"error": "no points to undo"}), 400
    removed = points.pop()
    return jsonify({"removed": removed, "remaining": len(points)})


@app.route("/points/<int:idx>", methods=["DELETE"])
def delete_point(idx):
    if idx < 0 or idx >= len(points):
        return jsonify({"error": "index out of range"}), 400
    removed = points.pop(idx)
    return jsonify({"removed": removed, "remaining": len(points)})


@app.route("/export")
def export_json():
    data = json.dumps(points, indent=2)
    return Response(
        data,
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=string_holes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"},
    )


@app.route("/import", methods=["POST"])
def import_json():
    body = request.get_json(force=True)
    if not isinstance(body, list):
        return jsonify({"error": "expected a JSON array of points"}), 400
    imported = 0
    for p in body:
        if not isinstance(p, dict):
            continue
        # Accept "a" key, fall back to "z" for backwards compatibility
        a_val = p.get("a", p.get("z", 0))
        points.append({
            "type": p.get("type", "inside"),
            "x": float(p.get("x", 0)),
            "y": float(p.get("y", 0)),
            "a": float(a_val),
            "timestamp": p.get("timestamp", datetime.now().isoformat()),
        })
        imported += 1
    return jsonify({"imported": imported, "total": len(points)})


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>String Hole Capture</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  h1 { padding: 12px 20px; background: #16213e; font-size: 1.3rem; }

  .status-bar { padding: 10px 20px; background: #0f3460; font-family: monospace;
                 display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
  .status-bar .pos { color: #8ab4f8; }
  .status-bar .state { font-weight: bold; }
  .state-idle { color: #4caf50; }
  .state-run { color: #ff9800; }
  .state-alarm { color: #f44336; }
  .tag-ok { color: #4caf50; }
  .tag-none { color: #f44336; }

  .main-row { display: flex; gap: 12px; padding: 12px 20px; }
  .camera-panel { flex: 3; background: #16213e; border-radius: 8px; padding: 12px;
                  min-width: 0; position: relative; }
  .camera-panel h3 { margin-bottom: 8px; font-size: 0.95rem; color: #8ab4f8; }
  .controls-panel { flex: 1; background: #16213e; border-radius: 8px; padding: 12px;
                    display: flex; flex-direction: column; gap: 10px; min-width: 180px; }
  .controls-panel h3 { margin-bottom: 4px; font-size: 0.95rem; color: #8ab4f8; }

  .wrap { position: relative; line-height: 0; width: 100%; aspect-ratio: 1920/1080;
          background: #111; border-radius: 4px; overflow: hidden; }
  .wrap img { display: block; width: 100%; height: 100%; object-fit: contain; }
  .wrap canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  .no-camera-placeholder { display: flex; align-items: center; justify-content: center;
                           background: #111; border-radius: 4px; height: 300px;
                           color: #666; font-size: 1.1rem; font-family: monospace; }

  #coord-tooltip {
    position: fixed; pointer-events: none; z-index: 9999;
    font-family: monospace; font-size: 13px; padding: 4px 8px;
    background: rgba(0,0,0,0.85); border-radius: 4px;
    white-space: nowrap; display: none; color: #0f0;
  }

  button { padding: 8px 18px; border: none; border-radius: 6px; font-weight: 600;
           cursor: pointer; font-size: 0.9rem; transition: opacity 0.15s; width: 100%; }
  button:hover { opacity: 0.85; }

  .type-toggle { display: flex; border-radius: 6px; overflow: hidden; }
  .type-toggle button { border-radius: 0; padding: 8px 0; width: 50%; }
  .btn-inside  { background: #333; color: #aaa; }
  .btn-outside { background: #333; color: #aaa; }
  .btn-inside.active  { background: #00bcd4; color: #fff; }
  .btn-outside.active { background: #ff9800; color: #fff; }

  .btn-capture { background: #4caf50; color: #fff; font-size: 1rem; padding: 10px 18px; }
  .btn-undo    { background: #607d8b; color: #fff; }
  .btn-clear   { background: #f44336; color: #fff; }
  .btn-export  { background: #9c27b0; color: #fff; }
  .btn-import  { background: #3f51b5; color: #fff; }
  .arduino-section-label { font-size: 0.8rem; color: #888; border-top: 1px solid #333;
                           padding-top: 8px; margin-top: 2px; }
  .arduino-conn { font-size: 0.75rem; }
  .motor-row { display: flex; align-items: center; gap: 4px; }
  .motor-row label { font-size: 0.78rem; min-width: 38px; color: #ccc; }
  .motor-input { width: 56px; padding: 5px 4px; border: 1px solid #444; border-radius: 4px;
                 background: #222; color: #e0e0e0; font-size: 0.82rem; text-align: center; }
  .motor-input:disabled { opacity: 0.4; }
  .btn-motor { background: #009688; color: #fff; padding: 6px 8px; font-size: 0.78rem;
               width: auto; min-width: 0; flex: 1; }
  .btn-motor:disabled { background: #444; color: #777; cursor: not-allowed; opacity: 0.6; }
  .btn-dc-fwd { background: #4caf50; color: #fff; }
  .btn-dc-rev { background: #ff9800; color: #fff; }
  .btn-dc-stop { background: #f44336; color: #fff; }
  .btn-reset { background: #607d8b; color: #fff; font-size: 0.78rem; width: auto; flex: 1; }

  .kbd { display: inline-block; background: #333; border: 1px solid #555; border-radius: 4px;
         padding: 1px 7px; font-family: monospace; font-size: 0.75rem; color: #aaa; }

  .bottom-row { display: flex; gap: 12px; padding: 0 20px 12px 20px; flex-wrap: wrap; }
  .panel { background: #16213e; border-radius: 8px; padding: 12px; min-width: 300px; }
  .panel h3 { margin-bottom: 8px; font-size: 0.95rem; color: #8ab4f8; }
  .panel-plot { flex: 2; }
  .panel-table { flex: 1; }

  #plot { width: 100%; height: 400px; }

  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 8px; }
  th, td { padding: 5px 8px; text-align: left; border-bottom: 1px solid #333; }
  th { color: #8ab4f8; }
  .del-btn { background: #c62828; color: #fff; border: none; border-radius: 4px;
             padding: 2px 8px; cursor: pointer; font-size: 0.75rem; width: auto; }
  .del-btn:hover { opacity: 0.8; }
  .badge-inside  { color: #00bcd4; }
  .badge-outside { color: #ff9800; }

  input[type=file] { display: none; }
</style>
</head><body>

<h1>String Hole Capture</h1>

<div class="status-bar">
  <span>CNC: <span class="pos" id="cnc-pos">X=?.?? Y=?.?? A(angle)=?.??&deg;</span></span>
  <span style="color:#666; margin-left:4px;">|</span>
  <span class="pos" id="arduino-pos">X:-- Z:-- B1:-- B2:--</span>
  <span class="state" id="cnc-state">--</span>
  <span style="color:#666" id="cnc-conn">disconnected</span>
  <a href="/sequencer" style="margin-left:auto; color:#ce93d8; text-decoration:none; font-weight:600;">Block Sequencer</a>
  <a href="/navigator" style="color:#8ab4f8; text-decoration:none; font-weight:600;">Point Navigator &rarr;</a>
  <span id="tag-status"></span>
</div>

<div class="main-row">
  <div class="camera-panel" id="camera-panel">
    <h3>Camera Feed</h3>
    <div class="wrap" id="camera-wrap" style="display:none">
      <img id="stream" src="/feed">
      <canvas id="overlay"></canvas>
    </div>
    <div class="no-camera-placeholder" id="no-camera-msg">No Camera</div>
  </div>
  <div class="controls-panel">
    <h3>Controls</h3>
    <div class="type-toggle">
      <button class="btn-inside active" id="btn-inside" onclick="setType('inside')">Inside</button>
      <button class="btn-outside" id="btn-outside" onclick="setType('outside')">Outside</button>
    </div>
    <button class="btn-capture" onclick="capturePoint()">Capture <span class="kbd">Space</span></button>
    <button class="btn-undo" onclick="undoLast()">Undo</button>
    <button class="btn-clear" onclick="clearAll()">Clear All</button>
    <button class="btn-export" onclick="exportJSON()">Export</button>
    <button class="btn-import" onclick="document.getElementById('import-file').click()">Import</button>
    <input type="file" id="import-file" accept=".json" onchange="importJSON(this)">
    <div class="arduino-section-label">Arduino Motors <span class="arduino-conn" id="arduino-conn-label">(disconnected)</span></div>
    <div class="motor-row"><label>Z</label><input type="number" class="motor-input" id="steps-z" value="100" min="1"><button class="btn-motor" onclick="moveMotor('z',1)">&#x25B2; Up</button><button class="btn-motor" onclick="moveMotor('z',-1)">&#x25BC; Down</button></div>
    <div class="motor-row"><label>X</label><input type="number" class="motor-input" id="steps-x" value="100" min="1"><button class="btn-motor" onclick="moveMotor('x',1)">&#x2192; Fwd</button><button class="btn-motor" onclick="moveMotor('x',-1)">&#x2190; Back</button></div>
    <div class="motor-row"><label>BYJ1</label><input type="number" class="motor-input" id="steps-byj1" value="100" min="1"><button class="btn-motor" onclick="moveMotor('byj1',1)">CW</button><button class="btn-motor" onclick="moveMotor('byj1',-1)">CCW</button></div>
    <div class="motor-row"><label>BYJ2</label><input type="number" class="motor-input" id="steps-byj2" value="100" min="1"><button class="btn-motor" onclick="moveMotor('byj2',1)">CW</button><button class="btn-motor" onclick="moveMotor('byj2',-1)">CCW</button></div>
    <div class="motor-row"><label>Servo</label><input type="number" class="motor-input" id="servo-angle" value="90" min="0" max="180"><button class="btn-motor" onclick="sendServo()">Set</button></div>
    <div class="motor-row"><label>DC</label><input type="number" class="motor-input" id="dc-speed" value="50" min="0" max="100"><button class="btn-motor btn-dc-fwd" onclick="dcControl('forward')">Fwd</button><button class="btn-motor btn-dc-rev" onclick="dcControl('reverse')">Rev</button><button class="btn-motor btn-dc-stop" onclick="dcControl('stop')">Stop</button></div>
    <div class="motor-row"><button class="btn-reset btn-motor" onclick="resetArduino('steppers')">Reset Steppers</button><button class="btn-reset btn-motor" onclick="resetArduino('all')">Reset All</button></div>
  </div>
</div>

<div class="bottom-row">
  <div class="panel panel-plot">
    <h3>Scatter Plot (X vs Y)</h3>
    <div id="plot"></div>
  </div>
  <div class="panel panel-table">
    <h3>Captured Points</h3>
    <table>
      <thead><tr><th>#</th><th>Type</th><th>X</th><th>Y</th><th>A&deg;</th><th>Time</th><th></th></tr></thead>
      <tbody id="pts-body"></tbody>
    </table>
  </div>
</div>

<div id="coord-tooltip"></div>

<script>
const CAMERA_AVAILABLE = __CAMERA_AVAILABLE__;
let ARDUINO_CONNECTED = __ARDUINO_CONNECTED__;
const CAM_W = 1920, CAM_H = 1080;
let currentType = 'inside';
let plotInitialized = false;

// --- Camera setup ---
if (CAMERA_AVAILABLE) {
  document.getElementById('camera-wrap').style.display = '';
  document.getElementById('no-camera-msg').style.display = 'none';
} else {
  document.getElementById('camera-wrap').style.display = 'none';
  document.getElementById('no-camera-msg').style.display = 'flex';
}

// --- Hover tooltip ---
const tooltip = document.getElementById('coord-tooltip');
const overlay = document.getElementById('overlay');
const streamImg = document.getElementById('stream');
let wcAborter = null;
let wcLastTime = 0;
const WC_INTERVAL = 100;

function scaleF() { return streamImg.clientWidth / CAM_W; }

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

if (CAMERA_AVAILABLE) {
  overlay.addEventListener('mousemove', e => {
    const rect = overlay.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const sc = scaleF();
    const camU = mx / sc, camV = my / sc;
    fetchWorldCoord(camU, camV, e.clientX, e.clientY);
  });
  overlay.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

  // Resize canvas to match img
  function resizeOverlay() {
    overlay.width = streamImg.clientWidth;
    overlay.height = streamImg.clientHeight;
  }
  streamImg.addEventListener('load', resizeOverlay);
  window.addEventListener('resize', resizeOverlay);
}

// --- CNC polling ---
function pollCNC() {
  fetch('/cnc/status').then(r => r.json()).then(s => {
    const p = s.wpos;
    document.getElementById('cnc-pos').textContent =
      `X=${p.x.toFixed(2)} Y=${p.y.toFixed(2)} A(angle)=${p.z.toFixed(2)}\u00B0`;
    const st = document.getElementById('cnc-state');
    st.textContent = s.state;
    st.className = 'state state-' + s.state.toLowerCase();
    document.getElementById('cnc-conn').textContent = s.connected ? 'connected' : 'disconnected';

    // Tag status
    const tagEl = document.getElementById('tag-status');
    if (s.camera_available) {
      if (s.tag_detected) {
        tagEl.innerHTML = '<span class="tag-ok">Tag: detected</span>';
      } else {
        tagEl.innerHTML = '<span class="tag-none">Tag: none</span>';
      }
    } else {
      tagEl.textContent = '';
    }
  }).catch(() => {});

  // Poll Arduino motor controller
  fetch('/arduino/status').then(r => r.json()).then(s => {
    ARDUINO_CONNECTED = s.connected;
    const posEl = document.getElementById('arduino-pos');
    const p = s.positions;
    if (s.connected) {
      posEl.textContent = `X:${p.x} Z:${p.z} B1:${p.byj1} B2:${p.byj2}`;
    } else {
      posEl.textContent = 'X:-- Z:-- B1:-- B2:--';
    }
    const label = document.getElementById('arduino-conn-label');
    label.textContent = s.connected ? '(connected)' : '(disconnected)';
    label.style.color = s.connected ? '#4caf50' : '#888';
    document.querySelectorAll('.btn-motor').forEach(b => b.disabled = !s.connected);
    document.querySelectorAll('.motor-input').forEach(i => i.disabled = !s.connected);
  }).catch(() => {});
}
setInterval(pollCNC, 500);
pollCNC();

// --- Type toggle ---
function setType(t) {
  currentType = t;
  document.getElementById('btn-inside').className =
    'btn-inside' + (t === 'inside' ? ' active' : '');
  document.getElementById('btn-outside').className =
    'btn-outside' + (t === 'outside' ? ' active' : '');
}

// --- Capture ---
function capturePoint() {
  fetch('/capture', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({type: currentType}),
  }).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); return; }
    refreshPoints();
    refreshPlot();
  }).catch(e => alert('Capture failed: ' + e));
}

// --- Refresh table ---
function refreshPoints() {
  fetch('/points').then(r => r.json()).then(data => {
    const tb = document.getElementById('pts-body');
    tb.innerHTML = '';
    data.forEach((p, i) => {
      const tr = document.createElement('tr');
      const badge = p.type === 'inside' ? 'badge-inside' : 'badge-outside';
      tr.innerHTML = `<td>${i + 1}</td>` +
        `<td class="${badge}">${p.type}</td>` +
        `<td>${p.x.toFixed(2)}</td>` +
        `<td>${p.y.toFixed(2)}</td>` +
        `<td>${p.a.toFixed(2)}</td>` +
        `<td>${p.timestamp.slice(11, 19)}</td>` +
        `<td><button class="del-btn" onclick="deletePoint(${i})">&#x2715;</button></td>`;
      tb.appendChild(tr);
    });
  });
}

// --- Refresh plot ---
function refreshPlot() {
  fetch('/points').then(r => r.json()).then(data => {
    const ins = data.filter(p => p.type === 'inside');
    const out = data.filter(p => p.type === 'outside');

    const traceIn = {
      x: ins.map(p => p.x),
      y: ins.map(p => p.y),
      text: ins.map((p, i) => {
        const idx = data.indexOf(p) + 1;
        return `#${idx} (inside)<br>X: ${p.x.toFixed(2)} mm<br>Y: ${p.y.toFixed(2)} mm<br>A: ${p.a.toFixed(2)}\u00B0`;
      }),
      hoverinfo: 'text',
      mode: 'markers',
      type: 'scatter',
      name: 'Inside',
      marker: { color: '#00bcd4', size: 10, symbol: 'circle' },
    };
    const traceOut = {
      x: out.map(p => p.x),
      y: out.map(p => p.y),
      text: out.map((p, i) => {
        const idx = data.indexOf(p) + 1;
        return `#${idx} (outside)<br>X: ${p.x.toFixed(2)} mm<br>Y: ${p.y.toFixed(2)} mm<br>A: ${p.a.toFixed(2)}\u00B0`;
      }),
      hoverinfo: 'text',
      mode: 'markers',
      type: 'scatter',
      name: 'Outside',
      marker: { color: '#ff9800', size: 10, symbol: 'diamond' },
    };

    const layout = {
      paper_bgcolor: '#16213e',
      plot_bgcolor: '#16213e',
      font: { color: '#e0e0e0' },
      xaxis: { title: 'X (mm)', gridcolor: '#333', zerolinecolor: '#555',
               scaleanchor: 'y', scaleratio: 1 },
      yaxis: { title: 'Y (mm)', gridcolor: '#333', zerolinecolor: '#555' },
      legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' },
      margin: { l: 60, r: 20, t: 20, b: 50 },
    };

    Plotly.react('plot', [traceIn, traceOut], layout);
    plotInitialized = true;
  });
}

// --- Undo ---
function undoLast() {
  fetch('/points/last', {method: 'DELETE'}).then(r => r.json()).then(data => {
    if (data.error) return;
    refreshPoints();
    refreshPlot();
  });
}

// --- Clear ---
function clearAll() {
  if (!confirm('Clear all captured points?')) return;
  fetch('/points', {method: 'DELETE'}).then(() => {
    refreshPoints();
    refreshPlot();
  });
}

// --- Delete specific ---
function deletePoint(i) {
  fetch(`/points/${i}`, {method: 'DELETE'}).then(r => r.json()).then(data => {
    if (data.error) return;
    refreshPoints();
    refreshPlot();
  });
}

// --- Export ---
function exportJSON() {
  window.location.href = '/export';
}

// --- Import ---
function importJSON(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    let data;
    try { data = JSON.parse(e.target.result); } catch { alert('Invalid JSON file'); return; }
    fetch('/import', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data),
    }).then(r => r.json()).then(res => {
      if (res.error) { alert(res.error); return; }
      refreshPoints();
      refreshPlot();
    }).catch(err => alert('Import failed: ' + err));
  };
  reader.readAsText(file);
  input.value = '';
}

// --- Arduino motor controls ---
function moveMotor(motor, direction) {
  if (!ARDUINO_CONNECTED) return;
  const input = document.getElementById('steps-' + motor);
  const steps = parseInt(input.value, 10) * direction;
  fetch('/arduino/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({motor: motor, steps: steps}),
  }).then(r => r.json()).catch(() => {});
}
function sendServo() {
  if (!ARDUINO_CONNECTED) return;
  const angle = parseInt(document.getElementById('servo-angle').value, 10);
  fetch('/arduino/servo', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({angle: angle}),
  }).then(r => r.json()).catch(() => {});
}
function dcControl(action) {
  if (!ARDUINO_CONNECTED) return;
  const speed = parseInt(document.getElementById('dc-speed').value, 10);
  fetch('/arduino/dc', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: action, speed: speed}),
  }).then(r => r.json()).catch(() => {});
}
function resetArduino(mode) {
  if (!ARDUINO_CONNECTED) return;
  fetch('/arduino/reset', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: mode}),
  }).then(r => r.json()).catch(() => {});
}

// --- Keyboard shortcut ---
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
    e.preventDefault();
    capturePoint();
  }
});

// Initial load
refreshPoints();
refreshPlot();
</script>
</body></html>"""


@app.route("/navigator")
def navigator():
    return NAVIGATOR_HTML


# ---------------------------------------------------------------------------
# Navigator HTML
# ---------------------------------------------------------------------------
NAVIGATOR_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Point Navigator</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  h1 { padding: 12px 20px; background: #16213e; font-size: 1.3rem; }

  .status-bar { padding: 10px 20px; background: #0f3460; font-family: monospace;
                 display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
  .status-bar .pos { color: #8ab4f8; }
  .status-bar .state { font-weight: bold; }
  .state-idle { color: #4caf50; }
  .state-run { color: #ff9800; }
  .state-jog { color: #ff9800; }
  .state-alarm { color: #f44336; }

  .nav-bar { padding: 10px 20px; background: #16213e; display: flex; gap: 16px;
             align-items: center; flex-wrap: wrap; border-bottom: 1px solid #333; }
  .nav-bar a { color: #8ab4f8; text-decoration: none; font-weight: 600; font-size: 0.9rem; }
  .nav-bar a:hover { text-decoration: underline; }
  .btn-cancel { background: #f44336; color: #fff; border: none; border-radius: 6px;
                padding: 7px 18px; font-weight: 600; cursor: pointer; font-size: 0.85rem; }
  .btn-cancel:hover { opacity: 0.85; }
  .btn-cancel:disabled { background: #555; color: #888; cursor: not-allowed; }
  .nav-status { font-family: monospace; font-size: 0.85rem; color: #aaa; flex: 1; }
  .feedrate-input { width: 70px; padding: 5px 6px; border: 1px solid #444; border-radius: 4px;
                    background: #222; color: #e0e0e0; font-size: 0.85rem; text-align: center; }
  .feedrate-label { font-size: 0.85rem; color: #aaa; }

  .bottom-row { display: flex; gap: 12px; padding: 12px 20px; flex-wrap: wrap; }
  .panel { background: #16213e; border-radius: 8px; padding: 12px; min-width: 300px; }
  .panel h3 { margin-bottom: 8px; font-size: 0.95rem; color: #8ab4f8; }
  .panel-plot { flex: 2; }
  .panel-table { flex: 1; max-height: 500px; overflow-y: auto; }

  #plot { width: 100%; height: 450px; }

  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 8px; }
  th, td { padding: 5px 8px; text-align: left; border-bottom: 1px solid #333; }
  th { color: #8ab4f8; }
  .badge-inside  { color: #00bcd4; }
  .badge-outside { color: #ff9800; }

  .btn-go { background: #4caf50; color: #fff; border: none; border-radius: 4px;
            padding: 3px 12px; cursor: pointer; font-size: 0.78rem; font-weight: 600; }
  .btn-go:hover { opacity: 0.85; }
  .btn-go:disabled { background: #555; color: #888; cursor: not-allowed; }

  .row-active { background: rgba(76, 175, 80, 0.15); }
</style>
</head><body>

<h1>Point Navigator</h1>

<div class="status-bar">
  <span>CNC: <span class="pos" id="cnc-pos">X=?.?? Y=?.?? A=?.??&deg;</span></span>
  <span class="state" id="cnc-state">--</span>
  <span style="color:#666" id="cnc-conn">disconnected</span>
</div>

<div class="nav-bar">
  <a href="/">&larr; Back to Capture</a>
  <a href="/sequencer">Block Sequencer</a>
  <button class="btn-cancel" id="btn-cancel" onclick="cancelMove()" disabled>Cancel Move</button>
  <span class="nav-status" id="nav-status"></span>
  <span class="feedrate-label">Feedrate:</span>
  <input type="number" class="feedrate-input" id="feedrate" value="1000" min="100" max="5000">
  <span class="feedrate-label">mm/min</span>
</div>

<div class="bottom-row">
  <div class="panel panel-plot">
    <h3>Click a point to move CNC</h3>
    <div id="plot"></div>
  </div>
  <div class="panel panel-table">
    <h3>Captured Points</h3>
    <table>
      <thead><tr><th>#</th><th>Type</th><th>X</th><th>Y</th><th>A&deg;</th><th></th></tr></thead>
      <tbody id="pts-body"></tbody>
    </table>
  </div>
</div>

<script>
let allPoints = [];
let cncConnected = false;
let cncState = 'Unknown';
let activeIdx = -1;
let moving = false;

// --- Poll CNC status ---
function pollCNC() {
  fetch('/cnc/status').then(r => r.json()).then(s => {
    const p = s.wpos;
    document.getElementById('cnc-pos').textContent =
      `X=${p.x.toFixed(2)} Y=${p.y.toFixed(2)} A=${p.z.toFixed(2)}\u00B0`;
    const st = document.getElementById('cnc-state');
    cncState = s.state;
    st.textContent = s.state;
    st.className = 'state state-' + s.state.toLowerCase();
    cncConnected = s.connected;
    document.getElementById('cnc-conn').textContent = s.connected ? 'connected' : 'disconnected';

    // Detect arrival (was moving, now Idle)
    if (moving && s.state === 'Idle') {
      moving = false;
      document.getElementById('nav-status').textContent = `Arrived at #${activeIdx + 1}`;
      document.getElementById('btn-cancel').disabled = true;
    }

    // Update Go button states
    document.querySelectorAll('.btn-go').forEach(b => b.disabled = !s.connected);
  }).catch(() => {});
}
setInterval(pollCNC, 500);
pollCNC();

// --- Fetch and render points ---
function refreshPoints() {
  fetch('/points').then(r => r.json()).then(data => {
    allPoints = data;
    renderTable();
    renderPlot();
  });
}

function renderTable() {
  const tb = document.getElementById('pts-body');
  tb.innerHTML = '';
  allPoints.forEach((p, i) => {
    const tr = document.createElement('tr');
    if (i === activeIdx) tr.className = 'row-active';
    const badge = p.type === 'inside' ? 'badge-inside' : 'badge-outside';
    tr.innerHTML = `<td>${i + 1}</td>` +
      `<td class="${badge}">${p.type}</td>` +
      `<td>${p.x.toFixed(2)}</td>` +
      `<td>${p.y.toFixed(2)}</td>` +
      `<td>${p.a.toFixed(2)}</td>` +
      `<td><button class="btn-go" onclick="goToPoint(${i})"${!cncConnected ? ' disabled' : ''}>Go</button></td>`;
    tb.appendChild(tr);
  });
}

function renderPlot() {
  const ins = allPoints.filter(p => p.type === 'inside');
  const out = allPoints.filter(p => p.type === 'outside');

  function makeTrace(pts, name, color, symbol) {
    return {
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      customdata: pts.map(p => [allPoints.indexOf(p), p.x, p.y, p.a]),
      text: pts.map(p => {
        const idx = allPoints.indexOf(p) + 1;
        return `#${idx} (${p.type})<br>X: ${p.x.toFixed(2)}<br>Y: ${p.y.toFixed(2)}<br>A: ${p.a.toFixed(2)}\u00B0`;
      }),
      hoverinfo: 'text',
      mode: 'markers',
      type: 'scatter',
      name: name,
      marker: { color: color, size: 10, symbol: symbol },
    };
  }

  const traces = [
    makeTrace(ins, 'Inside', '#00bcd4', 'circle'),
    makeTrace(out, 'Outside', '#ff9800', 'diamond'),
  ];

  const layout = {
    paper_bgcolor: '#16213e',
    plot_bgcolor: '#16213e',
    font: { color: '#e0e0e0' },
    xaxis: { title: 'X (mm)', gridcolor: '#333', zerolinecolor: '#555',
             scaleanchor: 'y', scaleratio: 1 },
    yaxis: { title: 'Y (mm)', gridcolor: '#333', zerolinecolor: '#555' },
    legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' },
    margin: { l: 60, r: 20, t: 20, b: 50 },
  };

  Plotly.react('plot', traces, layout);

  // Attach click handler
  const plotEl = document.getElementById('plot');
  plotEl.removeAllListeners && plotEl.removeAllListeners('plotly_click');
  plotEl.on('plotly_click', function(data) {
    if (data.points.length > 0) {
      const cd = data.points[0].customdata;
      if (cd) goToPoint(cd[0]);
    }
  });
}

// --- Go to point ---
async function goToPoint(idx) {
  if (!cncConnected || idx < 0 || idx >= allPoints.length) return;
  const p = allPoints[idx];
  const feedrate = clampFeedrate();

  // Cancel current move if in progress
  if (moving) {
    await fetch('/cnc/jog/cancel', { method: 'POST' });
    await new Promise(r => setTimeout(r, 200));
  }

  activeIdx = idx;
  moving = true;
  renderTable();
  document.getElementById('nav-status').textContent = `Moving to #${idx + 1}...`;
  document.getElementById('btn-cancel').disabled = false;

  try {
    const resp = await fetch('/cnc/jog', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: p.x, y: p.y, a: p.a, feedrate: feedrate }),
    });
    const data = await resp.json();
    if (data.error) {
      moving = false;
      document.getElementById('nav-status').textContent = `Error: ${data.error}`;
      document.getElementById('btn-cancel').disabled = true;
    }
  } catch (e) {
    moving = false;
    document.getElementById('nav-status').textContent = `Error: ${e.message}`;
    document.getElementById('btn-cancel').disabled = true;
  }
}

// --- Cancel move ---
async function cancelMove() {
  try {
    await fetch('/cnc/jog/cancel', { method: 'POST' });
  } catch (e) {}
  moving = false;
  document.getElementById('nav-status').textContent = 'Move cancelled';
  document.getElementById('btn-cancel').disabled = true;
}

// --- Feedrate helper ---
function clampFeedrate() {
  const el = document.getElementById('feedrate');
  let v = parseInt(el.value, 10);
  if (isNaN(v) || v < 100) v = 100;
  if (v > 5000) v = 5000;
  el.value = v;
  return v;
}

// --- Keyboard: Escape to cancel ---
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    cancelMove();
  }
});

// Initial load
refreshPoints();
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Sequencer routes
# ---------------------------------------------------------------------------
@app.route("/sequencer")
def sequencer():
    return SEQUENCER_HTML


@app.route("/sequencer/save", methods=["POST"])
def sequencer_save():
    body = request.get_json(force=True)
    name = body.get("name", "").strip()
    blocks = body.get("blocks", [])
    if not name:
        return jsonify({"error": "Name required"}), 400
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    path = os.path.join("data/sequences", safe + ".json")
    with open(path, "w") as f:
        json.dump({"name": name, "blocks": blocks}, f, indent=2)
    return jsonify({"ok": True})


@app.route("/sequencer/list")
def sequencer_list():
    names = []
    for fname in sorted(os.listdir("data/sequences")):
        if fname.endswith(".json"):
            names.append(fname[:-5])
    return jsonify(names)


@app.route("/sequencer/load")
def sequencer_load():
    name = request.args.get("name", "").strip()
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    path = os.path.join("data/sequences", safe + ".json")
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/sequencer/macros/save", methods=["POST"])
def sequencer_macro_save():
    body = request.get_json(force=True)
    name = body.get("name", "").strip()
    blocks = body.get("blocks", [])
    if not name:
        return jsonify({"error": "Name required"}), 400
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    path = os.path.join("data/macros", safe + ".json")
    with open(path, "w") as f:
        json.dump({"name": name, "blocks": blocks}, f, indent=2)
    return jsonify({"ok": True})


@app.route("/sequencer/macros")
def sequencer_macros():
    macros = []
    for fname in sorted(os.listdir("data/macros")):
        if fname.endswith(".json"):
            with open(os.path.join("data/macros", fname)) as f:
                macros.append(json.load(f))
    return jsonify(macros)


# ---------------------------------------------------------------------------
# Sequencer HTML
# ---------------------------------------------------------------------------
SEQUENCER_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Block Sequencer</title>
<script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.6/Sortable.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body { font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; display: flex; flex-direction: column;
         overflow: hidden; }

  h1 { padding: 10px 20px; background: #16213e; font-size: 1.2rem; display: flex; align-items: center; gap: 16px; }
  h1 .status-info { font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #8ab4f8; }
  h1 .nav-links { margin-left: auto; display: flex; gap: 12px; }
  h1 .nav-links a { color: #8ab4f8; text-decoration: none; font-size: 0.85rem; font-weight: 600; }
  h1 .nav-links a:hover { text-decoration: underline; }

  .workspace { display: flex; flex: 1; overflow: hidden; }

  /* Palette sidebar */
  .palette { width: 220px; min-width: 220px; background: #0f3460; overflow-y: auto;
             padding: 10px; border-right: 1px solid #333; }
  .palette-category { margin-bottom: 14px; }
  .palette-category h3 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;
                          color: #888; margin-bottom: 6px; padding: 0 4px; }
  .palette-block { background: #16213e; border-radius: 6px; padding: 8px 10px; margin-bottom: 5px;
                   cursor: grab; font-size: 0.8rem; font-weight: 600; border-top: 3px solid;
                   opacity: 0.85; transition: opacity 0.15s; user-select: none; }
  .palette-block:hover { opacity: 1; }
  .palette-block.sortable-ghost { opacity: 0.4; }

  /* Canvas */
  .canvas-wrap { flex: 1; position: relative; overflow: hidden; }
  .canvas { width: 100%; height: 100%; overflow: hidden; background: #1a1a2e; position: relative; cursor: grab;
            background-image: radial-gradient(circle, #2a2a4e 1px, transparent 1px);
            background-size: 24px 24px; }
  .canvas.panning { cursor: grabbing; }
  .canvas-inner { position: absolute; top: 0; left: 0; transform-origin: 0 0;
                  min-width: 200px; padding: 16px; }
  .canvas-empty { color: #555; text-align: center; padding-top: 80px; font-size: 0.95rem;
                  position: absolute; top: 0; left: 0; right: 0; pointer-events: none; z-index: 1; }
  .zoom-controls { position: absolute; bottom: 10px; right: 10px; display: flex; gap: 0;
                   background: #16213e; border: 1px solid #333; border-radius: 6px; z-index: 2;
                   font-family: 'Fira Code', monospace; user-select: none; overflow: hidden; }
  .zoom-controls button, .zoom-controls span { background: none; border: none; color: #888;
                   font-size: 0.78rem; padding: 5px 10px; cursor: pointer; }
  .zoom-controls button:hover { background: #0f3460; color: #ccc; }
  .zoom-controls span { cursor: pointer; min-width: 44px; text-align: center; border-left: 1px solid #333;
                        border-right: 1px solid #333; }
  .zoom-controls span:hover { color: #ccc; }
  .canvas-inner .block { background: #16213e; border-radius: 8px; margin-bottom: 8px; border-top: 4px solid;
                         transition: box-shadow 0.15s, opacity 0.15s; position: relative; }
  .canvas-inner .block.selected { box-shadow: 0 0 0 2px #4caf50; }
  .canvas-inner .block.running { box-shadow: 0 0 0 2px #ff9800; animation: pulse 1s infinite; }
  .canvas-inner .block.error { box-shadow: 0 0 0 2px #f44336; }
  .canvas-inner .block.done { opacity: 0.5; }
  .canvas-inner .block .block-header { padding: 8px 12px; font-size: 0.8rem; font-weight: 700;
                                  cursor: grab; display: flex; align-items: center; gap: 8px;
                                  user-select: none; border-radius: 4px 4px 0 0; }
  .canvas-inner .block .block-header .grip { color: #666; font-size: 0.7rem; }
  .canvas-inner .block .block-header .block-title { flex: 1; }
  .canvas-inner .block .block-header .block-num { color: #666; font-size: 0.7rem; }
  .canvas-inner .block .block-header .btn-del { background: none; border: none; color: #f44336;
                                           cursor: pointer; font-size: 1rem; padding: 0 4px;
                                           opacity: 0.5; }
  .canvas-inner .block .block-header .btn-del:hover { opacity: 1; }
  .canvas-inner .block .block-body { padding: 6px 12px 10px; display: flex; flex-wrap: wrap; gap: 8px; }
  .canvas-inner .block .block-body label { font-size: 0.75rem; color: #aaa; display: flex;
                                      align-items: center; gap: 4px; }
  .canvas-inner .block .block-body input,
  .canvas-inner .block .block-body select { background: #0d1b3e; border: 1px solid #333; border-radius: 4px;
                                       color: #e0e0e0; padding: 4px 6px; font-size: 0.8rem;
                                       font-family: 'Fira Code', monospace; width: 80px; }
  .canvas-inner .block .block-body select { width: auto; min-width: 80px; }
  .canvas-inner .block.sortable-ghost { opacity: 0.3; }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 2px #ff9800; }
    50% { box-shadow: 0 0 12px 2px #ff9800; }
  }

  /* Context menu */
  .ctx-menu { position: fixed; background: #16213e; border: 1px solid #444; border-radius: 6px;
              padding: 4px 0; z-index: 1000; display: none; min-width: 160px;
              box-shadow: 0 4px 16px rgba(0,0,0,0.5); }
  .ctx-menu div { padding: 7px 16px; font-size: 0.82rem; cursor: pointer; }
  .ctx-menu div:hover { background: #0f3460; }
  .ctx-menu .sep { height: 1px; background: #333; margin: 4px 0; padding: 0; cursor: default; }
  .ctx-menu .sep:hover { background: #333; }

  /* Toolbar */
  .toolbar { display: flex; gap: 8px; padding: 10px 16px 14px; background: #16213e;
             border-top: 1px solid #333; align-items: center; flex-wrap: wrap;
             flex-shrink: 0; }
  .toolbar button { padding: 7px 16px; border: none; border-radius: 6px; font-weight: 600;
                    font-size: 0.82rem; cursor: pointer; transition: opacity 0.15s; }
  .toolbar button:hover { opacity: 0.85; }
  .btn-play { background: #4caf50; color: #fff; }
  .btn-stop { background: #f44336; color: #fff; }
  .btn-save { background: #1565C0; color: #fff; }
  .btn-load { background: #0f3460; color: #e0e0e0; border: 1px solid #444 !important; }
  .btn-export { background: #6A1B9A; color: #fff; }
  .btn-import { background: #0f3460; color: #e0e0e0; border: 1px solid #444 !important; }
  .toolbar .spacer { flex: 1; }
  .toolbar .run-status { font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #aaa; }
</style>
</head><body>

<h1>
  Block Sequencer
  <span class="status-info" id="status-info">CNC: -- | Arduino: --</span>
  <span class="nav-links">
    <a href="/">&larr; Capture</a>
    <a href="/navigator">Navigator</a>
  </span>
</h1>

<div class="workspace">
  <div class="palette" id="palette"></div>
  <div class="canvas-wrap">
    <div class="canvas-empty" id="canvas-empty">Drag blocks from the palette to build a sequence</div>
    <div class="canvas" id="canvas">
      <div class="canvas-inner" id="canvas-inner"></div>
    </div>
    <div class="zoom-controls">
      <button onclick="zoomBy(-ZOOM_STEP)" title="Zoom out">&minus;</button>
      <span id="zoom-badge" onclick="resetView()" title="Reset view">100%</span>
      <button onclick="zoomBy(ZOOM_STEP)" title="Zoom in">&plus;</button>
    </div>
  </div>
</div>

<div class="toolbar">
  <button class="btn-play" id="btn-play" onclick="runSequence()">&#9654; Play</button>
  <button class="btn-stop" id="btn-stop" onclick="stopSequence()" disabled>&#9632; Stop</button>
  <span class="spacer"></span>
  <span class="run-status" id="run-status"></span>
  <span class="spacer"></span>
  <button class="btn-save" onclick="saveSequence()">Save</button>
  <button class="btn-load" onclick="showLoadDialog()">Load</button>
  <button class="btn-export" onclick="exportSequence()">Export</button>
  <button class="btn-import" onclick="importSequence()">Import</button>
</div>

<div class="ctx-menu" id="ctx-menu">
  <div onclick="ctxCopy()">Copy</div>
  <div onclick="ctxPaste()">Paste</div>
  <div class="sep"></div>
  <div onclick="ctxDuplicate()">Duplicate</div>
  <div onclick="ctxDelete()">Delete</div>
  <div class="sep"></div>
  <div onclick="ctxSaveAsMacro()">Save as Macro</div>
</div>

<input type="file" id="file-import" accept=".json" style="display:none" onchange="handleImport(event)">

<script>
// =========================================================================
// Block Type Registry
// =========================================================================

async function waitForCncIdle(timeoutSec) {
  const deadline = Date.now() + timeoutSec * 1000;
  await new Promise(r => setTimeout(r, 300));
  while (Date.now() < deadline) {
    if (stopFlag) return;
    const r = await fetch('/cnc/status');
    const d = await r.json();
    if (d.state === 'Idle') return;
    await new Promise(r => setTimeout(r, 200));
  }
  throw new Error('Timeout waiting for CNC idle');
}

async function waitForArduinoMotorStop(motor, timeoutSec) {
  const deadline = Date.now() + timeoutSec * 1000;
  await new Promise(r => setTimeout(r, 400));
  let lastPos = null;
  let stableCount = 0;
  while (Date.now() < deadline) {
    if (stopFlag) return;
    const r = await fetch('/arduino/status');
    const d = await r.json();
    const pos = d.positions ? d.positions[motor] : null;
    if (pos !== null && pos === lastPos) {
      stableCount++;
      if (stableCount >= 3) return;
    } else {
      stableCount = 0;
    }
    lastPos = pos;
    await new Promise(r => setTimeout(r, 300));
  }
  throw new Error('Timeout waiting for Arduino motor to stop');
}

const BLOCK_TYPES = {
  'cnc-goto': {
    name: 'CNC Go To', category: 'CNC', color: '#1565C0',
    params: [
      { key: 'point', label: 'Point', type: 'point-select' },
      { key: 'x', label: 'X', type: 'number', default: 0, step: 0.1 },
      { key: 'y', label: 'Y', type: 'number', default: 0, step: 0.1 },
      { key: 'a', label: 'A', type: 'number', default: 0, step: 0.1 },
      { key: 'feedrate', label: 'Feed', type: 'number', default: 1000, step: 100 }
    ],
    execute: async (p) => {
      const r = await fetch('/cnc/jog', { method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ x: +p.x, y: +p.y, a: +p.a, feedrate: +p.feedrate })
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      await waitForCncIdle(120);
    }
  },
  'cnc-cancel': {
    name: 'CNC Cancel', category: 'CNC', color: '#1565C0', params: [],
    execute: async () => {
      await fetch('/cnc/jog/cancel', { method: 'POST' });
    }
  },
  'arduino-move': {
    name: 'Arduino Move', category: 'Arduino', color: '#2E7D32',
    params: [
      { key: 'motor', label: 'Motor', type: 'select', options: ['x','z','byj1','byj2'], default: 'z' },
      { key: 'steps', label: 'Steps', type: 'number', default: 100, step: 1 }
    ],
    execute: async (p) => {
      const r = await fetch('/arduino/move', { method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ motor: p.motor, steps: +p.steps })
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      await waitForArduinoMotorStop(p.motor, 120);
    }
  },
  'arduino-servo': {
    name: 'Arduino Servo', category: 'Arduino', color: '#2E7D32',
    params: [
      { key: 'angle', label: 'Angle', type: 'number', default: 90, min: 0, max: 180, step: 1 }
    ],
    execute: async (p) => {
      const r = await fetch('/arduino/servo', { method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ angle: +p.angle })
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
    }
  },
  'arduino-dc': {
    name: 'Arduino DC', category: 'Arduino', color: '#2E7D32',
    params: [
      { key: 'action', label: 'Action', type: 'select', options: ['forward','reverse','stop'], default: 'forward' },
      { key: 'speed', label: 'Speed', type: 'number', default: 50, min: 0, max: 255, step: 1 },
      { key: 'duration', label: 'Time (s)', type: 'number', default: 0, min: 0, step: 0.1 }
    ],
    execute: async (p) => {
      const r = await fetch('/arduino/dc', { method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ action: p.action, speed: +p.speed })
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      const dur = +p.duration;
      if (dur > 0) {
        await new Promise(r => setTimeout(r, dur * 1000));
        if (!stopFlag) {
          await fetch('/arduino/dc', { method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ action: 'stop' })
          });
        }
      }
    }
  },
  'arduino-reset': {
    name: 'Arduino Reset', category: 'Arduino', color: '#2E7D32',
    params: [
      { key: 'mode', label: 'Mode', type: 'select', options: ['all','steppers'], default: 'all' }
    ],
    execute: async (p) => {
      const r = await fetch('/arduino/reset', { method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ mode: p.mode })
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
    }
  },
  'wait': {
    name: 'Wait', category: 'Control', color: '#E65100',
    params: [
      { key: 'seconds', label: 'Seconds', type: 'number', default: 1, step: 0.1, min: 0 }
    ],
    execute: async (p) => {
      await new Promise(r => setTimeout(r, (+p.seconds) * 1000));
    }
  },
  'wait-idle': {
    name: 'Wait for Idle', category: 'Control', color: '#E65100',
    params: [
      { key: 'timeout', label: 'Timeout (s)', type: 'number', default: 30, step: 1, min: 1 }
    ],
    execute: async (p) => {
      const deadline = Date.now() + (+p.timeout) * 1000;
      while (Date.now() < deadline) {
        if (stopFlag) return;
        const r = await fetch('/cnc/status');
        const d = await r.json();
        if (d.state === 'Idle') return;
        await new Promise(r => setTimeout(r, 250));
      }
      throw new Error('Timeout waiting for idle');
    }
  },
  'macro': {
    name: 'Macro', category: 'Macros', color: '#6A1B9A',
    params: [
      { key: 'name', label: 'Macro', type: 'select', options: [], default: '' }
    ],
    execute: async (p, depth) => {
      if (depth > 10) throw new Error('Macro recursion limit (10)');
      const macro = loadedMacros.find(m => m.name === p.name);
      if (!macro) throw new Error('Macro not found: ' + p.name);
      for (const b of macro.blocks) {
        if (stopFlag) return;
        const bt = BLOCK_TYPES[b.type];
        if (bt) await bt.execute(b.params, (depth || 0) + 1);
      }
    }
  }
};

// =========================================================================
// State
// =========================================================================
let blocks = [];        // [{id, type, params:{}}]
let nextId = 1;
let selectedIds = new Set();
let clipboard = [];
let stopFlag = false;
let running = false;
let loadedMacros = [];
let cachedPoints = [];  // [{type, x, y, a, timestamp}]

// =========================================================================
// Palette rendering
// =========================================================================
function buildPalette() {
  const pal = document.getElementById('palette');
  pal.innerHTML = '';
  const cats = {};
  for (const [key, bt] of Object.entries(BLOCK_TYPES)) {
    if (!cats[bt.category]) cats[bt.category] = [];
    cats[bt.category].push({ key, ...bt });
  }
  for (const [cat, items] of Object.entries(cats)) {
    const sec = document.createElement('div');
    sec.className = 'palette-category';
    sec.innerHTML = '<h3>' + cat + '</h3>';
    const list = document.createElement('div');
    list.className = 'palette-list';
    list.dataset.category = cat;
    for (const item of items) {
      const el = document.createElement('div');
      el.className = 'palette-block';
      el.style.borderTopColor = item.color;
      el.textContent = item.name;
      el.dataset.blockType = item.key;
      list.appendChild(el);
    }
    sec.appendChild(list);
    pal.appendChild(sec);
    new Sortable(list, {
      group: { name: 'blocks', pull: 'clone', put: false },
      sort: false,
      animation: 150,
    });
  }
}

// =========================================================================
// Canvas rendering
// =========================================================================
function createBlockId() { return nextId++; }

function makeBlockFromType(typeKey) {
  const bt = BLOCK_TYPES[typeKey];
  if (!bt) return null;
  const params = {};
  for (const p of bt.params) params[p.key] = p.default;
  return { id: createBlockId(), type: typeKey, params };
}

function renderBlock(b, idx) {
  const bt = BLOCK_TYPES[b.type];
  if (!bt) return null;
  const el = document.createElement('div');
  el.className = 'block';
  el.dataset.id = b.id;
  el.style.borderTopColor = bt.color;
  if (selectedIds.has(b.id)) el.classList.add('selected');

  // Header
  const hdr = document.createElement('div');
  hdr.className = 'block-header';
  hdr.style.background = bt.color + '22';
  hdr.innerHTML = '<span class="grip">&#9776;</span>' +
    '<span class="block-title">' + bt.name + '</span>' +
    '<span class="block-num">#' + (idx + 1) + '</span>';
  const btnDel = document.createElement('button');
  btnDel.className = 'btn-del';
  btnDel.innerHTML = '&times;';
  btnDel.onclick = (e) => { e.stopPropagation(); deleteBlock(b.id); };
  hdr.appendChild(btnDel);
  el.appendChild(hdr);

  // Body with params
  if (bt.params.length > 0) {
    const body = document.createElement('div');
    body.className = 'block-body';
    for (const p of bt.params) {
      const lbl = document.createElement('label');
      lbl.textContent = p.label + ' ';
      let inp;
      if (p.type === 'point-select') {
        inp = document.createElement('select');
        const none = document.createElement('option');
        none.value = ''; none.textContent = '-- manual --';
        inp.appendChild(none);
        cachedPoints.forEach((pt, i) => {
          const opt = document.createElement('option');
          opt.value = i;
          opt.textContent = `#${i+1} ${pt.type} (${pt.x.toFixed(1)}, ${pt.y.toFixed(1)}, ${pt.a.toFixed(1)})`;
          if (b.params.point !== undefined && +b.params.point === i) opt.selected = true;
          inp.appendChild(opt);
        });
        inp.style.width = 'auto'; inp.style.minWidth = '140px';
        inp.dataset.paramKey = 'point';
        inp.dataset.blockId = b.id;
        inp.addEventListener('change', (e) => {
          const block = blocks.find(x => x.id == e.target.dataset.blockId);
          if (!block) return;
          const idx = e.target.value;
          block.params.point = idx;
          if (idx !== '' && cachedPoints[+idx]) {
            const pt = cachedPoints[+idx];
            block.params.x = pt.x; block.params.y = pt.y; block.params.a = pt.a;
          }
          autoSave();
          renderAllBlocks();
        });
        lbl.appendChild(inp);
        body.appendChild(lbl);
        continue;
      } else if (p.type === 'select') {
        inp = document.createElement('select');
        const opts = p.key === 'name' && b.type === 'macro' ? macroNames() : p.options;
        for (const o of opts) {
          const opt = document.createElement('option');
          opt.value = o; opt.textContent = o;
          if (b.params[p.key] === o) opt.selected = true;
          inp.appendChild(opt);
        }
      } else {
        inp = document.createElement('input');
        inp.type = 'number';
        inp.value = b.params[p.key];
        if (p.step != null) inp.step = p.step;
        if (p.min != null) inp.min = p.min;
        if (p.max != null) inp.max = p.max;
      }
      inp.dataset.paramKey = p.key;
      inp.dataset.blockId = b.id;
      inp.addEventListener('change', (e) => {
        const block = blocks.find(x => x.id == e.target.dataset.blockId);
        if (block) {
          block.params[e.target.dataset.paramKey] = e.target.value;
          autoSave();
        }
      });
      lbl.appendChild(inp);
      body.appendChild(lbl);
    }
    el.appendChild(body);
  }

  // Click to select
  el.addEventListener('click', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'BUTTON') return;
    handleSelect(b.id, e);
  });

  // Right-click context menu
  el.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (!selectedIds.has(b.id)) {
      selectedIds.clear();
      selectedIds.add(b.id);
      renderAllBlocks();
    }
    showContextMenu(e.clientX, e.clientY);
  });

  return el;
}

function renderAllBlocks() {
  const inner = document.getElementById('canvas-inner');
  const empty = document.getElementById('canvas-empty');
  inner.innerHTML = '';
  empty.style.display = blocks.length === 0 ? '' : 'none';
  blocks.forEach((b, i) => {
    const el = renderBlock(b, i);
    if (el) inner.appendChild(el);
  });
}

// =========================================================================
// SortableJS on canvas
// =========================================================================
let canvasSortable;
function initCanvasSortable() {
  canvasSortable = new Sortable(document.getElementById('canvas-inner'), {
    group: { name: 'blocks', pull: false, put: true },
    animation: 150,
    handle: '.block-header',
    onAdd: function(evt) {
      const typeKey = evt.item.dataset.blockType;
      const newBlock = makeBlockFromType(typeKey);
      evt.item.remove();
      if (!newBlock) return;
      const idx = Math.min(evt.newIndex, blocks.length);
      blocks.splice(idx, 0, newBlock);
      renderAllBlocks();
      autoSave();
    },
    onUpdate: function(evt) {
      if (evt.oldIndex === evt.newIndex) return;
      const item = blocks.splice(evt.oldIndex, 1)[0];
      blocks.splice(evt.newIndex, 0, item);
      renderAllBlocks();
      autoSave();
    }
  });
}

// =========================================================================
// Selection
// =========================================================================
function handleSelect(id, e) {
  if (e.ctrlKey || e.metaKey) {
    if (selectedIds.has(id)) selectedIds.delete(id);
    else selectedIds.add(id);
  } else if (e.shiftKey && selectedIds.size > 0) {
    const lastId = [...selectedIds].pop();
    const lastIdx = blocks.findIndex(b => b.id === lastId);
    const curIdx = blocks.findIndex(b => b.id === id);
    const [from, to] = lastIdx < curIdx ? [lastIdx, curIdx] : [curIdx, lastIdx];
    for (let i = from; i <= to; i++) selectedIds.add(blocks[i].id);
  } else {
    selectedIds.clear();
    selectedIds.add(id);
  }
  renderAllBlocks();
}

function deleteBlock(id) {
  blocks = blocks.filter(b => b.id !== id);
  selectedIds.delete(id);
  renderAllBlocks();
  autoSave();
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (selectedIds.size > 0) {
      blocks = blocks.filter(b => !selectedIds.has(b.id));
      selectedIds.clear();
      renderAllBlocks();
      autoSave();
    }
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
    e.preventDefault();
    selectedIds.clear();
    blocks.forEach(b => selectedIds.add(b.id));
    renderAllBlocks();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
    if (selectedIds.size > 0) {
      clipboard = blocks.filter(b => selectedIds.has(b.id)).map(b => JSON.parse(JSON.stringify(b)));
    }
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
    if (clipboard.length > 0) {
      const newBlocks = clipboard.map(b => ({ ...JSON.parse(JSON.stringify(b)), id: createBlockId() }));
      // Insert after last selected or at end
      let insertIdx = blocks.length;
      if (selectedIds.size > 0) {
        const lastSelId = [...selectedIds].pop();
        const lastSelIdx = blocks.findIndex(b => b.id === lastSelId);
        if (lastSelIdx >= 0) insertIdx = lastSelIdx + 1;
      }
      blocks.splice(insertIdx, 0, ...newBlocks);
      selectedIds.clear();
      newBlocks.forEach(b => selectedIds.add(b.id));
      renderAllBlocks();
      autoSave();
    }
  }
});

// Click on canvas background to deselect
document.getElementById('canvas').addEventListener('click', (e) => {
  if (e.target.id === 'canvas' || e.target.id === 'canvas-inner') {
    selectedIds.clear();
    renderAllBlocks();
  }
});

// =========================================================================
// Context menu
// =========================================================================
function showContextMenu(x, y) {
  const menu = document.getElementById('ctx-menu');
  menu.style.left = x + 'px';
  menu.style.top = y + 'px';
  menu.style.display = 'block';
}

document.addEventListener('click', () => {
  document.getElementById('ctx-menu').style.display = 'none';
});

function ctxCopy() {
  clipboard = blocks.filter(b => selectedIds.has(b.id)).map(b => JSON.parse(JSON.stringify(b)));
}

function ctxPaste() {
  if (clipboard.length === 0) return;
  const newBlocks = clipboard.map(b => ({ ...JSON.parse(JSON.stringify(b)), id: createBlockId() }));
  let insertIdx = blocks.length;
  if (selectedIds.size > 0) {
    const lastSelId = [...selectedIds].pop();
    const lastSelIdx = blocks.findIndex(b => b.id === lastSelId);
    if (lastSelIdx >= 0) insertIdx = lastSelIdx + 1;
  }
  blocks.splice(insertIdx, 0, ...newBlocks);
  selectedIds.clear();
  newBlocks.forEach(b => selectedIds.add(b.id));
  renderAllBlocks();
  autoSave();
}

function ctxDuplicate() {
  const sel = blocks.filter(b => selectedIds.has(b.id));
  if (sel.length === 0) return;
  const dupes = sel.map(b => ({ ...JSON.parse(JSON.stringify(b)), id: createBlockId() }));
  const lastIdx = blocks.findIndex(b => b.id === sel[sel.length - 1].id);
  blocks.splice(lastIdx + 1, 0, ...dupes);
  selectedIds.clear();
  dupes.forEach(b => selectedIds.add(b.id));
  renderAllBlocks();
  autoSave();
}

function ctxDelete() {
  blocks = blocks.filter(b => !selectedIds.has(b.id));
  selectedIds.clear();
  renderAllBlocks();
  autoSave();
}

function ctxSaveAsMacro() {
  const sel = blocks.filter(b => selectedIds.has(b.id));
  if (sel.length === 0) return;
  const name = prompt('Macro name:');
  if (!name) return;
  const macroBlocks = sel.map(b => ({ type: b.type, params: { ...b.params } }));
  fetch('/sequencer/macros/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, blocks: macroBlocks })
  }).then(() => loadMacros());
}

// =========================================================================
// Execution engine
// =========================================================================
async function runSequence() {
  if (running) return;
  if (blocks.length === 0) return;
  running = true;
  stopFlag = false;
  document.getElementById('btn-play').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  setRunStatus('Running...');

  // Clear previous states
  document.querySelectorAll('.canvas .block').forEach(el => {
    el.classList.remove('done', 'error', 'running');
  });

  for (let i = 0; i < blocks.length; i++) {
    if (stopFlag) break;
    const b = blocks[i];
    const bt = BLOCK_TYPES[b.type];
    if (!bt) continue;

    // Highlight running
    const el = document.querySelector(`.block[data-id="${b.id}"]`);
    if (el) { el.classList.add('running'); el.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
    setRunStatus(`Running block ${i + 1}/${blocks.length}: ${bt.name}`);

    try {
      await bt.execute(b.params, 0);
      if (el) { el.classList.remove('running'); el.classList.add('done'); }
    } catch (err) {
      if (el) { el.classList.remove('running'); el.classList.add('error'); }
      setRunStatus(`Error at block ${i + 1}: ${err.message}`);
      break;
    }
  }

  if (stopFlag) setRunStatus('Stopped');
  else if (!document.querySelector('.canvas .block.error')) setRunStatus('Complete');
  running = false;
  document.getElementById('btn-play').disabled = false;
  document.getElementById('btn-stop').disabled = true;
}

async function stopSequence() {
  stopFlag = true;
  // Also cancel any CNC jog in progress
  try { await fetch('/cnc/jog/cancel', { method: 'POST' }); } catch (e) {}
  setRunStatus('Stopping...');
}

function setRunStatus(msg) {
  document.getElementById('run-status').textContent = msg;
}

// =========================================================================
// Macro system
// =========================================================================
function macroNames() {
  return loadedMacros.map(m => m.name);
}

async function loadMacros() {
  try {
    const r = await fetch('/sequencer/macros');
    loadedMacros = await r.json();
  } catch (e) {
    loadedMacros = [];
  }
  // Update macro block type options
  BLOCK_TYPES['macro'].params[0].options = macroNames();
  if (loadedMacros.length > 0 && !BLOCK_TYPES['macro'].params[0].default) {
    BLOCK_TYPES['macro'].params[0].default = loadedMacros[0].name;
  }
  buildPalette();
  renderAllBlocks(); // re-render to update any macro select dropdowns
}

// =========================================================================
// Persistence — localStorage auto-save
// =========================================================================
function autoSave() {
  try {
    localStorage.setItem('sequencer_blocks', JSON.stringify(blocks));
    localStorage.setItem('sequencer_nextId', nextId);
  } catch (e) {}
}

function autoLoad() {
  try {
    const saved = localStorage.getItem('sequencer_blocks');
    if (saved) {
      blocks = JSON.parse(saved);
      nextId = parseInt(localStorage.getItem('sequencer_nextId') || '1', 10);
      // Ensure nextId is higher than any existing block id
      for (const b of blocks) { if (b.id >= nextId) nextId = b.id + 1; }
    }
  } catch (e) {}
}

// Save/Load named sequences to server
async function saveSequence() {
  const name = prompt('Sequence name:');
  if (!name) return;
  const payload = blocks.map(b => ({ type: b.type, params: { ...b.params } }));
  const r = await fetch('/sequencer/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, blocks: payload })
  });
  const d = await r.json();
  if (d.error) alert('Error: ' + d.error);
  else setRunStatus('Saved: ' + name);
}

async function showLoadDialog() {
  const r = await fetch('/sequencer/list');
  const names = await r.json();
  if (names.length === 0) { alert('No saved sequences'); return; }
  const name = prompt('Load sequence:\n\n' + names.join('\n') + '\n\nEnter name:');
  if (!name) return;
  const r2 = await fetch('/sequencer/load?name=' + encodeURIComponent(name));
  const d = await r2.json();
  if (d.error) { alert('Error: ' + d.error); return; }
  blocks = d.blocks.map(b => ({ id: createBlockId(), type: b.type, params: { ...b.params } }));
  selectedIds.clear();
  renderAllBlocks();
  autoSave();
  setRunStatus('Loaded: ' + name);
}

// Export/Import JSON files
function exportSequence() {
  const payload = blocks.map(b => ({ type: b.type, params: { ...b.params } }));
  const blob = new Blob([JSON.stringify({ blocks: payload }, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'sequence.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

function importSequence() {
  document.getElementById('file-import').click();
}

function handleImport(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      if (data.blocks && Array.isArray(data.blocks)) {
        blocks = data.blocks.map(b => ({ id: createBlockId(), type: b.type, params: { ...b.params } }));
        selectedIds.clear();
        renderAllBlocks();
        autoSave();
        setRunStatus('Imported ' + blocks.length + ' blocks');
      } else {
        alert('Invalid sequence file');
      }
    } catch (err) {
      alert('Error reading file: ' + err.message);
    }
  };
  reader.readAsText(file);
  event.target.value = '';
}

// =========================================================================
// Captured points
// =========================================================================
async function loadPoints() {
  try {
    const r = await fetch('/points');
    cachedPoints = await r.json();
  } catch (e) { cachedPoints = []; }
}

// =========================================================================
// Status polling
// =========================================================================
async function pollStatus() {
  try {
    const [cncR, ardR] = await Promise.all([fetch('/cnc/status'), fetch('/arduino/status')]);
    const cnc = await cncR.json();
    const ard = await ardR.json();
    const wp = cnc.wpos || {};
    const ap = ard.positions || {};
    document.getElementById('status-info').textContent =
      `CNC: X=${(wp.x||0).toFixed(1)} Y=${(wp.y||0).toFixed(1)} A=${(wp.z||0).toFixed(1)} [${cnc.state||'?'}]` +
      ` | Arduino: X=${ap.x||0} Z=${ap.z||0}`;
  } catch (e) {}
}

setInterval(pollStatus, 1000);

// =========================================================================
// Pan & Zoom
// =========================================================================
let panX = 0, panY = 0, zoom = 1;
const ZOOM_MIN = 0.25, ZOOM_MAX = 3, ZOOM_STEP = 0.1;
let isPanning = false, panStartX = 0, panStartY = 0, panStartPX = 0, panStartPY = 0;

function applyTransform() {
  const inner = document.getElementById('canvas-inner');
  const cvs = document.getElementById('canvas');
  inner.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
  // Move the dot grid with the pan
  cvs.style.backgroundPosition = `${panX}px ${panY}px`;
  cvs.style.backgroundSize = `${24 * zoom}px ${24 * zoom}px`;
  document.getElementById('zoom-badge').textContent = Math.round(zoom * 100) + '%';
}

function resetView() {
  panX = 0; panY = 0; zoom = 1;
  applyTransform();
}

function zoomBy(delta) {
  const cvs = document.getElementById('canvas');
  const rect = cvs.getBoundingClientRect();
  const cx = rect.width / 2, cy = rect.height / 2;
  const bx = (cx - panX) / zoom, by = (cy - panY) / zoom;
  zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom + delta));
  panX = cx - bx * zoom; panY = cy - by * zoom;
  applyTransform();
}

(function initPanZoom() {
  const cvs = document.getElementById('canvas');

  // Zoom with mouse wheel
  cvs.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = cvs.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // Point under mouse in canvas-inner coordinates before zoom
    const bx = (mx - panX) / zoom;
    const by = (my - panY) / zoom;
    const oldZoom = zoom;
    if (e.deltaY < 0) zoom = Math.min(ZOOM_MAX, zoom + ZOOM_STEP);
    else zoom = Math.max(ZOOM_MIN, zoom - ZOOM_STEP);
    // Adjust pan so the point under mouse stays fixed
    panX = mx - bx * zoom;
    panY = my - by * zoom;
    applyTransform();
  }, { passive: false });

  // Pan with middle mouse button or left-click on canvas background
  cvs.addEventListener('mousedown', (e) => {
    // Middle button always pans
    const isMiddle = e.button === 1;
    // Left button pans only if clicking on canvas or canvas-inner background
    const isLeftOnBg = e.button === 0 && (e.target.id === 'canvas' || e.target.id === 'canvas-inner');
    if (!isMiddle && !isLeftOnBg) return;
    e.preventDefault();
    isPanning = true;
    panStartX = e.clientX; panStartY = e.clientY;
    panStartPX = panX; panStartPY = panY;
    cvs.classList.add('panning');
  });

  window.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    panX = panStartPX + (e.clientX - panStartX);
    panY = panStartPY + (e.clientY - panStartY);
    applyTransform();
  });

  window.addEventListener('mouseup', (e) => {
    if (!isPanning) return;
    isPanning = false;
    cvs.classList.remove('panning');
  });
})();

// =========================================================================
// Init
// =========================================================================
autoLoad();
loadPoints().then(() => loadMacros());   // loadMacros also calls buildPalette + renderAllBlocks
initCanvasSortable();
pollStatus();
applyTransform();
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"Starting string hole capture tool on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
