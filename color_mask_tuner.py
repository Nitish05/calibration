"""
Interactive Color Mask Tuner — live HSV + LAB threshold tuning with sliders
and hover color readout.

Two modes:
  - Local:    cv2.imshow with trackbars + mouse hover
  - Headless: Flask/MJPEG stream with HTML sliders + JS tooltip

Adjust sliders to isolate the racket frame, then Ctrl+C to print final
values as CLI args for segment_racket_v2.py.
"""

import argparse
import threading

import cv2
import numpy as np
from flask import request, jsonify

from camera import Camera
from stream import StreamServer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Interactive color mask tuner")
parser.add_argument("--headless", action="store_true", help="Stream via HTTP instead of cv2.imshow")
parser.add_argument("--port", type=int, default=8082, help="HTTP port (default: 8082)")
parser.add_argument("--use-picamera", action="store_true", help="Use picamera2 for CSI camera")
parser.add_argument("--h-low", type=int, default=0, help="Initial HSV H lower bound")
parser.add_argument("--h-high", type=int, default=180, help="Initial HSV H upper bound")
parser.add_argument("--s-low", type=int, default=0, help="Initial HSV S lower bound")
parser.add_argument("--s-high", type=int, default=255, help="Initial HSV S upper bound")
parser.add_argument("--v-low", type=int, default=0, help="Initial HSV V lower bound")
parser.add_argument("--v-high", type=int, default=255, help="Initial HSV V upper bound")
parser.add_argument("--l-thresh", type=int, default=90, help="Initial LAB L-channel threshold")
parser.add_argument("--invert", action="store_true",
                    help="Invert HSV mask (define background range to exclude)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------
slider_lock = threading.Lock()
shared_sliders = {
    "h_low": args.h_low,
    "h_high": args.h_high,
    "s_low": args.s_low,
    "s_high": args.s_high,
    "v_low": args.v_low,
    "v_high": args.v_high,
    "l_thresh": args.l_thresh,
    "invert": 1 if args.invert else 0,
}

frame_lock = threading.Lock()
shared_frame = None  # raw BGR for color readout endpoint

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def apply_masks(frame, sliders):
    """Apply HSV range mask + LAB L-threshold and return visualization."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([sliders["h_low"], sliders["s_low"], sliders["v_low"]])
    high = np.array([sliders["h_high"], sliders["s_high"], sliders["v_high"]])
    hsv_mask = cv2.inRange(hsv, low, high)

    if sliders["invert"]:
        hsv_mask = cv2.bitwise_not(hsv_mask)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    lab_mask = (L < sliders["l_thresh"]).astype(np.uint8) * 255

    combined = hsv_mask & lab_mask

    # Overlay: masked-in full brightness, masked-out dimmed to 30%
    dimmed = (frame * 0.3).astype(np.uint8)
    mask_3ch = cv2.merge([combined, combined, combined])
    vis = np.where(mask_3ch > 0, frame, dimmed)

    # Pixel count text
    px_count = int(np.count_nonzero(combined))
    inv_tag = " [INV]" if sliders["invert"] else ""
    text = (f"H:{sliders['h_low']}-{sliders['h_high']}  "
            f"S:{sliders['s_low']}-{sliders['s_high']}  "
            f"V:{sliders['v_low']}-{sliders['v_high']}  "
            f"L<{sliders['l_thresh']}{inv_tag}  "
            f"px:{px_count}")
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Mask thumbnail (200px height) in bottom-left
    th = 200
    tw = int(combined.shape[1] * th / combined.shape[0])
    thumb = cv2.resize(combined, (tw, th))
    thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    fh = vis.shape[0]
    vis[fh - th:fh, 0:tw] = thumb_bgr

    return vis


def get_pixel_colors(frame, u, v):
    """Return BGR, HSV, LAB values at pixel (u, v)."""
    h, w = frame.shape[:2]
    u, v = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
    bgr = frame[v, u].tolist()
    hsv = cv2.cvtColor(frame[v:v+1, u:u+1], cv2.COLOR_BGR2HSV)[0, 0].tolist()
    lab = cv2.cvtColor(frame[v:v+1, u:u+1], cv2.COLOR_BGR2LAB)[0, 0].tolist()
    return {"bgr": bgr, "hsv": hsv, "lab": lab}


# ---------------------------------------------------------------------------
# Headless HTML
# ---------------------------------------------------------------------------
TUNER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #111; display: flex; flex-direction: column;
               align-items: center; font-family: monospace; color: #ddd;
               min-height: 100vh; padding: 8px; }
        h1 { color: #0f0; margin: 6px 0; font-size: 1.1em; }
        .main { display: flex; gap: 12px; align-items: flex-start;
                max-width: 100vw; flex-wrap: wrap; justify-content: center; }
        .stream-wrap { position: relative; display: inline-block; line-height: 0; }
        .stream-wrap img { display: block; max-width: 72vw; max-height: 85vh; }
        .stream-wrap canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
        .panel { background: #1a1a1a; border: 1px solid #333; border-radius: 6px;
                 padding: 10px 14px; min-width: 240px; }
        .panel h2 { color: #0ff; font-size: 0.95em; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }
        .slider-row { display: flex; align-items: center; gap: 6px; margin: 5px 0; }
        .slider-row label { width: 60px; text-align: right; font-size: 13px; color: #aaa; }
        .slider-row input[type=range] { flex: 1; accent-color: #0f0; }
        .slider-row .val { width: 32px; text-align: center; font-size: 13px; color: #0f0; }
        #color-tooltip {
            position: fixed; pointer-events: none; z-index: 9999;
            font-family: monospace; font-size: 12px; padding: 6px 10px;
            background: rgba(0,0,0,0.85); border: 1px solid #555;
            border-radius: 4px; white-space: nowrap; display: none;
        }
        #color-tooltip .swatch { display: inline-block; width: 14px; height: 14px;
                                  border: 1px solid #777; vertical-align: middle;
                                  margin-right: 6px; border-radius: 2px; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="main">
        <div class="stream-wrap">
            <img id="stream" src="/feed">
            <canvas id="overlay"></canvas>
        </div>
        <div class="panel">
            <h2>HSV Range</h2>
            <div class="slider-row">
                <label>H Low</label>
                <input type="range" id="h_low" min="0" max="180" value="">
                <span class="val" id="h_low_val"></span>
            </div>
            <div class="slider-row">
                <label>H High</label>
                <input type="range" id="h_high" min="0" max="180" value="">
                <span class="val" id="h_high_val"></span>
            </div>
            <div class="slider-row">
                <label>S Low</label>
                <input type="range" id="s_low" min="0" max="255" value="">
                <span class="val" id="s_low_val"></span>
            </div>
            <div class="slider-row">
                <label>S High</label>
                <input type="range" id="s_high" min="0" max="255" value="">
                <span class="val" id="s_high_val"></span>
            </div>
            <div class="slider-row">
                <label>V Low</label>
                <input type="range" id="v_low" min="0" max="255" value="">
                <span class="val" id="v_low_val"></span>
            </div>
            <div class="slider-row">
                <label>V High</label>
                <input type="range" id="v_high" min="0" max="255" value="">
                <span class="val" id="v_high_val"></span>
            </div>
            <h2>LAB L Threshold</h2>
            <div class="slider-row">
                <label>L Thresh</label>
                <input type="range" id="l_thresh" min="0" max="255" value="">
                <span class="val" id="l_thresh_val"></span>
            </div>
            <h2>Mode</h2>
            <div class="slider-row">
                <label>Invert</label>
                <input type="range" id="invert" min="0" max="1" value="">
                <span class="val" id="invert_val"></span>
            </div>
        </div>
    </div>
    <div id="color-tooltip">
        <span class="swatch" id="ct-swatch"></span>
        <span id="ct-text"></span>
    </div>
<script>
const CAM_W = 1920, CAM_H = 1080;
const img = document.getElementById('stream');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('color-tooltip');
const ctSwatch = document.getElementById('ct-swatch');
const ctText = document.getElementById('ct-text');

const SLIDER_KEYS = ['h_low','h_high','s_low','s_high','v_low','v_high','l_thresh','invert'];

// --- fetch initial slider values ---
fetch('/sliders').then(r=>r.json()).then(d => {
    SLIDER_KEYS.forEach(k => {
        const el = document.getElementById(k);
        el.value = d[k];
        document.getElementById(k+'_val').textContent = d[k];
    });
});

// --- slider change handler ---
SLIDER_KEYS.forEach(k => {
    const el = document.getElementById(k);
    el.addEventListener('input', () => {
        document.getElementById(k+'_val').textContent = el.value;
        sendSliders();
    });
});

let sliderTimeout = null;
function sendSliders() {
    clearTimeout(sliderTimeout);
    sliderTimeout = setTimeout(() => {
        const data = {};
        SLIDER_KEYS.forEach(k => { data[k] = parseInt(document.getElementById(k).value); });
        fetch('/sliders', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(data)
        });
    }, 50);
}

// --- color tooltip (same AbortController pattern as segment_racket_v2.py) ---
let colorAborter = null;
let colorLastTime = 0;
const COLOR_INTERVAL = 100;

function scaleToCanvas() { return img.clientWidth / CAM_W; }

canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const sc = scaleToCanvas();
    const camU = Math.round(mx / sc), camV = Math.round(my / sc);

    const now = Date.now();
    if (now - colorLastTime < COLOR_INTERVAL) return;
    colorLastTime = now;

    if (colorAborter) colorAborter.abort();
    colorAborter = new AbortController();

    fetch('/pixel_color', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({u: camU, v: camV}),
        signal: colorAborter.signal
    })
    .then(r => r.json())
    .then(d => {
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 16) + 'px';
        tooltip.style.top  = (e.clientY + 16) + 'px';
        const [b,g,r] = d.bgr;
        ctSwatch.style.background = `rgb(${r},${g},${b})`;
        ctText.innerHTML =
            `<span style="color:#f88">BGR</span> ${b},${g},${r}  ` +
            `<span style="color:#8f8">HSV</span> ${d.hsv[0]},${d.hsv[1]},${d.hsv[2]}  ` +
            `<span style="color:#88f">LAB</span> ${d.lab[0]},${d.lab[1]},${d.lab[2]}`;
    })
    .catch(() => {});
});

canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

// resize overlay canvas to match image
function resizeCanvas() {
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
}
img.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Camera + stream setup
# ---------------------------------------------------------------------------
cap = Camera(use_picamera=args.use_picamera)
stream = None

if args.headless:
    stream = StreamServer(port=args.port, title="Color Mask Tuner", html=TUNER_HTML)

    @stream.app.route("/sliders", methods=["GET", "POST"])
    def sliders_endpoint():
        global shared_sliders
        if request.method == "POST":
            data = request.get_json()
            with slider_lock:
                for k in shared_sliders:
                    if k in data:
                        shared_sliders[k] = int(data[k])
            return jsonify(ok=True)
        with slider_lock:
            return jsonify(dict(shared_sliders))

    @stream.app.route("/pixel_color", methods=["POST"])
    def pixel_color_endpoint():
        data = request.get_json()
        u, v = int(data["u"]), int(data["v"])
        with frame_lock:
            f = shared_frame
        if f is None:
            return jsonify(error="no frame")
        colors = get_pixel_colors(f, u, v)
        return jsonify(colors)

    stream.start()

# ---------------------------------------------------------------------------
# Local mode: trackbar callback + mouse hover
# ---------------------------------------------------------------------------
WINDOW = "Color Mask Tuner"
hover_pos = None  # (u, v) in pixel coords or None

if not args.headless:
    cv2.namedWindow(WINDOW)

    def make_cb(key):
        def cb(val):
            with slider_lock:
                shared_sliders[key] = val
        return cb

    cv2.createTrackbar("H Low",   WINDOW, args.h_low,   180, make_cb("h_low"))
    cv2.createTrackbar("H High",  WINDOW, args.h_high,  180, make_cb("h_high"))
    cv2.createTrackbar("S Low",   WINDOW, args.s_low,   255, make_cb("s_low"))
    cv2.createTrackbar("S High",  WINDOW, args.s_high,  255, make_cb("s_high"))
    cv2.createTrackbar("V Low",   WINDOW, args.v_low,   255, make_cb("v_low"))
    cv2.createTrackbar("V High",  WINDOW, args.v_high,  255, make_cb("v_high"))
    cv2.createTrackbar("L Thresh", WINDOW, args.l_thresh, 255, make_cb("l_thresh"))
    cv2.createTrackbar("Invert",  WINDOW, 1 if args.invert else 0, 1, make_cb("invert"))

    def mouse_cb(event, x, y, flags, param):
        global hover_pos
        if event == cv2.EVENT_MOUSEMOVE:
            hover_pos = (x, y)
        elif event == cv2.EVENT_MOUSELEAVE:
            hover_pos = None

    cv2.setMouseCallback(WINDOW, mouse_cb)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print(f"Color Mask Tuner | mode={'headless' if args.headless else 'local'}")
print(f"Initial: H={args.h_low}-{args.h_high}  S={args.s_low}-{args.s_high}  "
      f"V={args.v_low}-{args.v_high}  L<{args.l_thresh}")
if args.headless:
    print(f"Open http://0.0.0.0:{args.port} in browser")
print("Press Ctrl+C to exit and print final values")
print("-" * 60)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        with frame_lock:
            shared_frame = frame.copy()

        with slider_lock:
            sliders = dict(shared_sliders)

        vis = apply_masks(frame, sliders)

        # Local mode: draw hover color readout
        if not args.headless and hover_pos is not None:
            u, v = hover_pos
            h, w = frame.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                colors = get_pixel_colors(frame, u, v)
                b, g, r = colors["bgr"]
                ch, cs, cv_ = colors["hsv"]
                cl, ca, cb_ = colors["lab"]
                text = f"BGR:{b},{g},{r}  HSV:{ch},{cs},{cv_}  LAB:{cl},{ca},{cb_}"
                # Draw text near cursor
                tx = min(u + 20, w - 400)
                ty = max(v - 10, 25)
                cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1)
                # Color swatch
                cv2.rectangle(vis, (tx - 18, ty - 12), (tx - 4, ty + 2),
                              (int(b), int(g), int(r)), -1)
                cv2.rectangle(vis, (tx - 18, ty - 12), (tx - 4, ty + 2),
                              (255, 255, 255), 1)

        if args.headless:
            stream.update_frame(vis)
        else:
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

except KeyboardInterrupt:
    pass

# ---------------------------------------------------------------------------
# Print final values on exit
# ---------------------------------------------------------------------------
with slider_lock:
    final = dict(shared_sliders)

inv_flag = " --invert-hsv" if final["invert"] else ""
print("\n" + "=" * 60)
print("Final slider values:")
print(f"  --h-low {final['h_low']} --h-high {final['h_high']} "
      f"--s-low {final['s_low']} --s-high {final['s_high']} "
      f"--v-low {final['v_low']} --v-high {final['v_high']} "
      f"--l-thresh {final['l_thresh']}{inv_flag}")
print("=" * 60)

cap.release()
if not args.headless:
    cv2.destroyAllWindows()
