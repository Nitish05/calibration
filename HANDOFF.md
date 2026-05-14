# Cold-Start Hand-Off

This document is the first thing a new engineer should read when picking up
this repo. It describes what the system is, what hardware it drives, how
to bring it up from a fresh shell, and where everything lives.

Companion documents:

- [`README.md`](README.md) — quick reference / elevator pitch.
- [`SERIAL_COMMANDS.md`](SERIAL_COMMANDS.md) — full Arduino serial protocol.

---

## 1. Project at a Glance

This is an **ongoing project to build an automatic tennis-racket
stringing machine** — a robot that maps the string holes on a racket,
threads string through them, tensions, and ties off without manual
stringing. This repo is the **Raspberry Pi 5 control software** for that
rig; it is **not feature-complete** — the Block Sequencer and motor
control are in place, but the end-to-end stringing routine is still
under active development.

The repo is named `calibration` for historical reasons; it began as the
hole-mapping and camera-calibration scaffolding and is now growing into
the full machine controller.

A Raspberry Pi 5 runs a Flask web app that:

- streams an overhead camera with **AprilTag 6-DoF pose overlay**,
- drives a **Grbl CNC** (X, Y, and a rotational A axis) over USB serial,
- drives an **Arduino Uno + CNC-Shield V3** with six actuators (two NEMA
  steppers, two 28BYJ-48 steppers, an SG90 servo, and a 5 V DC motor)
  over a custom serial protocol,
- lets the operator click holes on the live feed to capture string-hole
  positions, and
- provides a visual **Block Sequencer** for composing CNC + Arduino
  macros without writing code.

Several stand-alone CV utilities (`hole_vector.py`, `segment_racket_v2.py`,
`color_mask_tuner.py`) live alongside the main app for offline tuning.

```
                +-----------------+
                |  Raspberry Pi 5 |
                |  Flask app on   |
                |  :5001 (+ :5000 |
                |   :8080 etc.)   |
                +--------+--------+
                         |
        +----------------+----------------+
        |                |                |
   USB serial        USB serial        USB camera
   /dev/cnc_main     /dev/cnc_aux      /dev/videoN
        |                |                |
   +----+----+      +----+----+     +-----+------+
   |  Grbl   |      | Arduino |     |  Logitech  |
   |  CNC    |      | Uno +   |     |  C920 or   |
   | (X/Y/A) |      | CNC-V3  |     |  Pi CSI    |
   +----+----+      +----+----+     +-----+------+
        |                |                |
   X, Y steppers    X-stepper, Z-       overhead view
   + rotational A   stepper, BYJ1,      with AprilTag
   on Z-output      BYJ2, servo,        for world frame
                    DC motor
                         |
                +--------+---------+
                |  Tennis racket   |
                |  on jig          |
                +------------------+
```

**Critical wiring quirk:** the main Grbl board only has three motor outputs
(X / Y / Z). Its **Z output drives the rotational axis** (`$102` was
re-configured for rotation in deg/step). The actual linear vertical motion
of the gantry is driven by the **Arduino's Z stepper**, not Grbl's Z. So
"Grbl Z" in this codebase means "rotation in degrees", and "Arduino Z"
means "vertical travel in steps". This is non-obvious and will bite you.

---

## 2. Hardware Inventory

| Subsystem | Part | Notes |
|---|---|---|
| Host | **Raspberry Pi 5** | All hardware (CNC, Arduino, camera) plugs into the Pi. The repo is intended to run on the Pi — not as a portable desktop app. |
| Main CNC controller | Grbl board with CH340 USB serial (VID `0x1A86`) | Auto-detect in `cnc.py:17-28`. Pin to `/dev/cnc_main` via udev. |
| Aux board | Arduino Uno (VID `0x2341`) + CNC Shield V3 | Auto-detect in `arduino_controller.py:28-60`. Pin to `/dev/cnc_aux` via udev. |
| Stepper 1 | NEMA on **X** (CNC Shield) | HR4988 driver (A4988 clone). Commanded as `X<steps>`. |
| Stepper 2 | NEMA on **Z** (CNC Shield) — actual linear vertical | HR4988. Commanded as `Z<steps>`. |
| Stepper 3 | 28BYJ-48 #1 | ULN2003 driver. Commanded as `B<steps>`. |
| Stepper 4 | 28BYJ-48 #2 | ULN2003 driver. Commanded as `J<steps>`. |
| Servo | SG90 | Direct PWM. Commanded as `O<angle>` (0–180°, absolute). |
| DC motor | 5 V via L298N (motor A) | `F<speed>` fwd, `G<speed>` rev, `S` stop (0–100). |
| Camera (primary) | Logitech C920 (USB, V4L2) | MJPEG at 1920×1080. Controls locked in `camera.py:72-91`. |
| Camera (fallback) | Pi CSI via `picamera2` | Used if `use_picamera=True` is passed to `Camera()`. |
| Fiducial | AprilTag `tag36h11`, 77.69 mm | Defined in `transforms.py:18` (`TAG_SIZE` in metres). |

The Arduino firmware **is not in this repo** — it lives on the Arduino's
flash. The serial-protocol contract is documented in
[`SERIAL_COMMANDS.md`](SERIAL_COMMANDS.md). If you ever need to reflash
or modify it, that source is an open question (track down before you
touch the board).

### 2.1 Arduino + CNC-Shield-V3 Wiring

All six actuators are driven by an **Arduino Uno** on a CNC Shield V3.
The PlatformIO sketch is in this repo at [`firmware/`](firmware/);
`platformio.ini` is `board = uno` and the sketch itself is
[`firmware/src/main.cpp`](firmware/src/main.cpp). Library deps
(`arduino-libraries/Servo`, `waspinator/AccelStepper`) are declared in
`platformio.ini` and a vendored copy is also committed under
`firmware/.pio/libdeps/uno/` for offline builds.

Build / flash from the repo root:

```bash
cd firmware
pio run -t upload   # compile + flash to /dev/cnc_aux
```

The table below is **verified** against `main.cpp` lines 6-30 — every
actuator, every pin.

#### NEMA steppers (CNC-Shield V3 X and Z sockets, A4988 drivers)

| Actuator | STEP | DIR | ENABLE (active-low) | Driver | AccelStepper config |
|---|---|---|---|---|---|
| **X stepper** | D2 | D5 | D8 (shared) | A4988 / HR4988 in X socket | `maxSpeed 1000`, `accel 500` |
| **Z stepper** (linear vertical) | D4 | D7 | D8 (shared) | A4988 / HR4988 in Z socket | `maxSpeed 1000`, `accel 500` |

`ENABLE_PIN` (D8) is driven LOW in `setup()`, so the drivers are always
enabled while the Arduino is powered. Microstepping is whatever the
three MS-jumpers under each socket are set to — record the pattern when
you set the rig up because the host-side `steps/mm` calibration depends
on it.

The CNC Shield's **Y stepper socket is empty**; the firmware repurposes
its STEP pin (D3) for the servo signal — see below.

#### Servo (CNC-Shield Y_STEP pin, repurposed)

| Actuator | Signal pin | Library | Notes |
|---|---|---|---|
| SG90 grip servo | **D3** (`SERVO_PIN`, the Y_STEP header on the shield) | `Servo.h` | Attached in `setup()`; init position 0°. Range clamped to 0–180° in firmware. |

#### 28BYJ-48 steppers (ULN2003 modules, half-step mode)

| Actuator | IN1 | IN2 | IN3 | IN4 | AccelStepper config |
|---|---|---|---|---|---|
| **BYJ1** | A0 | A1 | A2 | A3 | `HALF4WIRE`, `maxSpeed 800`, `accel 400` |
| **BYJ2** | D11 | D12 | A4 | A5 | `HALF4WIRE`, `maxSpeed 800`, `accel 400` |

⚠️ The AccelStepper `HALF4WIRE` constructors pass pins in the order
**`(IN1, IN3, IN2, IN4)`** — i.e. the firmware swaps the middle two.
This is the coil-energization order, not a wiring change — wire the
ULN2003 modules in the natural IN1→IN1, IN2→IN2, IN3→IN3, IN4→IN4 order
and the pin-swap inside `AccelStepper::HALF4WIRE(...)` does the rest.

#### DC motor (L298N motor A side)

| Pin name | Arduino pin | Role |
|---|---|---|
| `L298N_IN1` | D9 | Direction A |
| `L298N_IN2` | D10 | Direction B |
| `L298N_ENA` | D6 | Speed PWM (`analogWrite`, 0–255 native; firmware constrains the protocol value to 0–100 before writing) |

`F<speed>` drives IN1 HIGH, IN2 LOW; `G<speed>` reverses; `S` sets all
three pins low (ENA = 0, coast stop).

#### Pin-allocation reference

```
D2  X_STEP      A0  BYJ1_IN1
D3  SERVO       A1  BYJ1_IN2
D4  Z_STEP      A2  BYJ1_IN3
D5  X_DIR       A3  BYJ1_IN4
D6  L298N_ENA   A4  BYJ2_IN3
D7  Z_DIR       A5  BYJ2_IN4
D8  ENABLE (NEMA drivers, active-low)
D9  L298N_IN1
D10 L298N_IN2
D11 BYJ2_IN1
D12 BYJ2_IN2
D13 (unused)
```

Free pins: D0, D1 (UART — keep clear), D13 (LED only).

#### Power rails

- **Vmot** (12 V terminal block on the CNC Shield) → NEMA X and Z
  drivers only. Do **not** route this to the Pi or the BYJ / servo /
  L298N.
- **5 V** (Arduino's 5 V pin or a separate 5 V supply) → ULN2003
  modules for both BYJ steppers and the SG90 servo. If both BYJs run at
  once and dip the rail, give them their own 5 V supply with common
  ground.
- **L298N motor supply** — independent 5–12 V terminal on the L298N
  module; do not back-feed off the Arduino. Logic-Vcc on the L298N can
  share the Arduino 5 V.
- All grounds must be tied together (Arduino GND ↔ CNC-Shield GND ↔
  ULN2003 GND ↔ L298N GND ↔ external-supply GND).

#### Pi 5 → Arduino link

The Arduino is a USB device on the Pi. The udev rule in §5.4 pins it to
`/dev/cnc_aux`. The Arduino draws its logic power from USB; the motor
supplies (12 V for steppers, 5 V for BYJ/servo, separate for L298N) are
all external — do **not** try to power any motor off the Pi's 5 V rail.

---

## 3. Software Stack

- **Python 3.12.13** (CPython) in a `uv`-managed `.venv`.
  - `.venv/pyvenv.cfg` records `uv = 0.10.9`, `version_info = 3.12.13`.
  - `include-system-site-packages = false`.
- **Flask 3.1.3** for the web app.
- **OpenCV 4.13** (`opencv-python`).
- **pupil-apriltags** for tag detection + pose.
- **numpy 2.4**, **scipy** (via robotpy-wpimath), **matplotlib**,
  **toml**.
- **camerakit 2.0.0** (CLI) for one-shot intrinsic calibration.
- **Plotly 2.35** — loaded from CDN at runtime, *not* a Python dep.
- **pyserial** — required to talk to CNC or Arduino. **Not currently in
  the dev `.venv`** (see [§5](#5-virtual-environment--first-run-setup)).
- **picamera2** — required only if you use the Pi CSI camera. **Not in
  the dev `.venv`** (it's normally installed via `apt` on the Pi).

No `requirements.txt`, `pyproject.toml`, `uv.lock`, or `Pipfile` is
committed. Dependencies live only inside `.venv/`, which is gitignored.
That means a fresh checkout will have **no working environment** until
you run the bootstrap commands in §5.

No Node, no npm, no build step, no Dockerfile, no CI.

---

## 4. Repository Layout

```
/home/nitish/work/calibration/
├── README.md                # short overview, quick start
├── HANDOFF.md               # this file
├── SERIAL_COMMANDS.md       # Arduino serial protocol reference
├── Config.toml              # CameraKit checkerboard settings (8×5, 9.37 mm)
├── Calib_board_outer.toml   # camerakit output (1280×720 intrinsics) — gitignored
├── .gitignore
├── .venv/                   # uv-managed Python 3.12 venv — gitignored
│
├── string_holes.py          # 2642 LOC — MAIN APP (Flask, embeds 3 HTML UIs)
├── cnc.py                   # 167 LOC — Grbl CNC serial driver
├── arduino_controller.py    # 191 LOC — 6-actuator serial driver
├── camera.py                # 105 LOC — Camera wrapper (picamera2 + OpenCV V4L2)
├── transforms.py            # 96 LOC — intrinsics + pixel↔world math
├── stream.py                # 81 LOC — reusable MJPEG Flask server
│
├── detect_apriltag.py       # 115 LOC — standalone AprilTag overlay viewer
├── record_calibration.py    # 80 LOC — record video for CameraKit
├── hole_vector.py           # 882 LOC — dual-perspective hole-vector tool
├── pi_stream.py             # 51 LOC — picamera2 MJPEG-only helper (spawned by hole_vector)
├── color_mask_tuner.py      # 466 LOC — interactive HSV/LAB tuner
├── segment_racket_v2.py     # 1387 LOC — racket-frame segmenter (ellipse fit, preferred)
├── segment_racket.py        # 543 LOC — v1 segmenter (deprecated, kept for reference)
│
├── intrinsics/cam_00/       # CameraKit working dir (videos gitignored)
├── calibration/             # camerakit output dir — gitignored
└── data/                    # created on first run of string_holes.py
    ├── sequences/           # saved Block Sequencer workflows (*.json)
    └── macros/              # saved macro definitions (*.json)
```

`calibration/` and `intrinsics/cam_00/*.mp4` are gitignored — they're
re-generated locally for each camera.

---

## 5. Virtual Environment & First-Run Setup

### 5.1 Day-1 commands (you already have the .venv)

```bash
cd /home/nitish/work/calibration
source .venv/bin/activate
python --version           # → Python 3.12.13
```

Then jump to §6 to pick an entry point.

### 5.2 Rebuilding the venv from scratch (e.g. fresh Pi)

The committed `.venv` was built with `uv`. The fastest reproducible
recipe:

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/
cd /home/nitish/work/calibration
uv venv --python 3.12
source .venv/bin/activate
uv pip install \
  flask opencv-python pupil-apriltags numpy matplotlib toml \
  camerakit pyserial
```

Plain-pip equivalent (slower):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install flask opencv-python pupil-apriltags numpy matplotlib toml \
            camerakit pyserial
```

On the **Pi 5**, additionally:

```bash
# picamera2 is installed system-wide via apt; expose it to the venv:
sudo apt install -y python3-picamera2
# Recreate the venv with --system-site-packages so the apt picamera2 is visible:
uv venv --python 3.12 --system-site-packages
```

### 5.3 ⚠️ Local-venv state (snapshot 2026-05-14)

The whole rig runs on the Pi 5; the venv inspected on this local working
copy is **partially complete** and the same gap likely exists on the Pi
unless someone has fixed it there:

| Module | Installed? | Effect if missing |
|---|---|---|
| `flask`, `cv2`, `pupil_apriltags`, `numpy`, `toml` | ✅ | — |
| `serial` (pyserial) | ❌ | CNC and Arduino auto-detect both fail silently; web UI loads but shows "Disconnected" forever. Imports are deferred, so the app still boots. |
| `picamera2` | ❌ | Only matters if you set `use_picamera=True`; the default OpenCV USB path works fine. |

Fix before doing any hardware work:

```bash
uv pip install pyserial
# On the Pi, additionally:
sudo apt install -y python3-picamera2     # see §5.2
```

### 5.4 udev symlinks (recommended on the Pi)

The scripts default to `/dev/cnc_main` (Grbl) and `/dev/cnc_aux`
(Arduino) and only fall back to USB scanning if those paths are absent.
Pin both with udev rules in `/etc/udev/rules.d/99-cnc.rules`:

```udev
# Main Grbl (CH340)
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", \
  SYMLINK+="cnc_main", MODE="0666"

# Arduino Uno
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", \
  SYMLINK+="cnc_aux", MODE="0666"
```

Reload:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Auto-detect precedence (no symlinks):

- `cnc.py:17-28` picks any port whose USB VID is `0x1A86` or whose
  description contains `CH340`.
- `arduino_controller.py:28-60` prefers VID `0x2341`, then anything
  matching `Arduino` in the description, then any `/dev/ttyACM*` that
  isn't the CH340 board.

If only one device is plugged in, the auto-detect for the other will
quietly grab it. Use the symlinks.

---

## 6. Entry-Point Scripts

All scripts are run from the repo root with the venv active.

### 6.1 `string_holes.py` — main Flask app (port **5001**)

```bash
python string_holes.py [--port 5001] [--baud 115200] \
                       [--serial-port /dev/cnc_main] \
                       [--serial-port-z /dev/cnc_aux] \
                       [--no-camera]
```

Boots three things: the Grbl CNC, the Arduino, and (unless `--no-camera`)
the AprilTag camera thread. Serves three browser pages:

| Path | Purpose |
|---|---|
| `/` | Live feed + click-to-capture string-hole positions, motor jog panels, Plotly scatter of captured points. |
| `/navigator` | Click-to-jog: click a Plotly point and the CNC jogs there. |
| `/sequencer` | Visual Block Sequencer for composing CNC + Arduino macros (see §8). |

Required: `pyserial` (for hardware), OpenCV + pupil-apriltags (for camera
overlay). The app degrades gracefully if any of these are missing — the
UI just shows "Disconnected" / "Camera unavailable".

### 6.2 `hole_vector.py` — dual-perspective hole-vector tool (port **5000**)

```bash
python hole_vector.py [--port 5000] [--pi-host 192.168.0.123] \
                      [--pi-port 8080] [--baud 115200] \
                      [--serial-port auto] [--threshold 128] \
                      [--min-area 100] [--z-working <mm>] \
                      [--hole-diameter 3.0] \
                      [--fx ...] [--fy ...] [--cx-intrinsic ...] [--cy-intrinsic ...]
```

Spawns `pi_stream.py` as a subprocess to publish the Pi CSI camera over
`http://<pi-host>:<pi-port>/stream`, then fetches frames, detects circular
grommet holes, and computes a 3-D vector through each hole using two
camera perspectives. Useful for measuring drilled-hole angles. Intrinsics
default to the values in `transforms.py` unless overridden on the CLI.

### 6.3 `detect_apriltag.py` — standalone tag overlay (port **8080**)

```bash
python detect_apriltag.py [--headless] [--port 8080]
```

Minimal viewer that renders the AprilTag outline + RGB axes + XYZ/RPY
text on the live feed. `--headless` streams over HTTP (no X server);
without it, opens an OpenCV window.

### 6.4 `record_calibration.py` — record a calibration video (port **8080**)

```bash
python record_calibration.py [--headless] [--port 8080] \
                             [--duration 60] \
                             [--output intrinsics/cam_00/intrinsics.mp4]
```

Records a 60-second 1920×1080 MP4 while streaming live preview so you
can watch your checkerboard coverage. Step 1 of the calibration workflow
(§10).

### 6.5 `color_mask_tuner.py` — interactive HSV/LAB tuner (port **8082**)

```bash
python color_mask_tuner.py [--headless] [--port 8082] [--use-picamera] \
                           [--h-low 0] [--h-high 20] [--s-low 0] [--s-high 85] \
                           [--v-low 50] [--v-high 220] [--l-thresh 72] \
                           [--invert | --no-invert] [--no-lab]
```

Sliders for HSV ranges and an optional LAB L-channel cutoff with a live
"hover any pixel to see its colour values" readout. Used to dial in the
masks consumed by `segment_racket_v2.py`.

### 6.6 `segment_racket_v2.py` — racket-frame segmenter (port **8081**, preferred)

```bash
python segment_racket_v2.py [--headless] [--port 8081] \
                            [--output racket_contour] [--format csv|json] \
                            [--continuous] [--smooth-points 200] \
                            [--min-area 15000] [--roi X Y W H] \
                            [--l-thresh 72] [--h-low ... --h-high ... \
                             --s-low ... --s-high ... --v-low ... --v-high ...] \
                            [--invert-hsv | --no-invert-hsv]
```

Fits an ellipse to the racket head silhouette in LAB+HSV space and emits
the outline as CSV or JSON. `--continuous` keeps the stream open so you
can adjust ROI live.

### 6.7 `segment_racket.py` — v1 segmenter (deprecated)

Kept for reference, same flag set as v1 (`--epsilon`, `--hsv-low`,
`--hsv-high`, `--dark-thresh`, etc.). Use v2 instead.

### 6.8 `pi_stream.py` — bare picamera2 MJPEG helper (port **8080**, hardcoded)

```bash
python pi_stream.py
```

Standalone; serves `/`, `/stream`, `/capture`. Normally launched as a
subprocess by `hole_vector.py`, but can be run on its own for a quick
peek at the Pi CSI camera. Requires `picamera2` (only on the Pi).

---

## 7. Main App Tour — `string_holes.py`

The largest single file in the project (2.6 k lines). Layout:

| Lines | Section |
|---|---|
| 1–50 | Imports, deferred camera/vision imports, CLI argparse |
| 51–89 | CNC connect, Arduino connect, atexit cleanup |
| 86–112 | Shared state (locks, caches), data-dir bootstrap |
| 116–195 | Camera thread (AprilTag detection + JPEG encoding) |
| 199–236 | `cnc_poll_thread`, `arduino_poll_thread` (100 ms cadence) |
| 240–460 | Flask routes (capture / motors / points / import-export) |
| 462+ | `INDEX_HTML` (capture page) |
| ~940+ | `NAVIGATOR_HTML` (point navigator) |
| ~1220+ | Sequencer routes + `SEQUENCER_HTML` |
| 1620+ | `BLOCK_TYPES` definition (JS object) |
| Bottom | `app.run(host="0.0.0.0", port=args.port, threaded=True)` |

### 7.1 Background threads

- `camera_thread()` (line 133) — pulls frames from `Camera`, runs the
  AprilTag detector, draws overlay, JPEG-encodes, signals consumers via
  `frame_event`. Single producer / many consumers.
- `cnc_poll_thread()` (line 201) — queries Grbl status every 100 ms,
  caches the parsed dict under `cnc_status_lock`.
- `arduino_poll_thread()` (line 220) — queries `P` (position) on the
  Arduino every 100 ms, caches under `arduino_status_lock`.

All three are daemonised; they exit when Flask exits.

### 7.2 Flask route table

Verified by `grep -n "@app.route"` against the source as of commit
`4384077`.

| Route | Method | Defined at | Purpose |
|---|---|---|---|
| `/` | GET | 244 | Serve capture-tool page |
| `/cnc/status` | GET | 251 | Cached Grbl status (state, work pos, tag detect) |
| `/cnc/jog` | POST | 257 | `$J=G90 G21 X.. Y.. Z(=A).. F..` |
| `/cnc/jog/cancel` | POST | 274 | Real-time jog-cancel byte `0x85` |
| `/arduino/status` | GET | 282 | Cached `{positions: {x,z,byj1,byj2}}` |
| `/arduino/move` | POST | 288 | `{motor: x|z|byj1|byj2, steps: N}` |
| `/arduino/servo` | POST | 304 | `{angle: 0..180}` |
| `/arduino/dc` | POST | 314 | `{action: forward|reverse|stop, speed: 0..100}` |
| `/arduino/reset` | POST | 330 | `{mode: all|steppers}` |
| `/feed` | GET | 343 | MJPEG `multipart/x-mixed-replace` of annotated frames |
| `/world_coord` | POST | 361 | `{u, v}` pixel → `{x_mm, y_mm}` world XY via current pose |
| `/capture` | POST | 376 | Record current CNC pos as a hole (`type: inside|outside`) |
| `/points` | GET | 400 | Return all captured points |
| `/points` | DELETE | 405 | Clear all points |
| `/points/last` | DELETE | 411 | Undo last point |
| `/points/<idx>` | DELETE | 419 | Delete by index |
| `/export` | GET | 427 | Download points as `string_holes_<ts>.json` |
| `/import` | POST | 437 | Upload a points array (back-compat: accepts `a` or `z` key) |
| `/navigator` | GET | 939 | Serve point-navigator page |
| `/sequencer` | GET | 1219 | Serve block-sequencer page |
| `/sequencer/save` | POST | 1224 | Save sequence to `data/sequences/<name>.json` |
| `/sequencer/list` | GET | 1240 | List saved sequence names |
| `/sequencer/load` | GET | 1249 | Load sequence by name (query param `name`) |
| `/sequencer/macros/save` | POST | 1260 | Save macro to `data/macros/<name>.json` |
| `/sequencer/macros` | GET | 1274 | List saved macros (with block contents) |

---

## 8. Block Sequencer Model

### 8.1 Shape

The sequencer state is a `steps: Block[][]` array stored in
`localStorage` under `sequencer_steps`:

- Outer array → **sequential** steps.
- Inner array > 1 → **parallel** blocks within that step.

```js
[
  [ {id:1, type:'cnc-goto',     params:{x:10, y:20, a:0, feedrate:1000}} ],
  [ {id:2, type:'arduino-move', params:{motor:'z', steps:100}},
    {id:3, type:'arduino-servo',params:{angle:90}} ],          // parallel
  [ {id:4, type:'wait',         params:{seconds:2}} ],
  [ {id:5, type:'macro',        params:{name:'drill-and-insert'}} ],
]
```

### 8.2 Block types (`BLOCK_TYPES`, line 1620)

| Type | Params | Endpoint | Completion condition |
|---|---|---|---|
| `cnc-goto` | `x`, `y`, `a`, `feedrate`, `point` | `POST /cnc/jog` | Polls `/cnc/status` until `state == Idle` |
| `cnc-cancel` | — | `POST /cnc/jog/cancel` | Immediate |
| `arduino-move` | `motor` (`x`/`z`/`byj1`/`byj2`), `steps` | `POST /arduino/move` | Polls `/arduino/status` until position stable |
| `arduino-servo` | `angle` (0–180) | `POST /arduino/servo` | Immediate (servo PWM is fire-and-forget) |
| `arduino-dc` | `action` (`forward`/`reverse`/`stop`), `speed`, `duration` | `POST /arduino/dc` | Optional `duration` timer + auto-stop |
| `arduino-reset` | `mode` (`all`/`steppers`) | `POST /arduino/reset` | Immediate |
| `wait` | `seconds` | — | `setTimeout` |
| `wait-idle` | `timeout` (s) | polls `/cnc/status` | Until `state == Idle` or timeout |
| `macro` | `name` | resolves from saved macros | Recursive execution (max depth 10) |

### 8.3 Execution semantics

`runSequence()` in the sequencer JS iterates `steps` sequentially:

- If the step has one block, it `await`s that block's `execute`.
- If it has more, it does `Promise.all(step.map(b => execute(b)))`.
- Any rejected promise halts the sequence.

Macros are recursive — expanding a `macro` block runs its inner blocks
under the same scheduler. A depth counter (max 10) prevents infinite
self-reference.

### 8.4 Persistence

| Where | What |
|---|---|
| `localStorage.sequencer_steps` | Current canvas (auto-saved on every edit) |
| `localStorage.sequencer_nextId` | Block-id counter |
| `data/sequences/<name>.json` | Server-side saved sequences |
| `data/macros/<name>.json` | Server-side saved macros |
| Browser downloads | Import / export to flat JSON |

The `data/` directories are auto-created at startup (line 88-89) and
are not committed.

---

## 9. Coordinate System & Camera Math

`transforms.py` is the single source of truth for camera-to-world
geometry. Pipeline:

1. **Pixel → camera ray.** `(u, v)` → `((u-cx)/fx, (v-cy)/fy, 1)`.
2. **Camera → tag frame.** Apply `R^T · (ray − t)` using the AprilTag
   pose returned by `pupil_apriltags` for the detected tag.
3. **Intersect with Z = 0** (the tag's plane). Solves for the ray
   parameter `s` and reads off `(x, y)`.
4. **Metres → millimetres.** Multiply by 1000.

`pixels_to_world(contour, R, t)` and `world_to_pixels(world, R, t)` are
each ~30 lines of vectorised numpy.

### 9.1 Hardcoded intrinsics

`transforms.py:12-18`:

```python
FX = 1457.0856650406784
FY = 1452.6623072550426
CX = 959.5
CY = 539.5
TAG_SIZE = 0.07769  # 77.69 mm
```

These are for **1920 × 1080** capture. They must be re-entered by hand
after every calibration run — there's no auto-sync.

⚠️ **Mismatch in the repo today:** `Calib_board_outer.toml` (from
CameraKit) has 1280 × 720 intrinsics (`fx ≈ 1490.55`, `fy ≈ 1510.41`).
The values inlined in `transforms.py` are from a later 1920 × 1080
calibration that wasn't re-saved to the TOML. The TOML file is
gitignored, so this only matters if someone re-runs `camerakit report`
expecting it to match the live code.

---

## 10. Camera Calibration Workflow

For a fresh camera (e.g. you swapped lenses or moved to a different
unit):

```bash
source .venv/bin/activate

# 1. Initialise a CameraKit project (one-time)
camerakit init --path . --cameras 1
# Edit Config.toml if your checkerboard differs from 8×5 / 9.37 mm

# 2. Record a 60-second video while waving the checkerboard around
python record_calibration.py --headless --duration 60
# Open http://<pi-ip>:8080 to monitor coverage

# 3. Compute intrinsics
camerakit calibrate --config .
camerakit report --input Calib_board_outer.toml
# Note the FX, FY, CX, CY values

# 4. Paste them into transforms.py (lines 12-18) — there is no auto-sync
```

After step 4, restart any running app to pick up the new values.

---

## 11. Data Files & Persistence

| Location | Lifetime | Contents |
|---|---|---|
| In-memory `points[]` (string_holes.py:94) | Process | Captured holes. Lost on restart unless exported. |
| `/export` download | Manual | JSON snapshot of `points[]`. |
| `/import` upload | Manual | Restore points from a previous export. |
| `data/sequences/*.json` | Persistent | Saved Block Sequencer workflows. |
| `data/macros/*.json` | Persistent | Saved macros (reusable block groups). |
| `localStorage.sequencer_steps` | Browser-local | Auto-saved sequencer canvas. |
| `intrinsics/cam_00/*.mp4` | Persistent (gitignored) | Calibration recordings. |
| `calibration/` (gitignored) | Persistent | CameraKit working dir. |
| `Calib_board_outer.toml` | Local (gitignored) | CameraKit output, must be regenerated per camera. |

There is no database. Nothing in `data/` is committed.

---

## 12. Common Gotchas

- **"Arduino Connection failed: No module named 'serial'."**
  `pyserial` is not in the dev venv. `uv pip install pyserial`. See §5.3.
- **"Camera not available."** Most likely `cv2.VideoCapture` can't open
  the device. Run `v4l2-ctl --list-devices` to confirm `/dev/videoN`
  exists; the camera wrapper opens it with the V4L2 backend explicitly
  (`camera.py:62`) because integer indices misbehave on Pi 5.
- **Auto-detect grabs the wrong port.** If only one of CNC / Arduino is
  plugged in, the other auto-detect will sometimes claim it. Use the
  udev symlinks in §5.4, or pass `--serial-port` / `--serial-port-z`.
- **Grbl Z is rotation.** When you read jog code that says
  `$J=G90 G21 X10 Y20 Z45 F1000`, the `Z45` is 45° rotation, not
  45 mm vertical motion. Linear vertical is the **Arduino** Z stepper.
- **Calibration TOML doesn't match the code.** See §9.1. Trust
  `transforms.py`, not `Calib_board_outer.toml`, unless you've just
  re-run calibration.
- **Block Sequencer canvas vanished after a refresh.** It's auto-saved
  to `localStorage`; clearing site data wipes it. Use `Save` to push to
  `data/sequences/`.
- **Macros at depth > 10 silently stop expanding.** The recursion guard
  is a hard limit, not an error.
- **`Calib_board_outer.toml` gitignored.** Fresh clones won't have it;
  that's intentional — re-run `camerakit calibrate` on the target
  hardware.

---

## 13. Day-by-Day Onboarding Plan

**Day 1 — Environment.**

- Clone, `source .venv/bin/activate`, verify `python --version` is
  3.12.13.
- Run `python -c "import flask, cv2, pupil_apriltags, numpy, toml"`.
  Should be silent.
- Run `python -c "import serial"` — if this fails, `uv pip install
  pyserial`.
- Read this document and `SERIAL_COMMANDS.md`.

**Day 2 — Arduino, by hand.**

- Plug the Arduino into the dev box. Find its port: `ls /dev/ttyACM*`.
- Open a Python REPL and drive the motors directly using the snippet
  in `SERIAL_COMMANDS.md:87-121`. Move each actuator at least once.
  This builds an intuition the web UI hides.

**Day 3 — Camera + AprilTag.**

- Bring up the camera with `python detect_apriltag.py --headless`.
- Open `http://<host>:8080`, point at a tag, verify the XYZ readout
  changes when you move the tag.
- Optional: re-calibrate (§10) if your camera/lens isn't what was used
  for the values in `transforms.py`.

**Day 4 — Main app.**

- Plug in both CNC and Arduino. Symlinks ideally; otherwise pass
  `--serial-port` / `--serial-port-z`.
- `python string_holes.py --port 5001`. Open `http://<host>:5001`.
- Jog each motor from the side panel. Capture a few "inside" / "outside"
  points (you need a tag in view).
- Navigate to `/sequencer`. Drag a `CNC GoTo`, an `Arduino Move`, and a
  `Wait` onto the canvas. Click ▶ Play.

**Day 5 — Code deep-dive.** Read in this order:

1. `transforms.py` — the math foundation.
2. `cnc.py` — the simpler of the two drivers.
3. `arduino_controller.py` — the 6-actuator protocol.
4. `camera.py` and `stream.py` — capture + MJPEG.
5. `string_holes.py` — the giant glue file. Start with the imports and
   route table (§7); skim the embedded HTML.

---

## 14. Pointers

- [`README.md`](README.md) — short version of this.
- [`SERIAL_COMMANDS.md`](SERIAL_COMMANDS.md) — Arduino protocol.
- [CameraKit](https://github.com/saifkhichi96/camerakit) — upstream of
  the `camerakit` CLI.
- [Grbl wiki](https://github.com/gnea/grbl/wiki) — `$J`, `$102`,
  real-time commands, status report format.
- **Arduino firmware** — not in this repo. Currently lives only on the
  board. Tracking the source down is an open task before any firmware
  changes.

---

*Last verified against repo state at commit `4384077` on 2026-05-14.*
