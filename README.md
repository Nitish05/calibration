# Tennis Racket Calibration Rig

Control software for a Raspberry Pi 5 stringing rig — a Flask web app that
drives a Grbl CNC, an Arduino-based 6-actuator board, and a USB camera
with AprilTag pose overlay, plus a visual **Block Sequencer** for
composing CNC + Arduino macros. Includes standalone tools for camera
intrinsic calibration, hole-vector measurement, and racket-frame
segmentation.

> New to this repo? Start with **[`HANDOFF.md`](HANDOFF.md)** — full
> cold-start guide.

## Quick Start

```bash
cd /home/nitish/work/calibration
source .venv/bin/activate
python string_holes.py --port 5001
# open http://<host>:5001
```

If the venv is missing or you're on a fresh machine, see
[`HANDOFF.md` §5](HANDOFF.md#5-virtual-environment--first-run-setup) for
the bootstrap commands. Heads-up: the committed venv is **missing
`pyserial`** — `uv pip install pyserial` before talking to hardware.

## What's In Here

| Script | Port | Purpose |
|---|---|---|
| `string_holes.py` | 5001 | **Main app** — capture page, point navigator, block sequencer, motor control, AprilTag feed |
| `hole_vector.py` | 5000 | Dual-perspective hole-vector measurement (spawns `pi_stream.py`) |
| `detect_apriltag.py` | 8080 | Standalone AprilTag overlay viewer |
| `record_calibration.py` | 8080 | Record a CameraKit calibration video with live preview |
| `color_mask_tuner.py` | 8082 | Interactive HSV / LAB threshold tuner |
| `segment_racket_v2.py` | 8081 | Racket-frame segmenter (ellipse fit, preferred) |
| `segment_racket.py` | 8081 | v1 segmenter — deprecated, kept for reference |
| `pi_stream.py` | 8080 | Bare picamera2 MJPEG server (helper for `hole_vector.py`) |

Library modules: `cnc.py` (Grbl), `arduino_controller.py` (6 actuators),
`camera.py` (Picamera2/OpenCV), `transforms.py` (intrinsics + world math),
`stream.py` (reusable MJPEG Flask server).

## Hardware

| Subsystem | Wiring |
|---|---|
| Main Grbl CNC | CH340 USB → `/dev/cnc_main`. Drives X, Y, and a **rotational A on the Z output** (`$102` re-configured). |
| Arduino Uno + CNC Shield V3 | USB → `/dev/cnc_aux`. Six actuators: X / Z NEMA steppers, two 28BYJ-48s, an SG90 servo, and a 5 V DC motor on L298N. |
| Camera | Logitech C920 (USB, 1920×1080 MJPEG) primary; Pi CSI via `picamera2` optional. |

Arduino serial protocol: **[`SERIAL_COMMANDS.md`](SERIAL_COMMANDS.md)**.

## Camera Calibration

```bash
camerakit init --path . --cameras 1               # one-time
python record_calibration.py --headless --duration 60
camerakit calibrate --config .
camerakit report --input Calib_board_outer.toml
# copy FX/FY/CX/CY into transforms.py (lines 12-18)
```

Move the checkerboard slowly across the frame at multiple angles while
recording. The `Calib_board_outer.toml` file is gitignored — each camera
must be calibrated locally. See
[`HANDOFF.md` §10](HANDOFF.md#10-camera-calibration-workflow) for detail.

## Configuration

The intrinsics consumed by the AprilTag pipeline live in
[`transforms.py:12-18`](transforms.py):

| Constant | Description |
|---|---|
| `FX`, `FY`, `CX`, `CY` | Camera intrinsics from calibration |
| `TAG_SIZE` | Physical AprilTag size, metres (default `0.07769` ≈ 77.69 mm) |

The AprilTag family is hardcoded as `tag36h11` in the detector instances
in `detect_apriltag.py` and `string_holes.py`.

CameraKit settings (checkerboard corners, square size) live in
[`Config.toml`](Config.toml).

## Documentation

- **[HANDOFF.md](HANDOFF.md)** — cold-start guide: hardware, venv, every
  script, every Flask route, every block type, common gotchas, 5-day
  onboarding plan.
- **[SERIAL_COMMANDS.md](SERIAL_COMMANDS.md)** — Arduino serial protocol
  reference.

## Stack

Python 3.12.13 in a `uv`-managed `.venv`. Flask 3, OpenCV 4.13,
pupil-apriltags, pyserial, numpy, toml, camerakit 2.0. No Node / npm,
no build step, no Dockerfile, no CI. Plotly is loaded from CDN at
runtime.
