# Camera Calibration & AprilTag Pose Estimation

Camera intrinsic calibration using [CameraKit](https://github.com/saifkhichi96/camerakit) and real-time AprilTag 6DoF pose estimation with headless MJPEG streaming — designed to run on a Raspberry Pi 5.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install camerakit opencv-python pupil-apriltags flask
```

## Workflow

### 1. Calibrate Camera

Initialize a project and record a calibration video while viewing the live stream:

```bash
camerakit init --path . --cameras 1
# Edit Config.toml to match your checkerboard (corners, square size)
python record_calibration.py --headless --duration 60
```

Open `http://<device-ip>:8080` to see the live feed while recording. Move a checkerboard pattern slowly across the frame at various angles.

Then run calibration:

```bash
camerakit calibrate --config .
camerakit report --input Calib_board_outer.toml
```

Update the camera intrinsics in `detect_apriltag.py` with the values from the calibration output.

### 2. Detect AprilTags

Run the detector with pose estimation:

```bash
# With display
python detect_apriltag.py

# Headless (RPi)
python detect_apriltag.py --headless --port 8080
```

Open `http://<device-ip>:8080` in a browser to see the annotated feed with:
- Tag ID and green outline
- XYZ position (meters) and Roll/Pitch/Yaw (degrees)
- 3D axes overlay (RGB = XYZ)

## Configuration

Edit the constants at the top of `detect_apriltag.py`:

| Parameter | Description |
|-----------|-------------|
| `FX`, `FY`, `CX`, `CY` | Camera intrinsics from calibration |
| `TAG_SIZE` | Physical tag size in meters |
| `families` | AprilTag family (default: `tag36h11`) |

## Files

| File | Description |
|------|-------------|
| `detect_apriltag.py` | AprilTag detection + 6DoF pose estimation |
| `record_calibration.py` | Record calibration video with live preview |
| `stream.py` | Reusable MJPEG streaming server |
| `Config.toml` | CameraKit calibration configuration |
