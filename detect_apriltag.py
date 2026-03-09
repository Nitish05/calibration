import cv2
import numpy as np
import argparse
from pupil_apriltags import Detector
from stream import StreamServer
from camera import Camera

# Camera intrinsics from calibration (C920 @ 1920x1080)
FX = 2025.9561727564542
FY = 2026.9282796080697
CX = 959.5
CY = 539.5

# AprilTag settings
TAG_SIZE = 0.05684  # 56.84 mm

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Stream via HTTP instead of cv2.imshow")
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=2.0,
    quad_sigma=0.0,
    decode_sharpening=0.25,
    refine_edges=True,
)

CAMERA_MATRIX = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
])

cap = Camera()

stream = None
if args.headless:
    stream = StreamServer(port=args.port, title="AprilTag Detector")
    stream.start()

print(f"Tag family: tag36h11 | Tag size: {TAG_SIZE*100:.1f} cm")
print(f"Camera: fx={FX:.1f} fy={FY:.1f} cx={CX:.1f} cy={CY:.1f}")
if not args.headless:
    print("Press 'q' to quit")
else:
    print("Press Ctrl+C to quit")
print("-" * 60)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[FX, FY, CX, CY],
            tag_size=TAG_SIZE,
        )

        for det in detections:
            R = det.pose_R
            t = det.pose_t
            x, y, z = t.flatten()

            # Rotation matrix to Euler angles
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            if sy > 1e-6:
                roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                pitch = np.degrees(np.arctan2(-R[2, 0], sy))
                yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            else:
                roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
                pitch = np.degrees(np.arctan2(-R[2, 0], sy))
                yaw = 0

            # Draw tag outline
            corners = det.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

            # Draw center
            cx_tag, cy_tag = int(det.center[0]), int(det.center[1])
            cv2.circle(frame, (cx_tag, cy_tag), 5, (0, 0, 255), -1)

            # Draw 3D axes
            axis_len = TAG_SIZE * 0.5
            axis_pts = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,-axis_len]])
            rvec, _ = cv2.Rodrigues(R)
            tvec = t.reshape(3, 1)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, CAMERA_MATRIX, None)
            img_pts = img_pts.astype(int).reshape(-1, 2)
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), (0, 0, 255), 2)   # X red
            cv2.line(frame, origin, tuple(img_pts[2]), (0, 255, 0), 2)   # Y green
            cv2.line(frame, origin, tuple(img_pts[3]), (255, 0, 0), 2)   # Z blue

            # On-screen text
            cv2.putText(frame, f"ID:{det.tag_id}", (corners[0][0], corners[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"XYZ: {x:.3f}, {y:.3f}, {z:.3f} m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"RPY: {roll:.1f}, {pitch:.1f}, {yaw:.1f} deg",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            print(f"Tag {det.tag_id:3d} | "
                  f"X={x:+.3f} Y={y:+.3f} Z={z:+.3f} m | "
                  f"Roll={roll:+6.1f} Pitch={pitch:+6.1f} Yaw={yaw:+6.1f} deg")

        if args.headless:
            stream.update_frame(frame)
        else:
            cv2.imshow("AprilTag Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopping...")

cap.release()
if not args.headless:
    cv2.destroyAllWindows()
