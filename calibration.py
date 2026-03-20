"""
Calibration Module
==================
Maps camera-space fingertip positions to screen pixel coordinates
using a homography (projective transform).

Calibration flow:
1. Detect monitor layout
2. Place cursor at calibration points (corners + center)
3. User points at cursor, holds steady, presses SPACE to confirm
4. Compute homography from collected point pairs
5. Save to calibration.json
"""

import cv2
import numpy as np
import json
import os
import time
from collections import deque

from platform_utils import get_monitor_bounds, set_cursor_pos

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")


def get_calibration_points(margin=200):
    """
    Return calibration target points (screen coordinates).
    4 corners + center of the virtual desktop.
    """
    left, top, right, bottom = get_monitor_bounds()
    print(f"Virtual desktop: ({left},{top}) to ({right},{bottom})")

    points = [
        (left + margin, top + margin),           # top-left
        (right - margin, top + margin),           # top-right
        (right - margin, bottom - margin),        # bottom-right
        (left + margin, bottom - margin),         # bottom-left
        ((left + right) // 2, (top + bottom) // 2),  # center
    ]
    return points


class Calibrator:
    def __init__(self):
        self.screen_points = []
        self.camera_points = []
        self.homography = None
        self.bounds = get_monitor_bounds()

    def run_calibration(self, hand_tracker):
        """Interactive calibration using the camera feed."""
        targets = get_calibration_points()
        self.screen_points = []
        self.camera_points = []

        print("\n=== CALIBRATION ===")
        print("The cursor will move to 5 positions.")
        print("Point your index finger at the cursor each time.")
        print("Hold steady and press SPACE to confirm.")
        print("Press 'q' to abort.\n")

        # Brief pause so user can read instructions
        time.sleep(1)

        position_history = deque(maxlen=30)

        for i, (sx, sy) in enumerate(targets):
            print(f"\n--- Point {i+1}/5: Move cursor to ({sx}, {sy}) ---")

            # Move cursor to target
            set_cursor_pos(sx, sy)

            confirmed = False
            position_history.clear()

            while not confirmed:
                ret, frame = hand_tracker.get_frame()
                if not ret:
                    break

                result = hand_tracker.process(frame)
                hand_tracker.draw_debug(frame, result)

                # Show calibration info
                cv2.putText(frame, f"Calibration {i+1}/5", (10, frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Hold open hand at cursor, press SPACE", (10, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Track stability — accept any hand during calibration
                if result:
                    position_history.append(result["position"])
                    gesture_label = result["gesture"]

                    if len(position_history) >= 10:
                        positions = np.array(list(position_history))
                        std_x = np.std(positions[:, 0])
                        std_y = np.std(positions[:, 1])
                        stability = max(std_x, std_y)

                        # Show stability meter
                        bar_width = int(max(0, min(200, 200 - stability * 5)))
                        color = (0, 255, 0) if stability < 8 else (0, 165, 255)
                        cv2.rectangle(frame, (10, frame.shape[0] - 90),
                                      (10 + bar_width, frame.shape[0] - 75), color, -1)
                        cv2.putText(frame, f"Stability: {stability:.1f}px  [{gesture_label}]",
                                    (220, frame.shape[0] - 77),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Stark Control - Calibration", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(" ") and result:
                    if len(position_history) >= 5:
                        # Use average of recent positions
                        positions = np.array(list(position_history))
                        avg_pos = np.mean(positions[-15:], axis=0)
                        self.camera_points.append(tuple(avg_pos))
                        self.screen_points.append((sx, sy))
                        print(f"  Recorded: camera ({avg_pos[0]:.0f}, {avg_pos[1]:.0f}) -> screen ({sx}, {sy})")
                        confirmed = True
                    else:
                        print("  Hold steady a bit longer...")

                elif key == ord("q"):
                    print("Calibration aborted!")
                    cv2.destroyWindow("Stark Control - Calibration")
                    return False

        cv2.destroyWindow("Stark Control - Calibration")

        # Compute homography
        self._compute_homography()
        self.save()
        print("\nCalibration complete and saved!")
        return True

    def _compute_homography(self):
        src = np.float32(self.camera_points)
        dst = np.float32(self.screen_points)
        self.homography, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self.homography is None:
            print("WARNING: Homography computation failed! Using fallback linear mapping.")
            # Fallback: simple affine
            self.homography, _ = cv2.findHomography(src, dst, 0)

    def camera_to_screen(self, cam_x, cam_y):
        """Map camera pixel coords to screen coords via homography."""
        if self.homography is None:
            return (int(cam_x), int(cam_y))

        pt = np.array([cam_x, cam_y, 1.0])
        mapped = self.homography @ pt
        sx = mapped[0] / mapped[2]
        sy = mapped[1] / mapped[2]

        # Clamp to virtual desktop bounds
        left, top, right, bottom = self.bounds
        sx = max(left, min(right - 1, sx))
        sy = max(top, min(bottom - 1, sy))

        return (int(sx), int(sy))

    def save(self):
        data = {
            "homography": self.homography.tolist() if self.homography is not None else None,
            "screen_points": self.screen_points,
            "camera_points": self.camera_points,
            "bounds": list(self.bounds),
        }
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved calibration to {CALIBRATION_FILE}")

    def load(self):
        if not os.path.exists(CALIBRATION_FILE):
            return False
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            self.homography = np.array(data["homography"]) if data["homography"] else None
            self.screen_points = [tuple(p) for p in data["screen_points"]]
            self.camera_points = [tuple(p) for p in data["camera_points"]]
            self.bounds = tuple(data["bounds"])
            print(f"Loaded calibration ({len(self.screen_points)} points)")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
