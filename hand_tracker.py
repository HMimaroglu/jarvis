"""
Hand Tracking Module
====================
MediaPipe-based hand tracking using the Tasks API.
Tracks hand position in camera space.
Open hand = active tracking, closed fist = inactive.
"""

import cv2
import numpy as np
import os
import sys

from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp

# Landmark indices
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17
THUMB_TIP = 4
THUMB_MCP = 2
WRIST = 0

# Palm center approximation landmarks
PALM_LANDMARKS = [0, 5, 9, 13, 17]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


class HandTracker:
    def __init__(self, camera_index=None):
        self.cap = None
        if camera_index is not None:
            self.cap = cv2.VideoCapture(camera_index)
        else:
            for idx in [0, 1, 2]:
                test = cv2.VideoCapture(idx)
                if test.isOpened():
                    ret, _ = test.read()
                    if ret:
                        print(f"Found camera at index {idx}")
                        self.cap = test
                        break
                    test.release()

        if self.cap is None:
            print("ERROR: No camera found!")
            sys.exit(1)

        # Read a test frame to get actual resolution
        ret, test_frame = self.cap.read()
        if ret:
            self.h, self.w = test_frame.shape[:2]
        else:
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {self.w}x{self.h}")

        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Hand landmarker model not found at {MODEL_PATH}")
            print("Download it with:")
            print('  curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"')
            sys.exit(1)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
        return ret, frame

    def process(self, frame):
        """
        Returns dict with:
          - position: (x, y) in camera pixels — palm center, used for screen mapping
          - active: True if hand is open (tracking active), False if fist (paused)
          - gesture: 'open_palm', 'fist', or 'back_hand'
          - landmarks: list of NormalizedLandmark (for drawing)
        Or None if no hand detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]

        # Palm center: average of wrist + finger MCPs
        palm_x = np.mean([lm[i].x for i in PALM_LANDMARKS]) * self.w
        palm_y = np.mean([lm[i].y for i in PALM_LANDMARKS]) * self.h

        # Detect palm orientation and gesture
        palm_facing = self._is_palm_facing_camera(lm)
        fingers_open = self._count_fingers_extended(lm) >= 3
        gesture = self._classify_gesture(palm_facing, fingers_open)

        return {
            "position": (float(palm_x), float(palm_y)),
            "active": gesture == "open_palm",
            "gesture": gesture,
            "palm_facing": palm_facing,
            "landmarks": lm,
        }

    def _is_palm_facing_camera(self, lm):
        """
        Check if the palm faces the camera using the palm normal vector.
        Cross product of (wrist->index_mcp) x (wrist->pinky_mcp) gives the palm normal.
        If the z-component is negative, the palm faces the camera.
        """
        wrist = np.array([lm[WRIST].x, lm[WRIST].y, lm[WRIST].z])
        index_mcp = np.array([lm[INDEX_MCP].x, lm[INDEX_MCP].y, lm[INDEX_MCP].z])
        pinky_mcp = np.array([lm[PINKY_MCP].x, lm[PINKY_MCP].y, lm[PINKY_MCP].z])

        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        normal = np.cross(v1, v2)
        # Positive z = palm faces camera (flipped due to mirror/frame flip)
        return normal[2] > 0

    def _count_fingers_extended(self, lm):
        """Count how many fingers are extended using 3D landmarks."""
        wrist = np.array([lm[WRIST].x, lm[WRIST].y, lm[WRIST].z])

        count = 0
        finger_pairs = [
            (INDEX_TIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_MCP),
            (RING_TIP, RING_MCP),
            (PINKY_TIP, PINKY_MCP),
        ]

        for tip_idx, mcp_idx in finger_pairs:
            tip = np.array([lm[tip_idx].x, lm[tip_idx].y, lm[tip_idx].z])
            mcp = np.array([lm[mcp_idx].x, lm[mcp_idx].y, lm[mcp_idx].z])

            tip_dist = np.linalg.norm(tip - wrist)
            mcp_dist = np.linalg.norm(mcp - wrist)

            if tip_dist > mcp_dist * 1.1:
                count += 1

        return count

    def _classify_gesture(self, palm_facing, fingers_open):
        """Classify the hand gesture based on palm orientation and finger state."""
        if palm_facing and fingers_open:
            return "open_palm"   # active tracking
        elif not fingers_open:
            return "fist"        # paused
        else:
            return "back_hand"   # palm facing away, not tracking

    def draw_debug(self, frame, result):
        """Draw hand skeleton and tracking overlay on frame."""
        if result is None:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return

        # Draw landmarks using the new Tasks API
        drawing_utils.draw_landmarks(
            frame,
            result["landmarks"],
            HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            drawing_utils.DrawingSpec(color=(0, 200, 0), thickness=2),
        )

        lm = result["landmarks"]
        pos_x = int(np.mean([lm[i].x for i in PALM_LANDMARKS]) * frame.shape[1])
        pos_y = int(np.mean([lm[i].y for i in PALM_LANDMARKS]) * frame.shape[0])
        pos = (pos_x, pos_y)

        gesture = result["gesture"]

        if gesture == "open_palm":
            # Active tracking — green crosshair
            cv2.circle(frame, pos, 15, (0, 255, 255), 2)
            cv2.line(frame, (pos[0] - 20, pos[1]), (pos[0] + 20, pos[1]), (0, 255, 255), 1)
            cv2.line(frame, (pos[0], pos[1] - 20), (pos[0], pos[1] + 20), (0, 255, 255), 1)
            cv2.putText(frame, "TRACKING (PALM)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif gesture == "fist":
            cv2.circle(frame, pos, 15, (0, 0, 255), 2)
            cv2.putText(frame, "PAUSED (FIST)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif gesture == "back_hand":
            cv2.circle(frame, pos, 15, (0, 165, 255), 2)
            cv2.putText(frame, "PALM AWAY - face palm to camera", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    def release(self):
        self.cap.release()
        self.landmarker.close()
