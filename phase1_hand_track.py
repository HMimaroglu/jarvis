"""
Phase 1: Hand Tracking + Pointing Direction Detection
======================================================
Uses MediaPipe Hands (Tasks API) to detect your hand and figure out where you're pointing.
Draws a ray from your index finger showing the pointing direction.
"""

import cv2
import numpy as np
import os
import sys

import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)
from mediapipe.tasks.python import BaseOptions

# Landmark indices we care about
INDEX_TIP = 8
INDEX_DIP = 7
INDEX_PIP = 6
INDEX_MCP = 5
WRIST = 0

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def get_pointing_vector(landmarks, w, h):
    """
    Compute the pointing direction from the index finger.
    Returns: (base_point, tip_point, direction_vector) in pixel coords,
             or None if hand isn't in a pointing pose.
    """
    # Get key points as numpy arrays (pixel coords)
    index_tip = np.array([landmarks[INDEX_TIP].x * w, landmarks[INDEX_TIP].y * h])
    index_mcp = np.array([landmarks[INDEX_MCP].x * w, landmarks[INDEX_MCP].y * h])

    # Check if index finger is extended (tip is far from MCP)
    finger_length = np.linalg.norm(index_tip - index_mcp)

    # Basic check: finger should be relatively straight (extended)
    if finger_length < 50:  # finger too curled
        return None

    # Direction: from MCP through the tip, extended outward
    direction = index_tip - index_mcp
    direction_norm = direction / (np.linalg.norm(direction) + 1e-6)

    # Extend the ray far out from the tip
    ray_end = index_tip + direction_norm * 1000

    return index_mcp, index_tip, ray_end, direction_norm


def main():
    # Try to find the camera
    cap = None
    for cam_index in [0, 1, 2]:
        test = cv2.VideoCapture(cam_index)
        if test.isOpened():
            ret, frame = test.read()
            if ret:
                print(f"Found camera at index {cam_index}")
                cap = test
                break
            test.release()

    if cap is None:
        print("ERROR: No camera found! Make sure your camera is plugged in.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {w}x{h}")
    print("Press 'q' to quit")
    print("Point with your index finger - you should see a ray drawn from your finger!")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Hand landmarker model not found at {MODEL_PATH}")
        print("Download it with:")
        print('  curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"')
        sys.exit(1)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        # Flip horizontally so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            for landmarks in results.hand_landmarks:
                # Draw hand skeleton
                drawing_utils.draw_landmarks(
                    frame, landmarks, HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    drawing_utils.DrawingSpec(color=(0, 200, 0), thickness=2),
                )

                # Get pointing direction
                result = get_pointing_vector(landmarks, w, h)
                if result is not None:
                    base, tip, ray_end, direction = result

                    # Draw the pointing ray (bright cyan line)
                    cv2.line(
                        frame,
                        tuple(tip.astype(int)),
                        tuple(ray_end.astype(int)),
                        (0, 255, 255),  # cyan
                        3,
                    )

                    # Draw a circle at the fingertip
                    cv2.circle(frame, tuple(tip.astype(int)), 10, (0, 0, 255), -1)

                    # Show direction info
                    angle = np.degrees(np.arctan2(-direction[1], direction[0]))
                    cv2.putText(
                        frame,
                        f"Pointing: {angle:.0f} deg",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )

                    # Show normalized direction
                    cv2.putText(
                        frame,
                        f"Dir: ({direction[0]:.2f}, {direction[1]:.2f})",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 200),
                        2,
                    )
        else:
            cv2.putText(
                frame,
                "No hand detected - show your hand!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Show the frame
        cv2.imshow("Stark Control - Phase 1: Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
