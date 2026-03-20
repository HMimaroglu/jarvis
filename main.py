"""
Stark Control - Main Orchestrator
==================================
Tony Stark-style gesture-driven computer control:
  - Open palm: cursor follows your hand
  - Close fist: clicks at cursor position, starts voice recording
  - Open palm again: stops recording, transcribes, types + Enter
  - Hand leaves while recording: cancels recording

Controls (keyboard fallbacks):
  C - Recalibrate
  Q - Quit
"""

import cv2
import numpy as np
import time
import sys

from hand_tracker import HandTracker
from calibration import Calibrator
from window_manager import get_window_at, get_window_info, focus_window
from voice_input import VoiceInput

# States
STATE_IDLE = "IDLE"             # no hand detected
STATE_TRACKING = "TRACKING"     # open palm, cursor follows
STATE_RECORDING = "RECORDING"   # fist closed, recording voice


class StarkControl:
    def __init__(self):
        self.tracker = HandTracker()
        self.calibrator = Calibrator()
        self.voice = VoiceInput()

        # State
        self.state = STATE_IDLE
        self.smooth_pos = None
        self.current_hwnd = None
        self.focused_hwnd = None
        self.alpha = 0.35
        self.dead_zone = 5
        self.last_cam_pos = None
        self.last_screen = (None, None)

    def run(self):
        # Load or run calibration
        if not self.calibrator.load():
            print("No calibration found. Starting calibration...")
            if not self.calibrator.run_calibration(self.tracker):
                print("Calibration failed. Exiting.")
                return

        print("\n=== STARK CONTROL ACTIVE ===")
        print("Open palm = move cursor")
        print("Close fist = click + record voice")
        print("Open palm = stop recording + type + Enter")
        print("C = Recalibrate | Q = Quit\n")

        # Make window always-on-top
        cv2.namedWindow("Stark Control", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Stark Control", cv2.WND_PROP_TOPMOST, 1)

        while True:
            ret, frame = self.tracker.get_frame()
            if not ret:
                break

            result = self.tracker.process(frame)
            screen_x, screen_y = None, None

            if result:
                cam_x, cam_y = result["position"]

                # Dead zone filter
                if self.last_cam_pos is not None:
                    dx = abs(cam_x - self.last_cam_pos[0])
                    dy = abs(cam_y - self.last_cam_pos[1])
                    if dx < self.dead_zone and dy < self.dead_zone:
                        cam_x, cam_y = self.last_cam_pos
                self.last_cam_pos = (cam_x, cam_y)

                # Map to screen coordinates
                raw_sx, raw_sy = self.calibrator.camera_to_screen(cam_x, cam_y)

                # EMA smoothing
                if self.smooth_pos is None:
                    self.smooth_pos = (float(raw_sx), float(raw_sy))
                else:
                    self.smooth_pos = (
                        self.smooth_pos[0] * (1 - self.alpha) + raw_sx * self.alpha,
                        self.smooth_pos[1] * (1 - self.alpha) + raw_sy * self.alpha,
                    )

                screen_x = int(self.smooth_pos[0])
                screen_y = int(self.smooth_pos[1])
                self.last_screen = (screen_x, screen_y)

                # Find window at position
                hwnd = get_window_at(screen_x, screen_y)
                if hwnd:
                    self.current_hwnd = hwnd

                gesture = result["gesture"]

                # --- State transitions ---
                if self.state == STATE_IDLE:
                    if gesture == "open_palm":
                        self.state = STATE_TRACKING

                elif self.state == STATE_TRACKING:
                    if gesture == "open_palm":
                        # Move cursor
                        try:
                            import win32api
                            win32api.SetCursorPos((screen_x, screen_y))
                        except Exception:
                            pass
                    elif gesture == "fist":
                        # Click at current position and start recording
                        self._do_click(screen_x, screen_y)
                        self.voice.start_recording()
                        self.state = STATE_RECORDING

                elif self.state == STATE_RECORDING:
                    if gesture == "open_palm":
                        # Stop recording, transcribe, type
                        self._finish_recording()
                        self.state = STATE_TRACKING

            else:
                # No hand detected
                if self.state == STATE_RECORDING:
                    # Hand left the screen — cancel recording
                    self.voice.cancel_recording()
                    self.state = STATE_IDLE
                    print("Hand lost — recording cancelled.")
                elif self.state == STATE_TRACKING:
                    self.state = STATE_IDLE

            # Draw debug overlay
            self.tracker.draw_debug(frame, result)
            self._draw_hud(frame, result, screen_x, screen_y)

            cv2.imshow("Stark Control", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                if self.state == STATE_RECORDING:
                    self.voice.cancel_recording()
                break
            elif key == ord("c"):
                if self.state == STATE_RECORDING:
                    self.voice.cancel_recording()
                print("Recalibrating...")
                self.state = STATE_IDLE
                self.calibrator.run_calibration(self.tracker)

        self.tracker.release()
        cv2.destroyAllWindows()
        print("Stark Control shut down.")

    def _do_click(self, screen_x, screen_y):
        """Click at the screen position to focus the element under cursor."""
        try:
            import win32api
            import win32con
            # Move cursor to position
            win32api.SetCursorPos((screen_x, screen_y))
            time.sleep(0.05)
            # Click
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
            time.sleep(0.02)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
            time.sleep(0.05)

            hwnd = get_window_at(screen_x, screen_y)
            if hwnd:
                self.focused_hwnd = hwnd
                info = get_window_info(hwnd)
                title = info["title"] if info else "unknown"
                print(f"Clicked: ({screen_x},{screen_y}) — {title}")
        except Exception as e:
            print(f"Click failed: {e}")

    def _finish_recording(self):
        """Stop recording, transcribe, and type into focused window."""
        text = self.voice.stop_recording()
        if text and self.focused_hwnd:
            # Re-focus the window before typing
            focus_window(self.focused_hwnd)
            time.sleep(0.1)
            self.voice.type_text(text)
            print(f'Typed: "{text}"')

    def _draw_hud(self, frame, result, screen_x, screen_y):
        """Draw the heads-up display overlay."""
        h, w = frame.shape[:2]

        # Status bar background
        cv2.rectangle(frame, (0, h - 120), (w, h), (30, 30, 30), -1)

        # State indicator (large, top-left area)
        state_colors = {
            STATE_IDLE: ((100, 100, 100), "IDLE"),
            STATE_TRACKING: ((0, 255, 255), "TRACKING"),
            STATE_RECORDING: ((0, 0, 255), "RECORDING"),
        }
        color, label = state_colors[self.state]
        cv2.putText(frame, label, (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Recording indicator — pulsing red dot
        if self.state == STATE_RECORDING:
            pulse = int(abs(time.time() % 1 - 0.5) * 2 * 255)
            cv2.circle(frame, (w - 220, 25), 8, (0, 0, max(100, pulse)), -1)

        if screen_x is not None:
            cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", (10, h - 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            if self.current_hwnd:
                info = get_window_info(self.current_hwnd)
                if info:
                    title = info["title"][:50]
                    cv2.putText(frame, f"Target: {title}", (10, h - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if self.focused_hwnd:
            info = get_window_info(self.focused_hwnd)
            if info:
                cv2.putText(frame, f"Focused: {info['title'][:40]}", (10, h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Instructions
        cv2.putText(frame, "Palm=Move  Fist=Click+Record  Palm=Type  |  C:Calibrate  Q:Quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Draw minimap
        self._draw_minimap(frame, screen_x, screen_y)

    def _draw_minimap(self, frame, screen_x, screen_y):
        """Draw a minimap showing monitor layout and pointer position."""
        h, w = frame.shape[:2]
        left, top, right, bottom = self.calibrator.bounds
        desk_w = right - left
        desk_h = bottom - top

        map_w = 160
        map_h = int(map_w * desk_h / desk_w)
        map_x = w - map_w - 10
        map_y = 45

        cv2.rectangle(frame, (map_x - 2, map_y - 2),
                      (map_x + map_w + 2, map_y + map_h + 2), (80, 80, 80), -1)
        cv2.rectangle(frame, (map_x, map_y),
                      (map_x + map_w, map_y + map_h), (20, 20, 20), -1)

        scale = map_w / desk_w

        try:
            import win32api
            monitors = win32api.EnumDisplayMonitors()
            for mon in monitors:
                mx1, my1, mx2, my2 = mon[2]
                rx1 = map_x + int((mx1 - left) * scale)
                ry1 = map_y + int((my1 - top) * scale)
                rx2 = map_x + int((mx2 - left) * scale)
                ry2 = map_y + int((my2 - top) * scale)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (60, 60, 60), -1)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (100, 100, 100), 1)
        except Exception:
            cv2.rectangle(frame, (map_x, map_y),
                          (map_x + map_w, map_y + map_h), (60, 60, 60), -1)

        if screen_x is not None:
            px = map_x + int((screen_x - left) * scale)
            py = map_y + int((screen_y - top) * scale)
            px = max(map_x, min(map_x + map_w, px))
            py = max(map_y, min(map_y + map_h, py))

            dot_color = (0, 0, 255) if self.state == STATE_RECORDING else (0, 200, 255)
            cv2.circle(frame, (px, py), 6, dot_color, -1)
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)


def main():
    ctrl = StarkControl()
    ctrl.run()


if __name__ == "__main__":
    main()
