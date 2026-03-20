"""
Window Manager Module
=====================
Find and focus windows at screen coordinates using Win32 API.
"""

import ctypes
import ctypes.wintypes
import time

try:
    import win32gui
    import win32api
    import win32con
    import win32process
except ImportError:
    raise ImportError("pywin32 required: pip install pywin32")


def get_window_at(screen_x, screen_y):
    """Get the top-level window handle at the given screen coordinates."""
    hwnd = win32gui.WindowFromPoint((screen_x, screen_y))
    if not hwnd:
        return None

    # Walk up to the top-level (owned) window
    root = win32gui.GetAncestor(hwnd, 2)  # GA_ROOT = 2
    if root:
        hwnd = root

    return hwnd


def get_window_info(hwnd):
    """Get info about a window."""
    if not hwnd or not win32gui.IsWindow(hwnd):
        return None
    try:
        title = win32gui.GetWindowText(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        cls = win32gui.GetClassName(hwnd)
        return {
            "hwnd": hwnd,
            "title": title[:60] if title else "(no title)",
            "rect": rect,
            "class": cls,
        }
    except Exception:
        return None


def focus_window(hwnd):
    """
    Bring a window to the foreground.
    Uses the Alt-key trick to bypass Windows' focus-stealing prevention.
    """
    if not hwnd or not win32gui.IsWindow(hwnd):
        return False

    try:
        # If minimized, restore it
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        # The Alt-key trick: simulate an Alt press to allow SetForegroundWindow
        # This works because Windows allows the foreground process to set focus
        user32 = ctypes.windll.user32

        # Get current foreground window's thread
        fg_hwnd = user32.GetForegroundWindow()
        fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
        cur_thread = user32.GetWindowThreadProcessId(hwnd, None)

        # Attach thread input to trick Windows into allowing focus change
        if fg_thread != cur_thread:
            user32.AttachThreadInput(fg_thread, cur_thread, True)

        user32.SetForegroundWindow(hwnd)
        user32.BringWindowToTop(hwnd)

        if fg_thread != cur_thread:
            user32.AttachThreadInput(fg_thread, cur_thread, False)

        return True

    except Exception as e:
        print(f"Focus failed: {e}")
        # Fallback: simulate Alt key
        try:
            import pyautogui
            pyautogui.press("alt")
            time.sleep(0.05)
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False
