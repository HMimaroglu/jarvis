"""
Platform Utilities
==================
Cross-platform abstractions for cursor control, mouse events,
window management, and monitor detection.
Supports Windows (win32) and macOS (Quartz/AppKit).
"""

import sys
import time
import platform

PLATFORM = platform.system()  # "Windows", "Darwin", "Linux"


# ── Cursor Control ──────────────────────────────────────────────────

def set_cursor_pos(x, y):
    """Move the mouse cursor to (x, y) screen coordinates."""
    if PLATFORM == "Windows":
        import win32api
        win32api.SetCursorPos((int(x), int(y)))
    elif PLATFORM == "Darwin":
        import Quartz
        event = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved,
            (float(x), float(y)), Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    else:
        import pyautogui
        pyautogui.moveTo(int(x), int(y))


def mouse_click(x, y):
    """Click the left mouse button at (x, y)."""
    if PLATFORM == "Windows":
        import win32api
        import win32con
        win32api.SetCursorPos((int(x), int(y)))
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, int(x), int(y), 0, 0)
        time.sleep(0.02)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(x), int(y), 0, 0)
    elif PLATFORM == "Darwin":
        import Quartz
        point = (float(x), float(y))
        event_down = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseDown,
            point, Quartz.kCGMouseButtonLeft
        )
        event_up = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseUp,
            point, Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
        time.sleep(0.02)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
    else:
        import pyautogui
        pyautogui.click(int(x), int(y))


# ── Monitor / Screen Bounds ────────────────────────────────────────

def get_monitor_bounds():
    """Return (left, top, right, bottom) of the primary monitor."""
    if PLATFORM == "Windows":
        import ctypes
        user32 = ctypes.windll.user32
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        return (0, 0, w, h)
    elif PLATFORM == "Darwin":
        import Quartz
        main = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return (
            int(main.origin.x),
            int(main.origin.y),
            int(main.origin.x + main.size.width),
            int(main.origin.y + main.size.height),
        )
    else:
        # Linux fallback via screeninfo
        try:
            from screeninfo import get_monitors
            m = get_monitors()[0]
            return (m.x, m.y, m.x + m.width, m.y + m.height)
        except Exception:
            return (0, 0, 1920, 1080)


def get_all_monitors():
    """Return list of (left, top, right, bottom) for each monitor."""
    if PLATFORM == "Windows":
        try:
            import win32api
            monitors = win32api.EnumDisplayMonitors()
            return [mon[2] for mon in monitors]
        except Exception:
            return [get_monitor_bounds()]
    elif PLATFORM == "Darwin":
        import Quartz
        result = []
        for display_id in Quartz.CGGetActiveDisplayList(32, None, None)[1]:
            bounds = Quartz.CGDisplayBounds(display_id)
            result.append((
                int(bounds.origin.x),
                int(bounds.origin.y),
                int(bounds.origin.x + bounds.size.width),
                int(bounds.origin.y + bounds.size.height),
            ))
        return result if result else [get_monitor_bounds()]
    else:
        try:
            from screeninfo import get_monitors
            return [(m.x, m.y, m.x + m.width, m.y + m.height) for m in get_monitors()]
        except Exception:
            return [get_monitor_bounds()]


# ── Window Management ──────────────────────────────────────────────

def get_window_at(screen_x, screen_y):
    """Get an opaque window handle at the given screen coordinates."""
    if PLATFORM == "Windows":
        import win32gui
        hwnd = win32gui.WindowFromPoint((int(screen_x), int(screen_y)))
        if not hwnd:
            return None
        root = win32gui.GetAncestor(hwnd, 2)  # GA_ROOT
        return root if root else hwnd
    elif PLATFORM == "Darwin":
        import Quartz
        # Get list of on-screen windows, front-to-back
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        for win in window_list:
            bounds = win.get("kCGWindowBounds", {})
            wx = bounds.get("X", 0)
            wy = bounds.get("Y", 0)
            ww = bounds.get("Width", 0)
            wh = bounds.get("Height", 0)
            if wx <= screen_x <= wx + ww and wy <= screen_y <= wy + wh:
                return win  # return the window dict as the "handle"
        return None
    else:
        return None


def get_window_info(handle):
    """Get info dict about a window. Returns None if invalid."""
    if handle is None:
        return None
    if PLATFORM == "Windows":
        import win32gui
        hwnd = handle
        if not win32gui.IsWindow(hwnd):
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
    elif PLATFORM == "Darwin":
        win = handle
        owner = win.get("kCGWindowOwnerName", "")
        name = win.get("kCGWindowName", "")
        title = name if name else owner
        bounds = win.get("kCGWindowBounds", {})
        return {
            "hwnd": win.get("kCGWindowNumber", 0),
            "title": title[:60] if title else "(no title)",
            "rect": (bounds.get("X", 0), bounds.get("Y", 0),
                     bounds.get("X", 0) + bounds.get("Width", 0),
                     bounds.get("Y", 0) + bounds.get("Height", 0)),
            "class": owner,
        }
    return None


def focus_window(handle):
    """Bring a window to the foreground."""
    if handle is None:
        return False
    if PLATFORM == "Windows":
        import ctypes
        import win32gui
        import win32con
        hwnd = handle
        if not win32gui.IsWindow(hwnd):
            return False
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            user32 = ctypes.windll.user32
            fg_hwnd = user32.GetForegroundWindow()
            fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
            cur_thread = user32.GetWindowThreadProcessId(hwnd, None)
            if fg_thread != cur_thread:
                user32.AttachThreadInput(fg_thread, cur_thread, True)
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
            if fg_thread != cur_thread:
                user32.AttachThreadInput(fg_thread, cur_thread, False)
            return True
        except Exception as e:
            print(f"Focus failed: {e}")
            try:
                import pyautogui
                pyautogui.press("alt")
                time.sleep(0.05)
                win32gui.SetForegroundWindow(hwnd)
                return True
            except Exception:
                return False
    elif PLATFORM == "Darwin":
        win = handle
        pid = win.get("kCGWindowOwnerPID")
        if pid:
            try:
                from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
                app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
                if app:
                    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                    return True
            except Exception as e:
                print(f"Focus failed: {e}")
        return False
    return False


# ── Clipboard / Paste ──────────────────────────────────────────────

def clipboard_paste(text):
    """Copy text to clipboard and paste it."""
    import pyautogui

    if PLATFORM == "Windows":
        import subprocess
        process = subprocess.Popen(
            ["clip.exe"], stdin=subprocess.PIPE, shell=True,
        )
        process.communicate(text.encode("utf-16-le"))
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
    elif PLATFORM == "Darwin":
        import subprocess
        process = subprocess.Popen(
            ["pbcopy"], stdin=subprocess.PIPE,
        )
        process.communicate(text.encode("utf-8"))
        time.sleep(0.05)
        pyautogui.hotkey("command", "v")
    else:
        import subprocess
        try:
            process = subprocess.Popen(
                ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE,
            )
            process.communicate(text.encode("utf-8"))
            time.sleep(0.05)
            pyautogui.hotkey("ctrl", "v")
        except FileNotFoundError:
            pyautogui.typewrite(text, interval=0.02)
