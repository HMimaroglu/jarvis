"""
Voice Input Module
==================
Non-blocking audio recording with gesture control.
Starts/stops recording based on external signals.
Transcribes with faster-whisper and types + Enter into the active window.
"""

import numpy as np
import threading
import time

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("WARNING: sounddevice not installed. Voice input disabled.")

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
    print("WARNING: faster-whisper not installed. Voice input disabled.")

try:
    import pyautogui
except ImportError:
    pyautogui = None


class VoiceInput:
    def __init__(self, model_size="base.en"):
        self.model = None
        self.sample_rate = 16000
        self.available = sd is not None and WhisperModel is not None

        # Recording state
        self.is_recording = False
        self.chunks = []
        self.stream = None
        self._record_thread = None
        self._cancelled = False

        if self.available:
            print(f"Loading Whisper model '{model_size}'...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("Whisper model loaded.")

    def start_recording(self):
        """Start recording audio in the background."""
        if not self.available or self.is_recording:
            return

        self.chunks = []
        self.is_recording = True
        self._cancelled = False

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        print("Recording started...")

    def _record_loop(self):
        """Background thread that continuously records audio chunks."""
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
            )
            self.stream.start()

            while self.is_recording and not self._cancelled:
                data, overflowed = self.stream.read(chunk_size)
                self.chunks.append(data.copy())

            self.stream.stop()
            self.stream.close()
        except Exception as e:
            print(f"Recording error: {e}")
        self.stream = None

    def stop_recording(self):
        """Stop recording and transcribe. Returns text or empty string."""
        if not self.is_recording:
            return ""

        self.is_recording = False
        if self._record_thread:
            self._record_thread.join(timeout=1.0)

        if self._cancelled or len(self.chunks) == 0:
            print("Recording cancelled.")
            return ""

        audio = np.concatenate(self.chunks, axis=0).flatten()
        duration = len(audio) / self.sample_rate

        if duration < 0.3:
            print("Recording too short.")
            return ""

        print(f"Transcribing {duration:.1f}s...")
        segments, info = self.model.transcribe(audio, beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if text:
            print(f'Heard: "{text}"')
        else:
            print("(no speech recognized)")

        return text

    def cancel_recording(self):
        """Cancel recording without transcribing."""
        self._cancelled = True
        self.is_recording = False
        if self._record_thread:
            self._record_thread.join(timeout=1.0)
        self.chunks = []
        print("Recording cancelled.")

    def type_text(self, text):
        """Type text into the currently focused window, then press Enter."""
        if not text or pyautogui is None:
            return

        try:
            from platform_utils import clipboard_paste
            clipboard_paste(text)
            time.sleep(0.05)
            pyautogui.press("enter")
        except Exception:
            try:
                pyautogui.typewrite(text, interval=0.02)
                pyautogui.press("enter")
            except Exception as e:
                print(f"Typing failed: {e}")
