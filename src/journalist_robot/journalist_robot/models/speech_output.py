"""Offline text-to-speech for the audio version of an article (optional)."""

from __future__ import annotations

import subprocess
import sys

# pyttsx3 keeps one cached engine per process, and repeated runAndWait() calls from a
# worker thread of a long-running process can hang. Each synthesis therefore runs in
# a short-lived child process.
_SCRIPT = """
import sys, pyttsx3
engine = pyttsx3.init()
engine.setProperty("rate", int(sys.argv[2]))
engine.save_to_file(sys.stdin.read(), sys.argv[1])
engine.runAndWait()
"""


def synthesize_to_file(text: str, path: str, rate: int = 170, timeout_s: float = 300.0) -> bool:
    """Write ``text`` as speech to ``path`` with pyttsx3. Returns False if unavailable or failed."""
    try:
        result = subprocess.run([sys.executable, "-c", _SCRIPT, path, str(rate)], input=text, text=True,
                                capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
