"""Backwards-compatible alias for scripts/launch.py."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    print("[launch] scripts/launch_cockpit.py is now scripts/launch.py; forwarding.")
    sys.argv[0] = str(Path(__file__).with_name("launch.py"))
    runpy.run_path(sys.argv[0], run_name="__main__")
