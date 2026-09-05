"""
One command to start URA-Shree.

Builds the frontend if the bundle is missing or stale, then serves the API and
the compiled UI from a single port. Pass --dev to run Vite's dev server
alongside the API instead, which gives hot reload while editing the interface.

    python scripts/launch.py
    python scripts/launch.py --dev
    python scripts/launch.py --port 9000 --workspace ../some-project
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = PROJECT_ROOT / "frontend"
DIST = FRONTEND / "dist"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def npm() -> str:
    """npm on Windows is a .cmd shim, which subprocess will not find unaided."""
    return shutil.which("npm") or shutil.which("npm.cmd") or "npm"


def bundle_is_stale() -> bool:
    """True when a source file is newer than the built bundle."""
    index = DIST / "index.html"
    if not index.exists():
        return True
    built_at = index.stat().st_mtime
    # public/ is copied into the bundle verbatim (the landing page lives there),
    # so a change under it is just as stale-making as one under src/.
    for root in ((FRONTEND / "src"), (FRONTEND / "public")):
        for path in root.rglob("*"):
            if path.is_file() and path.stat().st_mtime > built_at:
                return True
    return (FRONTEND / "index.html").stat().st_mtime > built_at


def build_frontend() -> bool:
    if not (FRONTEND / "package.json").exists():
        print("[launch] No frontend directory; serving the API only.")
        return False

    if not (FRONTEND / "node_modules").exists():
        print("[launch] Installing frontend dependencies (first run only)...")
        if subprocess.call([npm(), "install"], cwd=str(FRONTEND)) != 0:
            print("[launch] npm install failed.")
            return False

    print("[launch] Building the interface...")
    if subprocess.call([npm(), "run", "build"], cwd=str(FRONTEND)) != 0:
        print("[launch] Build failed. Fix the errors above and try again.")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the URA-Shree server and interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workspace", default=None,
                        help="Directory the agent works in (defaults to ./workspace)")
    parser.add_argument("--dev", action="store_true",
                        help="Run the Vite dev server for hot reload on port 5173")
    parser.add_argument("--no-build", action="store_true", help="Serve the existing bundle as-is")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    parser.add_argument("--reload", action="store_true", help="Reload the API on Python changes")
    args = parser.parse_args()

    if args.workspace:
        workspace = Path(args.workspace).resolve()
        if not workspace.is_dir():
            sys.exit(f"[launch] No such directory: {workspace}")
    else:
        workspace = (PROJECT_ROOT / "workspace").resolve()
        workspace.mkdir(parents=True, exist_ok=True)
    os.environ["SHREE_WORKSPACE"] = str(workspace)

    vite = None
    if args.dev:
        print("[launch] Starting the Vite dev server on http://127.0.0.1:5173")
        vite = subprocess.Popen([npm(), "run", "dev"], cwd=str(FRONTEND))
        ui_url = "http://127.0.0.1:5173"
    else:
        if not args.no_build and bundle_is_stale():
            build_frontend()
        elif args.no_build:
            print("[launch] Skipping the frontend build.")
        else:
            print("[launch] Frontend bundle is up to date.")
        ui_url = f"http://{args.host}:{args.port}"

    print()
    print("=" * 62)
    print("  URA-Shree")
    print(f"  Interface   {ui_url}")
    print(f"  Overview    {ui_url}/landing.html")
    print(f"  API         http://{args.host}:{args.port}/api/status")
    print(f"  Workspace   {workspace}")
    print("=" * 62)
    print("  Paste an API key under Settings, scan for models, and pick one.")
    print("  Ctrl+C to stop.")
    print()

    if not args.no_browser:
        # A short delay so the server is accepting connections when the tab opens.
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(ui_url)).start()

    import uvicorn

    try:
        uvicorn.run(
            "server.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass
    finally:
        if vite is not None:
            vite.terminate()
        print("\n[launch] Stopped.")


if __name__ == "__main__":
    main()
