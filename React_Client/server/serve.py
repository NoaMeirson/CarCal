"""Serves the built React client as a standalone desktop app.

Packaged into a single executable with PyInstaller (see build.ps1). The exe
bundles the built `dist/` assets; `config.json` stays external, next to the
executable, so the API URL can be changed without rebuilding.
"""
import http.server
import json
import os
import socket
import sys
import threading
import webbrowser

DEFAULT_API_URL = "http://localhost:8002"
DEFAULT_PORT = 5173


def get_exe_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_dist_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "dist")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist"))


def load_config() -> dict:
    config_path = os.path.join(get_exe_dir(), "config.json")
    config = {"apiUrl": DEFAULT_API_URL, "port": DEFAULT_PORT}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: could not read config.json ({e}); using defaults")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"apiUrl": DEFAULT_API_URL}, f, indent=2)
        print(f"Created default config.json at {config_path}")

    return config


def find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def make_handler(config: dict):
    dist_dir = get_dist_dir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=dist_dir, **kwargs)

        def do_GET(self):
            if self.path == "/config.js":
                body = f"window.__APP_CONFIG__ = {json.dumps({'apiUrl': config['apiUrl']})};".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    config = load_config()
    port = find_free_port(config.get("port", DEFAULT_PORT))
    url = f"http://127.0.0.1:{port}"

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(config))

    print(f"CarCal client running at {url}")
    print(f"API URL: {config['apiUrl']}  (edit config.json next to this executable to change it)")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
