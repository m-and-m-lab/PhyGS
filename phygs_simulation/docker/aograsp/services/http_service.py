"""Minimal JSON HTTP service helpers for the AO-Grasp sidecars."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Callable, Dict, Optional, Tuple


JsonHandler = Callable[[Optional[dict]], Tuple[int, dict]]


class _JsonServiceHandler(BaseHTTPRequestHandler):
    server_version = "AoGraspJsonService/1.0"

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def log_message(self, fmt: str, *args) -> None:
        return None

    def _dispatch(self, method: str) -> None:
        route_key = (method.upper(), self.path)
        handler = getattr(self.server, "routes", {}).get(route_key)
        if handler is None:
            self._write_json(404, {"error": f"Unknown route: {method} {self.path}"})
            return

        payload = None
        if method.upper() == "POST":
            try:
                payload = self._read_json_body()
            except ValueError as exc:
                self._write_json(400, {"error": str(exc)})
                return

        try:
            status, body = handler(payload)
        except Exception as exc:  # pragma: no cover - runtime safety net
            self._write_json(500, {"error": str(exc)})
            return
        self._write_json(int(status), body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _write_json(self, status: int, body: dict) -> None:
        response = json.dumps(body).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class JsonServiceServer(ThreadingHTTPServer):
    def __init__(self, server_address, routes: Dict[Tuple[str, str], JsonHandler]):
        super().__init__(server_address, _JsonServiceHandler)
        self.routes = routes


def run_json_service(host: str, port: int, routes: Dict[Tuple[str, str], JsonHandler]) -> None:
    server = JsonServiceServer((host, int(port)), routes)
    try:
        server.serve_forever()
    finally:
        server.server_close()
