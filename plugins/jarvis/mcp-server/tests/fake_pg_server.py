"""A fake PostgreSQL that rejects every login like a crash-looping server.

Answers each startup packet with ``FATAL 57P03 the database system is in
recovery mode`` — exactly what the embedded PostgreSQL returned for ~15h during
the 2026-09 disk-full outage. Binds 127.0.0.1 on an ephemeral port; use it as a
context manager so the listener is always closed.
"""

from __future__ import annotations

import socket
import struct
import threading

_SSL_OR_GSS_REQUEST = (80877103, 80877104)


def _recovery_mode_error() -> bytes:
    fields = (
        b"SFATAL\x00VFATAL\x00C57P03\x00"
        b"Mthe database system is in recovery mode\x00\x00"
    )
    return b"E" + struct.pack("!I", len(fields) + 4) + fields


class FakeRecoveringPostgres:
    """Listener on 127.0.0.1:<ephemeral> that refuses every login with 57P03."""

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(64)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self.connections = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def url(self, user: str = "jarvis", password: str = "hunter2-secret") -> str:
        return f"postgresql://{user}:{password}@127.0.0.1:{self.port}/jarvis"

    def __enter__(self) -> "FakeRecoveringPostgres":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            self.connections += 1
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    @staticmethod
    def _handle(conn: socket.socket) -> None:
        reject_login_in_recovery(conn)


def reject_login_in_recovery(conn: socket.socket) -> None:
    """Refuse SSL/GSS, then answer the startup packet with 57P03 and hang up."""
    conn.settimeout(2)
    try:
        while True:
            header = conn.recv(4)
            if len(header) < 4:
                return
            (length,) = struct.unpack("!I", header)
            body = b""
            while len(body) < length - 4:
                chunk = conn.recv(length - 4 - len(body))
                if not chunk:
                    return
                body += chunk
            if struct.unpack("!I", body[:4])[0] in _SSL_OR_GSS_REQUEST:
                conn.sendall(b"N")
                continue
            conn.sendall(_recovery_mode_error())
            return
    except OSError:
        return
    finally:
        conn.close()
