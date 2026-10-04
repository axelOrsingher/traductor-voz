"""Transporte motor <-> UI: JSON-lines sobre TCP localhost.

El motor publica eventos ({"type": ...}) a todos los clientes conectados y
recibe comandos ({"cmd": ...}). La UI nunca llama al motor directamente.
En Linux (fase 3) el mismo protocolo va sobre un unix socket.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable


class JsonLinesServer:
    def __init__(self, host: str, port: int, on_command: Callable[[dict], None]):
        self.on_command = on_command
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._srv = socket.create_server((host, port))
        threading.Thread(target=self._accept, name="transport", daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._clients.append(conn)
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn: socket.socket):
        buf = b""
        try:
            while chunk := conn.recv(4096):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        try:
                            self.on_command(json.loads(line))
                        except Exception as e:  # un comando roto no puede tirar la conexión
                            self.publish({"type": "error", "msg": f"comando inválido: {e!r}"})
        except OSError:
            pass
        finally:
            self._drop(conn)

    def _drop(self, conn):
        with self._lock:
            if conn in self._clients:
                self._clients.remove(conn)
        conn.close()

    def publish(self, event: dict):
        data = (json.dumps(event, ensure_ascii=False) + "\n").encode()
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                c.sendall(data)
            except OSError:
                self._drop(c)

    def close(self):
        self._srv.close()
