"""How the evaluation process and the model process talk: frames over one local TCP connection.

A message is a JSON-like tree (dict, list, str, int, float, bool, None) that may hold NumPy arrays. A frame is

    8 bytes   length of the header (big-endian)
    header    UTF-8 JSON; every array is replaced by {"__array__": index, "dtype": ..., "shape": [...]}
    then for each array: 8 bytes length, the array's C-ordered bytes

Standard library and NumPy only, Python 3.8 and later, so that no model environment needs a package for it.
The connection is local to one machine (127.0.0.1); a model and its simulator always run on the same host.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, List, Tuple

import numpy as np

_LENGTH = struct.Struct(">Q")


def _encode(node: Any, arrays: List[np.ndarray]) -> Any:
    if isinstance(node, np.ndarray):
        arrays.append(np.ascontiguousarray(node))
        return {"__array__": len(arrays) - 1, "dtype": node.dtype.str, "shape": list(node.shape)}
    if isinstance(node, np.generic):
        return node.item()
    if isinstance(node, dict):
        return {str(key): _encode(value, arrays) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_encode(value, arrays) for value in node]
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    raise TypeError("cannot send a %s" % type(node).__name__)


def _decode(node: Any, arrays: List[np.ndarray]) -> Any:
    if isinstance(node, dict):
        if "__array__" in node:
            return arrays[node["__array__"]]
        return {key: _decode(value, arrays) for key, value in node.items()}
    if isinstance(node, list):
        return [_decode(value, arrays) for value in node]
    return node


def pack(message: Any) -> bytes:
    arrays: List[np.ndarray] = []
    header = json.dumps(_encode(message, arrays), allow_nan=True).encode("utf-8")
    parts = [_LENGTH.pack(len(header)), header]
    for array in arrays:
        data = array.tobytes()
        parts += [_LENGTH.pack(len(data)), data]
    return b"".join(parts)


def _read(connection: socket.socket, count: int) -> bytes:
    chunks, left = [], count
    while left:
        chunk = connection.recv(min(left, 1 << 20))
        if not chunk:
            raise ConnectionError("the other side closed the connection")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def send(connection: socket.socket, message: Any) -> None:
    connection.sendall(pack(message))


def receive(connection: socket.socket) -> Any:
    (length,) = _LENGTH.unpack(_read(connection, _LENGTH.size))
    header = json.loads(_read(connection, length).decode("utf-8"))
    specs: List[Tuple[int, str, List[int]]] = []

    def find(node: Any) -> None:
        if isinstance(node, dict):
            if "__array__" in node:
                specs.append((node["__array__"], node["dtype"], node["shape"]))
            else:
                for value in node.values():
                    find(value)
        elif isinstance(node, list):
            for value in node:
                find(value)

    find(header)
    arrays: List[Any] = [None] * len(specs)
    for index, dtype, shape in sorted(specs):
        (size,) = _LENGTH.unpack(_read(connection, _LENGTH.size))
        arrays[index] = np.frombuffer(_read(connection, size), dtype=np.dtype(dtype)).reshape(shape).copy()
    return _decode(header, arrays)


def call(connection: socket.socket, kind: str, **body: Any) -> Dict[str, Any]:
    """One request and its answer. The model process answers {"ok": True, ...} or {"ok": False, "error": text}."""
    send(connection, dict(body, kind=kind))
    answer = receive(connection)
    if not answer.get("ok"):
        raise RuntimeError("the model process failed on %r:\n%s" % (kind, answer.get("error")))
    return answer
