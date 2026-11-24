"""The model process: load one row's adapter in the row's environment and answer the evaluation's queries.

    <the row's python> -m evaluation.rows.serve --row <row> --port-file <file>

Started by `run.py`, not by hand. It imports `evaluation.rows.<row>.adapter`, calls its `load(row)`, binds a free
local port, writes the port to `--port-file` when the model is ready, and serves one evaluation process:

    report         -> Adapter.prepare(the block's instructions), then the adapter's report (checked for the
                      keys the config record needs)
    begin_episode  -> Adapter.begin_episode(Episode)
    act            -> Adapter.act(Observation) as Actions
    close          -> the end of one evaluation process; the model stays loaded for the next
    shutdown       -> Adapter.close(), then exit (sent by `run.py` when its blocks are done)

Python 3.8 and later; beyond the adapter's own imports it needs NumPy only.
"""

from __future__ import annotations

import argparse
import importlib
import os
import socket
import traceback
from pathlib import Path

from evaluation import interface
from evaluation import layout
from evaluation import wire


def answer(adapter: interface.Adapter, message: dict) -> dict:
    kind = message["kind"]
    if kind == "report":
        adapter.prepare([str(text) for text in message.get("instructions") or []])
        report = adapter.report()
        problems = interface.check_report(report)
        if problems:
            raise ValueError("the adapter's report is incomplete: " + "; ".join(problems))
        return {"report": report}
    if kind == "begin_episode":
        adapter.begin_episode(interface.Episode(**message["episode"]))
        return {}
    if kind == "act":
        actions = adapter.act(interface.Observation.from_wire(message["observation"]))
        actions.check()
        return actions.to_wire()
    raise ValueError("unknown request %r" % kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--row", required=True)
    parser.add_argument("--port-file", required=True)
    args = parser.parse_args()

    row = layout.load_row(args.row)
    adapter = importlib.import_module("evaluation.rows.%s.adapter" % args.row).load(row)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    temporary = Path(args.port_file + ".tmp")
    temporary.write_text(str(listener.getsockname()[1]))
    os.replace(str(temporary), args.port_file)  # the file appears only when the model is loaded

    try:
        serve(adapter, listener)
    finally:
        listener.close()


def serve(adapter: interface.Adapter, listener: socket.socket) -> None:
    """Serves one evaluation process after another on `listener` (one launch may run several blocks of the row
    with one loaded model), until a connection asks for `shutdown`.

    `close` ends a connection and leaves the model loaded; `shutdown` calls `Adapter.close()` and returns.
    """
    while True:
        connection, _ = listener.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                try:
                    message = wire.receive(connection)
                except ConnectionError:  # the evaluation process went away without a word
                    break
                if message["kind"] in ("close", "shutdown"):
                    if message["kind"] == "shutdown":
                        adapter.close()
                    wire.send(connection, {"ok": True})
                    if message["kind"] == "shutdown":
                        return
                    break
                try:
                    wire.send(connection, dict(answer(adapter, message), ok=True))
                except Exception:  # the evaluation decides what a failed query means for the episode
                    wire.send(connection, {"ok": False, "error": traceback.format_exc()})
        finally:
            connection.close()


if __name__ == "__main__":
    main()
