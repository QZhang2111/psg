"""The machine check: on a machine file and weights laid out in a temporary folder. No model, no card."""

from __future__ import annotations

import json
import sys

from evaluation import check_machine
from evaluation import layout


def machine(tmp_path, monkeypatch, leave_out=()):
    """A machine file naming this Python for every environment, and every row's weights laid out on disk."""
    variables = {"OPENPI_DATA_HOME": str(tmp_path / "openpi"), "HF_HOME": str(tmp_path / "hf")}
    environments = {layout.load_row(r)["environment"] for r in layout.rows()}
    environments |= {layout.load_block(b)["environment"] for b in layout.blocks()}
    table = {name: {"python": sys.executable, "env": dict(variables)} for name in environments if name not in leave_out}
    (tmp_path / "machine.json").write_text(json.dumps(table))
    monkeypatch.setenv("PSG_ENVIRONMENTS", str(tmp_path / "machine.json"))
    for row in layout.rows():
        for checkpoint in check_machine.checkpoints(layout.load_row(row)):
            path = check_machine.checkpoint_path(checkpoint, variables)
            if checkpoint.get("file"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"w")
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / "params").write_bytes(b"w")


def failures(rows=None, blocks=None):
    report = check_machine.check(layout.rows() if rows is None else rows, layout.blocks() if blocks is None else blocks)
    return [(title, line) for title, checks in report for good, line in checks if not good]


def test_a_complete_machine_passes(tmp_path, monkeypatch):
    machine(tmp_path, monkeypatch)
    assert failures() == []
    assert check_machine.main([]) == 0


def test_the_openpi_environments_keep_the_deterministic_operations(tmp_path, monkeypatch):
    machine(tmp_path, monkeypatch)
    lines = [line for _, checks in check_machine.check(["pi05"], []) for _, line in checks]
    assert any("--xla_gpu_deterministic_ops=true" in line for line in lines)
    assert any(line.startswith("TF_CUDNN_DETERMINISTIC=1") for line in lines)


def test_missing_weights_fail_their_row_only(tmp_path, monkeypatch):
    machine(tmp_path, monkeypatch)
    checkpoint = layout.load_row("pi05_rss")["checkpoint"]
    path = check_machine.checkpoint_path(checkpoint, {"HF_HOME": str(tmp_path / "hf")})
    (path / "params").unlink()
    assert [title for title, _ in failures()] == ["row pi05_rss: weights"]
    assert check_machine.main(["--row", "pi05_rss"]) == 1


def test_an_environment_without_an_entry_fails(tmp_path, monkeypatch):
    machine(tmp_path, monkeypatch, leave_out=("apt",))
    titles = {title for title, _ in failures(rows=["apt"], blocks=[])}
    assert titles == {"environment apt (row apt)", "row apt: weights"}


def test_a_missing_path_in_the_machine_entry_fails(tmp_path, monkeypatch):
    machine(tmp_path, monkeypatch)
    table = json.loads((tmp_path / "machine.json").read_text())
    table["sim-libero-cf"]["env"]["LIBERO_CONFIG_PATH"] = str(tmp_path / "nowhere")
    (tmp_path / "machine.json").write_text(json.dumps(table))
    assert failures(rows=[], blocks=["CF-O"]) == [("environment sim-libero-cf (block CF-O)", "LIBERO_CONFIG_PATH: %s" % (tmp_path / "nowhere"))]
