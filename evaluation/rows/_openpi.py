"""What the rows on openpi's models share: the release's LIBERO input, its policy object, the paired noise.

The release is openpi's own LIBERO evaluation (`examples/libero/main.py`): both camera images rotated by 180
degrees and resized with padding to 224 px, an 8-D state (end-effector position, axis-angle orientation, gripper
joints), the instruction as the prompt, the first five actions of each chunk executed. The policy object is
built by openpi's `create_trained_policy`, unchanged.

Runs in the `openpi` environment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

from evaluation import interface

IMAGE_SIZE = 224  # the release resizes each camera image to this, with padding
EXECUTE = 5  # actions of each chunk executed before the next query
NOISE_DIM = 32  # the models' padded action dimension
OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"  # upstream openpi, as the environment is built


def query_noise(episode: interface.Episode, query: int, horizon: int) -> np.ndarray:
    """The initial noise of one query, float32 [horizon, 32], drawn from the shared rule's generator
    (`interface.noise_generator`): the rows on these models get the same noise in the same episode and query, so
    that they differ by their method only."""
    return interface.noise_generator(episode, query).standard_normal((int(horizon), NOISE_DIM)).astype(np.float32)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> axis-angle, as openpi's LIBERO example converts it."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def request(observation: interface.Observation) -> Dict[str, Any]:
    """The shared observation as openpi's LIBERO policy expects it."""
    from openpi_client import image_tools

    def image(name: str) -> np.ndarray:
        upright = np.ascontiguousarray(observation.images[name][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(upright, IMAGE_SIZE, IMAGE_SIZE))

    state = observation.state
    return {
        "observation/image": image("main"),
        "observation/wrist_image": image("wrist"),
        "observation/state": np.concatenate(
            (state["eef_pos"], quat2axisangle(state["eef_quat"]), state["gripper_qpos"])
        ),
        "prompt": observation.instruction,
    }


def tree_sha256(directory: Path) -> str:
    """One digest for a checkpoint directory: sha256 over the lines '<file sha256>  ./<relative path>' of every
    file, sorted by path (`find . -type f | sort | xargs sha256sum | sha256sum`)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        file_digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                file_digest.update(chunk)
        digest.update(f"{file_digest.hexdigest()}  ./{path.relative_to(directory).as_posix()}\n".encode())
    return digest.hexdigest()


class OpenpiRow(interface.Adapter):
    """A released openpi LIBERO checkpoint, run as released. Subclasses add a method or a setting."""

    #: False: z-score normalisation where openpi's config would choose quantiles (see the π0-FAST row).
    use_quantile_norm: Optional[bool] = None
    noise_seed_controls = (
        "The initial noise of every query, standard_normal((horizon, 32)) as float32 from "
        + interface.NOISE_RULE
        + "."
    )

    def __init__(self, row: Dict[str, Any]) -> None:
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_config

        self.row = row
        checkpoint = row["checkpoint"]
        self.train_config = openpi_config.get_config(checkpoint["openpi_config"])
        if self.use_quantile_norm is not None:
            self.train_config = _with_normalisation(self.train_config, self.use_quantile_norm)
        self.directory = Path(download.maybe_download(checkpoint["address"]))
        self.tree_sha256 = tree_sha256(self.directory)
        if self.tree_sha256 != checkpoint["tree_sha256"]:
            raise SystemExit(
                f"the checkpoint at {checkpoint['address']} has tree sha256 {self.tree_sha256} on this machine; "
                f"row.json expects {checkpoint['tree_sha256']}"
            )
        self.policy = policy_config.create_trained_policy(self.train_config, self.directory, default_prompt=None)
        self.horizon = int(self.train_config.model.action_horizon)
        self.episode: Optional[interface.Episode] = None

    def begin_episode(self, episode: interface.Episode) -> None:
        self.episode = episode

    def act(self, observation: interface.Observation) -> interface.Actions:
        noise = query_noise(self.episode, observation.query, self.horizon)
        actions = np.asarray(self.policy.infer(request(observation), noise=noise)["actions"])
        return interface.Actions(actions, EXECUTE)

    # ------------------------------------------------------------------------------------------- the report

    def sampler_settings(self) -> Dict[str, Any]:
        """The keyword defaults of the model's own sampler: what is in effect when nothing is passed."""
        signature = inspect.signature(type(self.policy._model).sample_actions)
        return {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and isinstance(parameter.default, (int, float, str, bool))
        }

    def inference(self) -> Dict[str, Any]:
        model = self.train_config.model
        data = self.train_config.data.create(self.train_config.assets_dirs, model)
        return {
            "openpi_config": self.train_config.name,
            "model_type": str(model.model_type),
            "action_horizon": self.horizon,
            "action_dim": int(model.action_dim),
            "max_token_len": int(model.max_token_len),
            "sampler": self.sampler_settings(),
            "actions_executed_per_query": EXECUTE,
            "images": f"main and wrist, each rotated by 180 degrees, resized with padding to {IMAGE_SIZE} x "
            f"{IMAGE_SIZE}, uint8; a third image slot, if the model has one, is masked",
            "state": "end-effector position, axis-angle orientation, gripper joints (8 numbers), passed to "
            "openpi's transforms; "
            + (
                "the model's token sequence does not contain it (discrete_state_input = false)"
                if getattr(model, "pi05", False) and not getattr(model, "discrete_state_input", True)
                else "read by the model as its configuration defines"
            ),
            "prompt": "the instruction as given, tokenised by openpi's transforms",
            "normalisation": ("quantile" if data.use_quantile_norm else "z-score") + ", with the checkpoint's statistics",
            "precision": "parameters restored as bfloat16 by openpi's create_trained_policy",
            # Read from the process, not assumed: with deterministic GPU operations an episode repeats from
            # launch to launch on one card; without them it does not (`evaluation/environments/openpi.json`, "why_the_variables").
            "gpu_operations": {
                "deterministic": "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""),
                "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
                "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC", ""),
            },
        }

    def method(self) -> Dict[str, Any]:
        return {"name": "none"}

    def report(self) -> Dict[str, Any]:
        from importlib import metadata

        packages = {}
        for name in ("jax", "flax", "numpy", "openpi", "openpi-client", "sentencepiece"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = "not installed as a distribution"
        checkpoint = self.row["checkpoint"]
        return {
            "model": {
                "name": self.row["name"],
                "checkpoint": {
                    "address": checkpoint["address"],
                    "revision": "none: the address has no revisions; the digest identifies the files",
                    "sha256": {"tree": self.tree_sha256, "computed_as": tree_sha256.__doc__.split("\n\n")[0]},
                    "licence": checkpoint.get("licence", "not stated"),
                },
                # Declared, not read from the installation: the address and the commit the environment is built
                # from (evaluation/environments/openpi.json). This repository's own commit is the record's `code`.
                "code": {
                    "address": "github.com/Physical-Intelligence/openpi, installed in the row's environment",
                    "revision": OPENPI_REVISION + " (as the environment's declaration pins it; not read from "
                    "the installation)",
                },
            },
            "inference": self.inference(),
            "method": self.method(),
            "noise_seed_controls": self.noise_seed_controls,
            "environment": {"name": self.row["environment"], "python": sys.version.split()[0], "packages": packages},
            "requires": {
                "camera_height": interface.DEFAULT_CAMERA_SIZE,
                "camera_width": interface.DEFAULT_CAMERA_SIZE,
            },
        }


def _with_normalisation(train_config: Any, use_quantile_norm: bool) -> Any:
    """The same openpi training config with its data factory's normalisation kind replaced."""
    factory = train_config.data

    @dataclasses.dataclass(frozen=True)
    class Factory(type(factory)):  # type: ignore[misc]
        def create(self, assets_dirs: Any, model_config: Any) -> Any:
            return dataclasses.replace(super().create(assets_dirs, model_config), use_quantile_norm=use_quantile_norm)

    fields = {field.name: getattr(factory, field.name) for field in dataclasses.fields(factory)}
    return dataclasses.replace(train_config, data=Factory(**fields))
